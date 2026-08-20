import argparse
import inspect
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from common.config import load_settings

sys.path.append(os.path.dirname(__file__))
from build_training_datasets import build_datasets  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json_dataset(path: str):
    from datasets import load_dataset

    return load_dataset("json", data_files=path, split="train")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_manifest(path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _safe_save_pretrained(model: object, output_dir: str, safe_serialization: bool) -> None:
    kwargs: Dict[str, Any] = {}
    if safe_serialization:
        try:
            save_params = model.save_pretrained.__code__.co_varnames  # type: ignore[attr-defined]
            if "safe_serialization" in save_params:
                kwargs["safe_serialization"] = True
        except Exception:
            kwargs = {}
    try:
        model.save_pretrained(output_dir, **kwargs)  # type: ignore[attr-defined]
    except TypeError:
        if kwargs:
            model.save_pretrained(output_dir)  # type: ignore[attr-defined]
        else:
            raise


def _build_training_args(cls: object, **kwargs: Any) -> object:
    try:
        sig = inspect.signature(cls.__init__)  # type: ignore[attr-defined]
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        if "evaluation_strategy" not in allowed and "evaluate_during_training" in allowed:
            if "eval_steps" in filtered or kwargs.get("evaluation_strategy"):
                filtered["evaluate_during_training"] = True
        return cls(**filtered)  # type: ignore[misc]
    except Exception:
        return cls(**kwargs)  # type: ignore[misc]


def _parse_lora_targets(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _supports_target_module(model: object, target: str) -> bool:
    try:
        for name, _ in model.named_modules():  # type: ignore[attr-defined]
            if name.endswith(target):
                return True
    except Exception:
        return False
    return False


def _infer_lora_targets(model: object) -> List[str]:
    config = getattr(model, "config", None)
    model_type = ""
    if config is not None:
        model_type = str(getattr(config, "model_type", "") or "").lower()
        if not model_type:
            archs = getattr(config, "architectures", None)
            if isinstance(archs, (list, tuple)) and archs:
                model_type = str(archs[0]).lower()

    candidates: List[str] = []
    if "qwen" in model_type or "llama" in model_type or "mistral" in model_type or "gemma" in model_type:
        candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "phi3" in model_type or "phi" in model_type:
        candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "gpt_neox" in model_type:
        candidates = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "gpt2" in model_type:
        candidates = ["c_attn", "c_proj", "c_fc"]

    if not candidates:
        return []
    return [name for name in candidates if _supports_target_module(model, name)]


def _train_router(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    label_source: str,
    max_length: int,
    epochs: float,
    batch_size: int,
) -> Dict[str, Any]:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    data = _load_json_dataset(dataset_path)

    def assign_label(example: Dict[str, Any]) -> Dict[str, Any]:
        label = example.get("label")
        if label_source == "router":
            label = example.get("label_router")
        if label not in ("GENERAL", "AGENT"):
            return {"labels": None}
        return {"labels": 1 if label == "AGENT" else 0}

    data = data.map(assign_label)
    data = data.filter(lambda x: x["labels"] is not None)
    if len(data) < 2:
        raise RuntimeError("Not enough labeled router samples.")

    split = data.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    eval_data = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    args = _build_training_args(
        TrainingArguments,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {"output_dir": output_dir}


def _train_sft(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    finetune_type: str,
    max_length: int,
    epochs: float,
    batch_size: int,
    merge_lora: bool = False,
    merged_output_dir: Optional[str] = None,
    merged_safetensors: bool = False,
    lora_target_modules: Optional[List[str]] = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_chat_template: bool = False,
) -> Dict[str, Any]:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    finetune_type = finetune_type.lower()
    if finetune_type == "fst":
        finetune_type = "sft"

    data = _load_json_dataset(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    has_chat_template = hasattr(tokenizer, "apply_chat_template")

    def format_text(example: Dict[str, Any]) -> Dict[str, Any]:
        system = (example.get("system") or "").strip()
        prompt = example.get("prompt") or ""
        response = example.get("response") or ""
        if use_chat_template and has_chat_template:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            if system:
                text = system + "\n" + prompt + response
            else:
                text = prompt + response
        return {"text": text}

    data = data.map(format_text)
    if len(data) < 2:
        raise RuntimeError("Not enough SFT samples.")

    split = data.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"]
    eval_data = split["test"]

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    is_lora = finetune_type == "lora"
    if finetune_type == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        targets = list(lora_target_modules or [])
        if not targets:
            targets = _infer_lora_targets(model)
        if not targets:
            raise RuntimeError(
                "LoRA target modules could not be inferred. "
                "Pass --lora-target-modules (comma-separated), e.g. "
                "--lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
            )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_config)

    args = _build_training_args(
        TrainingArguments,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    merged_dir = None
    if merge_lora and not is_lora:
        print("[warn] --merge-lora ignored because finetune-type is not lora.")
    if merge_lora and is_lora:
        merged_dir = merged_output_dir or f"{output_dir}_merged"
        merged = model.merge_and_unload()
        _ensure_dir(merged_dir)
        _safe_save_pretrained(merged, merged_dir, merged_safetensors)
        tokenizer.save_pretrained(merged_dir)
    return {
        "output_dir": output_dir,
        "merged_dir": merged_dir,
        "is_lora": is_lora,
        "lora_target_modules": lora_target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_chat_template": use_chat_template,
    }


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Train router or SFT models from logs.")
    parser.add_argument("--task", choices=["router", "sft"], required=True)
    parser.add_argument("--dataset-dir", default=os.path.join(settings.workspace_root, "evolve", "datasets"))
    parser.add_argument("--output-dir", default=os.path.join(settings.workspace_root, "evolve", "output"))
    parser.add_argument("--base-model", default="")
    parser.add_argument("--mode", choices=["general", "agent", "complex"], default="general")
    parser.add_argument("--finetune-type", default=settings.evolve.get("finetune_type", "sft"))
    parser.add_argument("--label-source", choices=["tool", "router"], default="tool")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--build", action="store_true", help="Build datasets before training.")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA into base weights after training.")
    parser.add_argument("--merged-output-dir", default="", help="Override merged output directory.")
    parser.add_argument(
        "--merged-safetensors",
        action="store_true",
        help="Save merged weights as safetensors when supported.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="",
        help="Comma-separated module names for LoRA injection (e.g., q_proj,k_proj,v_proj,o_proj).",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8).")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16).")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05).")
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Format SFT samples using tokenizer.apply_chat_template when available.",
    )
    args = parser.parse_args()

    counts = None
    if args.build:
        counts = build_datasets(
            record_path=os.path.join(settings.data_dir, "record.log"),
            out_dir=args.dataset_dir,
        )

    _ensure_dir(args.output_dir)

    if args.task == "router":
        dataset_path = os.path.join(args.dataset_dir, "router.jsonl")
        if not args.base_model:
            raise RuntimeError("Router training requires --base-model")
        result = _train_router(
            dataset_path=dataset_path,
            base_model=args.base_model,
            output_dir=os.path.join(args.output_dir, "router"),
            label_source=args.label_source,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        manifest = {
            "task": "router",
            "base_model": args.base_model,
            "dataset_path": dataset_path,
            "label_source": args.label_source,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "output_dir": result["output_dir"],
            "dataset_counts": counts,
            "created_at": _now_iso(),
        }
        _write_manifest(os.path.join(result["output_dir"], "manifest.json"), manifest)
        return

    if args.task == "sft":
        if not args.base_model:
            raise RuntimeError("SFT training requires --base-model")
        dataset_file = f"sft_{args.mode}.jsonl"
        dataset_path = os.path.join(args.dataset_dir, dataset_file)
        merged_output_dir = args.merged_output_dir or None
        lora_targets = _parse_lora_targets(args.lora_target_modules)
        result = _train_sft(
            dataset_path=dataset_path,
            base_model=args.base_model,
            output_dir=os.path.join(args.output_dir, f"sft_{args.mode}"),
            finetune_type=args.finetune_type,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            merge_lora=args.merge_lora,
            merged_output_dir=merged_output_dir,
            merged_safetensors=args.merged_safetensors,
            lora_target_modules=lora_targets,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_chat_template=args.use_chat_template,
        )
        manifest = {
            "task": "sft",
            "mode": args.mode,
            "finetune_type": args.finetune_type,
            "base_model": args.base_model,
            "dataset_path": dataset_path,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "output_dir": result["output_dir"],
            "merged_dir": result["merged_dir"],
            "lora": {
                "target_modules": result["lora_target_modules"],
                "r": result["lora_r"],
                "alpha": result["lora_alpha"],
                "dropout": result["lora_dropout"],
            }
            if result["is_lora"]
            else None,
            "use_chat_template": result["use_chat_template"],
            "dataset_counts": counts,
            "created_at": _now_iso(),
        }
        _write_manifest(os.path.join(result["output_dir"], "manifest.json"), manifest)
        return


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        raise SystemExit(
            "Missing training dependencies. Install transformers/datasets/peft/accelerate and torch."
        ) from e
