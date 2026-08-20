import argparse
import json
import os
import time


def _resolve_dtype(value: str):
    if not value or value == "auto":
        return "auto"
    try:
        import torch
    except Exception:
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(value).lower(), "auto")


def _safe_save_pretrained(model: object, output_dir: str, safe_serialization: bool) -> None:
    kwargs = {}
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model", required=True, help="HF base model id or local path.")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory.")
    parser.add_argument("--out", required=True, help="Output directory for merged weights.")
    parser.add_argument("--safetensors", action="store_true", help="Save merged weights as safetensors.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading the base model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for loading (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="dtype override (auto/float16/bfloat16/float32).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.adapter):
        print(f"[error] adapter not found: {args.adapter}")
        return 2

    os.makedirs(args.out, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        print(f"[error] missing deps: {e}")
        print("Install: pip install -U transformers peft")
        return 1

    dtype = _resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    merged = model.merge_and_unload()

    _safe_save_pretrained(merged, args.out, args.safetensors)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=bool(args.trust_remote_code))
    tokenizer.save_pretrained(args.out)

    manifest = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "output_dir": args.out,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "safe_serialization": bool(args.safetensors),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f"[ok] merged weights saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
