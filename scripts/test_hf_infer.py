import argparse
import json
import os


def _load_system_text(path: str) -> str:
    if not path:
        return ""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    desc = ""
    props = data.get("properties") if isinstance(data.get("properties"), dict) else {}
    if props:
        desc = props.get("description", "") or ""
    name = data.get("name", "") or ""
    aliases = data.get("aliases", []) or []
    parts = []
    if name:
        parts.append(f"System name: {name}.")
    if aliases:
        parts.append(f"Aliases: {', '.join(aliases)}.")
    if desc:
        parts.append(f"Description: {desc}.")
    return " ".join(parts)


def _build_messages(user_text: str, system_text: str) -> list[dict]:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Test HF model inference with chat template.")
    parser.add_argument("--model", required=True, help="HF model path or id.")
    parser.add_argument("--prompt", required=True, help="User prompt.")
    parser.add_argument(
        "--system-entity",
        default="",
        help="Optional path to system_entity.json to build a system prompt.",
    )
    parser.add_argument(
        "--system-text",
        default="",
        help="Optional explicit system prompt string (overrides --system-entity).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.system_text:
        system_text = args.system_text
    elif args.system_entity:
        system_text = _load_system_text(args.system_entity)
    else:
        system_text = ""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        raise SystemExit(f"[error] missing deps: {e}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    messages = _build_messages(args.prompt, system_text)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise SystemExit("[error] tokenizer does not support apply_chat_template")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.temperature and args.temperature > 0),
        temperature=float(args.temperature),
    )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    print(decoded.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
