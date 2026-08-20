import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from common.config import load_settings


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _abs_if_exists(path: str) -> str:
    if not path:
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return path


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a model or adapter for Ollama.")
    parser.add_argument("--name", required=True, help="Ollama model name to create.")
    parser.add_argument("--from-model", default="", help="Base model name or path for Modelfile FROM.")
    parser.add_argument("--adapter", default="", help="Path to LoRA adapter directory or GGUF adapter.")
    parser.add_argument("--model-dir", default="", help="Path to merged/full model directory.")
    parser.add_argument("--gguf", default="", help="Path to a GGUF model file.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to write Modelfile and manifest (default: evolve/output/ollama).",
    )
    parser.add_argument("--quantize", default="", help="Quantization type for ollama create --quantize.")
    parser.add_argument("--create", action="store_true", help="Run ollama create after writing Modelfile.")
    parser.add_argument("--timestamped", action="store_true", help="Write outputs into a timestamped subdir.")
    args = parser.parse_args()

    if args.gguf and args.model_dir:
        print("[error] Use only one of --gguf or --model-dir.")
        return 2
    if args.adapter and (args.gguf or args.model_dir):
        print("[error] --adapter cannot be combined with --gguf or --model-dir.")
        return 2
    if args.adapter and not args.from_model:
        print("[error] --adapter requires --from-model.")
        return 2
    if not args.adapter and not args.gguf and not args.model_dir and not args.from_model:
        print("[error] Provide --from-model, --model-dir, or --gguf.")
        return 2

    adapter_path = _abs_if_exists(args.adapter)
    gguf_path = _abs_if_exists(args.gguf)
    model_dir = _abs_if_exists(args.model_dir)
    from_model = _abs_if_exists(args.from_model)

    if adapter_path and not os.path.exists(adapter_path):
        print(f"[error] adapter not found: {adapter_path}")
        return 2
    if gguf_path and not os.path.exists(gguf_path):
        print(f"[error] gguf not found: {gguf_path}")
        return 2
    if model_dir and not os.path.exists(model_dir):
        print(f"[error] model dir not found: {model_dir}")
        return 2

    if gguf_path:
        from_line = gguf_path
    elif model_dir:
        from_line = model_dir
    else:
        from_line = from_model

    settings = load_settings()
    base_out = args.output_dir or os.path.join(settings.workspace_root, "evolve", "output", "ollama")
    target_dir = os.path.join(base_out, _safe_name(args.name))
    if args.timestamped:
        target_dir = os.path.join(target_dir, _now_stamp())
    os.makedirs(target_dir, exist_ok=True)

    modelfile_path = os.path.join(target_dir, "Modelfile")
    lines = [f"FROM {from_line}"]
    if adapter_path:
        lines.append(f"ADAPTER {adapter_path}")
    modelfile_text = "\n".join(lines) + "\n"
    _write_text(modelfile_path, modelfile_text)

    cmd = ["ollama", "create", args.name, "-f", modelfile_path]
    if args.quantize:
        cmd.extend(["--quantize", args.quantize])
    command_text = " ".join(cmd)

    manifest_path = os.path.join(target_dir, "manifest.json")
    manifest = {
        "name": args.name,
        "from": from_line,
        "adapter": adapter_path or None,
        "model_dir": model_dir or None,
        "gguf": gguf_path or None,
        "quantize": args.quantize or None,
        "modelfile": modelfile_path,
        "create_command": command_text,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_text(manifest_path, json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")

    print(f"[ok] wrote {modelfile_path}")
    print(f"[ok] wrote {manifest_path}")
    print(f"[cmd] {command_text}")

    if args.create:
        try:
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except FileNotFoundError:
            print("[error] ollama not found on PATH.")
            return 127
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
