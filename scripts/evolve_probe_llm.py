import argparse
import json
import os
import sys
from typing import Any, Dict, List


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from common.config import load_settings
from model_server.hf_client import HfError
from model_server.llamacpp_client import LlamacppError
from model_server.local_generation import generate_local_text
from model_server.ollama_client import OllamaError


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("No valid JSON object found in output.")


def _build_prompt(payload: Dict[str, Any], max_symbols: int, max_files: int, enable_thinking: bool) -> str:
    target = payload.get("target", {})
    path = target.get("path", "")
    start_line = target.get("start_line", 0)
    end_line = target.get("end_line", 0)
    content = target.get("content", "")
    objectives = payload.get("objectives", {})
    objectives_block = "\n".join([f"- {k}: {v}" for k, v in objectives.items()]) if objectives else "None"

    prompt = (
        "You are an expert software engineer. Identify what extra context is needed to improve the target. "
        "Return JSON only. Do not include markdown fences or commentary.\n\n"
        "Return up to:\n"
        f"- {max_symbols} symbols (functions/classes/constants)\n"
        f"- {max_files} file paths (relative to repo root)\n\n"
        "Only request symbols/files that are necessary to understand or improve the target.\n\n"
        f"Objectives:\n{objectives_block}\n\n"
        f"Target path: {path}\n"
        f"Target lines: {start_line}-{end_line}\n"
        "Target content:\n"
        f"{content}\n\n"
        "Output JSON format:\n"
        "{\"symbols\": [\"Foo\", \"bar\"], \"files\": [\"src/x.py\"], \"notes\": \"...\"}"
    )
    if enable_thinking:
        return "/think\n" + prompt
    return prompt


def _build_local_probe_output(reason: str) -> Dict[str, Any]:
    note = f"local_fallback: context probe backend unavailable: {reason}"
    return {"symbols": [], "files": [], "notes": note}


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM probe for context needs.")
    parser.add_argument("--model", default="", help="Override model id for generation.")
    parser.add_argument("--provider", default="", choices=["", "ollama", "hf", "llamacpp"], help="Override provider.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=800, help="Max new tokens.")
    parser.add_argument("--enable-thinking", action="store_true", help="Prepend /think to the prompt.")
    args = parser.parse_args()

    payload_text = sys.stdin.read()
    if not payload_text.strip():
        raise SystemExit("[error] expected JSON payload on stdin")
    payload = json.loads(payload_text)
    max_symbols = int(payload.get("max_symbols") or 10)
    max_files = int(payload.get("max_files") or 4)

    settings = load_settings()
    provider = args.provider or str(settings.models.get("provider", "ollama")).lower()
    model_id = args.model or settings.evolve.get("evolution_model") or settings.models.get("text_model", "")
    if not model_id:
        sys.stdout.write(json.dumps(_build_local_probe_output("no model id configured"), ensure_ascii=True))
        return 0

    prompt = _build_prompt(payload, max_symbols=max_symbols, max_files=max_files, enable_thinking=args.enable_thinking)
    try:
        output = generate_local_text(
            prompt,
            model_id,
            provider=provider,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            system="Return only valid JSON as instructed.",
            response_format="json",
        )
        data = _extract_json(output)
    except (OllamaError, HfError, LlamacppError, ValueError) as e:
        sys.stdout.write(json.dumps(_build_local_probe_output(str(e)), ensure_ascii=True))
        return 0
    symbols = data.get("symbols", [])
    files = data.get("files", [])
    notes = data.get("notes", "")
    if not isinstance(symbols, list):
        symbols = []
    if not isinstance(files, list):
        files = []
    out = {"symbols": symbols[:max_symbols], "files": files[:max_files], "notes": notes}
    sys.stdout.write(json.dumps(out, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
