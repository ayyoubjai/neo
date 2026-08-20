import argparse
import ast
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from common.config import load_settings
from model_server.hf_client import generate_text as hf_generate_text
from model_server.ollama_client import generate as ollama_generate


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        sliced = text[start : end + 1]
        data = json.loads(sliced)
        if isinstance(data, dict):
            return data
    raise ValueError("No valid JSON object found in LLM output.")


def _json_compatible(value: Any, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_json_compatible(item, depth + 1) for item in value)
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return False
            if not _json_compatible(item, depth + 1):
                return False
        return True
    return False


def _module_from_target_rel(target_rel: str) -> str:
    normalized = _normalize_path(target_rel).lstrip("./")
    if normalized.startswith("src/") and normalized.endswith(".py"):
        module_rel = normalized[len("src/") : -3]
        return module_rel.replace("/", ".")
    return ""


def _build_parent_map(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    parents: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _line_span(node: ast.AST) -> Tuple[int, int]:
    start = int(getattr(node, "lineno", 0) or 0)
    end = int(getattr(node, "end_lineno", start) or start)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            starts = [int(getattr(dec, "lineno", start) or start) for dec in decorators]
            if starts:
                start = min(start, min(starts))
    return start, end


def _smallest_enclosing_object_id(
    repo_root: str,
    target_rel: str,
    start_line: int,
    end_line: int,
) -> str:
    if start_line <= 0:
        return ""
    if end_line <= 0:
        end_line = start_line
    module_name = _module_from_target_rel(target_rel)
    if not module_name:
        return ""
    target_abs = os.path.join(repo_root, _normalize_path(target_rel))
    if not os.path.exists(target_abs):
        return ""
    try:
        with open(target_abs, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return ""

    parents = _build_parent_map(tree)
    candidates: List[Tuple[int, int, ast.AST]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        node_start, node_end = _line_span(node)
        if node_start <= start_line and node_end >= end_line:
            candidates.append((node_start, node_end, node))

    if not candidates:
        fallback: List[Tuple[int, int, ast.AST]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            node_start, node_end = _line_span(node)
            if node_start >= start_line - 2:
                fallback.append((node_start, node_end, node))
        if not fallback:
            return ""
        fallback.sort(key=lambda item: (abs(item[0] - start_line), item[1] - item[0], item[0]))
        _, _, best = fallback[0]
    else:
        candidates.sort(key=lambda item: (item[1] - item[0], item[0]))
        _, _, best = candidates[0]

    parts: List[str] = []
    current: Optional[ast.AST] = best
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(current.name)
        current = parents.get(current)
    parts.reverse()
    if not parts:
        return ""
    return module_name + "." + ".".join(parts)


def _read_target_snippet(repo_root: str, target_rel: str, start_line: int, end_line: int) -> str:
    target_abs = os.path.join(repo_root, _normalize_path(target_rel))
    if not os.path.exists(target_abs):
        return ""
    try:
        with open(target_abs, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return ""
    if not lines:
        return ""
    start = max(1, int(start_line) if start_line > 0 else 1)
    end = int(end_line) if end_line > 0 else len(lines)
    start = min(start, len(lines))
    end = min(max(start, end), len(lines))
    return "".join(lines[start - 1 : end])


def _build_prompt(
    target_rel: str,
    start_line: int,
    end_line: int,
    object_ids: List[str],
    snippet: str,
    max_cases: int,
) -> str:
    objects_block = "\n".join(f"- {obj}" for obj in object_ids) if object_ids else "- (none)"
    return (
        "You generate runtime benchmark scenarios for Python callables.\n"
        "Return JSON only (no markdown).\n\n"
        "Output schema:\n"
        "{"
        "\"scenarios\": ["
        "{"
        "\"object_id\": \"module.path.callable\","
        "\"call\": {\"args\": [...], \"kwargs\": {...}},"
        "\"note\": \"short reason\""
        "}"
        "]"
        "}\n\n"
        "Hard constraints:\n"
        "- object_id must be one of the allowed object ids.\n"
        "- args/kwargs must be JSON-serializable only (null, bool, number, string, list, object).\n"
        "- Do not use placeholders like <...>.\n"
        f"- Return at most {max_cases} scenarios.\n\n"
        f"Target file: {target_rel}\n"
        f"Target lines: {start_line}-{end_line}\n"
        f"Allowed object ids:\n{objects_block}\n\n"
        "Target code snippet:\n"
        f"{snippet or '(snippet unavailable)'}\n\n"
        "Generate realistic scenarios including common and edge cases."
    )


def _normalize_scenarios(
    raw_scenarios: Any,
    allowed_object_ids: List[str],
    max_cases: int,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_scenarios, list):
        return []
    allowed = set(allowed_object_ids)
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for entry in raw_scenarios:
        if not isinstance(entry, dict):
            continue
        object_id = str(entry.get("object_id", "") or "").strip()
        if not object_id:
            continue
        if allowed and object_id not in allowed:
            continue
        call = entry.get("call")
        if not isinstance(call, dict):
            continue
        args = call.get("args", [])
        kwargs = call.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            continue
        if not _json_compatible(args) or not _json_compatible(kwargs):
            continue
        scenario = {
            "object_id": object_id,
            "call": {"args": args, "kwargs": kwargs},
        }
        note = entry.get("note")
        if isinstance(note, str) and note.strip():
            scenario["note"] = note.strip()
        key = json.dumps(scenario, ensure_ascii=True, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(scenario)
        if max_cases > 0 and len(normalized) >= max_cases:
            break
    return normalized


def _resolve_model_and_provider(provider_arg: str, model_arg: str) -> Tuple[str, str]:
    settings = load_settings()
    provider = provider_arg.strip().lower()
    if not provider or provider == "auto":
        provider = str(settings.models.get("provider", "ollama")).lower()
    model = model_arg.strip()
    if not model:
        model = str(settings.evolve.get("evolution_model", "") or "").strip()
    if not model:
        model = str(settings.models.get("text_model", "") or "").strip()
    return provider, model


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JSON-only macro scenarios with an LLM.")
    parser.add_argument("--target", required=True, help="Target file path relative to repo root.")
    parser.add_argument("--target-start-line", type=int, default=0)
    parser.add_argument("--target-end-line", type=int, default=0)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--object-ids", default="", help="Comma-separated callable ids.")
    parser.add_argument("--max-cases", type=int, default=8, help="Maximum scenario count.")
    parser.add_argument("--provider", default="auto", choices=["auto", "ollama", "hf"])
    parser.add_argument("--model", default="", help="Override model id.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output", default="", help="Optional output JSON file path.")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    target_rel = _normalize_path(args.target)
    max_cases = max(1, int(args.max_cases))

    object_ids = [part.strip() for part in args.object_ids.split(",") if part.strip()]
    if not object_ids:
        resolved = _smallest_enclosing_object_id(
            repo_root=repo_root,
            target_rel=target_rel,
            start_line=int(args.target_start_line),
            end_line=int(args.target_end_line),
        )
        if resolved:
            object_ids = [resolved]

    snippet = _read_target_snippet(
        repo_root=repo_root,
        target_rel=target_rel,
        start_line=int(args.target_start_line),
        end_line=int(args.target_end_line),
    )
    prompt = _build_prompt(
        target_rel=target_rel,
        start_line=int(args.target_start_line),
        end_line=int(args.target_end_line),
        object_ids=object_ids,
        snippet=snippet,
        max_cases=max_cases,
    )
    if args.enable_thinking:
        prompt = "/think\n" + prompt

    provider, model = _resolve_model_and_provider(args.provider, args.model)
    if not model:
        raise SystemExit("[error] no model configured (settings.models.text_model or evolve.evolution_model)")

    if provider == "hf":
        output = hf_generate_text(
            prompt,
            model,
            temperature=float(args.temperature),
            max_new_tokens=max(32, int(args.max_new_tokens)),
        )
    else:
        output = ollama_generate(
            prompt,
            model,
            system="Return only strict JSON for the requested schema.",
            options={"temperature": float(args.temperature)},
            response_format="json",
        )

    data = _extract_json(output)
    raw_scenarios = data.get("scenarios")
    if raw_scenarios is None:
        raw_scenarios = data.get("cases", [])
    scenarios = _normalize_scenarios(raw_scenarios, allowed_object_ids=object_ids, max_cases=max_cases)

    payload = {
        "schema_version": 1,
        "kind": "llm_scenarios",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": target_rel,
        "object_ids": object_ids,
        "scenarios": scenarios,
        "count": len(scenarios),
    }

    output_file = args.output.strip()
    if output_file:
        output_path = output_file if os.path.isabs(output_file) else os.path.join(repo_root, output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
