#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_ROOT = Path(__file__).resolve().parent
SYSTEM_ROOT = SCRIPT_ROOT.parent
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common.config import load_settings
from autonomy.cognition_client import CognitionClient, CognitionClientError, CognitionJsonError
from common.soul_mirror import SoulConfig, load_config, sync_soul
from model_server.hf_client import HfError, generate_text as hf_generate_text
from model_server.ollama_client import OllamaError, generate as ollama_generate


DEFAULT_INCLUDE = (
    "README.md",
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "config",
    "docs",
    "scripts",
    "src",
    "tests",
)
DEFAULT_EXCLUDE = (
    "soul/**",
    ".git/**",
    ".venv/**",
    "venv/**",
    "node_modules/**",
    "data/**",
    "workspace/**",
    "evolve/runs/**",
    "evolve/output/**",
    "**/__pycache__/**",
    "**/*.pyc",
)
MAX_PROMPT_CHARS = 60000


@dataclass(frozen=True)
class Snippet:
    path: str
    start_line: int
    end_line: int
    content: str
    score: int


def _repo_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[error] repo root is not a directory: {root}")
    return root


def _load_soul_config(repo_root: Path, output_dir: str) -> SoulConfig:
    config_path = repo_root / "config" / "soul.json"
    if config_path.exists():
        cfg = load_config(config_path)
        if output_dir and output_dir != cfg.output_dir:
            return SoulConfig(
                include=cfg.include,
                exclude=cfg.exclude,
                max_file_bytes=cfg.max_file_bytes,
                output_dir=output_dir,
            )
        return cfg
    include = tuple(item for item in DEFAULT_INCLUDE if (repo_root / item).exists())
    if not include:
        include = (".",)
    return SoulConfig(
        include=include,
        exclude=DEFAULT_EXCLUDE,
        max_file_bytes=384 * 1024,
        output_dir=output_dir,
    )


def _read_text(path: Path, limit: Optional[int] = None) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 20)] + "\n...[truncated]...\n"
    return text


def _load_jsonl(path: Path, limit: int = 5000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _task_terms(task: str) -> List[str]:
    terms = []
    seen = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", task.lower()):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:40]


def _score_path(path: str, terms: Sequence[str]) -> int:
    lowered = path.lower()
    return sum(3 for term in terms if term in lowered)


def _score_text(text: str, terms: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term) for term in terms)


def _iter_candidate_files(repo_root: Path, files_meta: Sequence[Dict[str, Any]], terms: Sequence[str]) -> List[Tuple[int, str]]:
    scored: List[Tuple[int, str]] = []
    for row in files_meta:
        rel_path = str(row.get("path") or row.get("rel_path") or "")
        if not rel_path:
            continue
        if rel_path.endswith((".png", ".jpg", ".jpeg", ".gif", ".pt", ".safetensors", ".bin")):
            continue
        score = _score_path(rel_path, terms)
        abs_path = repo_root / rel_path
        if abs_path.exists() and abs_path.is_file():
            try:
                score += _score_text(_read_text(abs_path, limit=12000), terms)
            except OSError:
                pass
        if score > 0:
            scored.append((score, rel_path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _make_snippet(repo_root: Path, rel_path: str, terms: Sequence[str], max_chars: int) -> Optional[Snippet]:
    path = repo_root / rel_path
    if not path.exists() or not path.is_file():
        return None
    try:
        text = _read_text(path)
    except OSError:
        return None
    if len(text) <= max_chars:
        return Snippet(rel_path, 1, max(1, text.count("\n") + 1), text, _score_text(text, terms))

    lines = text.splitlines()
    best_line = 0
    best_score = -1
    for idx, line in enumerate(lines):
        score = _score_text(line, terms)
        if score > best_score:
            best_score = score
            best_line = idx
    radius = 70
    start = max(0, best_line - radius)
    end = min(len(lines), best_line + radius + 1)
    content = "\n".join(lines[start:end])
    if len(content) > max_chars:
        content = content[: max(0, max_chars - 20)] + "\n...[truncated]...\n"
    return Snippet(rel_path, start + 1, end, content, max(0, best_score))


def _select_snippets(repo_root: Path, soul_root: Path, task: str, explicit_files: Sequence[str], max_files: int) -> List[Snippet]:
    terms = _task_terms(task)
    files_meta = _load_jsonl(soul_root / "meta" / "files.jsonl")
    selected: List[str] = []
    seen = set()
    for rel_path in explicit_files:
        normalized = rel_path.strip().replace("\\", "/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            selected.append(normalized)
    for _, rel_path in _iter_candidate_files(repo_root, files_meta, terms):
        if rel_path not in seen:
            seen.add(rel_path)
            selected.append(rel_path)
        if len(selected) >= max_files:
            break

    snippets: List[Snippet] = []
    for rel_path in selected[:max_files]:
        snippet = _make_snippet(repo_root, rel_path, terms, max_chars=9000)
        if snippet:
            snippets.append(snippet)
    return snippets


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("model output did not contain a valid JSON object")


def _short_git_status(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if proc.returncode != 0:
        return "not a git repository or git status failed"
    return proc.stdout.strip() or "clean"


def _build_prompt(repo_root: Path, soul_root: Path, task: str, snippets: Sequence[Snippet]) -> str:
    overview_path = soul_root / "meta" / "overview.md"
    tree_path = soul_root / "meta" / "tree.txt"
    overview = _read_text(overview_path, limit=12000) if overview_path.exists() else ""
    tree = _read_text(tree_path, limit=12000) if tree_path.exists() else ""
    status = _short_git_status(repo_root)
    snippet_block = "\n\n".join(
        f"[{item.path}:{item.start_line}-{item.end_line}]\n{item.content}" for item in snippets
    )
    prompt = (
        "You are a local coding agent working inside a repository. "
        "Implement the requested change by returning a structured candidate. "
        "Return JSON only. Do not include markdown fences or commentary.\n\n"
        "Candidate schema:\n"
        "{\n"
        '  "summary": "short implementation summary",\n'
        '  "edits": [{"path": "relative/file.py", "before": "exact existing text", "after": "replacement text", "occurrence": 1}],\n'
        '  "files": [{"path": "relative/new_file.py", "content": "full file content"}],\n'
        '  "tests": ["pytest tests/test_x.py"]\n'
        "}\n\n"
        "Rules:\n"
        "- Use edits for existing files and files for new files.\n"
        "- Every edit.before must be an exact substring from the repository snippets.\n"
        "- Keep the change focused on the task.\n"
        "- Do not add dependencies unless the task explicitly requires them.\n"
        "- Do not include shell commands that modify the repo.\n"
        "- If you cannot safely implement it from the provided context, return empty edits/files and explain the missing context in summary.\n\n"
        f"Repository root: {repo_root}\n"
        f"Git status:\n{status}\n\n"
        f"Task:\n{task}\n\n"
        f"Soul overview:\n{overview or 'None'}\n\n"
        f"Repository tree excerpt:\n{tree or 'None'}\n\n"
        f"Relevant snippets:\n{snippet_block or 'None'}\n"
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[: MAX_PROMPT_CHARS - 80] + "\n...[prompt truncated; use focused exact edits]...\n"
    return prompt


def _generate_candidate_direct(prompt: str, model: str, provider: str, temperature: float, max_new_tokens: int) -> Dict[str, Any]:
    if provider == "hf":
        output = hf_generate_text(prompt, model, temperature=temperature, max_new_tokens=max_new_tokens)
    else:
        output = ollama_generate(
            prompt,
            model,
            system="Return only valid JSON for a code editing candidate.",
            #options={"temperature": temperature},
            #response_format="json",
        )
        print(f"[model] raw output: {output}")
    return _extract_json(output)

 
async def _generate_candidate_cognition_async(
    prompt: str,
    *,
    timeout_s: int,
    permission_policy: str,
) -> Dict[str, Any]:
    client = CognitionClient(
        source_id="local_code_agent",
        permission_policy=permission_policy,
        timeout_s=timeout_s,
    )
    return await client.generate_json(prompt, timeout=timeout_s)


def _generate_candidate_cognition(prompt: str, *, timeout_s: int, permission_policy: str) -> Dict[str, Any]:
    return asyncio.run(
        _generate_candidate_cognition_async(
            prompt,
            timeout_s=timeout_s,
            permission_policy=permission_policy,
        )
    )


def _normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(candidate)
    edits = normalized.get("edits")
    files = normalized.get("files")
    tests = normalized.get("tests")
    if not isinstance(edits, list):
        normalized["edits"] = []
    if not isinstance(files, list):
        normalized["files"] = []
    if tests is not None and not isinstance(tests, list):
        normalized["tests"] = []
    if not isinstance(normalized.get("summary"), str):
        normalized["summary"] = ""
    return normalized


def _resolve_repo_path(repo_root: Path, rel_path: str) -> Path:
    cleaned = rel_path.replace("\\", "/").lstrip("/")
    full_path = (repo_root / cleaned).resolve()
    try:
        full_path.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(f"path escapes repo root: {rel_path}") from exc
    return full_path


def _replace_nth(text: str, old: str, new: str, occurrence: int) -> str:
    start = -1
    offset = 0
    for _ in range(occurrence):
        start = text.find(old, offset)
        if start < 0:
            return text
        offset = start + len(old)
    return text[:start] + new + text[start + len(old) :]


def _apply_edit(repo_root: Path, edit: Dict[str, Any]) -> str:
    rel_path = str(edit.get("path") or "")
    before = edit.get("before")
    after = edit.get("after")
    if not rel_path or not isinstance(before, str) or not isinstance(after, str):
        raise RuntimeError("edit requires path, before, and after")
    full_path = _resolve_repo_path(repo_root, rel_path)
    if not full_path.exists():
        raise RuntimeError(f"edit path not found: {rel_path}")
    current = _read_text(full_path)
    match_count = current.count(before)
    if match_count == 0:
        raise RuntimeError(f"edit before snippet not found: {rel_path}")
    occurrence_raw = edit.get("occurrence")
    occurrence = int(occurrence_raw) if occurrence_raw is not None else 1
    if occurrence < 1:
        raise RuntimeError(f"edit occurrence must be >= 1: {rel_path}")
    if match_count > 1 and occurrence_raw is None:
        raise RuntimeError(f"edit before snippet is not unique; add occurrence: {rel_path}")
    if occurrence > match_count:
        raise RuntimeError(f"edit occurrence exceeds match count: {rel_path}")
    updated = _replace_nth(current, before, after, occurrence)
    full_path.write_text(updated, encoding="utf-8")
    return rel_path


def _write_candidate_file(repo_root: Path, entry: Dict[str, Any]) -> str:
    rel_path = str(entry.get("path") or "")
    content = entry.get("content")
    if not rel_path or not isinstance(content, str):
        raise RuntimeError("file entry requires path and content")
    full_path = _resolve_repo_path(repo_root, rel_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return rel_path


def _apply_candidate(repo_root: Path, candidate: Dict[str, Any]) -> List[str]:
    changed: List[str] = []
    edits = candidate.get("edits", [])
    if isinstance(edits, list):
        for edit in edits:
            if isinstance(edit, dict):
                changed.append(_apply_edit(repo_root, edit))
    files = candidate.get("files", [])
    if isinstance(files, list):
        for entry in files:
            if isinstance(entry, dict):
                changed.append(_write_candidate_file(repo_root, entry))
    return sorted(set(changed))


def _run_command(repo_root: Path, command: str, timeout_s: int) -> Tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=str(repo_root),
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout[-12000:]


def _write_run_artifact(repo_root: Path, candidate: Dict[str, Any], prompt: str) -> Path:
    out_dir = repo_root / "data" / "local_code_agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    path = out_dir / f"candidate_{stamp}.json"
    payload = {"candidate": candidate, "prompt_excerpt": prompt[:8000]}
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Codex-like coding agent powered by soul + local models.")
    parser.add_argument("task", nargs="+", help="Implementation request.")
    parser.add_argument("--repo-root", default=".", help="Repository to edit. Defaults to the current directory.")
    parser.add_argument("--model", default="", help="Override local model id.")
    parser.add_argument("--provider", default="", choices=["", "ollama", "hf"], help="Override provider.")
    parser.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "cognition"],
        help="Generation mode: direct calls the configured model; cognition submits through the running orchestrator.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--max-files", type=int, default=8, help="Maximum relevant files to include.")
    parser.add_argument("--file", action="append", default=[], help="Force include a relative file path. May repeat.")
    parser.add_argument("--soul-output-dir", default="soul", help="Soul mirror output directory in the target repo.")
    parser.add_argument("--no-sync-soul", action="store_true", help="Use existing soul mirror without refreshing it.")
    parser.add_argument("--cognition-timeout-s", type=int, default=240, help="Timeout for --mode cognition.")
    parser.add_argument(
        "--cognition-permission-policy",
        default="deny",
        choices=["deny", "approve"],
        help="How the cognition client answers orchestrator permission prompts.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply the generated candidate to the repo.")
    parser.add_argument("--test-command", default="", help="Command to run after applying changes.")
    parser.add_argument("--test-timeout-s", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root(args.repo_root)
    task = " ".join(args.task).strip()
    if not task:
        raise SystemExit("[error] task is required")

    soul_config = _load_soul_config(repo_root, args.soul_output_dir)
    if not args.no_sync_soul:
        result = sync_soul(repo_root, soul_config)
        status = "updated" if result.changed else "current"
        print(f"[soul] {status}: files={result.manifest['file_count']} symbols={result.manifest['symbol_count']}")

    soul_root = repo_root / soul_config.output_dir
    snippets = _select_snippets(repo_root, soul_root, task, args.file, max(1, args.max_files))
    prompt = _build_prompt(repo_root, soul_root, task, snippets)
    try:
        if args.mode == "cognition":
            candidate = _generate_candidate_cognition(
                prompt,
                timeout_s=max(1, int(args.cognition_timeout_s)),
                permission_policy=args.cognition_permission_policy,
            )
        else:
            settings = load_settings()
            provider = args.provider or str(settings.models.get("provider", "ollama")).lower()
            model = args.model or str(settings.evolve.get("evolution_model") or settings.models.get("text_model") or "")
            if not model:
                raise SystemExit("[error] no local model configured; set config/settings.json models.text_model or pass --model")
            candidate = _generate_candidate_direct(prompt, model, provider, args.temperature, args.max_new_tokens)
    except (OllamaError, HfError, ValueError, CognitionClientError, CognitionJsonError) as exc:
        if args.mode == "cognition":
            raise SystemExit(
                "[error] cognition generation failed. Make sure the local stack is running "
                f"(for example scripts/run_all.sh or scripts/run_all.ps1): {exc}"
            ) from exc
        raise SystemExit(f"[error] local model generation failed: {exc}") from exc
    candidate = _normalize_candidate(candidate)

    artifact = _write_run_artifact(repo_root, candidate, prompt)
    print(f"[agent] candidate written: {artifact}")
    print(f"[agent] summary: {candidate.get('summary', '')}")

    if not args.apply:
        print("[agent] dry run only; rerun with --apply to modify files")
        print(json.dumps(candidate, ensure_ascii=True, indent=2))
        return 0

    changed = _apply_candidate(repo_root, candidate)
    if changed:
        print("[agent] changed files: " + ", ".join(changed))
    else:
        print("[agent] no file changes were produced")

    tests = []
    if args.test_command:
        tests = [args.test_command]
    elif isinstance(candidate.get("tests"), list):
        tests = [str(item) for item in candidate["tests"] if str(item).strip()]
    for command in tests[:3]:
        print(f"[test] {command}")
        code, output = _run_command(repo_root, command, args.test_timeout_s)
        print(output)
        if code != 0:
            print(f"[test] failed with exit code {code}")
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
