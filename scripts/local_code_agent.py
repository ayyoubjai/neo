#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import fnmatch
import os
import uuid
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


SCRIPT_ROOT = Path(__file__).resolve().parent
SYSTEM_ROOT = SCRIPT_ROOT.parent
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from coding_agent.session import SessionStore
from coding_agent.tools import CodingTools, MUTATIONS
from common.config import load_settings
from autonomy.cognition_client import CognitionClient, CognitionClientError, CognitionJsonError
from common.soul_mirror import SoulConfig, load_config, sync_soul
from model_server.hf_client import HfError
from model_server.llamacpp_client import LlamacppError
from model_server.local_generation import generate_local_text
from model_server.ollama_client import OllamaError


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
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "**/node_modules/**",
    "**/.git/**",
    "**/.venv/**",
)
MAX_PROMPT_CHARS = 60000
SESSION_DEFAULTS = {
    "apply": False, "allow_exec": False, "allow_network": False, "allow_model_tests": False,
    "test_command": "", "mode": "direct", "model": "", "provider": "",
    "temperature": 0.1, "max_new_tokens": 6000, "max_files": 8, "file": [],
    "soul_output_dir": "soul", "no_sync_soul": False, "cognition_timeout_s": 240,
    "cognition_permission_policy": "deny", "test_timeout_s": 180,
}


class PatchRollbackError(RuntimeError):
    """Filesystem failure prevented restoring all files; stop automatic repair."""


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
    include = tuple(_discover_files(repo_root, output_dir))
    return SoulConfig(
        include=include,
        exclude=DEFAULT_EXCLUDE,
        max_file_bytes=384 * 1024,
        output_dir=output_dir,
    )


def _discover_files(repo_root: Path, output_dir: str = "soul") -> List[str]:
    """Respect Git ignores when available; include unfamiliar repository layouts."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "."],
            cwd=repo_root, capture_output=True, timeout=10, check=False,
        )
        paths = proc.stdout.decode("utf-8").split("\0") if proc.returncode == 0 else None
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        paths = None
    if paths is None:
        paths = []
        for directory, dirs, files in os.walk(repo_root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in {
                ".git", ".venv", "venv", "node_modules", "__pycache__", "soul", "data"
            } and not (Path(directory) / name).is_symlink()]
            paths.extend((Path(directory) / name).relative_to(repo_root).as_posix() for name in files)
    excluded = (*DEFAULT_EXCLUDE, f"{output_dir.rstrip('/')}/**")
    return sorted({name for name in paths if name
                   and not any(fnmatch.fnmatch(name, pattern) for pattern in excluded)
                   and (repo_root / name).is_file()
                   and not (repo_root / name).is_symlink()
                   and (repo_root / name).resolve().is_relative_to(repo_root)})


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
    path = _resolve_repo_path(repo_root, rel_path)
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
    overview = _read_text(overview_path, limit=4000) if overview_path.exists() else ""
    tree = _read_text(tree_path, limit=4000) if tree_path.exists() else ""
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
        '  "tests": ["pytest tests/test_x.py"],\n'
        '  "reads": [{"path": "src/example.py", "start_line": 1, "end_line": 160}],\n'
        '  "searches": ["literal symbol or text"],\n'
        '  "tools": [{"tool_id": "git.diff", "args": {}}],\n'
        '  "plan": [{"id": "setup", "description": "Set up project", "status": "pending"}],\n'
        '  "notes": "Durable design decisions and remaining context",\n'
        '  "verify": false, "status": "continue"\n'
        "}\n\n"
        "Rules:\n"
        "- Request reads/searches with empty edits/files when you need more context; results arrive next step.\n"
        "- Searches match literal text and file paths. Read ranges are inclusive and limited to 200 lines.\n"
        "- Choose one action per response: reads/searches, tools, edits/files, verify=true, or status=complete/blocked.\n"
        "- Return only the next patch, never reapply earlier patches.\n"
        "- Maintain the whole task plan with stable milestone ids and pending/in_progress/completed statuses.\n"
        "- Passing checks is feedback, not completion: continue implementing outstanding milestones.\n"
        "- Return status=complete with no actions only after all milestones are completed and current checks passed.\n"
        "- Use verify=true to rerun authorized checks, including after process or file tools.\n"
        "- Keep important decisions in notes; older observations may leave the prompt but remain on disk.\n"
        "- Tool commands use argv arrays; use enabled proc tools for installs/builds/servers when needed.\n"
        "- Stop all background process handles before final verification and completion.\n"
        "- Use edits for existing files and files for new files.\n"
        "- Every edit.before must be an exact substring from the repository snippets.\n"
        "- Keep the change focused on the task.\n"
        "- Do not add dependencies unless the task explicitly requires them.\n"
        "- Test commands are for verification; use tools for setup and other execution.\n"
        "- Inspect missing context before editing; return status=blocked with a reason if unable to proceed.\n\n"
        f"Repository root: {repo_root}\n"
        f"Git status:\n{status}\n\n"
        f"Task:\n{task}\n\n"
        f"Repository instructions (AGENTS.md):\n{_read_text(repo_root / 'AGENTS.md', 8000) if (repo_root / 'AGENTS.md').is_file() else 'None'}\n\n"
        f"Soul overview:\n{overview or 'None'}\n\n"
        f"Repository tree excerpt:\n{tree or 'None'}\n\n"
        f"Relevant snippets:\n{snippet_block or 'None'}\n"
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[: MAX_PROMPT_CHARS - 80] + "\n...[prompt truncated; use focused exact edits]...\n"
    return prompt


def _generate_candidate_direct(prompt: str, model: str, provider: str, temperature: float, max_new_tokens: int) -> Dict[str, Any]:
    output = generate_local_text(
        prompt,
        model,
        provider=provider,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        system="Return only valid JSON for a code editing candidate.",
        response_format="json",
    )
    return _extract_json(output)


async def _generate_candidate_cognition_async(
    prompt: str,
    *,
    timeout_s: int,
    permission_policy: str,
) -> Dict[str, Any]:
    client = CognitionClient(
        source_id="neo_code",
        permission_policy=permission_policy,
        timeout_s=timeout_s,
    )
    return await client.generate_json(prompt, timeout=timeout_s, max_actions=0)


def _generate_candidate_cognition(prompt: str, *, timeout_s: int, permission_policy: str) -> Dict[str, Any]:
    return asyncio.run(
        _generate_candidate_cognition_async(
            prompt,
            timeout_s=timeout_s,
            permission_policy=permission_policy,
        )
    )


def _normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a JSON object")
    normalized = dict(candidate)
    for key in ("edits", "files", "tests", "reads", "searches", "tools"):
        value = normalized.get(key)
        if value is None:
            normalized[key] = []
        elif not isinstance(value, list):
            raise ValueError(f"candidate.{key} must be a list")
    for key in ("edits", "files", "reads", "tools"):
        if any(not isinstance(item, dict) for item in normalized[key]):
            raise ValueError(f"candidate.{key} entries must be objects")
    for key in ("tests", "searches"):
        if any(not isinstance(item, str) or not item.strip() for item in normalized[key]):
            raise ValueError(f"candidate.{key} entries must be nonempty strings")
    normalized.setdefault("status", "continue")
    normalized.setdefault("verify", False)
    if not isinstance(normalized["status"], str) or normalized["status"] not in {"continue", "complete", "blocked"}:
        raise ValueError("status must be continue, complete, or blocked")
    if type(normalized["verify"]) is not bool:
        raise ValueError("verify must be a boolean")
    actions = sum(bool(group) for group in (
        normalized["reads"] or normalized["searches"], normalized["tools"],
        normalized["edits"] or normalized["files"], normalized["verify"],
    ))
    if actions > 1 or (actions and normalized["status"] != "continue"):
        raise ValueError("Choose one action type; completion/blocked responses must have no actions")
    if len(normalized["tools"]) > 4:
        raise ValueError("At most four tool calls per step")
    for request in normalized["tools"]:
        if not isinstance(request.get("tool_id"), str) or not isinstance(request.get("args", {}), dict):
            raise ValueError("tools require tool_id and an args object")
    if "notes" in normalized and (not isinstance(normalized["notes"], str) or len(normalized["notes"]) > 8000):
        raise ValueError("notes must be a string of at most 8000 characters")
    if not isinstance(normalized.get("summary"), str):
        normalized["summary"] = ""
    return normalized


def _resolve_repo_path(repo_root: Path, rel_path: str) -> Path:
    cleaned = rel_path.replace("\\", "/")
    if not cleaned or cleaned.startswith("/") or re.match(r"^[A-Za-z]:", cleaned):
        raise RuntimeError(f"expected a relative repository path: {rel_path}")
    if ".git" in Path(cleaned).parts:
        raise RuntimeError("Git metadata cannot be accessed by the coding agent")
    full_path = (repo_root / cleaned).resolve()
    try:
        relative = full_path.relative_to(repo_root)
        if relative.parts[:2] == ("data", "neo_code"):
            raise RuntimeError("Session storage cannot be accessed by patch or inspection actions")
        if ".git" in relative.parts:
            raise RuntimeError("Git metadata cannot be accessed by the coding agent")
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


def _edited_text(current: str, edit: Dict[str, Any]) -> str:
    before, after = edit.get("before"), edit.get("after")
    if not isinstance(before, str) or not before or not isinstance(after, str):
        raise RuntimeError("edit requires nonempty before text and string after text")
    match_count = current.count(before)
    if match_count == 0:
        raise RuntimeError(f"edit before snippet not found: {edit.get('path')}")
    occurrence_raw = edit.get("occurrence")
    occurrence = occurrence_raw if occurrence_raw is not None else 1
    if type(occurrence) is not int or occurrence < 1 or occurrence > match_count:
        raise RuntimeError("edit occurrence must be an integer within the match count")
    if match_count > 1 and occurrence_raw is None:
        raise RuntimeError("edit before snippet is not unique; add occurrence")
    return _replace_nth(current, before, after, occurrence)


def _stage_candidate(repo_root: Path, candidate: Dict[str, Any]) -> Tuple[Dict[Path, bytes], Dict[Path, Optional[bytes]]]:
    staged: Dict[Path, bytes] = {}
    originals: Dict[Path, Optional[bytes]] = {}
    for edit in candidate.get("edits", []):
        path = _resolve_repo_path(repo_root, str(edit.get("path") or ""))
        if not path.is_file():
            raise RuntimeError(f"edit path not found: {edit.get('path')}")
        if path not in originals:
            originals[path] = path.read_bytes()
        current = staged.get(path, originals[path]).decode("utf-8")
        staged[path] = _edited_text(current, edit).encode("utf-8")
    for entry in candidate.get("files", []):
        path = _resolve_repo_path(repo_root, str(entry.get("path") or ""))
        if path.exists() or path in staged:
            raise RuntimeError(f"new file already exists: {entry.get('path')}; use edits")
        content = entry.get("content")
        if not isinstance(content, str):
            raise RuntimeError("file entry requires string content")
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise RuntimeError(f"file parent is not a directory: {parent}")
        originals[path] = None
        staged[path] = content.encode("utf-8")
    for path in staged:
        if any(parent in staged for parent in path.parents):
            raise RuntimeError("candidate contains conflicting file and directory paths")
    return staged, originals


def _apply_edit(repo_root: Path, edit: Dict[str, Any]) -> str:
    _apply_candidate(repo_root, {"edits": [edit]})
    return str(edit["path"])


def _write_candidate_file(repo_root: Path, entry: Dict[str, Any]) -> str:
    _apply_candidate(repo_root, {"files": [entry]})
    return str(entry["path"])


def _apply_candidate(repo_root: Path, candidate: Dict[str, Any]) -> List[str]:
    candidate = _normalize_candidate(candidate)
    staged, originals = _stage_candidate(repo_root, candidate)
    written: List[Path] = []
    created_dirs: List[Path] = []
    try:
        for path, content in staged.items():
            original = originals[path]
            if (path.read_bytes() if path.exists() else None) != original:
                raise RuntimeError(f"file changed during patch application: {path}")
            missing = []
            parent = path.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                directory.mkdir()
                created_dirs.append(directory)
            written.append(path)
            path.write_bytes(content)
    except BaseException as exc:
        rollback_errors = []
        for path in reversed(written):
            try:
                original = originals[path]
                if original is None:
                    path.unlink(missing_ok=True)
                elif path.read_bytes() != original:
                    path.write_bytes(original)
            except OSError as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        for directory in reversed(created_dirs):
            try:
                directory.rmdir()
            except OSError as rollback_exc:
                rollback_errors.append(f"{directory}: {rollback_exc}")
        if rollback_errors:
            raise PatchRollbackError("Patch failed; rollback incomplete: " + "; ".join(rollback_errors)) from exc
        raise
    return sorted(path.relative_to(repo_root).as_posix() for path in staged)


def _run_command(repo_root: Path, command: str, timeout_s: int) -> Tuple[int, str]:
    from tool_runtime.processes import run_process
    result = run_process(command, cwd=repo_root, timeout_s=timeout_s, shell=True)
    output = (result["stdout"] + result["stderr"])[-12000:]
    if result["timed_out"]:
        return 124, output + f"\nCommand timed out after {timeout_s}s"
    return result["returncode"], output


def _write_run_artifact(repo_root: Path, candidate: Dict[str, Any], prompt: str) -> Path:
    out_dir = repo_root / "data" / "neo_code"
    if not out_dir.resolve().is_relative_to(repo_root):
        raise ValueError("Candidate storage resolves outside the repository")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    path = out_dir / f"candidate_{stamp}.json"
    payload = {"candidate": candidate, "prompt_excerpt": prompt[:8000]}
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="neo-code", description="Neo Code: persistent coding sessions powered by soul and local models or cognition.")
    parser.add_argument("task", nargs="*", help="Implementation request.")
    parser.add_argument("--repo-root", default=".", help="Repository to edit. Defaults to the current directory.")
    parser.add_argument("--model", default=None, help="Override local model id.")
    parser.add_argument("--provider", default=None, choices=["", "ollama", "hf", "llamacpp"], help="Override provider.")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["direct", "cognition"],
        help="Generation mode: direct calls the configured model; cognition submits through the running orchestrator.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None, help="Maximum relevant files to include.")
    parser.add_argument("--file", action="append", default=None, help="Force include a relative file path. May repeat.")
    parser.add_argument("--soul-output-dir", default=None, help="Soul mirror output directory in the target repo.")
    parser.add_argument("--no-sync-soul", action="store_true", default=None, help="Use existing soul mirror without refreshing it.")
    parser.add_argument("--cognition-timeout-s", type=int, default=None, help="Timeout for --mode cognition.")
    parser.add_argument(
        "--cognition-permission-policy",
        default=None,
        choices=["deny", "approve"],
        help="How the cognition client answers orchestrator permission prompts.",
    )
    parser.add_argument("--max-steps", type=int, default=24, help="Maximum inspection/generation/repair steps.")
    parser.add_argument("--resume", help="Resume a saved Neo Code session; max-steps is an additional budget.")
    parser.add_argument("--allow-exec", action=argparse.BooleanOptionalAction, default=None,
                        help="Authorize model-selected local commands and managed background servers (unsandboxed).")
    parser.add_argument("--allow-network", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable runtime web search and GET/HEAD HTTP requests.")
    parser.add_argument("--candidate", help="Load a saved candidate without regenerating it; use --apply to apply.")
    parser.add_argument("--allow-model-tests", action=argparse.BooleanOptionalAction, default=None, help="Authorize model-suggested shell test commands (unsandboxed).")
    parser.add_argument("--apply", action=argparse.BooleanOptionalAction, default=None, help="Apply the generated candidate to the repo.")
    parser.add_argument("--test-command", default=None, help="Command to run after applying changes.")
    parser.add_argument("--test-timeout-s", type=int, default=None)
    args = parser.parse_args()
    if args.max_steps < 1 or (args.test_timeout_s is not None and args.test_timeout_s < 1):
        parser.error("--max-steps and --test-timeout-s must be positive")
    if args.resume and args.candidate:
        parser.error("--resume and --candidate cannot be combined")
    if not args.task and not args.candidate and not args.resume:
        parser.error("provide a task, --resume, or --candidate")
    return args


def _inspect_requests(repo_root: Path, candidate: Dict[str, Any]) -> str:
    results = []
    for request in candidate["reads"][:8]:
        try:
            path = _resolve_repo_path(repo_root, str(request.get("path") or ""))
            start = max(1, int(request.get("start_line", 1)))
            end = min(start + 199, int(request.get("end_line", start + 159)))
            if end < start:
                raise ValueError("end_line must be >= start_line")
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("file exceeds 1 MiB read limit")
            lines = path.read_text(encoding="utf-8").splitlines()
            excerpt = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines) if start <= i + 1 <= end)
            results.append(f"READ {request['path']} ({len(lines)} lines):\n{excerpt[:12000]}")
        except (OSError, ValueError, RuntimeError) as exc:
            results.append(f"READ {request}: {exc}")
    paths = _discover_files(repo_root) if candidate["searches"] else []
    for query in candidate["searches"][:4]:
        matches = []
        for name in paths:
            if query.lower() in name.lower():
                matches.append(f"PATH {name}")
            try:
                path = repo_root / name
                if path.stat().st_size > 384 * 1024:
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if query.lower() in line.lower():
                        matches.append(f"{name}:{number}: {line[:300]}")
                    if len(matches) >= 80:
                        break
            except (OSError, UnicodeError):
                continue
            if len(matches) >= 80:
                break
        results.append(f"SEARCH {query!r}:\n" + ("\n".join(matches[:80]) or "No matches"))
    return "\n\n".join(results)[:24000]


def _verify_candidate(repo_root: Path, candidate: Dict[str, Any], args: argparse.Namespace) -> Tuple[Optional[int], str]:
    commands = [args.test_command] if args.test_command else candidate["tests"][:3] if args.allow_model_tests or getattr(args, "allow_exec", False) else []
    if not commands:
        detail = "No authorized verification command. Supply --test-command, --allow-model-tests, or --allow-exec."
        if candidate["tests"]:
            detail += " Suggested checks: " + json.dumps(candidate["tests"])
        return None, detail
    results = []
    for command in commands:
        print(f"[test] {command}")
        code, output = _run_command(repo_root, command, args.test_timeout_s)
        print(output)
        results.append(f"$ {command}\nexit={code}\n{output}")
        if code:
            return code, "\n".join(results)
    return 0, "\n".join(results)


def _repo_fingerprint(repo_root: Path, soul_root: Path, changed: Sequence[str]) -> str:
    digest = hashlib.sha256()
    names = set(_discover_files(repo_root, soul_root.relative_to(repo_root).as_posix())) | set(changed)
    for name in sorted(names):
        path = _resolve_repo_path(repo_root, name)
        digest.update(name.encode("utf-8"))
        if not path.is_file():
            digest.update(b"missing")
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _record_verification(repo_root: Path, soul_root: Path, candidate: Dict[str, Any],
                         args: argparse.Namespace, session: SessionStore) -> None:
    session.begin_action({"verify": True})
    code, output = _verify_candidate(repo_root, candidate, args)
    session.state["verification"] = {
        "passed": code == 0, "revision": session.state["revision"],
        "fingerprint": _repo_fingerprint(repo_root, soul_root, session.state["changed"]) if code == 0 else None,
        "output": output, "tests": candidate["tests"],
    }
    session.observe(("Verification passed. Continue remaining milestones or explicitly complete.\n" if code == 0
                     else "Verification failed or unavailable. Repair or arrange authorized checks.\n") + output)
    session.finish_action()


def _run_agent_loop(repo_root: Path, soul_root: Path, task: str, args: argparse.Namespace,
                    generate: Callable[[str], Dict[str, Any]]) -> int:
    with SessionStore(repo_root, task, getattr(args, "resume", None)) as session:
        print(f"[neo-code] session: {session.path}")
        for key, default in SESSION_DEFAULTS.items():
            session.state["options"][key] = getattr(args, key, default)
        adapter = CodingTools(repo_root, apply=args.apply, allow_exec=getattr(args, "allow_exec", False),
                              allow_network=getattr(args, "allow_network", False))
        try:
            result = _continue_session(repo_root, soul_root, args, generate, session, adapter)
            return result
        except BaseException:
            session.state["status"] = "interrupted"
            raise
        finally:
            try:
                if adapter.processes.handles:
                    session.observe("Stopped owned background processes on session exit; restart them if needed on resume.")
                    session.state["verification"] = None
                adapter.close()
            finally:
                session.save()


def _continue_session(repo_root: Path, soul_root: Path, args: argparse.Namespace,
                      generate: Callable[[str], Dict[str, Any]], session: SessionStore, adapter: CodingTools) -> int:
    state = session.state
    for step in range(1, args.max_steps + 1):
        state["steps"] += 1
        snippets = _select_snippets(repo_root, soul_root, state["task"], args.file, max(1, args.max_files))
        base = _build_prompt(repo_root, soul_root, state["task"], snippets)
        verification = state.get("verification") or {}
        durable = json.dumps({
            "plan": state["plan"], "notes": state["notes"], "revision": state["revision"],
            "recent_changed_files": state["changed"][-50:], "changed_count": len(state["changed"]),
            "verification": {"passed": verification.get("passed", False),
                             "revision": verification.get("revision"),
                             "output": verification.get("output", "")[-2000:]},
        }, ensure_ascii=False)
        catalog = json.dumps(adapter.catalog())
        feedback_budget = max(0, min(20000, MAX_PROMPT_CHARS - len(durable) - len(catalog) - 14000))
        transcript = "\n\n".join(item["text"] for item in state["history"])
        feedback = transcript[-feedback_budget:] if feedback_budget else ""
        controls = (f"\nStep {step}/{args.max_steps}; lifetime steps {state['steps']}. "
                    "Earlier successful actions are already on disk; rejected/reviewed patches are not.\n"
                    f"Durable task state:\n{durable}\nAvailable tool catalog:\n{catalog}\n"
                    f"Execution observations:\n{feedback}")
        prompt = base[:max(4000, MAX_PROMPT_CHARS - len(controls))] + controls
        print(f"[neo-code] step {step}/{args.max_steps}")
        session.save()
        try:
            candidate = _normalize_candidate(generate(prompt))
            session.update_plan(candidate.get("plan"))
        except (ValueError, TypeError, CognitionJsonError) as exc:
            session.observe(f"Invalid response: {exc}. Return a valid candidate object.")
            continue
        if "notes" in candidate:
            state["notes"] = candidate["notes"]
        artifact = _write_run_artifact(repo_root, candidate, prompt)
        print(f"[neo-code] candidate: {artifact}")
        print(f"[neo-code] {candidate['summary']}")
        state["history"].append({"role": "assistant", "text": json.dumps(candidate, ensure_ascii=False)})
        session.save()
        if candidate["status"] == "blocked":
            state["status"] = "blocked"
            return 2
        if candidate["status"] == "complete":
            verification = state.get("verification") or {}
            if any(item["status"] != "completed" for item in state["plan"]):
                session.observe("Completion rejected: unfinished milestones remain. Update the plan and finish them.")
            elif adapter.processes.handles:
                session.observe("Completion rejected: stop background handles, then verify again.")
            elif not verification.get("passed") or verification.get("revision") != state["revision"]:
                session.observe("Completion rejected: current changes need successful authorized verification.")
            elif verification["fingerprint"] != _repo_fingerprint(repo_root, soul_root, state["changed"]):
                state["verification"] = None
                session.observe("Completion rejected: repository changed since checks passed. Verify again.")
            else:
                state["status"] = "complete"
                print("[neo-code] complete: all milestones marked complete and current checks passed")
                return 0
            continue
        if candidate["reads"] or candidate["searches"]:
            session.observe(_inspect_requests(repo_root, candidate))
            continue
        if candidate["tools"]:
            for request in candidate["tools"]:
                mutating = request["tool_id"] in MUTATIONS or request["tool_id"] == "proc.stop"
                session.begin_action(request, mutating=mutating)
                result = adapter.execute(request)
                session.observe("Tool result: " + json.dumps(result, ensure_ascii=False)[:24000])
                # Record affected files even when an operation reports a partial failure.
                if mutating:
                    for key in ("path", "src", "dst"):
                        value = request.get("args", {}).get(key)
                        if isinstance(value, str):
                            try:
                                normalized = adapter._path(value).removeprefix("workspace:/")
                                if normalized not in state["changed"]:
                                    state["changed"].append(normalized)
                            except ValueError:
                                pass
                session.finish_action()
                if not result["ok"]:
                    break
            continue
        if candidate["verify"]:
            if not args.apply:
                session.observe("Review mode does not execute verification commands. Resume with --apply.")
            else:
                _record_verification(repo_root, soul_root, candidate, args, session)
            continue
        if candidate["edits"] or candidate["files"]:
            try:
                if not args.apply:
                    _stage_candidate(repo_root, candidate)
                    print(json.dumps(candidate, ensure_ascii=True, indent=2))
                    print(f"[neo-code] review only; apply exactly with --candidate {artifact} --apply")
                    session.observe("Patch validated for review only; no source changes applied.")
                    state["status"] = "review"
                    return 0
                session.begin_action({"candidate": str(artifact)}, mutating=True)
                applied = _apply_candidate(repo_root, candidate)
            except PatchRollbackError:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                session.observe(f"Patch rejected: {exc}. Inspect current files and submit a corrected patch.")
                session.finish_action()
                continue
            state["changed"] = sorted(set(state["changed"]) | set(applied))
            session.observe("Applied patch to: " + ", ".join(applied))
            session.finish_action()
            print("[neo-code] changed files: " + ", ".join(applied))
            _record_verification(repo_root, soul_root, candidate, args, session)
            continue
        if candidate.get("plan") or candidate.get("notes"):
            session.observe("Plan/notes saved. Continue with the next action.")
            continue
        state["status"] = "blocked"
        print("[neo-code] blocked: no action or patch produced. " + candidate["summary"])
        return 2
    state["status"] = "budget_exhausted"
    print(f"[neo-code] step budget exhausted; resume with --resume {session.path}")
    return 2


def _main(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args.repo_root)
    saved = SessionStore.read(Path(args.resume).expanduser().resolve()) if args.resume else None
    for key, default in SESSION_DEFAULTS.items():
        if getattr(args, key) is None:
            value = saved["options"].get(key) if saved else None
            setattr(args, key, value if value is not None else default)
    if saved and Path(saved["repo_root"]).resolve() != repo_root:
        raise ValueError("Session belongs to a different repository; use its --repo-root")
    if args.candidate:
        payload = json.loads(Path(args.candidate).expanduser().read_text(encoding="utf-8"))
        candidate = _normalize_candidate(payload.get("candidate", payload) if isinstance(payload, dict) else payload)
        if candidate["reads"] or candidate["searches"] or candidate["tools"] or candidate["verify"] or not (candidate["edits"] or candidate["files"]):
            raise SystemExit("[error] saved candidate must contain a patch")
        if not args.apply:
            _stage_candidate(repo_root, candidate)
            print(json.dumps(candidate, indent=2))
            return 0
        changed = _apply_candidate(repo_root, candidate)
        print("[neo-code] changed files: " + ", ".join(changed))
        code, output = _verify_candidate(repo_root, candidate, args)
        print(output)
        return 2 if code is None else code

    task = " ".join(args.task).strip()
    if saved:
        if task and task != saved["task"]:
            raise ValueError("Resume without a new task")
        task = saved["task"]
    soul_config = _load_soul_config(repo_root, args.soul_output_dir)
    soul_root = _resolve_repo_path(repo_root, soul_config.output_dir)
    if soul_root == repo_root:
        raise SystemExit("[error] soul output must be a subdirectory")
    if not args.no_sync_soul:
        result = sync_soul(repo_root, soul_config)
        status = "updated" if result.changed else "current"
        print(f"[soul] {status}: files={result.manifest['file_count']} symbols={result.manifest['symbol_count']}")

    if args.mode == "cognition":
        def generate(prompt: str) -> Dict[str, Any]:
            return _generate_candidate_cognition(prompt, timeout_s=max(1, args.cognition_timeout_s),
                                                permission_policy=args.cognition_permission_policy)
    else:
        settings = load_settings()
        provider = args.provider or str(settings.models.get("provider", "ollama")).lower()
        model = args.model or str(settings.evolve.get("evolution_model") or settings.models.get("text_model") or "")
        if not model:
            raise SystemExit("[error] no local model configured; pass --model or configure models.text_model")
        def generate(prompt: str) -> Dict[str, Any]:
            return _generate_candidate_direct(prompt, model, provider, args.temperature, args.max_new_tokens)
    try:
        return _run_agent_loop(repo_root, soul_root, task, args, generate)
    except (OllamaError, HfError, LlamacppError, CognitionClientError, OSError) as exc:
        raise SystemExit(f"[error] coding agent failed: {exc}") from exc


def main() -> int:
    args = _parse_args()
    try:
        return _main(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[error] coding agent failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
