import atexit
import argparse
import difflib
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
_PYTHON_COMMAND_WORKS: Optional[bool] = None
_ACTIVE_EVOLUTION_LOCKS: Dict[str, int] = {}

from common.config import load_settings
from reflection_engine import run_autonomous_reflection


@dataclass
class Target:
    path: str
    start_line: int = 0
    end_line: int = 0
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    score: Optional[float]
    passed: Optional[bool]
    required: bool
    details: str = ""
    raw_stdout: str = ""
    raw_stderr: str = ""
    returncode: Optional[int] = None


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_target(target: Target) -> Dict[str, Any]:
    full_path = os.path.join(REPO_ROOT, target.path)
    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start = max(1, int(target.start_line)) if target.start_line else 1
    end = int(target.end_line) if target.end_line else len(lines)
    start = min(start, len(lines))
    end = min(end, len(lines))
    selected = "".join(lines[start - 1 : end])
    return {
        "path": target.path,
        "start_line": start,
        "end_line": end,
        "notes": target.notes,
        "content": selected,
        "full_path": full_path,
    }


def _count_changed_lines(old_text: str, new_text: str) -> int:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed


def _candidate_declared_paths(candidate: Dict[str, Any]) -> List[str]:
    rows: List[str] = []
    seen = set()
    edits = candidate.get("edits", [])
    if isinstance(edits, list):
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            path = str(edit.get("path", "") or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            rows.append(path)
    files = candidate.get("files", [])
    if isinstance(files, list):
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            path = str(file_entry.get("path", "") or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            rows.append(path)
    return rows


def _candidate_signature(candidate: Dict[str, Any]) -> str:
    payload: Dict[str, Any] = {}
    edits = candidate.get("edits", [])
    if isinstance(edits, list):
        normalized_edits: List[Dict[str, Any]] = []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            row = {
                "path": str(edit.get("path", "") or ""),
                "before": str(edit.get("before", "") or ""),
                "after": str(edit.get("after", "") or ""),
            }
            if "occurrence" in edit and edit.get("occurrence") is not None:
                row["occurrence"] = int(edit.get("occurrence"))
            normalized_edits.append(row)
        if normalized_edits:
            payload["edits"] = normalized_edits
    files = candidate.get("files", [])
    if isinstance(files, list):
        normalized_files: List[Dict[str, Any]] = []
        for entry in files:
            if not isinstance(entry, dict):
                continue
            normalized_files.append(
                {
                    "path": str(entry.get("path", "") or ""),
                    "content": str(entry.get("content", "") or ""),
                }
            )
        if normalized_files:
            payload["files"] = normalized_files
    if not payload:
        payload["summary"] = str(candidate.get("summary", "") or "")
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def _read_optional_text(path: str) -> Optional[str]:
    if not os.path.exists(path) or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _summarize_candidate_effect(candidate_dir: str, baseline_root: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    declared_paths = _candidate_declared_paths(candidate)
    changed_files: List[str] = []
    changed_line_count = 0
    for rel_path in declared_paths:
        current_path = _resolve_run_path(candidate_dir, rel_path)
        baseline_path = os.path.abspath(os.path.join(baseline_root, rel_path))
        current_exists = os.path.exists(current_path)
        baseline_exists = os.path.exists(baseline_path)
        current_text = _read_optional_text(current_path)
        baseline_text = _read_optional_text(baseline_path)
        if current_exists == baseline_exists and current_text == baseline_text:
            continue
        changed_files.append(rel_path)
        if current_text is None or baseline_text is None:
            changed_line_count += max(
                len((current_text or "").splitlines()),
                len((baseline_text or "").splitlines()),
                1,
            )
        else:
            changed_line_count += _count_changed_lines(baseline_text, current_text)
    return {
        "declared_paths": declared_paths,
        "changed_files": changed_files,
        "changed_file_count": len(changed_files),
        "changed_line_count": changed_line_count,
        "effective_change": bool(changed_files),
    }


def _load_evolve_runtime_settings() -> Dict[str, Any]:
    settings_path = os.path.join(REPO_ROOT, "config", "settings.json")
    if not os.path.exists(settings_path):
        return {}
    try:
        settings = load_settings(settings_path)
    except Exception:
        return {}
    return dict(settings.evolve) if isinstance(settings.evolve, dict) else {}


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_lock_payload(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _release_evolution_lock(path: str, owner_pid: int) -> None:
    current_owner = _ACTIVE_EVOLUTION_LOCKS.get(path)
    if current_owner is not None and current_owner != owner_pid:
        return
    payload = _read_lock_payload(path)
    if payload and int(payload.get("pid", 0) or 0) not in {0, owner_pid}:
        return
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        return
    _ACTIVE_EVOLUTION_LOCKS.pop(path, None)


def _hold_evolution_lock(lock_path: str) -> None:
    owner_pid = os.getpid()
    if _ACTIVE_EVOLUTION_LOCKS.get(lock_path) == owner_pid:
        return
    parent = os.path.dirname(lock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {"pid": owner_pid, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_lock_payload(lock_path)
            existing_pid = int(existing.get("pid", 0) or 0)
            if existing_pid > 0 and not _pid_is_running(existing_pid):
                try:
                    os.unlink(lock_path)
                    continue
                except OSError:
                    pass
            raise RuntimeError(f"evolution run already in progress: {lock_path}")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
        _ACTIVE_EVOLUTION_LOCKS[lock_path] = owner_pid
        atexit.register(_release_evolution_lock, lock_path, owner_pid)
        return


def _resolve_targets_from_autonomous_reflection(config: Dict[str, Any]) -> Tuple[List[Target], Dict[str, Any]]:
    autonomous_cfg = config.get("autonomous", {})
    if not isinstance(autonomous_cfg, dict) or not bool(autonomous_cfg.get("enabled", False)):
        return [], {}

    reflection_config_path = os.path.join(
        REPO_ROOT,
        str(autonomous_cfg.get("reflection_config", "config/reflection.json") or "config/reflection.json"),
    )
    soul_config_path = os.path.join(
        REPO_ROOT,
        str(autonomous_cfg.get("soul_config", "config/soul.json") or "config/soul.json"),
    )
    plan = run_autonomous_reflection(
        Path(REPO_ROOT),
        Path(reflection_config_path),
        Path(soul_config_path),
        refresh_soul=bool(autonomous_cfg.get("refresh_soul", True)),
        max_dossiers=max(1, _int_with_default(autonomous_cfg.get("max_dossiers"), 3)),
        max_targets_per_dossier=max(1, _int_with_default(autonomous_cfg.get("max_targets_per_dossier"), 1)),
        target_kinds=[
            str(item)
            for item in autonomous_cfg.get(
                "target_kinds",
                ["trace_failure", "generated_capability_review", "structural_hotspot"],
            )
            if str(item).strip()
        ],
        prefer_symbols=bool(autonomous_cfg.get("prefer_symbols", True)),
    )
    targets = [
        Target(
            path=item.path,
            start_line=item.start_line,
            end_line=item.end_line,
            notes=item.notes,
            metadata=dict(item.metadata),
        )
        for item in plan.targets
    ]
    info = {
        "enabled": True,
        "reflection_run_dir": str(plan.run_dir),
        "selected_dossier_count": len(plan.selected_dossiers),
        "target_count": len(targets),
        "selected_dossiers": [
            {
                "dossier_id": str(dossier.get("dossier_id", "") or ""),
                "kind": str(dossier.get("kind", "") or ""),
                "title": str(dossier.get("title", "") or ""),
                "score": float(dossier.get("score", 0.0) or 0.0),
            }
            for dossier in plan.selected_dossiers
        ],
    }
    return targets, info


def _normalize_python_command(command: str) -> str:
    global _PYTHON_COMMAND_WORKS
    if _PYTHON_COMMAND_WORKS is None:
        py = shutil.which("python")
        if not py:
            _PYTHON_COMMAND_WORKS = False
        else:
            try:
                probe = subprocess.run(
                    [py, "-c", "import sys; sys.exit(0)"],
                    text=True,
                    capture_output=True,
                )
                _PYTHON_COMMAND_WORKS = probe.returncode == 0
            except OSError:
                _PYTHON_COMMAND_WORKS = False
    if _PYTHON_COMMAND_WORKS:
        return command
    stripped = command.lstrip()
    match = re.match(r"^python(?=\s|$)", stripped)
    if not match:
        return command
    prefix_len = len(command) - len(stripped)
    rest = stripped[match.end() :]
    interpreter = shlex.quote(sys.executable or "python3")
    return command[:prefix_len] + interpreter + rest


def _run_command(command: str, cwd: str, timeout_s: int) -> subprocess.CompletedProcess:
    resolved_command = _normalize_python_command(command)
    return subprocess.run(
        resolved_command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def _run_json_command(command: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    resolved_command = _normalize_python_command(command)
    proc = subprocess.run(
        resolved_command,
        input=json.dumps(payload, ensure_ascii=True),
        text=True,
        shell=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"generator failed: {proc.stderr.strip() or proc.stdout.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError("generator did not return valid JSON") from e


def _tree_fingerprint(root: str) -> str:
    if not os.path.isdir(root):
        return ""
    hasher = hashlib.sha256()
    for current_dir, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        rel_dir = os.path.relpath(current_dir, root).replace("\\", "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        hasher.update(f"D:{rel_dir}\n".encode("utf-8"))
        for name in filenames:
            path = os.path.join(current_dir, name)
            rel_path = os.path.join(rel_dir, name).replace("\\", "/") if rel_dir else name
            hasher.update(f"P:{rel_path}\n".encode("utf-8"))
            if os.path.islink(path):
                try:
                    target = os.readlink(path)
                except OSError:
                    target = "<broken>"
                hasher.update(f"L:{target}\n".encode("utf-8"))
                continue
            try:
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        hasher.update(chunk)
            except OSError:
                hasher.update(b"<unreadable>")
    return hasher.hexdigest()


def _assert_tree_unchanged(root: str, expected: str, label: str) -> None:
    if not expected:
        return
    current = _tree_fingerprint(root)
    if current != expected:
        raise RuntimeError(f"{label} changed during evolution run; aborting for safety")


def _run_git(args: List[str], cwd: str) -> subprocess.CompletedProcess:
    if shutil.which("git") is None:
        raise RuntimeError("git is not available on PATH")
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return proc


def _reset_dir_keep_git(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        return
    for name in os.listdir(path):
        if name == ".git":
            continue
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and not os.path.islink(full_path):
            _safe_rmtree(full_path)
        else:
            _safe_unlink(full_path)


def _make_writable(path: str) -> None:
    mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    try:
        os.chmod(path, mode)
    except OSError:
        return


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
        return
    except PermissionError:
        _make_writable(path)
    os.unlink(path)


def _rmtree_onerror(func: Any, path: str, exc_info: Any) -> None:
    _make_writable(path)
    func(path)


def _safe_rmtree(path: str) -> None:
    shutil.rmtree(path, onerror=_rmtree_onerror)


def _copy_repo_snapshot(src_root: str, dst_root: str, ignore: List[str]) -> None:
    normalized = _normalize_ignore_patterns(ignore)

    def _ignore(current_dir: str, names: List[str]) -> List[str]:
        rel_dir = os.path.relpath(current_dir, src_root)
        rel_dir = "" if rel_dir in (".", "") else rel_dir
        ignored: List[str] = []
        for entry in names:
            rel_path = os.path.join(rel_dir, entry) if rel_dir else entry
            if _path_matches_ignore(rel_path, entry, normalized):
                ignored.append(entry)
        return ignored

    shutil.copytree(src_root, dst_root, dirs_exist_ok=True, ignore=_ignore, copy_function=_safe_copy2)


def _safe_copy2(src: str, dst: str) -> str:
    try:
        return shutil.copy2(src, dst)
    except PermissionError:
        shutil.copyfile(src, dst)
        return dst


def _slug_for_branch(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._/-]+", "-", value).strip("-")
    return collapsed or "candidate"


def _prepare_evolution_repo(
    source_root: str,
    evolution_repo_root: str,
    ignore_patterns: List[str],
    run_id: str,
) -> Tuple[str, str]:
    os.makedirs(evolution_repo_root, exist_ok=True)
    git_dir = os.path.join(evolution_repo_root, ".git")
    if not os.path.isdir(git_dir):
        _run_git(["init"], evolution_repo_root)
        _run_git(["config", "user.name", "evolve-bot"], evolution_repo_root)
        _run_git(["config", "user.email", "evolve-bot@example.local"], evolution_repo_root)

    _reset_dir_keep_git(evolution_repo_root)
    _copy_repo_snapshot(source_root, evolution_repo_root, ignore_patterns)

    _run_git(["add", "-A"], evolution_repo_root)
    status_proc = _run_git(["status", "--porcelain"], evolution_repo_root)
    if status_proc.stdout.strip():
        _run_git(["commit", "-m", f"baseline {run_id}"], evolution_repo_root)
    else:
        # Ensure HEAD exists for worktree creation.
        try:
            _run_git(["rev-parse", "--verify", "HEAD"], evolution_repo_root)
        except RuntimeError:
            _run_git(["commit", "--allow-empty", "-m", f"baseline {run_id} (empty)"], evolution_repo_root)

    baseline_ref = _run_git(["rev-parse", "HEAD"], evolution_repo_root).stdout.strip()
    return evolution_repo_root, baseline_ref


def _create_candidate_worktree(
    evolution_repo_root: str,
    baseline_ref: str,
    candidate_dir: str,
    run_id: str,
    run_candidate_id: str,
) -> None:
    if os.path.exists(candidate_dir):
        shutil.rmtree(candidate_dir)
    branch_name = f"evolve/{_slug_for_branch(run_id)}/{_slug_for_branch(run_candidate_id)}"
    _run_git(
        ["worktree", "add", "-b", branch_name, candidate_dir, baseline_ref],
        evolution_repo_root,
    )


def _generate_from_command(command: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    data = _run_json_command(command, payload, timeout_s)
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("generator JSON must include a candidates list")
    return data


def _generate_from_inbox(inbox_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(inbox_dir):
        return []
    candidates: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(inbox_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(inbox_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "candidates" in data and isinstance(data["candidates"], list):
            candidates.extend([c for c in data["candidates"] if isinstance(c, dict)])
            continue
        if isinstance(data, dict):
            candidates.append(data)
    return candidates


def _collect_generation_candidates(
    *,
    generator_cmd: str,
    payload: Dict[str, Any],
    timeout_s: int,
    inbox_dir: str,
    fallback_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fallback_to_inbox_on_error = bool(fallback_cfg.get("fallback_to_inbox_on_error", True))
    fallback_to_inbox_on_empty = bool(fallback_cfg.get("fallback_to_inbox_on_empty", True))
    resolved_inbox_dir = ""
    if inbox_dir:
        resolved_inbox_dir = inbox_dir if os.path.isabs(inbox_dir) else os.path.join(REPO_ROOT, inbox_dir)

    def _inbox_result(mode: str, reason: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        inbox_candidates = _generate_from_inbox(resolved_inbox_dir)
        meta = {
            "mode": mode,
            "reason": reason,
            "candidate_count": len(inbox_candidates),
        }
        return inbox_candidates, meta

    if generator_cmd:
        try:
            data = _generate_from_command(generator_cmd, payload, timeout_s)
        except Exception as e:
            if resolved_inbox_dir and fallback_to_inbox_on_error:
                inbox_candidates, meta = _inbox_result("inbox_fallback", str(e))
                if inbox_candidates:
                    return inbox_candidates, meta
            raise
        candidates = data.get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError("generator JSON must include a candidates list")
        generator_meta = data.get("generator", {})
        if not isinstance(generator_meta, dict):
            generator_meta = {}
        if not candidates and resolved_inbox_dir and fallback_to_inbox_on_empty:
            inbox_candidates, meta = _inbox_result(
                "inbox_fallback",
                "generator returned no candidates",
            )
            if inbox_candidates:
                return inbox_candidates, meta
        if not generator_meta:
            generator_meta = {"mode": "llm", "candidate_count": len(candidates)}
        else:
            generator_meta["candidate_count"] = len(candidates)
        return candidates, generator_meta

    inbox_candidates, meta = _inbox_result("inbox", "generator command disabled")
    return inbox_candidates, meta


def _snapshot_candidate_paths(repo_root: str, candidate_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for rel_path in _candidate_declared_paths(candidate_payload):
        full_path = _resolve_run_path(repo_root, rel_path)
        snapshot[rel_path] = {
            "exists": os.path.exists(full_path),
            "content": _read_optional_text(full_path),
        }
    return snapshot


def _restore_candidate_paths(repo_root: str, snapshot: Dict[str, Dict[str, Any]]) -> None:
    for rel_path, state in snapshot.items():
        full_path = _resolve_run_path(repo_root, rel_path)
        existed = bool(state.get("exists", False))
        content = state.get("content")
        if existed:
            _atomic_write_text(full_path, str(content or ""))
            continue
        if os.path.exists(full_path):
            if os.path.isdir(full_path) and not os.path.islink(full_path):
                _safe_rmtree(full_path)
            else:
                _safe_unlink(full_path)


def _apply_promoted_candidates_to_repo(repo_root: str, candidate_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered_reports = sorted(
        [report for report in candidate_reports if isinstance(report, dict)],
        key=lambda report: (
            -float(report.get("score", 0.0) or 0.0),
            str(report.get("run_candidate_id", "") or report.get("id", "") or ""),
        ),
    )
    applied_paths: set[str] = set()
    results: List[Dict[str, Any]] = []
    for report in ordered_reports:
        run_candidate_id = str(report.get("run_candidate_id", "") or report.get("id", "") or "")
        payload = report.get("candidate_payload", {})
        if not isinstance(payload, dict):
            results.append(
                {
                    "run_candidate_id": run_candidate_id,
                    "status": "apply_failed",
                    "error": "missing candidate_payload",
                    "paths": [],
                }
            )
            continue
        declared_paths = _candidate_declared_paths(payload)
        conflicts = sorted(path for path in declared_paths if path in applied_paths)
        if conflicts:
            results.append(
                {
                    "run_candidate_id": run_candidate_id,
                    "status": "skipped_conflict",
                    "paths": declared_paths,
                    "conflicts": conflicts,
                }
            )
            continue
        snapshot = _snapshot_candidate_paths(repo_root, payload)
        try:
            _apply_candidate(repo_root, payload)
        except Exception as e:
            _restore_candidate_paths(repo_root, snapshot)
            results.append(
                {
                    "run_candidate_id": run_candidate_id,
                    "status": "apply_failed",
                    "error": str(e),
                    "paths": declared_paths,
                }
            )
            continue
        applied_paths.update(declared_paths)
        results.append(
            {
                "run_candidate_id": run_candidate_id,
                "status": "applied",
                "paths": declared_paths,
                "score": float(report.get("score", 0.0) or 0.0),
            }
        )
    return results


def _replace_nth(text: str, old: str, new: str, occurrence: int) -> str:
    if occurrence <= 1:
        return text.replace(old, new, 1)
    start = -1
    idx = 0
    while idx < occurrence:
        start = text.find(old, start + 1)
        if start == -1:
            return text
        idx += 1
    return text[:start] + new + text[start + len(old) :]


def _resolve_run_path(run_dir: str, rel_path: str) -> str:
    if not rel_path or not isinstance(rel_path, str):
        raise RuntimeError("edit path missing or invalid")
    full_path = os.path.abspath(os.path.join(run_dir, rel_path))
    run_root = os.path.abspath(run_dir)
    try:
        inside = os.path.commonpath([run_root, full_path]) == run_root
    except ValueError:
        inside = False
    if not inside:
        raise RuntimeError(f"path escapes run directory: {rel_path}")
    return full_path


def _atomic_write_text(path: str, content: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".evolve_tmp_", dir=parent or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _apply_single_edit(current: str, edit: Dict[str, Any], rel_path: str) -> str:
    before = edit.get("before")
    after = edit.get("after")
    if not isinstance(before, str) or not isinstance(after, str):
        raise RuntimeError(f"edit requires before/after strings: {rel_path}")
    match_count = current.count(before)
    if match_count <= 0:
        raise RuntimeError(f"edit 'before' snippet not found in {rel_path}")

    occurrence_raw = edit.get("occurrence")
    occurrence = int(occurrence_raw) if occurrence_raw is not None else 1
    if occurrence < 1:
        raise RuntimeError(f"edit occurrence must be >= 1 in {rel_path}")
    if match_count > 1 and occurrence_raw is None:
        raise RuntimeError(f"edit 'before' snippet is not unique in {rel_path}; add occurrence")
    if occurrence > match_count:
        raise RuntimeError(
            f"edit occurrence {occurrence} exceeds {match_count} matches in {rel_path}"
        )

    return _replace_nth(current, before, after, occurrence)


def _apply_edits(run_dir: str, edits: List[Dict[str, Any]]) -> None:
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for edit in edits:
        path = edit.get("path")
        if not path or not isinstance(path, str):
            raise RuntimeError("edit missing path")
        by_path.setdefault(path, []).append(edit)

    for rel_path, file_edits in by_path.items():
        full_path = _resolve_run_path(run_dir, rel_path)
        if not os.path.exists(full_path):
            raise RuntimeError(f"edit path not found: {rel_path}")
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            original = f.read()
        updated = original
        for edit in file_edits:
            updated = _apply_single_edit(updated, edit, rel_path)
        if updated != original:
            _atomic_write_text(full_path, updated)


def _apply_candidate(run_dir: str, candidate: Dict[str, Any]) -> None:
    if "edits" in candidate and isinstance(candidate["edits"], list):
        _apply_edits(run_dir, candidate["edits"])
        return
    if "files" in candidate and isinstance(candidate["files"], list):
        for entry in candidate["files"]:
            rel_path = entry.get("path")
            content = entry.get("content")
            if not rel_path or content is None:
                raise RuntimeError("candidate files entry missing path/content")
            full_path = _resolve_run_path(run_dir, str(rel_path))
            _atomic_write_text(full_path, str(content))
        return
    raise RuntimeError("candidate missing edits/files")


def _normalize_ignore_patterns(patterns: List[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for pattern in patterns:
        if not pattern:
            continue
        p = str(pattern).strip().replace("\\", "/").rstrip("/")
        if not p or p in seen:
            continue
        seen.add(p)
        normalized.append(p)
    return normalized


def _path_matches_ignore(rel_path: str, name: str, patterns: List[str]) -> bool:
    rel_norm = rel_path.replace("\\", "/").lstrip("./")
    name_norm = name.replace("\\", "/")
    rel_segments = rel_norm.split("/") if rel_norm else []
    for pattern in patterns:
        if not pattern:
            continue
        pat = pattern.replace("\\", "/").rstrip("/")
        if not pat:
            continue
        wildcard = any(ch in pat for ch in "*?[]")
        if wildcard:
            if fnmatch.fnmatch(rel_norm, pat) or fnmatch.fnmatch(name_norm, pat):
                return True
            continue
        if "/" in pat:
            if rel_norm == pat or rel_norm.startswith(pat + "/"):
                return True
            continue
        if name_norm == pat:
            return True
        if pat in rel_segments:
            return True
    return False


def _copy_or_link(src: str, dst: str) -> None:
    _safe_copy2(src, dst)


def _copy_repo_with_mode(run_dir: str, ignore: List[str], copy_mode: str) -> None:
    normalized = _normalize_ignore_patterns(ignore)

    def _ignore(current_dir: str, names: List[str]) -> List[str]:
        rel_dir = os.path.relpath(current_dir, REPO_ROOT)
        rel_dir = "" if rel_dir in (".", "") else rel_dir
        ignored: List[str] = []
        for entry in names:
            rel_path = os.path.join(rel_dir, entry) if rel_dir else entry
            if _path_matches_ignore(rel_path, entry, normalized):
                ignored.append(entry)
        return ignored

    _ = copy_mode
    shutil.copytree(REPO_ROOT, run_dir, ignore=_ignore, copy_function=_safe_copy2)


def _score_candidate(metric_results: List[MetricResult], weights: Dict[str, float]) -> float:
    total_weight = 0.0
    score_sum = 0.0
    for result in metric_results:
        if result.score is None:
            continue
        weight = float(weights.get(result.name, 0.0))
        if weight <= 0:
            continue
        total_weight += weight
        score_sum += weight * float(result.score)
    if total_weight <= 0:
        return 0.0
    return score_sum / total_weight


def _int_with_default(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_with_default(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _run_benchmark_series(
    command: str,
    cwd: str,
    timeout_s: int,
    warmups: int,
    runs: int,
) -> Dict[str, Any]:
    effective_warmups = max(0, int(warmups))
    effective_runs = max(1, int(runs))
    total_runs = effective_warmups + effective_runs
    timings: List[float] = []
    out_tail = ""
    err_tail = ""

    for idx in range(total_runs):
        start = time.perf_counter()
        try:
            proc = _run_command(command, cwd, timeout_s)
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "details": f"timeout on run {idx + 1}/{total_runs}",
                "median_s": 0.0,
                "raw_stdout": out_tail,
                "raw_stderr": err_tail,
                "returncode": None,
            }
        elapsed = time.perf_counter() - start
        out_tail = proc.stdout[-2000:]
        err_tail = proc.stderr[-2000:]
        if proc.returncode != 0:
            return {
                "ok": False,
                "details": f"failed on run {idx + 1}/{total_runs} (rc={proc.returncode})",
                "median_s": 0.0,
                "raw_stdout": out_tail,
                "raw_stderr": err_tail,
                "returncode": proc.returncode,
            }
        if idx >= effective_warmups:
            timings.append(elapsed)

    if not timings:
        return {
            "ok": False,
            "details": "no measured runs",
            "median_s": 0.0,
            "raw_stdout": out_tail,
            "raw_stderr": err_tail,
            "returncode": 0,
        }
    return {
        "ok": True,
        "details": f"runs={len(timings)}",
        "median_s": _median(timings),
        "raw_stdout": out_tail,
        "raw_stderr": err_tail,
        "returncode": 0,
    }


def _build_metrics_from_profile(profile: Dict[str, Any], default_timeout_s: int) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    commands = profile.get("commands", [])
    if isinstance(commands, list):
        for entry in commands:
            if not isinstance(entry, dict):
                continue
            if not bool(entry.get("enabled", True)):
                continue
            name = str(entry.get("name", "") or "").strip()
            command = str(entry.get("command", "") or "").strip()
            if not name or not command:
                continue
            timeout_value = _int_with_default(entry.get("timeout_s"), default_timeout_s)
            if timeout_value <= 0:
                timeout_value = int(default_timeout_s)
            metrics.append(
                {
                    "name": name,
                    "type": "command",
                    "command": command,
                    "required": bool(entry.get("required", False)),
                    "weight": _float_with_default(entry.get("weight"), 0.0),
                    "parse_json": bool(entry.get("parse_json", False)),
                    "timeout_s": timeout_value,
                }
            )

    benchmark = profile.get("benchmark")
    if isinstance(benchmark, dict) and bool(benchmark.get("enabled", False)):
        name = str(benchmark.get("name", "workload") or "workload").strip()
        command = str(benchmark.get("command", "") or "").strip()
        if command:
            timeout_value = _int_with_default(benchmark.get("timeout_s"), default_timeout_s)
            if timeout_value <= 0:
                timeout_value = int(default_timeout_s)
            metrics.append(
                {
                    "name": name,
                    "type": "benchmark",
                    "command": command,
                    "required": bool(benchmark.get("required", False)),
                    "weight": _float_with_default(benchmark.get("weight"), 0.0),
                    "timeout_s": timeout_value,
                    "benchmark_warmups": max(0, _int_with_default(benchmark.get("warmups"), 1)),
                    "benchmark_runs": max(1, _int_with_default(benchmark.get("runs"), 5)),
                    "min_ratio": _float_with_default(benchmark.get("min_ratio"), 0.95),
                    "smaller_is_better": bool(benchmark.get("smaller_is_better", True)),
                }
            )
    return metrics


def _run_required_preflight(
    metrics_cfg: List[Dict[str, Any]],
    target: Target,
    default_timeout_s: int,
    repo_root: str,
) -> Optional[str]:
    for metric in metrics_cfg:
        if not bool(metric.get("required", False)):
            continue
        command = str(metric.get("command", "") or "").strip()
        if not command:
            continue
        name = str(metric.get("name", "metric") or "metric")
        timeout_value = _int_with_default(metric.get("timeout_s"), default_timeout_s)
        if timeout_value <= 0:
            timeout_value = int(default_timeout_s)
        command_fmt = command.format(
            target_path=target.path,
            target_start_line=target.start_line,
            target_end_line=target.end_line,
            run_dir=repo_root,
            repo_root=repo_root,
            generation=0,
            candidate_id="baseline",
        )
        mtype = str(metric.get("type", "command") or "command")
        if mtype == "benchmark":
            warmups = max(0, _int_with_default(metric.get("benchmark_warmups"), 1))
            runs = max(1, _int_with_default(metric.get("benchmark_runs"), 5))
            baseline = _run_benchmark_series(
                command=command_fmt,
                cwd=repo_root,
                timeout_s=timeout_value,
                warmups=warmups,
                runs=runs,
            )
            if not bool(baseline.get("ok", False)):
                return f"{name} preflight failed: {baseline.get('details', 'benchmark failed')}"
            continue

        try:
            proc = _run_command(command_fmt, repo_root, timeout_value)
        except subprocess.TimeoutExpired:
            return f"{name} preflight failed: timeout after {timeout_value}s"
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}"
            return f"{name} preflight failed: {detail}"
    return None


def _run_macro_prepare_once(
    metrics_cfg: List[Dict[str, Any]],
    target: Target,
    default_timeout_s: int,
    repo_root: str,
) -> Optional[str]:
    for metric in metrics_cfg:
        command = str(metric.get("command", "") or "").strip()
        if not command:
            continue
        if "evolve_score.py" not in command and "evolve_macro_score.py" not in command:
            continue
        name = str(metric.get("name", "macro") or "macro")
        timeout_value = _int_with_default(metric.get("timeout_s"), default_timeout_s)
        if timeout_value <= 0:
            timeout_value = int(default_timeout_s)
        command_fmt = command.format(
            target_path=target.path,
            target_start_line=target.start_line,
            target_end_line=target.end_line,
            run_dir=repo_root,
            repo_root=repo_root,
            generation=0,
            candidate_id="baseline",
        )
        if "--prepare-only" not in command_fmt:
            command_fmt = command_fmt + " --prepare-only"
        try:
            proc = _run_command(command_fmt, repo_root, timeout_value)
        except subprocess.TimeoutExpired:
            return f"{name} prepare failed: timeout after {timeout_value}s"
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}"
            return f"{name} prepare failed: {detail}"
    return None


def _rg_available() -> bool:
    return shutil.which("rg") is not None


def _rg_json_search(pattern: str, root: str, max_results: int) -> List[Tuple[str, int]]:
    if not _rg_available():
        return []
    cmd = ["rg", "--json", pattern, root]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    if proc.returncode not in (0, 1):
        return []
    results: List[Tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "match":
            continue
        data = record.get("data", {})
        path = data.get("path", {}).get("text")
        line_number = data.get("line_number")
        if not path or not line_number:
            continue
        results.append((path, int(line_number)))
        if max_results > 0 and len(results) >= max_results:
            break
    return results


def _should_skip_path(path: str, ignore_patterns: List[str]) -> bool:
    for pattern in ignore_patterns:
        if pattern.startswith("*.") and path.endswith(pattern[1:]):
            return True
        if pattern in path:
            return True
    return False


def _read_snippet(path: str, line_number: int, radius: int) -> Tuple[int, int, str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines:
        return 0, 0, ""
    idx = max(1, line_number)
    start = max(1, idx - radius)
    end = min(len(lines), idx + radius)
    snippet = "".join(lines[start - 1 : end])
    return start, end, snippet


def _collect_context(
    target_info: Dict[str, Any],
    symbols: List[str],
    files: List[str],
    providers: List[str],
    context_cfg: Dict[str, Any],
    ignore_patterns: List[str],
) -> Dict[str, Any]:
    max_chars = int(context_cfg.get("max_chars", 12000))
    max_snippets = int(context_cfg.get("max_snippets", 20))
    snippet_radius = int(context_cfg.get("snippet_radius", 6))
    max_files = int(context_cfg.get("max_files", 6))
    max_symbols = int(context_cfg.get("max_symbols", 12))
    adjacent_max_files = int(context_cfg.get("adjacent_max_files", 3))
    file_char_limit = int(context_cfg.get("max_file_chars", 2000))

    snippets: List[Dict[str, Any]] = []
    used_chars = 0

    def add_snippet(source: str, rel_path: str, start: int, end: int, content: str) -> None:
        nonlocal used_chars
        if not content:
            return
        if used_chars >= max_chars:
            return
        remaining = max_chars - used_chars
        clipped = content if len(content) <= remaining else content[:remaining]
        used_chars += len(clipped)
        snippets.append(
            {
                "source": source,
                "path": rel_path,
                "start_line": start,
                "end_line": end,
                "content": clipped,
            }
        )

    repo_root = REPO_ROOT
    if "explicit_files" in providers:
        for rel_path in files[:max_files]:
            full_path = os.path.join(repo_root, rel_path)
            if not os.path.exists(full_path) or _should_skip_path(rel_path, ignore_patterns):
                continue
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                add_snippet("explicit_files", rel_path, 1, content.count("\n") + 1, content[:file_char_limit])
            except OSError:
                continue

    if "adjacent_files" in providers:
        target_path = target_info.get("path", "")
        if target_path:
            folder = os.path.dirname(target_path)
            full_folder = os.path.join(repo_root, folder)
            if os.path.isdir(full_folder):
                entries = []
                for name in os.listdir(full_folder):
                    rel = os.path.join(folder, name) if folder else name
                    if rel == target_path:
                        continue
                    if _should_skip_path(rel, ignore_patterns):
                        continue
                    full = os.path.join(repo_root, rel)
                    if os.path.isfile(full):
                        entries.append(rel)
                for rel_path in entries[:adjacent_max_files]:
                    try:
                        with open(os.path.join(repo_root, rel_path), "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        add_snippet("adjacent_files", rel_path, 1, content.count("\n") + 1, content[:file_char_limit])
                    except OSError:
                        continue

    if "rg_symbols" in providers:
        symbol_list = symbols[:max_symbols]
        for symbol in symbol_list:
            if not symbol:
                continue
            matches = _rg_json_search(symbol, repo_root, max_snippets)
            for rel_path, line_number in matches:
                if _should_skip_path(rel_path, ignore_patterns):
                    continue
                full_path = os.path.join(repo_root, rel_path)
                if not os.path.exists(full_path):
                    continue
                start, end, content = _read_snippet(full_path, line_number, snippet_radius)
                add_snippet("rg_symbols", rel_path, start, end, content)
                if max_snippets > 0 and len(snippets) >= max_snippets:
                    break
            if max_snippets > 0 and len(snippets) >= max_snippets:
                break

    return {
        "symbols": symbols[:max_symbols],
        "files": files[:max_files],
        "snippets": snippets,
        "max_chars": max_chars,
    }


def _iter_result_generations(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    generations = result.get("generations")
    if isinstance(generations, list) and generations:
        return [g for g in generations if isinstance(g, dict)]
    candidates = result.get("candidates", [])
    promoted = result.get("promoted", [])
    return [{"generation": 1, "candidates": candidates, "promoted": promoted, "parents": []}]


def _build_evolution_log(
    run_id: str,
    created_at: str,
    report_path: str,
    results: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"Evolution Run: {run_id}")
    lines.append(f"Created At: {created_at}")
    lines.append(f"Report JSON: {report_path}")
    lines.append("")

    total_candidates = 0
    total_promoted = 0
    total_apply_failed = 0
    total_rejected = 0
    total_scored = 0
    best_score = 0.0
    best_id = ""

    for result in results:
        if not isinstance(result, dict):
            continue
        target = result.get("target", {}) if isinstance(result.get("target"), dict) else {}
        path = str(target.get("path", "unknown"))
        start_line = int(target.get("start_line", 0) or 0)
        end_line = int(target.get("end_line", 0) or 0)
        lines.append(f"Target: {path}:{start_line}-{end_line}")
        for gen in _iter_result_generations(result):
            generation = int(gen.get("generation", 1) or 1)
            parents = gen.get("parents", [])
            candidates = gen.get("candidates", [])
            promoted = gen.get("promoted", [])
            if not isinstance(candidates, list):
                candidates = []
            if not isinstance(promoted, list):
                promoted = []
            gen_scores = []
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                total_candidates += 1
                score = float(cand.get("score", 0.0) or 0.0)
                gen_scores.append(score)
                status = str(cand.get("status", "unknown"))
                if status == "apply_failed":
                    total_apply_failed += 1
                elif status == "rejected":
                    total_rejected += 1
                elif status == "scored":
                    total_scored += 1
                if score > best_score:
                    best_score = score
                    best_id = str(cand.get("run_candidate_id") or cand.get("id") or "")
            total_promoted += len(promoted)
            gen_best = max(gen_scores) if gen_scores else 0.0
            gen_avg = (sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0
            parent_text = ", ".join([str(p) for p in parents]) if parents else "none"
            lines.append(
                f"  Generation {generation}: candidates={len(candidates)} promoted={len(promoted)} "
                f"best={gen_best:.4f} avg={gen_avg:.4f} parents={parent_text}"
            )
            for p in promoted:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("run_candidate_id") or p.get("id") or "")
                pscore = float(p.get("score", 0.0) or 0.0)
                psummary = str(p.get("summary", "") or "").strip()
                if len(psummary) > 140:
                    psummary = psummary[:137] + "..."
                lines.append(f"    PROMOTED {pid} score={pscore:.4f} summary={psummary}")
        lines.append("")

    lines.append("Overview:")
    lines.append(f"  candidates={total_candidates}")
    lines.append(f"  promoted={total_promoted}")
    lines.append(f"  scored={total_scored}")
    lines.append(f"  rejected={total_rejected}")
    lines.append(f"  apply_failed={total_apply_failed}")
    lines.append(f"  best_candidate={best_id or 'n/a'}")
    lines.append(f"  best_score={best_score:.4f}")
    return "\n".join(lines) + "\n"


def _trim_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _compact_seed_candidate(report: Dict[str, Any], per_field_limit: int = 400) -> Dict[str, Any]:
    payload = report.get("candidate_payload", {})
    compact: Dict[str, Any] = {
        "id": report.get("run_candidate_id") or report.get("id") or "",
        "summary": str(report.get("summary", "") or ""),
        "score": float(report.get("score", 0.0) or 0.0),
        "generation": int(report.get("generation") or 0),
    }
    if not isinstance(payload, dict):
        return compact

    edits = payload.get("edits")
    if isinstance(edits, list):
        compact_edits: List[Dict[str, Any]] = []
        for edit in edits[:8]:
            if not isinstance(edit, dict):
                continue
            entry = {
                "path": edit.get("path"),
                "before": _trim_text(str(edit.get("before", "") or ""), per_field_limit),
                "after": _trim_text(str(edit.get("after", "") or ""), per_field_limit),
            }
            if "occurrence" in edit:
                entry["occurrence"] = edit.get("occurrence")
            compact_edits.append(entry)
        if compact_edits:
            compact["edits"] = compact_edits

    files = payload.get("files")
    if isinstance(files, list):
        compact_files: List[Dict[str, Any]] = []
        for file_entry in files[:4]:
            if not isinstance(file_entry, dict):
                continue
            compact_files.append(
                {
                    "path": file_entry.get("path"),
                    "content": _trim_text(str(file_entry.get("content", "") or ""), per_field_limit),
                }
            )
        if compact_files:
            compact["files"] = compact_files

    return compact


def _build_parent_pool(
    current_reports: List[Dict[str, Any]],
    archive_reports: List[Dict[str, Any]],
    pool_size: int,
    seed_from: str,
    max_chars: int,
) -> List[Dict[str, Any]]:
    if pool_size <= 0:
        return []

    source = seed_from.lower().strip()
    ranked_current = sorted(current_reports, key=lambda c: c.get("score", 0.0), reverse=True)
    scored_current = [c for c in ranked_current if c.get("status") == "scored"]
    promoted_current = [c for c in scored_current if bool(c.get("promoted", False))]

    if source == "top":
        primary = scored_current
    else:
        primary = promoted_current if promoted_current else scored_current

    combined = primary + [c for c in archive_reports if c.get("status") == "scored"]
    parents: List[Dict[str, Any]] = []
    seen_ids = set()
    used_chars = 0
    for report in combined:
        parent_id = report.get("run_candidate_id") or report.get("id")
        if not parent_id or parent_id in seen_ids:
            continue
        compact = _compact_seed_candidate(report)
        text_size = len(json.dumps(compact, ensure_ascii=True))
        if max_chars > 0 and used_chars + text_size > max_chars:
            continue
        parents.append(compact)
        seen_ids.add(parent_id)
        used_chars += text_size
        if len(parents) >= pool_size:
            break
    return parents


def _evaluate_candidates(
    candidates: List[Dict[str, Any]],
    run_root: str,
    run_id: str,
    generation_index: int,
    copy_ignore: List[str],
    copy_mode: str,
    execution_backend: str,
    execution_repo_root: str,
    execution_baseline_ref: str,
    metrics_cfg: List[Dict[str, Any]],
    metrics_weights: Dict[str, float],
    target: Target,
    timeout_s: int,
    parent_ids: List[str],
    source_guard_root: str,
    source_guard_fingerprint: str,
) -> List[Dict[str, Any]]:
    candidate_reports: List[Dict[str, Any]] = []
    benchmark_baseline_cache: Dict[str, Dict[str, Any]] = {}
    for idx, candidate in enumerate(candidates, start=1):
        _assert_tree_unchanged(source_guard_root, source_guard_fingerprint, "main source tree")
        if not isinstance(candidate, dict):
            continue
        raw_id = str(candidate.get("id") or f"candidate_{idx}")
        run_candidate_id = f"g{generation_index}_{raw_id}"
        candidate_dir = os.path.join(run_root, run_candidate_id)
        try:
            if execution_backend == "git_worktree":
                _create_candidate_worktree(
                    evolution_repo_root=execution_repo_root,
                    baseline_ref=execution_baseline_ref,
                    candidate_dir=candidate_dir,
                    run_id=run_id,
                    run_candidate_id=run_candidate_id,
                )
            else:
                _copy_repo_with_mode(candidate_dir, copy_ignore, copy_mode=copy_mode)
            _apply_candidate(candidate_dir, candidate)
        except Exception as e:
            candidate_reports.append(
                {
                    "id": raw_id,
                    "run_candidate_id": run_candidate_id,
                    "generation": generation_index,
                    "status": "apply_failed",
                    "error": str(e),
                    "parents": list(parent_ids),
                    "metrics": [],
                    "score": 0.0,
                    "summary": candidate.get("summary", ""),
                    "candidate_payload": candidate,
                }
            )
            _assert_tree_unchanged(source_guard_root, source_guard_fingerprint, "main source tree")
            continue

        change_summary = _summarize_candidate_effect(candidate_dir, execution_repo_root, candidate)
        if not bool(change_summary.get("effective_change", False)):
            candidate_reports.append(
                {
                    "id": raw_id,
                    "run_candidate_id": run_candidate_id,
                    "generation": generation_index,
                    "status": "rejected",
                    "error": "candidate produced no effective file changes",
                    "parents": list(parent_ids),
                    "metrics": [
                        MetricResult(
                            name="candidate_effect",
                            score=0.0,
                            passed=False,
                            required=True,
                            details="no_effective_change",
                        ).__dict__
                    ],
                    "score": 0.0,
                    "summary": candidate.get("summary", ""),
                    "candidate_payload": candidate,
                    "change_summary": change_summary,
                }
            )
            _assert_tree_unchanged(source_guard_root, source_guard_fingerprint, "main source tree")
            continue

        metric_results: List[MetricResult] = []
        rejected = False
        for metric in metrics_cfg:
            name = metric.get("name") or "metric"
            required = bool(metric.get("required", False))
            mtype = metric.get("type", "command")
            command = metric.get("command", "")
            timeout_value = _int_with_default(metric.get("timeout_s"), timeout_s)
            if timeout_value <= 0:
                timeout_value = int(timeout_s)
            parse_json = bool(metric.get("parse_json", False))
            if mtype not in {"command", "benchmark"} or not command:
                metric_results.append(
                    MetricResult(
                        name=name,
                        score=None,
                        passed=None,
                        required=required,
                        details="skipped (no command or unsupported type)",
                    )
                )
                continue
            if mtype == "benchmark":
                warmups = max(0, _int_with_default(metric.get("benchmark_warmups"), 1))
                runs = max(1, _int_with_default(metric.get("benchmark_runs"), 5))
                min_ratio = _float_with_default(metric.get("min_ratio"), 0.95)
                smaller_is_better = bool(metric.get("smaller_is_better", True))
                baseline_command = command.format(
                    target_path=target.path,
                    target_start_line=target.start_line,
                    target_end_line=target.end_line,
                    run_dir=execution_repo_root,
                    repo_root=execution_repo_root,
                    generation=0,
                    candidate_id="baseline",
                )
                candidate_command = command.format(
                    target_path=target.path,
                    target_start_line=target.start_line,
                    target_end_line=target.end_line,
                    run_dir=candidate_dir,
                    repo_root=execution_repo_root,
                    generation=generation_index,
                    candidate_id=run_candidate_id,
                )
                benchmark_key = f"{baseline_command}|{timeout_value}|{warmups}|{runs}"
                baseline = benchmark_baseline_cache.get(benchmark_key)
                if baseline is None:
                    baseline = _run_benchmark_series(
                        command=baseline_command,
                        cwd=execution_repo_root,
                        timeout_s=timeout_value,
                        warmups=warmups,
                        runs=runs,
                    )
                    benchmark_baseline_cache[benchmark_key] = baseline
                if not bool(baseline.get("ok", False)):
                    metric_results.append(
                        MetricResult(
                            name=name,
                            score=0.0,
                            passed=False,
                            required=required,
                            details=f"baseline_failed: {baseline.get('details', 'benchmark failed')}",
                            raw_stdout=str(baseline.get("raw_stdout", ""))[-2000:],
                            raw_stderr=str(baseline.get("raw_stderr", ""))[-2000:],
                            returncode=baseline.get("returncode"),
                        )
                    )
                    if required:
                        rejected = True
                    continue
                measured = _run_benchmark_series(
                    command=candidate_command,
                    cwd=candidate_dir,
                    timeout_s=timeout_value,
                    warmups=warmups,
                    runs=runs,
                )
                if not bool(measured.get("ok", False)):
                    metric_results.append(
                        MetricResult(
                            name=name,
                            score=0.0,
                            passed=False,
                            required=required,
                            details=f"candidate_failed: {measured.get('details', 'benchmark failed')}",
                            raw_stdout=str(measured.get("raw_stdout", ""))[-2000:],
                            raw_stderr=str(measured.get("raw_stderr", ""))[-2000:],
                            returncode=measured.get("returncode"),
                        )
                    )
                    if required:
                        rejected = True
                    continue
                baseline_median = float(baseline.get("median_s", 0.0) or 0.0)
                candidate_median = float(measured.get("median_s", 0.0) or 0.0)
                if baseline_median <= 0 or candidate_median <= 0:
                    ratio = 0.0
                elif smaller_is_better:
                    ratio = baseline_median / candidate_median
                else:
                    ratio = candidate_median / baseline_median
                passed = ratio >= min_ratio
                score = max(0.0, min(1.0, ratio))
                metric_results.append(
                    MetricResult(
                        name=name,
                        score=score,
                        passed=passed,
                        required=required,
                        details=(
                            f"baseline_median_s={baseline_median:.6f},"
                            f"candidate_median_s={candidate_median:.6f},"
                            f"ratio={ratio:.4f},min_ratio={min_ratio:.4f}"
                        ),
                        raw_stdout=str(measured.get("raw_stdout", ""))[-2000:],
                        raw_stderr=str(measured.get("raw_stderr", ""))[-2000:],
                        returncode=measured.get("returncode"),
                    )
                )
                if required and not passed:
                    rejected = True
                continue

            command_fmt = command.format(
                target_path=target.path,
                target_start_line=target.start_line,
                target_end_line=target.end_line,
                run_dir=candidate_dir,
                repo_root=execution_repo_root,
                generation=generation_index,
                candidate_id=run_candidate_id,
            )
            try:
                proc = _run_command(command_fmt, candidate_dir, timeout_value)
            except subprocess.TimeoutExpired:
                metric_results.append(
                    MetricResult(
                        name=name,
                        score=0.0,
                        passed=False,
                        required=required,
                        details="timeout",
                    )
                )
                if required:
                    rejected = True
                continue
            score = 1.0 if proc.returncode == 0 else 0.0
            details = ""
            if parse_json and proc.stdout:
                try:
                    parsed = json.loads(proc.stdout)
                    if isinstance(parsed, dict) and "score" in parsed:
                        score = float(parsed.get("score", score))
                        details = str(parsed.get("details", "")) if parsed.get("details") else ""
                        metric_status = str(parsed.get("status", "") or "").strip().lower()
                        if metric_status in {"neutral_fallback", "macro_skipped"}:
                            neutral_cap = _float_with_default(metric.get("neutral_score_cap"), 0.25)
                            score = min(score, neutral_cap)
                            if details:
                                details = f"{details}; neutral_score_cap={neutral_cap:.2f}"
                            else:
                                details = f"neutral_score_cap={neutral_cap:.2f}"
                except json.JSONDecodeError:
                    details = "invalid JSON output"
            metric_results.append(
                MetricResult(
                    name=name,
                    score=score,
                    passed=proc.returncode == 0,
                    required=required,
                    details=details,
                    raw_stdout=proc.stdout[-2000:],
                    raw_stderr=proc.stderr[-2000:],
                    returncode=proc.returncode,
                )
            )
            if required and proc.returncode != 0:
                rejected = True

        score = _score_candidate(metric_results, metrics_weights)
        status = "rejected" if rejected else "scored"
        candidate_report = {
            "id": raw_id,
            "run_candidate_id": run_candidate_id,
            "generation": generation_index,
            "status": status,
            "summary": candidate.get("summary", ""),
            "score": score,
            "parents": list(parent_ids),
            "metrics": [result.__dict__ for result in metric_results],
            "candidate_payload": candidate,
            "change_summary": change_summary,
        }
        _assert_tree_unchanged(source_guard_root, source_guard_fingerprint, "main source tree")
        candidate_reports.append(candidate_report)
    return candidate_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="General improvement pipeline runner.")
    parser.add_argument("--config", default="evolve/improve_config.json", help="Path to config JSON.")
    parser.add_argument("--target", default="", help="Override target path.")
    parser.add_argument("--generator-command", default="", help="Override generator command.")
    parser.add_argument("--inbox-dir", default="", help="Override inbox directory.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate and print candidates; do not apply or evaluate.",
    )
    args = parser.parse_args()

    config = _load_json(os.path.join(REPO_ROOT, args.config))
    runtime_evolve_settings = _load_evolve_runtime_settings()
    auto_apply_promoted = bool(runtime_evolve_settings.get("apply_promoted_candidates", False))
    candidate_output_dir = str(runtime_evolve_settings.get("candidate_dir", "") or "").strip()
    if not candidate_output_dir:
        candidate_output_dir = os.path.join(REPO_ROOT, "evolve", "candidates")
    run_lock_path = str(runtime_evolve_settings.get("run_lock_path", "") or "").strip()
    if not run_lock_path:
        run_lock_path = os.path.join(REPO_ROOT, "data", "evolve_improve.lock")
    elif not os.path.isabs(run_lock_path):
        run_lock_path = os.path.join(REPO_ROOT, run_lock_path)
    try:
        _hold_evolution_lock(run_lock_path)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 2
    autonomous_info: Dict[str, Any] = {}
    targets_cfg = config.get("targets") or []
    autonomous_targets: List[Target] = []
    if not args.target:
        try:
            autonomous_targets, autonomous_info = _resolve_targets_from_autonomous_reflection(config)
        except Exception as e:
            autonomous_cfg = config.get("autonomous", {})
            fallback_allowed = bool(autonomous_cfg.get("fallback_to_config_targets", True)) if isinstance(autonomous_cfg, dict) else True
            if not fallback_allowed or not targets_cfg:
                print(f"[error] autonomous reflection failed: {e}")
                return 2
            print(f"[warn] autonomous reflection failed; falling back to configured targets: {e}")
    if args.target:
        targets_cfg = [{"path": args.target}]
    targets = autonomous_targets if autonomous_targets else [Target(**t) for t in targets_cfg]
    if not targets:
        print("[error] no targets configured")
        return 2

    generator_cfg = config.get("generator", {})
    generator_cmd = args.generator_command or generator_cfg.get("command", "")
    generator_timeout = int(generator_cfg.get("timeout_s", 180))
    max_candidates = int(generator_cfg.get("max_candidates", 4))
    runs = int(generator_cfg.get("runs", 1))
    candidates_per_run = int(generator_cfg.get("candidates_per_run", 1))
    generator_fallback_cfg = generator_cfg.get("fallback", {}) if isinstance(generator_cfg.get("fallback"), dict) else {}
    inbox_dir = args.inbox_dir or config.get("candidate_source", {}).get("inbox_dir", "")
    configured_copy_ignore = config.get("copy_ignore", [])
    execution_cfg = config.get("execution", {})
    execution_backend = str(execution_cfg.get("backend", "git_worktree") or "git_worktree").lower().strip()
    if execution_backend not in {"copy", "git_worktree"}:
        print(f"[warn] unsupported execution.backend={execution_backend}; falling back to copy")
        execution_backend = "copy"
    execution_repo_dir_cfg = str(
        execution_cfg.get("evolution_repo_dir", "evolve/workspace/repo") or "evolve/workspace/repo"
    )
    requested_copy_mode = str(execution_cfg.get("copy_mode", "copy") or "copy").lower().strip()
    copy_mode = "copy"
    if requested_copy_mode != "copy":
        print(
            f"[warn] forcing execution.copy_mode=copy (requested={requested_copy_mode}) "
            "to prevent main-source edits"
        )
    default_copy_ignore = [
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "*.pyc",
        "data",
        "models",
        "screenshots",
        "reflection",
        "soul",
        "workspace",
        "evolve/base_models",
        "evolve/output",
        "evolve/runs",
        "evolve/candidates",
    ]
    copy_ignore = _normalize_ignore_patterns(default_copy_ignore + [str(p) for p in configured_copy_ignore])

    metrics_cfg = config.get("metrics", [])
    profile_cfg = config.get("project_profile", {})
    profile_enabled = bool(profile_cfg.get("enabled", False))
    profile_mode = str(profile_cfg.get("mode", "replace") or "replace").lower().strip()
    profile_preflight = bool(profile_cfg.get("baseline_preflight", True))
    profile_path = str(profile_cfg.get("file", "evolve/project_profile.json") or "evolve/project_profile.json")
    if profile_enabled:
        profile_abs = os.path.join(REPO_ROOT, profile_path)
        if not os.path.exists(profile_abs):
            print(f"[error] project_profile not found: {profile_path}")
            return 2
        try:
            profile_data = _load_json(profile_abs)
        except Exception as e:
            print(f"[error] failed to load project_profile: {e}")
            return 2
        profile_metrics = _build_metrics_from_profile(profile_data, generator_timeout)
        if not profile_metrics:
            print(f"[error] project_profile has no usable metrics: {profile_path}")
            return 2
        if profile_mode == "append":
            metrics_cfg = list(metrics_cfg) + profile_metrics
        else:
            metrics_cfg = profile_metrics

    metrics_weights = {m.get("name"): float(m.get("weight", 0.0)) for m in metrics_cfg if m.get("name")}
    source_guard_root = os.path.join(REPO_ROOT, "src")
    source_guard_fingerprint = _tree_fingerprint(source_guard_root)

    run_id = _now_stamp()
    run_root = os.path.join(REPO_ROOT, "evolve", "runs", run_id)
    os.makedirs(run_root, exist_ok=True)
    execution_repo_root = REPO_ROOT
    execution_baseline_ref = ""
    if execution_backend == "git_worktree" and not args.dry_run:
        execution_repo_root = (
            execution_repo_dir_cfg
            if os.path.isabs(execution_repo_dir_cfg)
            else os.path.join(REPO_ROOT, execution_repo_dir_cfg)
        )
        evolution_repo_rel = os.path.relpath(execution_repo_root, REPO_ROOT).replace("\\", "/")
        snapshot_ignore = _normalize_ignore_patterns(copy_ignore + [evolution_repo_rel])
        try:
            execution_repo_root, execution_baseline_ref = _prepare_evolution_repo(
                source_root=REPO_ROOT,
                evolution_repo_root=execution_repo_root,
                ignore_patterns=snapshot_ignore,
                run_id=run_id,
            )
        except Exception as e:
            print(f"[error] failed to prepare evolution repo: {e}")
            return 2

    selection_cfg = config.get("selection", {})
    min_score = float(selection_cfg.get("min_score", 0.0))
    top_k = int(selection_cfg.get("top_k", 1))

    context_cfg = config.get("context", {})
    context_enabled = bool(context_cfg.get("enabled", False))
    context_probe_command = context_cfg.get("probe_command", "")
    context_providers = context_cfg.get("providers", ["rg_symbols"])

    evolution_cfg = config.get("evolution", {})
    generations = max(1, int(evolution_cfg.get("generations", 1)))
    parent_pool_size = max(0, int(evolution_cfg.get("parent_pool_size", top_k)))
    seed_from = str(evolution_cfg.get("seed_from", "promoted"))
    seed_max_chars = max(0, int(evolution_cfg.get("seed_max_chars", 12000)))

    reporting_cfg = config.get("reporting", {})
    reporting_dashboard = bool(reporting_cfg.get("dashboard", True))
    dashboard_file = str(reporting_cfg.get("dashboard_file", "report.html") or "report.html")
    log_file = str(reporting_cfg.get("log_file", "evolution.log") or "evolution.log")
    dashboard_command = str(
        reporting_cfg.get(
            "dashboard_command",
            "python3 scripts/evolve_report.py --report {report_path} --output {output_path}",
        )
        or ""
    )

    all_reports: List[Dict[str, Any]] = []
    staged_candidates: List[Dict[str, Any]] = []
    dry_run_outputs: List[Dict[str, Any]] = []

    if autonomous_info.get("enabled"):
        print(
            "[ok] autonomous reflection: "
            f"run={autonomous_info.get('reflection_run_dir', '')} "
            f"dossiers={autonomous_info.get('selected_dossier_count', 0)} "
            f"targets={autonomous_info.get('target_count', 0)}"
        )

    for target in targets:
        target_info = _read_target(target)
        seen_candidate_signatures: set[str] = set()
        if profile_enabled and profile_preflight and not args.dry_run:
            preflight_error = _run_required_preflight(
                metrics_cfg=metrics_cfg,
                target=target,
                default_timeout_s=generator_timeout,
                repo_root=execution_repo_root,
            )
            if preflight_error:
                print(f"[error] baseline preflight failed for {target.path}: {preflight_error}")
                return 2
        if not args.dry_run:
            macro_prepare_error = _run_macro_prepare_once(
                metrics_cfg=metrics_cfg,
                target=target,
                default_timeout_s=generator_timeout,
                repo_root=execution_repo_root,
            )
            if macro_prepare_error:
                print(f"[error] macro prepare failed for {target.path}: {macro_prepare_error}")
                return 2
        base_payload = {
            "target": target_info,
            "objectives": {m.get("name"): m.get("weight", 0.0) for m in metrics_cfg if m.get("name")},
            "metrics": metrics_cfg,
            "max_candidates": max_candidates,
            "runs": runs,
            "candidates_per_run": candidates_per_run,
            "generator_fallback": generator_fallback_cfg,
        }
        reflection_payload = target.metadata.get("reflection") if isinstance(target.metadata, dict) else None
        if isinstance(reflection_payload, dict) and reflection_payload:
            base_payload["reflection"] = reflection_payload
        if context_enabled:
            probe_payload = {
                "target": target_info,
                "objectives": base_payload["objectives"],
                "max_symbols": int(context_cfg.get("max_symbols", 12)),
                "max_files": int(context_cfg.get("max_files", 6)),
            }
            symbols: List[str] = []
            files: List[str] = []
            notes = ""
            if context_probe_command:
                try:
                    probe_data = _run_json_command(context_probe_command, probe_payload, generator_timeout)
                    symbols = probe_data.get("symbols", []) if isinstance(probe_data.get("symbols"), list) else []
                    files = probe_data.get("files", []) if isinstance(probe_data.get("files"), list) else []
                    notes = str(probe_data.get("notes", "") or "")
                except Exception as e:
                    notes = f"probe_failed: {e}"
            symbols = [str(s) for s in symbols if s]
            files = [str(f) for f in files if f]
            context_data = _collect_context(
                target_info=target_info,
                symbols=symbols,
                files=files,
                providers=context_providers,
                context_cfg=context_cfg,
                ignore_patterns=copy_ignore,
            )
            context_data["notes"] = notes
            base_payload["context"] = context_data

        parent_pool: List[Dict[str, Any]] = []
        target_reports: List[Dict[str, Any]] = []
        dry_run_generations: List[Dict[str, Any]] = []
        archive_reports: List[Dict[str, Any]] = []
        for generation_index in range(1, generations + 1):
            payload = dict(base_payload)
            payload["generation"] = {
                "index": generation_index,
                "total": generations,
            }
            if parent_pool:
                payload["seed_candidates"] = parent_pool

            candidates, generation_meta = _collect_generation_candidates(
                generator_cmd=generator_cmd,
                payload=payload,
                timeout_s=generator_timeout,
                inbox_dir=inbox_dir,
                fallback_cfg=generator_fallback_cfg,
            )
            deduped_candidates: List[Dict[str, Any]] = []
            duplicate_count = 0
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                signature = _candidate_signature(candidate)
                if signature in seen_candidate_signatures:
                    duplicate_count += 1
                    continue
                seen_candidate_signatures.add(signature)
                deduped_candidates.append(candidate)
            candidates = deduped_candidates
            if max_candidates > 0:
                candidates = candidates[:max_candidates]
            if isinstance(generation_meta, dict):
                generation_meta["candidate_count"] = len(candidates)
                generation_meta["duplicate_candidate_count"] = duplicate_count
                fallback_mode = str(generation_meta.get("mode", "") or "")
                fallback_reason = str(generation_meta.get("reason", "") or "").strip()
                if fallback_mode and fallback_mode not in {"llm", "inbox"}:
                    message = f"[warn] generator fallback for {target.path} generation {generation_index}: {fallback_mode}"
                    if fallback_reason:
                        message += f" ({fallback_reason})"
                    if duplicate_count > 0:
                        message += f"; skipped_duplicates={duplicate_count}"
                    print(message)

            if args.dry_run:
                dry_run_generations.append(
                    {
                        "generation": generation_index,
                        "generator": generation_meta,
                        "parents": parent_pool,
                        "candidates": candidates,
                    }
                )
                continue

            parent_ids = [str(p.get("id")) for p in parent_pool if isinstance(p, dict) and p.get("id")]
            candidate_reports = _evaluate_candidates(
                candidates=candidates,
                run_root=run_root,
                run_id=run_id,
                generation_index=generation_index,
                copy_ignore=copy_ignore,
                copy_mode=copy_mode,
                execution_backend=execution_backend,
                execution_repo_root=execution_repo_root,
                execution_baseline_ref=execution_baseline_ref,
                metrics_cfg=metrics_cfg,
                metrics_weights=metrics_weights,
                target=target,
                timeout_s=generator_timeout,
                parent_ids=parent_ids,
                source_guard_root=source_guard_root,
                source_guard_fingerprint=source_guard_fingerprint,
            )
            _assert_tree_unchanged(source_guard_root, source_guard_fingerprint, "main source tree")
            candidate_reports.sort(key=lambda c: c.get("score", 0.0), reverse=True)
            top_candidates = [c for c in candidate_reports if c.get("status") == "scored"][:top_k]
            promoted: List[Dict[str, Any]] = []
            for cand in top_candidates:
                if cand.get("score", 0.0) < min_score:
                    cand["promoted"] = False
                    continue
                cand["promoted"] = True
                promoted.append(cand)
                staged_candidates.append(cand)
            for cand in candidate_reports:
                if "promoted" not in cand:
                    cand["promoted"] = False

            generation_report = {
                "generation": generation_index,
                "generator": generation_meta,
                "parents": parent_ids,
                "candidates": candidate_reports,
                "promoted": promoted,
            }
            target_reports.append(generation_report)
            archive_reports.extend(candidate_reports)
            archive_reports.sort(key=lambda c: c.get("score", 0.0), reverse=True)
            archive_reports = archive_reports[: max(20, parent_pool_size * 4)]

            parent_pool = _build_parent_pool(
                current_reports=candidate_reports,
                archive_reports=archive_reports,
                pool_size=parent_pool_size,
                seed_from=seed_from,
                max_chars=seed_max_chars,
            )

        if args.dry_run:
            dry_run_outputs.append({"target": target_info, "generations": dry_run_generations})
            continue

        if generations == 1 and target_reports:
            first = target_reports[0]
            all_reports.append(
                {
                    "target": target_info,
                    "reflection": reflection_payload if isinstance(reflection_payload, dict) else None,
                    "candidates": first.get("candidates", []),
                    "promoted": first.get("promoted", []),
                    "generations": target_reports,
                }
            )
        else:
            all_reports.append(
                {
                    "target": target_info,
                    "reflection": reflection_payload if isinstance(reflection_payload, dict) else None,
                    "generations": target_reports,
                }
            )

    if args.dry_run:
        sys.stdout.write(json.dumps({"targets": dry_run_outputs}, ensure_ascii=True, indent=2))
        return 0

    _assert_tree_unchanged(source_guard_root, source_guard_fingerprint, "main source tree")
    unique_staged: List[Dict[str, Any]] = []
    seen_staged_signatures = set()
    for cand in staged_candidates:
        signature = _candidate_signature(cand.get("candidate_payload", {}) if isinstance(cand, dict) else {})
        if signature in seen_staged_signatures:
            continue
        seen_staged_signatures.add(signature)
        unique_staged.append(cand)
    staged_candidates = unique_staged

    application_results: List[Dict[str, Any]] = []
    if auto_apply_promoted and staged_candidates:
        application_results = _apply_promoted_candidates_to_repo(REPO_ROOT, staged_candidates)

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_path = os.path.join(run_root, "report.json")
    _write_json(
        report_path,
        {
            "run_id": run_id,
            "created_at": created_at,
            "config": config,
            "autonomous": autonomous_info,
            "application": {
                "enabled": auto_apply_promoted,
                "results": application_results,
            },
            "results": all_reports,
        },
    )

    report_note = ""
    if reporting_dashboard:
        dashboard_path = os.path.join(run_root, dashboard_file)
        if dashboard_command:
            command = dashboard_command.format(
                report_path=report_path,
                output_path=dashboard_path,
                run_root=run_root,
                run_id=run_id,
                repo_root=REPO_ROOT,
            )
            try:
                proc = _run_command(command, REPO_ROOT, generator_timeout)
                if proc.returncode == 0:
                    report_note = f"[ok] dashboard: {dashboard_path}"
                else:
                    report_note = (
                        "[warn] dashboard generation failed: "
                        + (proc.stderr.strip() or proc.stdout.strip() or "unknown error")
                    )
            except Exception as e:
                report_note = f"[warn] dashboard generation failed: {e}"
    else:
        log_path = os.path.join(run_root, log_file)
        summary_log = _build_evolution_log(
            run_id=run_id,
            created_at=created_at,
            report_path=report_path,
            results=all_reports,
        )
        _write_text(log_path, summary_log)
        report_note = f"[ok] evolution log: {log_path}"

    application_by_id = {
        str(item.get("run_candidate_id", "") or ""): item
        for item in application_results
        if isinstance(item, dict)
    }

    if staged_candidates:
        candidates_dir = candidate_output_dir if os.path.isabs(candidate_output_dir) else os.path.join(REPO_ROOT, candidate_output_dir)
        os.makedirs(candidates_dir, exist_ok=True)
        for idx, cand in enumerate(staged_candidates, start=1):
            run_candidate_id = str(cand.get("run_candidate_id", "") or "")
            application = application_by_id.get(run_candidate_id)
            candidate_status = "STAGED"
            if isinstance(application, dict):
                application_status = str(application.get("status", "") or "").strip().upper()
                if application_status:
                    candidate_status = application_status
            out = {
                "candidate_id": f"candidate_{run_id}_{idx}",
                "source": "evolve_improve",
                "candidate": cand,
                "status": candidate_status,
                "application": application,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _write_json(os.path.join(candidates_dir, f"candidate_{run_id}_{idx}.json"), out)

    print(f"[ok] run report: {report_path}")
    if report_note:
        print(report_note)
    if auto_apply_promoted:
        applied = sum(1 for item in application_results if str(item.get("status", "") or "") == "applied")
        skipped = sum(1 for item in application_results if str(item.get("status", "") or "") == "skipped_conflict")
        failed = sum(1 for item in application_results if str(item.get("status", "") or "") == "apply_failed")
        print(f"[ok] auto-apply enabled: applied={applied} skipped_conflict={skipped} failed={failed}")
    if staged_candidates:
        print(f"[ok] staged {len(staged_candidates)} candidate(s) in {candidate_output_dir}")
    else:
        print("[warn] no candidates staged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
