from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REFLECTION_CANDIDATE_SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_IGNORE = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.pyc",
    "data",
    "models",
    "workspace",
    "soul",
    "evolve/base_models",
    "evolve/output",
    "evolve/runs",
    "evolve/candidates",
    "reflection/runs",
    "reflection/candidates",
    "reflection/workspace",
)


@dataclass(frozen=True)
class ReflectionImproveConfig:
    runs_dir: str = "reflection/runs"
    candidates_dir: str = "reflection/candidates"
    workspace_repo_dir: str = "reflection/workspace/repo"
    backend: str = "git_worktree"
    max_context_files: int = 10
    max_file_chars: int = 12000
    snippet_radius: int = 6
    max_symbol_snippets: int = 10
    max_edit_files: int = 6
    max_diff_lines: int = 240
    copy_ignore: Tuple[str, ...] = DEFAULT_CANDIDATE_IGNORE


@dataclass(frozen=True)
class ReflectionRunData:
    run_dir: Path
    manifest: Dict[str, Any]
    observations: Dict[str, Any]
    dossiers: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class PreparedCandidate:
    candidate_id: str
    candidate_dir: Path
    worktree_dir: Path
    manifest: Dict[str, Any]
    proposal: Dict[str, Any]
    instructions_markdown: str


def load_improve_config(path: Path) -> ReflectionImproveConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ReflectionImproveConfig(
        runs_dir=str(payload.get("runs_dir", "reflection/runs") or "reflection/runs"),
        candidates_dir=str(payload.get("candidates_dir", "reflection/candidates") or "reflection/candidates"),
        workspace_repo_dir=str(payload.get("workspace_repo_dir", "reflection/workspace/repo") or "reflection/workspace/repo"),
        backend=str(payload.get("backend", "git_worktree") or "git_worktree").strip().lower(),
        max_context_files=int(payload.get("max_context_files", 10) or 10),
        max_file_chars=int(payload.get("max_file_chars", 12000) or 12000),
        snippet_radius=int(payload.get("snippet_radius", 6) or 6),
        max_symbol_snippets=int(payload.get("max_symbol_snippets", 10) or 10),
        max_edit_files=int(payload.get("max_edit_files", 6) or 6),
        max_diff_lines=int(payload.get("max_diff_lines", 240) or 240),
        copy_ignore=tuple(_normalize_ignore_patterns([str(item) for item in payload.get("copy_ignore", list(DEFAULT_CANDIDATE_IGNORE))])),
    )


def find_latest_reflection_run(runs_root: Path) -> Path:
    candidates: List[Path] = []
    if not runs_root.exists():
        raise RuntimeError(f"reflection runs directory does not exist: {runs_root}")
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "manifest.json").exists() and (entry / "dossiers.json").exists():
            candidates.append(entry)
    if not candidates:
        raise RuntimeError(f"no reflection runs found under {runs_root}")
    candidates.sort(key=lambda item: item.name)
    return candidates[-1]


def load_reflection_run(run_dir: Path) -> ReflectionRunData:
    manifest = _read_json(run_dir / "manifest.json")
    observations = _read_json(run_dir / "observations.json")
    dossiers_payload = _read_json(run_dir / "dossiers.json")
    dossiers_raw = dossiers_payload.get("dossiers", [])
    dossiers = tuple(item for item in dossiers_raw if isinstance(item, dict))
    if not dossiers:
        raise RuntimeError(f"reflection run has no dossiers: {run_dir}")
    return ReflectionRunData(
        run_dir=run_dir,
        manifest=manifest,
        observations=observations,
        dossiers=dossiers,
    )


def select_dossier(
    run: ReflectionRunData,
    dossier_id: str = "",
    kind: str = "",
) -> Dict[str, Any]:
    if dossier_id:
        for dossier in run.dossiers:
            if str(dossier.get("dossier_id", "")) == dossier_id:
                return dossier
        raise RuntimeError(f"dossier not found: {dossier_id}")
    if kind:
        for dossier in run.dossiers:
            if str(dossier.get("kind", "")) == kind:
                return dossier
        raise RuntimeError(f"no dossier found for kind={kind}")
    return dict(run.dossiers[0])


def prepare_candidate(
    repo_root: Path,
    config: ReflectionImproveConfig,
    run: ReflectionRunData,
    dossier: Dict[str, Any],
) -> PreparedCandidate:
    candidate_id = _build_candidate_id(dossier)
    candidate_dir = repo_root / config.candidates_dir / candidate_id
    worktree_dir = candidate_dir / "worktree"
    context_dir = candidate_dir / "context"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    context_dir.mkdir(parents=True, exist_ok=True)

    backend = config.backend
    if backend not in {"git_worktree", "copy"}:
        raise RuntimeError(f"unsupported reflection improvement backend: {backend}")

    if backend == "git_worktree":
        workspace_repo_dir = repo_root / config.workspace_repo_dir
        baseline_ref = _prepare_workspace_repo(repo_root, workspace_repo_dir, list(config.copy_ignore), candidate_id)
        _create_candidate_worktree(workspace_repo_dir, baseline_ref, worktree_dir, candidate_id)
    else:
        _copy_repo_snapshot(repo_root, worktree_dir, list(config.copy_ignore))

    _populate_runtime_inputs(worktree_dir, run.manifest)

    dossier_copy = dict(dossier)
    proposal = _build_proposal(repo_root, worktree_dir, config, run, dossier_copy)
    instructions_markdown = _render_instructions(candidate_id, worktree_dir, dossier_copy, proposal)
    manifest = {
        "schema_version": REFLECTION_CANDIDATE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "candidate_id": candidate_id,
        "reflection_run_dir": str(run.run_dir),
        "reflection_run_generated_at": run.manifest.get("generated_at", ""),
        "reflection_source_digest": run.manifest.get("soul_source_digest", ""),
        "dossier_id": str(dossier_copy.get("dossier_id", "")),
        "dossier_kind": str(dossier_copy.get("kind", "")),
        "backend": backend,
        "candidate_dir": str(candidate_dir),
        "worktree_dir": str(worktree_dir),
        "target_file_count": len(proposal.get("scope", {}).get("target_files", [])),
        "target_symbol_count": len(proposal.get("scope", {}).get("target_symbols", [])),
        "verification_command_count": len(proposal.get("verification", {}).get("commands", [])),
    }

    _write_json(candidate_dir / "manifest.json", manifest)
    _write_json(candidate_dir / "dossier.json", dossier_copy)
    _write_json(candidate_dir / "proposal.json", proposal)
    (candidate_dir / "instructions.md").write_text(instructions_markdown, encoding="utf-8")
    _write_json(context_dir / "files.json", _build_file_contexts(repo_root, proposal["scope"]["target_files"], config))
    _write_json(context_dir / "symbols.json", _build_symbol_contexts(repo_root, proposal["scope"]["target_symbols"], config))
    _write_json(context_dir / "evidence.json", _build_evidence_contexts(dossier_copy))

    return PreparedCandidate(
        candidate_id=candidate_id,
        candidate_dir=candidate_dir,
        worktree_dir=worktree_dir,
        manifest=manifest,
        proposal=proposal,
        instructions_markdown=instructions_markdown,
    )


def _build_candidate_id(dossier: Dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw = str(dossier.get("dossier_id", "") or dossier.get("title", "") or "candidate")
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw.lower()).strip("-")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{stamp}_{slug[:48] or 'candidate'}_{digest}"


def _build_proposal(
    repo_root: Path,
    worktree_dir: Path,
    config: ReflectionImproveConfig,
    run: ReflectionRunData,
    dossier: Dict[str, Any],
) -> Dict[str, Any]:
    target_files = [str(item) for item in dossier.get("target_files", []) if isinstance(item, str)]
    target_symbols = [item for item in dossier.get("target_symbols", []) if isinstance(item, dict)]
    constraints = _proposal_constraints(dossier, config)
    verification_commands = _verification_commands(dossier, target_files)
    return {
        "candidate_id": _build_proposal_id(dossier),
        "objective": str(dossier.get("problem_statement", "") or dossier.get("title", "")),
        "summary": str(dossier.get("title", "") or ""),
        "dossier": {
            "id": str(dossier.get("dossier_id", "")),
            "kind": str(dossier.get("kind", "")),
            "severity": str(dossier.get("severity", "")),
            "confidence": float(dossier.get("confidence", 0.0) or 0.0),
            "score": float(dossier.get("score", 0.0) or 0.0),
            "trace_id": str(dossier.get("trace_id", "") or ""),
        },
        "workspace": {
            "repo_root": str(repo_root),
            "candidate_worktree": str(worktree_dir),
            "reflection_run_dir": str(run.run_dir),
        },
        "scope": {
            "affected_areas": [str(item) for item in dossier.get("affected_areas", []) if isinstance(item, str)],
            "affected_modules": [str(item) for item in dossier.get("affected_modules", []) if isinstance(item, str)],
            "target_files": target_files,
            "target_symbols": target_symbols,
            "strict_targeting": True,
        },
        "change_budget": {
            "max_files_to_edit": min(max(len(target_files) + 2, 2), config.max_edit_files),
            "max_diff_lines": config.max_diff_lines,
        },
        "hypotheses": _proposal_hypotheses(dossier),
        "constraints": constraints,
        "verification": {
            "success_criteria": [str(item) for item in dossier.get("verification_targets", []) if isinstance(item, str)],
            "commands": verification_commands,
            "known_tests": _known_test_commands(target_files),
        },
        "suggested_actions": [str(item) for item in dossier.get("suggested_actions", []) if isinstance(item, str)],
        "evidence": [item for item in dossier.get("evidence", []) if isinstance(item, dict)],
        "rationale": [str(item) for item in dossier.get("rationale", []) if isinstance(item, str)],
    }


def _build_proposal_id(dossier: Dict[str, Any]) -> str:
    raw = str(dossier.get("dossier_id", "") or "proposal")
    return f"proposal-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:10]}"


def _proposal_constraints(dossier: Dict[str, Any], config: ReflectionImproveConfig) -> List[str]:
    constraints = [
        "Keep the fix inside the targeted area unless a directly affected dependency must change.",
        f"Do not exceed {config.max_edit_files} edited files without re-scoping the candidate.",
        f"Keep the diff under roughly {config.max_diff_lines} changed lines unless a smaller fix is impossible.",
        "Preserve existing public contracts unless the dossier explicitly requires a contract change.",
    ]
    kind = str(dossier.get("kind", ""))
    if kind == "trace_failure":
        trace_focus = _trace_failure_focus(dossier)
        constraints.append("Make the final user-visible answer match real execution state.")
        if trace_focus == "model_unavailable":
            constraints.extend(
                [
                    "Treat model unavailability as a hard runtime degradation, not as a successful fallback response.",
                    "Preserve any useful fallback path only if it stays explicit about degraded execution state.",
                ]
            )
        else:
            constraints.append("Propagate create_tool and execution errors instead of hiding them behind optimistic success text.")
    elif kind == "generated_capability_review":
        constraints.extend(
            [
                "Treat repeated create_tool requests as duplicate-state scenarios, not as fresh success cases.",
                "Require execution-backed readiness before claiming a generated capability is usable.",
            ]
        )
    elif kind == "structural_hotspot":
        constraints.extend(
            [
                "Prefer seam extraction over behavior changes.",
                "Preserve runtime behavior while reducing concentration and fan-out.",
            ]
        )
    return _dedup_strings(constraints)


def _proposal_hypotheses(dossier: Dict[str, Any]) -> List[Dict[str, Any]]:
    kind = str(dossier.get("kind", ""))
    statements: List[str] = []
    if kind == "trace_failure":
        trace_focus = _trace_failure_focus(dossier)
        if trace_focus == "model_unavailable":
            statements = [
                "The runtime is collapsing offline model failures into a misleading generic fallback instead of surfacing degraded execution truthfully.",
                "The final-response path is missing a strict contract for model-unavailable states, so reflection sees repeated backend outages without a stable recovery path.",
            ]
        else:
            statements = [
                "The orchestration path is treating a failed create_tool attempt as recoverable without updating the final response contract.",
                "The create_tool flow is missing an idempotent duplicate-state path for already-registered tool ids.",
            ]
    elif kind == "generated_capability_review":
        statements = [
            "The system is overusing on-demand generated tools for a repeated capability gap that should be explicit and stable.",
            "The generated-tool lifecycle does not have a strict ready/not-ready boundary tied to execution evidence.",
        ]
    elif kind == "structural_hotspot":
        statements = [
            "The module is accumulating too many unrelated responsibilities in one file.",
            "Dependency fan-out is high enough that small changes likely have broad side effects.",
        ]
    else:
        statements = ["The dossier identifies a change candidate that needs a smaller, better isolated implementation path."]
    return [{"text": text, "confidence": 0.7 + 0.05 * idx} for idx, text in enumerate(statements, start=0)]


def _verification_commands(dossier: Dict[str, Any], target_files: Sequence[str]) -> List[Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []
    python_targets = [path for path in target_files if path.endswith(".py")]
    if python_targets:
        quoted = " ".join(_shell_quote(path) for path in python_targets)
        commands.append(
            {
                "name": "py_compile_targets",
                "required": True,
                "command": f"python3 -m py_compile {quoted}",
                "purpose": "Catch syntax and import-time parse errors in the directly targeted Python files.",
            }
        )
    commands.append(
        {
            "name": "rerun_reflection",
            "required": False,
            "command": "python3 scripts/reflection_dossier.py",
            "purpose": "Rebuild soul plus reflection dossiers against the copied runtime evidence and compare the selected issue.",
        }
    )
    for item in _known_test_commands(target_files):
        commands.append(
            {
                "name": item["name"],
                "required": item["required"],
                "command": item["command"],
                "purpose": item["purpose"],
            }
        )
    return commands


def _known_test_commands(target_files: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    targets = set(target_files)
    if any(
        path.startswith("src/reflection_engine/")
        or path.startswith("scripts/reflection_")
        or path.startswith("config/reflection")
        for path in targets
    ):
        rows.append(
            {
                "name": "reflection_tests",
                "required": True,
                "command": "python3 -m unittest discover -s tests -p 'test_reflection_engine.py' -v",
                "purpose": "Validate the reflection pipeline after changing its code or config.",
            }
        )
    if any(
        path == "src/common/soul_mirror.py"
        or path.startswith("scripts/soul_mirror")
        or path.startswith("config/soul")
        for path in targets
    ):
        rows.append(
            {
                "name": "soul_tests",
                "required": True,
                "command": "python3 -m unittest discover -s tests -p 'test_soul_mirror.py' -v",
                "purpose": "Validate soul metadata generation after changing mirror logic.",
            }
        )
    return rows


def _render_instructions(
    candidate_id: str,
    worktree_dir: Path,
    dossier: Dict[str, Any],
    proposal: Dict[str, Any],
) -> str:
    lines = [
        f"# Candidate {candidate_id}",
        "",
        "## Objective",
        str(proposal.get("objective", "")),
        "",
        "## Workspace",
        str(worktree_dir),
        "",
        "## Scope",
        f"- Kind: {proposal['dossier']['kind']}",
        f"- Severity: {proposal['dossier']['severity']}",
        f"- Target files: {', '.join(proposal['scope']['target_files']) or 'n/a'}",
        f"- Affected modules: {', '.join(proposal['scope']['affected_modules']) or 'n/a'}",
        "",
        "## Constraints",
    ]
    for item in proposal.get("constraints", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Suggested Actions"])
    for item in proposal.get("suggested_actions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Verification"])
    for item in proposal.get("verification", {}).get("success_criteria", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Commands"])
    for item in proposal.get("verification", {}).get("commands", []):
        lines.append(f"- {item['command']}")
    if dossier.get("user_request"):
        lines.extend(["", "## User Request", str(dossier.get("user_request", ""))])
    return "\n".join(lines).strip() + "\n"


def _build_file_contexts(
    repo_root: Path,
    target_files: Sequence[str],
    config: ReflectionImproveConfig,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for rel_path in list(target_files)[: config.max_context_files]:
        path = repo_root / rel_path
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        rows.append(
            {
                "path": rel_path,
                "size_bytes": path.stat().st_size,
                "char_count": len(text),
                "content": _truncate(text, config.max_file_chars),
            }
        )
    return {"files": rows}


def _build_symbol_contexts(
    repo_root: Path,
    target_symbols: Sequence[Dict[str, Any]],
    config: ReflectionImproveConfig,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for symbol in list(target_symbols)[: config.max_symbol_snippets]:
        rel_path = str(symbol.get("path", "") or "")
        if not rel_path:
            continue
        path = repo_root / rel_path
        if not path.exists() or not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start_line = max(1, int(symbol.get("start_line", 1) or 1))
        end_line = max(start_line, int(symbol.get("end_line", start_line) or start_line))
        snippet_start = max(1, start_line - config.snippet_radius)
        snippet_end = min(len(lines), end_line + config.snippet_radius)
        excerpt = "\n".join(lines[snippet_start - 1 : snippet_end])
        rows.append(
            {
                "path": rel_path,
                "qualname": str(symbol.get("qualname", "")),
                "kind": str(symbol.get("kind", "")),
                "start_line": start_line,
                "end_line": end_line,
                "snippet_start_line": snippet_start,
                "snippet_end_line": snippet_end,
                "content": excerpt,
            }
        )
    return {"symbols": rows}


def _build_evidence_contexts(dossier: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for evidence in dossier.get("evidence", []):
        if not isinstance(evidence, dict):
            continue
        item = dict(evidence)
        source = str(item.get("source", "") or "")
        line = int(item.get("line", 0) or 0)
        if source and line > 0 and os.path.exists(source):
            raw = _read_file_line(Path(source), line)
            if raw:
                item["raw_line"] = raw
                try:
                    item["raw_json"] = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        rows.append(item)
    return {"evidence": rows}


def _populate_runtime_inputs(worktree_dir: Path, reflection_manifest: Dict[str, Any]) -> None:
    inputs = reflection_manifest.get("inputs", {})
    if not isinstance(inputs, dict):
        return
    data_dir = worktree_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for source_path in inputs.values():
        if not isinstance(source_path, str) or not source_path:
            continue
        src = Path(source_path)
        if not src.exists() or not src.is_file():
            continue
        shutil.copy2(src, data_dir / src.name)


def _prepare_workspace_repo(
    source_root: Path,
    workspace_repo_dir: Path,
    ignore_patterns: List[str],
    candidate_id: str,
) -> str:
    workspace_repo_dir.mkdir(parents=True, exist_ok=True)
    git_dir = workspace_repo_dir / ".git"
    if not git_dir.exists():
        _run_git(["init"], workspace_repo_dir)
        _run_git(["config", "user.name", "reflection-bot"], workspace_repo_dir)
        _run_git(["config", "user.email", "reflection-bot@example.local"], workspace_repo_dir)

    _reset_dir_keep_git(workspace_repo_dir)
    _copy_repo_snapshot(source_root, workspace_repo_dir, ignore_patterns)
    _run_git(["add", "-A"], workspace_repo_dir)
    status = _run_git(["status", "--porcelain"], workspace_repo_dir)
    if status.stdout.strip():
        _run_git(["commit", "-m", f"baseline {candidate_id}"], workspace_repo_dir)
    else:
        try:
            _run_git(["rev-parse", "--verify", "HEAD"], workspace_repo_dir)
        except RuntimeError:
            _run_git(["commit", "--allow-empty", "-m", f"baseline {candidate_id} (empty)"], workspace_repo_dir)
    return _run_git(["rev-parse", "HEAD"], workspace_repo_dir).stdout.strip()


def _create_candidate_worktree(
    workspace_repo_dir: Path,
    baseline_ref: str,
    worktree_dir: Path,
    candidate_id: str,
) -> None:
    branch_name = f"reflection/{_slug(candidate_id)}"
    _run_git(["worktree", "add", "-b", branch_name, str(worktree_dir), baseline_ref], workspace_repo_dir)


def _run_git(args: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    if shutil.which("git") is None:
        raise RuntimeError("git is not available on PATH")
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return proc


def _reset_dir_keep_git(path: Path) -> None:
    for entry in path.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _copy_repo_snapshot(src_root: Path, dst_root: Path, ignore_patterns: List[str]) -> None:
    normalized = _normalize_ignore_patterns(ignore_patterns)

    def _ignore(current_dir: str, names: List[str]) -> List[str]:
        rel_dir = os.path.relpath(current_dir, str(src_root))
        rel_dir = "" if rel_dir in (".", "") else rel_dir
        ignored: List[str] = []
        for entry in names:
            rel_path = os.path.join(rel_dir, entry) if rel_dir else entry
            if _path_matches_ignore(rel_path, entry, normalized):
                ignored.append(entry)
        return ignored

    shutil.copytree(src_root, dst_root, dirs_exist_ok=True, ignore=_ignore, copy_function=shutil.copy2)


def _normalize_ignore_patterns(patterns: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for pattern in patterns:
        value = str(pattern).strip().replace("\\", "/").rstrip("/")
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _path_matches_ignore(rel_path: str, name: str, patterns: Sequence[str]) -> bool:
    rel_norm = rel_path.replace("\\", "/").lstrip("./")
    name_norm = name.replace("\\", "/")
    rel_segments = rel_norm.split("/") if rel_norm else []
    for pattern in patterns:
        pat = str(pattern).replace("\\", "/").rstrip("/")
        if not pat:
            continue
        if any(ch in pat for ch in "*?[]"):
            if fnmatch.fnmatch(rel_norm, pat) or fnmatch.fnmatch(name_norm, pat):
                return True
            continue
        if "/" in pat:
            if rel_norm == pat or rel_norm.startswith(pat + "/"):
                return True
            continue
        if name_norm == pat or pat in rel_segments:
            return True
    return False


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _read_file_line(path: Path, line_no: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for idx, line in enumerate(handle, start=1):
                if idx == line_no:
                    return line.rstrip("\n")
    except OSError:
        return ""
    return ""


def _slug(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._/-]+", "-", value).strip("-")
    return collapsed or "candidate"


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 15)] + "...(truncated)"


def _dedup_strings(values: Sequence[str]) -> List[str]:
    rows: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _trace_failure_focus(dossier: Dict[str, Any]) -> str:
    for evidence in dossier.get("evidence", []):
        if not isinstance(evidence, dict):
            continue
        event_type = str(evidence.get("event_type", "") or "").strip().lower()
        summary = str(evidence.get("summary", "") or "").strip().lower()
        if event_type == "model_error" or "offline" in summary:
            return "model_unavailable"
    target_files = [str(item) for item in dossier.get("target_files", []) if isinstance(item, str)]
    if "src/model_server/text_model.py" in target_files:
        return "model_unavailable"
    return "execution_orchestration"
