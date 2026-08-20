from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from common.config import load_settings
from common.soul_mirror import load_config as load_soul_config
from common.soul_mirror import sync_soul
from reflection_engine.dossier_builder import ReflectionRun, build_reflection_run, load_config as load_reflection_config, write_reflection_run
from reflection_engine.observer import collect_runtime_observations, load_soul_snapshot


@dataclass(frozen=True)
class AutonomousTarget:
    path: str
    start_line: int = 0
    end_line: int = 0
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomousReflectionRun:
    run: ReflectionRun
    run_dir: Path
    selected_dossiers: Tuple[Dict[str, Any], ...]
    targets: Tuple[AutonomousTarget, ...]


def run_autonomous_reflection(
    repo_root: Path,
    reflection_config_path: Path,
    soul_config_path: Path,
    *,
    refresh_soul: bool = True,
    max_dossiers: int = 3,
    max_targets_per_dossier: int = 1,
    target_kinds: Sequence[str] = (),
    prefer_symbols: bool = True,
) -> AutonomousReflectionRun:
    reflection_config = load_reflection_config(reflection_config_path)
    soul_config = load_soul_config(soul_config_path)
    if refresh_soul:
        sync_soul(repo_root, soul_config)

    soul_root = repo_root / soul_config.output_dir
    soul_snapshot = load_soul_snapshot(soul_root)
    settings_path = repo_root / "config" / "settings.json"
    settings = load_settings(str(settings_path)) if settings_path.exists() else load_settings()
    runtime = collect_runtime_observations(
        repo_root=repo_root,
        data_dir=Path(settings.data_dir),
        max_record_events=reflection_config.max_record_events,
        max_complex_events=reflection_config.max_complex_events,
        max_error_events=reflection_config.max_error_events,
        max_audit_events=reflection_config.max_audit_events,
    )
    run = build_reflection_run(repo_root, reflection_config, soul_snapshot, runtime)
    run_dir = write_reflection_run(repo_root, reflection_config, run)

    allowed_kinds = {str(item).strip() for item in target_kinds if str(item).strip()}
    selected_dossiers: List[Dict[str, Any]] = []
    for dossier in run.dossiers:
        kind = str(dossier.get("kind", "") or "")
        if allowed_kinds and kind not in allowed_kinds:
            continue
        selected_dossiers.append(dict(dossier))
        if max_dossiers > 0 and len(selected_dossiers) >= max_dossiers:
            break

    targets = _build_targets_from_dossiers(
        selected_dossiers,
        max_targets_per_dossier=max_targets_per_dossier,
        prefer_symbols=prefer_symbols,
    )
    return AutonomousReflectionRun(
        run=run,
        run_dir=run_dir,
        selected_dossiers=tuple(selected_dossiers),
        targets=tuple(targets),
    )


def _build_targets_from_dossiers(
    dossiers: Sequence[Dict[str, Any]],
    *,
    max_targets_per_dossier: int,
    prefer_symbols: bool,
) -> List[AutonomousTarget]:
    targets: List[AutonomousTarget] = []
    seen: set[Tuple[str, int, int]] = set()
    per_dossier_limit = max(1, int(max_targets_per_dossier or 1))

    for dossier in dossiers:
        added_for_dossier = 0
        symbols = dossier.get("target_symbols", [])
        files = dossier.get("target_files", [])
        if prefer_symbols and isinstance(symbols, list):
            for symbol in symbols:
                if not isinstance(symbol, dict):
                    continue
                path = str(symbol.get("path", "") or "")
                start_line = int(symbol.get("start_line", 0) or 0)
                end_line = int(symbol.get("end_line", 0) or 0)
                if not path or start_line <= 0:
                    continue
                key = (path, start_line, max(start_line, end_line))
                if key in seen:
                    continue
                seen.add(key)
                targets.append(
                    AutonomousTarget(
                        path=path,
                        start_line=start_line,
                        end_line=max(start_line, end_line),
                        notes=_build_target_notes(dossier, symbol=symbol),
                        metadata={"reflection": _build_reflection_metadata(dossier, symbol=symbol)},
                    )
                )
                added_for_dossier += 1
                if added_for_dossier >= per_dossier_limit:
                    break
        if added_for_dossier > 0:
            continue
        if not isinstance(files, list):
            continue
        for raw_path in files:
            path = str(raw_path or "").strip()
            if not path:
                continue
            key = (path, 0, 0)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                AutonomousTarget(
                    path=path,
                    start_line=0,
                    end_line=0,
                    notes=_build_target_notes(dossier),
                    metadata={"reflection": _build_reflection_metadata(dossier)},
                )
            )
            break
    return targets


def _build_target_notes(dossier: Dict[str, Any], symbol: Dict[str, Any] | None = None) -> str:
    lines = [
        f"Dossier: {str(dossier.get('title', '') or '').strip()}",
        f"Kind: {str(dossier.get('kind', '') or '').strip()}",
        f"Problem: {str(dossier.get('problem_statement', '') or '').strip()}",
    ]
    if isinstance(symbol, dict):
        qualname = str(symbol.get("qualname", "") or "").strip()
        if qualname:
            lines.append(f"Focused symbol: {qualname}")
    affected = dossier.get("affected_modules", [])
    if isinstance(affected, list) and affected:
        lines.append("Affected modules: " + ", ".join(str(item) for item in affected[:4]))
    actions = dossier.get("suggested_actions", [])
    if isinstance(actions, list):
        for item in actions[:3]:
            text = str(item).strip()
            if text:
                lines.append("Suggested action: " + text)
    verification = dossier.get("verification_targets", [])
    if isinstance(verification, list):
        for item in verification[:2]:
            text = str(item).strip()
            if text:
                lines.append("Verification target: " + text)
    return "\n".join(line for line in lines if line.strip())


def _build_reflection_metadata(
    dossier: Dict[str, Any],
    symbol: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "dossier_id": str(dossier.get("dossier_id", "") or ""),
        "kind": str(dossier.get("kind", "") or ""),
        "title": str(dossier.get("title", "") or ""),
        "problem_statement": str(dossier.get("problem_statement", "") or ""),
        "score": float(dossier.get("score", 0.0) or 0.0),
        "confidence": float(dossier.get("confidence", 0.0) or 0.0),
        "affected_areas": [str(item) for item in dossier.get("affected_areas", []) if isinstance(item, str)],
        "affected_modules": [str(item) for item in dossier.get("affected_modules", []) if isinstance(item, str)],
        "suggested_actions": [str(item) for item in dossier.get("suggested_actions", []) if isinstance(item, str)],
        "verification_targets": [str(item) for item in dossier.get("verification_targets", []) if isinstance(item, str)],
        "rationale": [str(item) for item in dossier.get("rationale", []) if isinstance(item, str)],
        "evidence": _compact_evidence(dossier.get("evidence", [])),
        "target_files": [str(item) for item in dossier.get("target_files", []) if isinstance(item, str)],
    }
    if isinstance(symbol, dict):
        metadata["selected_symbol"] = {
            "path": str(symbol.get("path", "") or ""),
            "qualname": str(symbol.get("qualname", "") or ""),
            "kind": str(symbol.get("kind", "") or ""),
            "start_line": int(symbol.get("start_line", 0) or 0),
            "end_line": int(symbol.get("end_line", 0) or 0),
        }
    return metadata


def _compact_evidence(values: Any) -> List[Dict[str, Any]]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return []
    rows: List[Dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "source": str(item.get("source", "") or ""),
                "line": int(item.get("line", 0) or 0),
                "event_type": str(item.get("event_type", "") or ""),
                "summary": str(item.get("summary", "") or ""),
            }
        )
        if len(rows) >= 6:
            break
    return rows
