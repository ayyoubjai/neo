from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EvidenceRef:
    source: str
    line: int
    event_type: str
    trace_id: str = ""
    summary: str = ""


@dataclass
class TraceObservation:
    trace_id: str
    user_text: str = ""
    final_text: str = ""
    router_modes: List[str] = field(default_factory=list)
    create_tool_requests: List[Dict[str, Any]] = field(default_factory=list)
    tool_failures: List[Dict[str, Any]] = field(default_factory=list)
    critic_failures: List[Dict[str, Any]] = field(default_factory=list)
    parse_failures: List[Dict[str, Any]] = field(default_factory=list)
    error_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SoulSnapshot:
    root: Path
    manifest: Dict[str, Any]
    files: Tuple[Dict[str, Any], ...]
    files_by_path: Dict[str, Dict[str, Any]]
    symbols: Tuple[Dict[str, Any], ...]
    symbols_by_path: Dict[str, Tuple[Dict[str, Any], ...]]
    symbols_by_qualname: Dict[str, Tuple[Dict[str, Any], ...]]
    module_graph: Dict[str, Any]
    module_nodes: Tuple[Dict[str, Any], ...]
    modules_by_name: Dict[str, Dict[str, Any]]
    area_summaries: Tuple[Dict[str, Any], ...]
    areas_by_name: Dict[str, Dict[str, Any]]
    entrypoints: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class RuntimeObservations:
    traces: Tuple[TraceObservation, ...]
    summary: Dict[str, Any]
    source_paths: Dict[str, str]


def load_soul_snapshot(soul_root: Path) -> SoulSnapshot:
    meta_root = soul_root / "meta"
    manifest = _read_json(meta_root / "manifest.json")
    files = tuple(_read_jsonl(meta_root / "files.jsonl"))
    symbols = tuple(_read_jsonl(meta_root / "symbols.jsonl"))
    module_graph = _read_json(meta_root / "module_graph.json")
    area_summary_payload = _read_json(meta_root / "area_summaries.json")
    entrypoints = tuple(_read_jsonl(meta_root / "entrypoints.jsonl"))

    files_by_path = {str(item.get("path", "")): item for item in files if str(item.get("path", ""))}

    symbols_by_path_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    symbols_by_qualname_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in symbols:
        path = str(item.get("path", ""))
        qualname = str(item.get("qualname", ""))
        if path:
            symbols_by_path_map[path].append(item)
        if qualname:
            symbols_by_qualname_map[qualname].append(item)
    symbols_by_path = {path: tuple(rows) for path, rows in symbols_by_path_map.items()}
    symbols_by_qualname = {qualname: tuple(rows) for qualname, rows in symbols_by_qualname_map.items()}

    module_nodes = tuple(module_graph.get("nodes", [])) if isinstance(module_graph.get("nodes", []), list) else ()
    modules_by_name = {
        str(item.get("module", "")): item for item in module_nodes if str(item.get("module", ""))
    }

    area_rows = tuple(area_summary_payload.get("areas", [])) if isinstance(area_summary_payload.get("areas", []), list) else ()
    areas_by_name = {str(item.get("area", "")): item for item in area_rows if str(item.get("area", ""))}

    return SoulSnapshot(
        root=soul_root,
        manifest=manifest,
        files=files,
        files_by_path=files_by_path,
        symbols=symbols,
        symbols_by_path=symbols_by_path,
        symbols_by_qualname=symbols_by_qualname,
        module_graph=module_graph,
        module_nodes=module_nodes,
        modules_by_name=modules_by_name,
        area_summaries=area_rows,
        areas_by_name=areas_by_name,
        entrypoints=entrypoints,
    )


def collect_runtime_observations(
    repo_root: Path,
    data_dir: Path,
    max_record_events: int = 5000,
    max_complex_events: int = 5000,
    max_error_events: int = 1000,
    max_audit_events: int = 1000,
) -> RuntimeObservations:
    traces: Dict[str, TraceObservation] = {}
    record_path = data_dir / "record.log"
    complex_path = data_dir / "complex_events.log"
    error_path = data_dir / "error.log"
    audit_path = data_dir / "audit.log"

    for line_no, record in _iter_recent_jsonl(record_path, max_record_events):
        _handle_record_event(traces, record_path, line_no, record)
    for line_no, record in _iter_recent_jsonl(complex_path, max_complex_events):
        _handle_complex_event(traces, complex_path, line_no, record)
    for line_no, record in _iter_recent_jsonl(error_path, max_error_events):
        _handle_error_event(traces, repo_root, error_path, line_no, record)

    trace_rows = tuple(sorted(traces.values(), key=lambda item: item.trace_id))
    audit_action_counts = _count_audit_actions(audit_path, max_audit_events)
    summary = {
        "trace_count": len(trace_rows),
        "trace_failure_count": sum(
            1 for trace in trace_rows if trace.critic_failures or trace.tool_failures or trace.parse_failures or trace.error_events
        ),
        "critic_failure_count": sum(len(trace.critic_failures) for trace in trace_rows),
        "tool_failure_count": sum(len(trace.tool_failures) for trace in trace_rows),
        "parse_failure_count": sum(len(trace.parse_failures) for trace in trace_rows),
        "error_event_count": sum(len(trace.error_events) for trace in trace_rows),
        "create_tool_request_count": sum(len(trace.create_tool_requests) for trace in trace_rows),
        "audit_action_counts": audit_action_counts,
    }
    source_paths = {
        "record_log": str(record_path),
        "complex_events_log": str(complex_path),
        "error_log": str(error_path),
        "audit_log": str(audit_path),
    }
    return RuntimeObservations(traces=trace_rows, summary=summary, source_paths=source_paths)


def _trace_for(traces: Dict[str, TraceObservation], trace_id: str) -> TraceObservation:
    trace = traces.get(trace_id)
    if trace is None:
        trace = TraceObservation(trace_id=trace_id)
        traces[trace_id] = trace
    return trace


def _handle_record_event(
    traces: Dict[str, TraceObservation],
    source_path: Path,
    line_no: int,
    record: Dict[str, Any],
) -> None:
    event_type = str(record.get("event_type", "") or "")
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return
    trace_id = str(payload.get("trace_id", "") or "")
    if not trace_id:
        return
    trace = _trace_for(traces, trace_id)
    evidence = _make_evidence(source_path, line_no, event_type, trace_id, payload)

    if event_type == "user_query":
        text = payload.get("text")
        if isinstance(text, str) and text:
            trace.user_text = text
        return

    if event_type == "router_decision":
        mode = payload.get("mode") or payload.get("requested_mode") or payload.get("forced_mode")
        if isinstance(mode, str) and mode and mode not in trace.router_modes:
            trace.router_modes.append(mode)
        return

    if event_type == "assistant_final":
        text = payload.get("text")
        if isinstance(text, str) and text:
            trace.final_text = text
        return

    if event_type == "tool_response":
        status = str(payload.get("status", "") or "")
        error = payload.get("error")
        if status and status != "APPROVED" or (isinstance(error, str) and error):
            trace.tool_failures.append(
                {
                    "tool_id": str(payload.get("tool_id", "") or ""),
                    "status": status,
                    "error": str(error or ""),
                    "evidence": evidence,
                }
            )


def _handle_complex_event(
    traces: Dict[str, TraceObservation],
    source_path: Path,
    line_no: int,
    record: Dict[str, Any],
) -> None:
    event_type = str(record.get("event_type", "") or "")
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return
    trace_id = str(payload.get("trace_id", "") or "")
    if not trace_id:
        return
    trace = _trace_for(traces, trace_id)
    evidence = _make_evidence(source_path, line_no, event_type, trace_id, payload)

    if event_type == "turn_start":
        text = payload.get("text")
        if isinstance(text, str) and text:
            trace.user_text = text
        return

    if event_type == "turn_final":
        text = payload.get("text")
        if isinstance(text, str) and text:
            trace.final_text = text
        return

    if event_type in {"react_parsed", "react_repaired"}:
        parsed = payload.get("parsed", {})
        if isinstance(parsed, dict) and parsed.get("type") == "create_tool":
            spec = parsed.get("spec", {})
            if isinstance(spec, dict):
                trace.create_tool_requests.append(
                    {
                        "tool_id": str(spec.get("tool_id", "") or ""),
                        "spec": spec,
                        "reason": str(parsed.get("reason", "") or ""),
                        "thought": str(parsed.get("thought", "") or ""),
                        "evidence": evidence,
                    }
                )
        return

    if event_type == "react_step":
        observation = payload.get("observation", {})
        action = payload.get("action", {})
        if not isinstance(observation, dict):
            return
        status = str(observation.get("status", "") or "")
        error = observation.get("error")
        if status and status != "APPROVED" or (isinstance(error, str) and error):
            tool_id = ""
            if isinstance(action, dict):
                tool_id = str(action.get("tool_id") or action.get("type") or "")
            trace.tool_failures.append(
                {
                    "tool_id": tool_id,
                    "status": status,
                    "error": str(error or ""),
                    "evidence": evidence,
                }
            )
        return

    if event_type == "final_critic_result":
        fulfilled = payload.get("fulfilled")
        if fulfilled is False:
            trace.critic_failures.append(
                {
                    "confidence": float(payload.get("confidence", 0.0) or 0.0),
                    "issues": [str(item) for item in payload.get("issues", [])] if isinstance(payload.get("issues"), list) else [],
                    "fix_instructions": [str(item) for item in payload.get("fix_instructions", [])]
                    if isinstance(payload.get("fix_instructions"), list)
                    else [],
                    "retry_count": int(payload.get("retry_count", 0) or 0),
                    "evidence": evidence,
                }
            )
        return

    if event_type.endswith("parse_failed") or event_type.endswith("validation_error"):
        trace.parse_failures.append(
            {
                "event_type": event_type,
                "error": str(payload.get("error", "") or ""),
                "evidence": evidence,
            }
        )


def _handle_error_event(
    traces: Dict[str, TraceObservation],
    repo_root: Path,
    source_path: Path,
    line_no: int,
    record: Dict[str, Any],
) -> None:
    event_type = str(record.get("event_type", "") or "")
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return
    trace_id = str(payload.get("trace_id", "") or "")
    if not trace_id:
        trace_id = f"error:{line_no}"
    trace = _trace_for(traces, trace_id)
    evidence = _make_evidence(source_path, line_no, event_type, trace_id, payload)
    traceback_text = str(payload.get("traceback", "") or "")
    file_refs = _traceback_refs(traceback_text, repo_root)
    trace.error_events.append(
        {
            "error_type": str(payload.get("error_type", "") or ""),
            "error": str(payload.get("error", "") or ""),
            "traceback_files": file_refs,
            "evidence": evidence,
        }
    )


def _count_audit_actions(path: Path, max_events: int) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for _, record in _iter_recent_jsonl(path, max_events):
        action = str(record.get("action", "") or "")
        if action:
            counter[action] += 1
    return dict(sorted(counter.items()))


def _make_evidence(source_path: Path, line_no: int, event_type: str, trace_id: str, payload: Dict[str, Any]) -> EvidenceRef:
    return EvidenceRef(
        source=str(source_path),
        line=line_no,
        event_type=event_type,
        trace_id=trace_id,
        summary=_payload_summary(event_type, payload),
    )


def _payload_summary(event_type: str, payload: Dict[str, Any]) -> str:
    if event_type in {"user_query", "turn_start"}:
        text = payload.get("text")
        if isinstance(text, str):
            return _truncate(text, 180)
    if event_type == "final_critic_result":
        issues = payload.get("issues", [])
        if isinstance(issues, list) and issues:
            return _truncate("; ".join(str(item) for item in issues[:3]), 180)
    if event_type == "react_step":
        observation = payload.get("observation", {})
        if isinstance(observation, dict):
            error = observation.get("error")
            status = observation.get("status")
            if isinstance(error, str) and error:
                return _truncate(error, 180)
            if isinstance(status, str) and status:
                return status
    if event_type == "tool_response":
        error = payload.get("error")
        if isinstance(error, str) and error:
            return _truncate(error, 180)
        status = payload.get("status")
        if isinstance(status, str) and status:
            return status
    if "error" in payload and payload.get("error"):
        return _truncate(str(payload.get("error")), 180)
    return _truncate(json.dumps(payload, ensure_ascii=True), 180)


def _traceback_refs(traceback_text: str, repo_root: Path) -> List[str]:
    refs: List[str] = []
    pattern = re.compile(r'File "([^"]+)", line (\d+)')
    repo_root_resolved = repo_root.resolve()
    for match in pattern.finditer(traceback_text):
        raw_path = Path(match.group(1))
        try:
            rel = raw_path.resolve().relative_to(repo_root_resolved).as_posix()
        except Exception:
            continue
        refs.append(f"{rel}:{match.group(2)}")
    return refs


def _iter_recent_jsonl(path: Path, max_items: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if max_items <= 0 or not path.exists():
        return ()
    buffer: Deque[Tuple[int, str]] = deque(maxlen=max_items)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            buffer.append((line_no, raw))
    rows: List[Tuple[int, Dict[str, Any]]] = []
    for line_no, raw in buffer:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append((line_no, payload))
    return tuple(rows)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, payload in _iter_recent_jsonl(path, max_items=1_000_000):
        rows.append(payload)
    return rows


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"
