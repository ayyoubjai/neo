from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from reflection_engine.observer import EvidenceRef, RuntimeObservations, SoulSnapshot, TraceObservation


DEFAULT_OUTPUT_DIR = "reflection/runs"
REFLECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReflectionConfig:
    output_dir: str = DEFAULT_OUTPUT_DIR
    max_record_events: int = 5000
    max_complex_events: int = 5000
    max_error_events: int = 1000
    max_audit_events: int = 1000
    max_dossiers: int = 8
    max_trace_failures: int = 4
    max_generated_capabilities: int = 3
    max_structural_hotspots: int = 3
    hotspot_score_threshold: float = 0.6
    hotspot_size_bytes_high: int = 180000
    hotspot_symbol_count_high: int = 80
    hotspot_dependency_count_high: int = 10
    hotspot_inbound_count_high: int = 10


@dataclass(frozen=True)
class ReflectionRun:
    manifest: Dict[str, Any]
    observations: Dict[str, Any]
    dossiers: Tuple[Dict[str, Any], ...]
    summary_markdown: str


def load_config(path: Path) -> ReflectionConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ReflectionConfig(
        output_dir=str(payload.get("output_dir", DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR),
        max_record_events=int(payload.get("max_record_events", 5000) or 5000),
        max_complex_events=int(payload.get("max_complex_events", 5000) or 5000),
        max_error_events=int(payload.get("max_error_events", 1000) or 1000),
        max_audit_events=int(payload.get("max_audit_events", 1000) or 1000),
        max_dossiers=int(payload.get("max_dossiers", 8) or 8),
        max_trace_failures=int(payload.get("max_trace_failures", 4) or 4),
        max_generated_capabilities=int(payload.get("max_generated_capabilities", 3) or 3),
        max_structural_hotspots=int(payload.get("max_structural_hotspots", 3) or 3),
        hotspot_score_threshold=float(payload.get("hotspot_score_threshold", 0.6) or 0.6),
        hotspot_size_bytes_high=int(payload.get("hotspot_size_bytes_high", 180000) or 180000),
        hotspot_symbol_count_high=int(payload.get("hotspot_symbol_count_high", 80) or 80),
        hotspot_dependency_count_high=int(payload.get("hotspot_dependency_count_high", 10) or 10),
        hotspot_inbound_count_high=int(payload.get("hotspot_inbound_count_high", 10) or 10),
    )


def build_reflection_run(
    repo_root: Path,
    config: ReflectionConfig,
    soul: SoulSnapshot,
    runtime: RuntimeObservations,
) -> ReflectionRun:
    trace_failure_dossiers = _build_trace_failure_dossiers(soul, runtime, config)
    generated_capability_dossiers = _build_generated_capability_dossiers(soul, runtime, config)
    structural_hotspot_dossiers = _build_structural_hotspot_dossiers(soul, config)
    dossier_rows = sorted(
        trace_failure_dossiers + generated_capability_dossiers + structural_hotspot_dossiers,
        key=_dossier_sort_key,
    )[: config.max_dossiers]

    observation_summary = {
        **runtime.summary,
        "source_paths": runtime.source_paths,
        "soul": {
            "schema_version": soul.manifest.get("schema_version"),
            "file_count": soul.manifest.get("file_count"),
            "symbol_count": soul.manifest.get("symbol_count"),
            "internal_dependency_edge_count": soul.manifest.get("internal_dependency_edge_count"),
            "area_count": soul.manifest.get("area_count"),
        },
    }
    manifest = {
        "schema_version": REFLECTION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "output_dir": config.output_dir,
        "soul_root": str(soul.root),
        "soul_source_digest": soul.manifest.get("source_digest", ""),
        "dossier_count": len(dossier_rows),
        "dossier_kinds": _count_by_key(dossier_rows, "kind"),
        "observation_counts": {
            "trace_count": runtime.summary.get("trace_count", 0),
            "trace_failure_count": runtime.summary.get("trace_failure_count", 0),
            "critic_failure_count": runtime.summary.get("critic_failure_count", 0),
            "tool_failure_count": runtime.summary.get("tool_failure_count", 0),
            "create_tool_request_count": runtime.summary.get("create_tool_request_count", 0),
        },
        "inputs": runtime.source_paths,
    }
    summary_markdown = _render_summary(manifest, observation_summary, dossier_rows)
    return ReflectionRun(
        manifest=manifest,
        observations=observation_summary,
        dossiers=tuple(dossier_rows),
        summary_markdown=summary_markdown,
    )


def write_reflection_run(repo_root: Path, config: ReflectionConfig, run: ReflectionRun) -> Path:
    run_root = repo_root / config.output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_hash = hashlib.sha256(run.summary_markdown.encode("utf-8")).hexdigest()[:8]
    run_dir = run_root / f"{stamp}_{run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "manifest.json", run.manifest)
    _write_json(run_dir / "observations.json", run.observations)
    _write_json(run_dir / "dossiers.json", {"schema_version": REFLECTION_SCHEMA_VERSION, "dossiers": list(run.dossiers)})
    (run_dir / "summary.md").write_text(run.summary_markdown, encoding="utf-8")
    return run_dir


def _build_trace_failure_dossiers(
    soul: SoulSnapshot,
    runtime: RuntimeObservations,
    config: ReflectionConfig,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    failing_traces = [
        trace
        for trace in runtime.traces
        if trace.critic_failures or trace.tool_failures or trace.parse_failures or trace.error_events
    ]
    ranked = sorted(
        failing_traces,
        key=lambda trace: (
            -_trace_failure_score(trace),
            trace.trace_id,
        ),
    )[: config.max_trace_failures]
    for trace in ranked:
        affected_areas = _trace_areas(trace)
        affected_modules = _trace_modules(trace, affected_areas)
        target_files = _trace_target_files(soul, trace, affected_areas)
        target_symbols = _trace_target_symbols(soul, trace, target_files)
        critic_issue_count = sum(len(item.get("issues", [])) for item in trace.critic_failures)
        tool_failure_count = len(trace.tool_failures)
        parse_failure_count = len(trace.parse_failures)
        score = _trace_failure_score(trace)
        severity = "high" if trace.critic_failures or trace.error_events else "medium"
        confidence = min(0.99, 0.65 + 0.08 * len(trace.critic_failures) + 0.06 * len(trace.tool_failures) + 0.05 * len(trace.parse_failures))
        evidence = _trace_evidence(trace)
        rationale = [
            f"The trace emitted {len(trace.critic_failures)} critic rejection(s), {tool_failure_count} tool failure(s), and {parse_failure_count} parse/validation failure(s).",
        ]
        first_critic = trace.critic_failures[0] if trace.critic_failures else None
        if isinstance(first_critic, dict) and first_critic.get("issues"):
            rationale.append(
                "Critic issues: "
                + "; ".join(str(item) for item in first_critic.get("issues", [])[:3])
            )
        if trace.tool_failures:
            rationale.append(
                "Tool failures: "
                + "; ".join(
                    f"{item.get('tool_id') or 'unknown'} -> {item.get('error') or item.get('status') or 'error'}"
                    for item in trace.tool_failures[:3]
                )
            )
        if trace.create_tool_requests:
            rationale.append(
                "The trace entered the create_tool path, so the failure spans orchestration, generated capability handling, and registry updates."
            )
        suggestions = _trace_suggestions(trace)
        verification = _trace_verification_targets(trace)
        problem_statement = _trace_problem_statement(trace)
        rows.append(
            {
                "dossier_id": _slug_id("trace_failure", trace.trace_id),
                "kind": "trace_failure",
                "title": f"Trace failure: {problem_statement}",
                "score": round(score, 3),
                "severity": severity,
                "confidence": round(confidence, 3),
                "problem_statement": problem_statement,
                "user_request": trace.user_text,
                "trace_id": trace.trace_id,
                "affected_areas": affected_areas,
                "affected_modules": affected_modules,
                "target_files": target_files,
                "target_symbols": target_symbols,
                "evidence": evidence,
                "rationale": rationale,
                "suggested_actions": suggestions,
                "verification_targets": verification,
                "metrics": {
                    "critic_issue_count": critic_issue_count,
                    "tool_failure_count": tool_failure_count,
                    "parse_failure_count": parse_failure_count,
                    "error_event_count": len(trace.error_events),
                },
            }
        )
    return rows


def _build_generated_capability_dossiers(
    soul: SoulSnapshot,
    runtime: RuntimeObservations,
    config: ReflectionConfig,
) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for trace in runtime.traces:
        if not trace.create_tool_requests:
            continue
        for request in trace.create_tool_requests:
            tool_id = str(request.get("tool_id", "") or "")
            if not tool_id:
                continue
            group = groups.setdefault(
                tool_id,
                {
                    "tool_id": tool_id,
                    "trace_ids": set(),
                    "requests": [],
                    "request_count": 0,
                    "critic_failures": 0,
                    "tool_failures": 0,
                    "user_texts": set(),
                    "evidence": [],
                    "counted_trace_ids": set(),
                },
            )
            group["trace_ids"].add(trace.trace_id)
            group["requests"].append(request)
            group["request_count"] += 1
            if trace.user_text:
                group["user_texts"].add(trace.user_text)
            if trace.trace_id not in group["counted_trace_ids"]:
                group["counted_trace_ids"].add(trace.trace_id)
                group["critic_failures"] += len(trace.critic_failures)
                group["tool_failures"] += len(trace.tool_failures)
            group["evidence"].append(request.get("evidence"))
            for item in trace.critic_failures[:1]:
                group["evidence"].append(item.get("evidence"))
            for item in trace.tool_failures[:1]:
                group["evidence"].append(item.get("evidence"))

    rows: List[Dict[str, Any]] = []
    ranked = sorted(
        groups.values(),
        key=lambda item: (
            -(0.55 + 0.12 * item["critic_failures"] + 0.08 * item["tool_failures"] + 0.05 * len(item["trace_ids"])),
            str(item["tool_id"]),
        ),
    )[: config.max_generated_capabilities]
    for group in ranked:
        tool_id = str(group["tool_id"])
        score = min(0.98, 0.55 + 0.12 * group["critic_failures"] + 0.08 * group["tool_failures"] + 0.05 * len(group["trace_ids"]))
        target_files = _existing_paths(
            soul,
            [
                "src/orchestrator/main.py",
                "src/tool_runtime/main.py",
                "src/tool_runtime/generated_tools.py",
                "config/tool_registry.json",
            ],
        )
        target_symbols = _resolve_symbol_refs(
            soul,
            [
                "Orchestrator._handle_create_tool",
                "Orchestrator._evaluate_tool_codegen_critic",
                "Orchestrator._validate_generated_tool_code",
                "Orchestrator._apply_final_response_critic",
            ],
            target_files=target_files,
        )
        rows.append(
            {
                "dossier_id": _slug_id("generated_capability", tool_id),
                "kind": "generated_capability_review",
                "title": f"Generated capability review: {tool_id}",
                "score": round(score, 3),
                "severity": "medium" if group["critic_failures"] == 0 and group["tool_failures"] == 0 else "high",
                "confidence": round(min(0.98, 0.7 + 0.06 * len(group["trace_ids"]) + 0.05 * group["critic_failures"]), 3),
                "problem_statement": (
                    f"{tool_id} was created on demand instead of being served by a stable first-class capability, "
                    "and the surrounding create_tool flow needs hardening."
                ),
                "affected_areas": ["src/orchestrator", "src/tool_runtime", "config"],
                "affected_modules": ["orchestrator.main"],
                "target_files": target_files,
                "target_symbols": target_symbols,
                "evidence": _compact_evidence(group["evidence"]),
                "rationale": [
                    f"The capability was requested {int(group['request_count'])} time(s) across {len(group['trace_ids'])} trace(s).",
                    f"It accumulated {group['critic_failures']} critic rejection(s) and {group['tool_failures']} execution error(s).",
                    "Generated capabilities are high leverage because they touch orchestration, registry state, and runtime execution paths at the same time.",
                ],
                "suggested_actions": [
                    f"Promote {tool_id} to a stable native capability or harden the generated-tool lifecycle around it.",
                    "Make repeated create_tool requests idempotent within the same trace and across existing registry state.",
                    "Require execution-backed evidence before the final answer claims that a generated capability is ready.",
                ],
                "verification_targets": [
                    f"Replay a request for {tool_id} and require zero tool execution errors.",
                    "Require final_critic_result.fulfilled=true for the same user intent.",
                    "Add a regression test for repeated create_tool requests that target the same tool_id.",
                ],
                "metrics": {
                    "trace_count": len(group["trace_ids"]),
                    "request_count": int(group["request_count"]),
                    "critic_failure_count": group["critic_failures"],
                    "tool_failure_count": group["tool_failures"],
                },
                "sample_requests": sorted(str(item) for item in group["user_texts"])[:3],
            }
        )
    return rows


def _build_structural_hotspot_dossiers(soul: SoulSnapshot, config: ReflectionConfig) -> List[Dict[str, Any]]:
    edges = soul.module_graph.get("edges", [])
    inbound_counts: Dict[str, int] = {}
    outbound_counts: Dict[str, int] = {}
    for edge in edges if isinstance(edges, list) else []:
        source_module = str(edge.get("source_module", ""))
        target_module = str(edge.get("target_module", ""))
        outbound_counts[source_module] = outbound_counts.get(source_module, 0) + 1
        inbound_counts[target_module] = inbound_counts.get(target_module, 0) + 1

    candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for node in soul.module_nodes:
        module_name = str(node.get("module", "") or "")
        path = str(node.get("path", "") or "")
        if not module_name or not path:
            continue
        file_meta = soul.files_by_path.get(path, {})
        size_bytes = int(file_meta.get("size_bytes", 0) or 0)
        symbol_count = int(node.get("symbol_count", 0) or 0)
        outbound = outbound_counts.get(module_name, 0)
        inbound = inbound_counts.get(module_name, 0)
        total_deps = outbound + inbound
        score = (
            0.4 * min(size_bytes / max(config.hotspot_size_bytes_high, 1), 1.5)
            + 0.3 * min(symbol_count / max(config.hotspot_symbol_count_high, 1), 1.5)
            + 0.2 * min(total_deps / max(config.hotspot_dependency_count_high, 1), 1.5)
            + 0.1 * min(inbound / max(config.hotspot_inbound_count_high, 1), 1.5)
        )
        if score < config.hotspot_score_threshold:
            continue
        area_name = str(node.get("area", "") or "")
        area_summary = soul.areas_by_name.get(area_name, {})
        candidates.append((score, node, file_meta, area_summary))

    rows: List[Dict[str, Any]] = []
    for score, node, file_meta, area_summary in sorted(candidates, key=lambda item: (-item[0], str(item[1].get("module", ""))))[
        : config.max_structural_hotspots
    ]:
        module_name = str(node.get("module", ""))
        path = str(node.get("path", ""))
        symbols = list(soul.symbols_by_path.get(path, ()))
        symbols.sort(key=lambda item: (-(int(item.get("end_line", 0) or 0) - int(item.get("start_line", 0) or 0)), str(item.get("qualname", ""))))
        target_symbols = [
            {
                "path": path,
                "qualname": str(item.get("qualname", "")),
                "kind": str(item.get("kind", "")),
                "start_line": int(item.get("start_line", 0) or 0),
                "end_line": int(item.get("end_line", 0) or 0),
            }
            for item in symbols[:5]
        ]
        inbound = sum(1 for edge in soul.module_graph.get("edges", []) if str(edge.get("target_module", "")) == module_name)
        outbound = sum(1 for edge in soul.module_graph.get("edges", []) if str(edge.get("source_module", "")) == module_name)
        rows.append(
            {
                "dossier_id": _slug_id("structural_hotspot", module_name),
                "kind": "structural_hotspot",
                "title": f"Structural hotspot: {module_name}",
                "score": round(score, 3),
                "severity": "high" if score >= 0.95 else "medium",
                "confidence": round(min(0.99, 0.68 + 0.2 * min(score, 1.0)), 3),
                "problem_statement": (
                    f"{module_name} concentrates unusually high structural weight and is a likely improvement target."
                ),
                "affected_areas": [str(node.get("area", ""))] if str(node.get("area", "")) else [],
                "affected_modules": [module_name],
                "target_files": _existing_paths(soul, [path]),
                "target_symbols": target_symbols,
                "evidence": [
                    {
                        "source": "soul/meta/files.jsonl",
                        "path": path,
                        "summary": f"size_bytes={int(file_meta.get('size_bytes', 0) or 0)} line_count={int(file_meta.get('line_count', 0) or 0)}",
                    },
                    {
                        "source": "soul/meta/module_graph.json",
                        "path": path,
                        "summary": f"symbol_count={int(node.get('symbol_count', 0) or 0)} inbound={inbound} outbound={outbound}",
                    },
                ],
                "rationale": [
                    f"The module has {int(file_meta.get('size_bytes', 0) or 0)} bytes and {int(node.get('symbol_count', 0) or 0)} indexed symbols.",
                    f"It participates in {outbound} outbound and {inbound} inbound internal dependency edges.",
                    f"Area summary: {str(area_summary.get('summary', '') or '').strip()}",
                ],
                "suggested_actions": _structural_suggestions(module_name, path, inbound, outbound),
                "verification_targets": [
                    f"Keep {module_name} behavior stable under focused tests.",
                    "Reduce or at least not increase its dependency fan-out while refactoring.",
                    "Preserve the current public entrypoints and contracts that depend on this module.",
                ],
                "metrics": {
                    "size_bytes": int(file_meta.get("size_bytes", 0) or 0),
                    "line_count": int(file_meta.get("line_count", 0) or 0),
                    "symbol_count": int(node.get("symbol_count", 0) or 0),
                    "inbound_dependency_count": inbound,
                    "outbound_dependency_count": outbound,
                },
            }
        )
    return rows


def _dossier_sort_key(item: Dict[str, Any]) -> Tuple[int, float, str]:
    kind_order = {
        "trace_failure": 0,
        "generated_capability_review": 1,
        "structural_hotspot": 2,
    }
    kind = str(item.get("kind", ""))
    return (kind_order.get(kind, 9), -float(item.get("score", 0.0)), str(item.get("title", "")))


def _trace_problem_statement(trace: TraceObservation) -> str:
    if _trace_has_model_unavailable_error(trace):
        return "model backend was unavailable during a user trace"
    if trace.create_tool_requests:
        tool_id = str(trace.create_tool_requests[0].get("tool_id", "") or "generated capability")
        return f"{tool_id} create_tool flow produced a non-robust result"
    if trace.tool_failures:
        tool_id = str(trace.tool_failures[0].get("tool_id", "") or "tool execution")
        return f"{tool_id} execution failed during a user trace"
    if trace.parse_failures:
        return "planner/react parsing failed during a user trace"
    return "the trace failed its internal quality checks"


def _trace_failure_score(trace: TraceObservation) -> float:
    score = 0.45
    score += 0.18 * len(trace.critic_failures)
    score += 0.12 * len(trace.tool_failures)
    score += 0.1 * len(trace.parse_failures)
    score += 0.08 * len(trace.error_events)
    if trace.create_tool_requests:
        score += 0.08
    return min(0.99, score)


def _trace_areas(trace: TraceObservation) -> List[str]:
    areas = {"src/orchestrator"}
    if trace.create_tool_requests:
        areas.update({"src/tool_runtime", "config"})
    for failure in trace.tool_failures:
        tool_id = str(failure.get("tool_id", "") or "")
        areas.update(_tool_id_areas(tool_id))
    for failure in trace.error_events:
        for ref in failure.get("traceback_files", []):
            area = _path_to_area(str(ref).split(":", 1)[0])
            if area:
                areas.add(area)
    return sorted(areas)


def _trace_modules(trace: TraceObservation, affected_areas: Sequence[str]) -> List[str]:
    modules = {"orchestrator.main"}
    if trace.create_tool_requests:
        modules.add("tool_runtime.main")
    for area in affected_areas:
        if area == "src/model_server":
            modules.add("model_server.main")
        elif area == "src/tool_runtime":
            modules.add("tool_runtime.main")
    for rel_path in _traceback_paths(trace):
        module_name = _path_to_module(rel_path)
        if module_name:
            modules.add(module_name)
    return sorted(modules)


def _trace_target_files(soul: SoulSnapshot, trace: TraceObservation, affected_areas: Sequence[str]) -> List[str]:
    candidates = ["src/orchestrator/main.py"]
    if trace.create_tool_requests:
        candidates.extend(["src/tool_runtime/main.py", "src/tool_runtime/generated_tools.py", "config/tool_registry.json"])
    for area in affected_areas:
        area_summary = soul.areas_by_name.get(area, {})
        if isinstance(area_summary.get("key_files"), list):
            for item in area_summary.get("key_files", [])[:3]:
                candidates.append(str(item))
    for failure in trace.error_events:
        for ref in failure.get("traceback_files", []):
            candidates.append(str(ref).split(":", 1)[0])
    return _existing_paths(soul, candidates)


def _trace_target_symbols(soul: SoulSnapshot, trace: TraceObservation, target_files: Sequence[str]) -> List[Dict[str, Any]]:
    names = ["Orchestrator._apply_final_response_critic", "Orchestrator._evaluate_final_response_critic"]
    if trace.create_tool_requests:
        names.extend(
            [
                "Orchestrator._handle_create_tool",
                "Orchestrator._evaluate_tool_codegen_critic",
                "Orchestrator._validate_generated_tool_code",
            ]
        )
    if trace.parse_failures:
        names.append("Orchestrator._run_cognition_loop")
    if _trace_has_model_unavailable_error(trace):
        names.extend(["generate", "ModelService.Generate"])
    rows = _resolve_symbol_refs(soul, names, target_files=target_files)
    if _trace_has_model_unavailable_error(trace):
        rows.sort(
            key=lambda item: (
                0 if str(item.get("path", "")) == "src/model_server/text_model.py" else 1,
                0 if str(item.get("qualname", "")) == "generate" else 1,
                str(item.get("path", "")),
                int(item.get("start_line", 0) or 0),
            )
        )
    return rows


def _trace_evidence(trace: TraceObservation) -> List[Dict[str, Any]]:
    refs: List[Any] = []
    if trace.user_text:
        refs.append(EvidenceRef(source="user_query", line=0, event_type="user_query", trace_id=trace.trace_id, summary=trace.user_text))
    for item in trace.critic_failures:
        refs.append(item.get("evidence"))
    for item in trace.tool_failures:
        refs.append(item.get("evidence"))
    for item in trace.parse_failures:
        refs.append(item.get("evidence"))
    for item in trace.error_events:
        refs.append(item.get("evidence"))
    return _compact_evidence(refs)


def _trace_suggestions(trace: TraceObservation) -> List[str]:
    suggestions = [
        "Make the final answer reflect execution reality instead of optimistic assumptions.",
        "Add a focused regression test that replays the same user intent end-to-end.",
    ]
    if _trace_has_model_unavailable_error(trace):
        suggestions.extend(
            [
                "Return a clear model-unavailable response instead of optimistic generic fallback text when the backend is offline.",
                "Treat offline model generation as a first-class failure signal in the final-response path and in reflection evidence.",
                "Add a regression test that simulates provider offline errors and verifies the user-visible response stays truthful.",
            ]
        )
    if trace.create_tool_requests:
        suggestions.extend(
            [
                "Make create_tool retries idempotent against the existing registry and generated tool set.",
                "Block 'tool is ready' claims until execution evidence shows the generated capability actually works.",
            ]
        )
    if trace.tool_failures:
        suggestions.append("Propagate tool execution errors into the orchestration decision instead of continuing as if the tool succeeded.")
    if trace.parse_failures:
        suggestions.append("Harden JSON validation and repair around the react/create_tool path.")
    return _dedup_strings(suggestions)


def _trace_verification_targets(trace: TraceObservation) -> List[str]:
    targets = ["Replay the same user request and require a clean final answer."]
    if _trace_has_model_unavailable_error(trace):
        targets.append("Simulate an offline model backend and require an explicit runtime-issue response instead of generic assistance text.")
    if trace.critic_failures:
        targets.append("Require final_critic_result.fulfilled=true for the replayed trace.")
    if trace.tool_failures:
        targets.append("Require zero tool execution errors for the replayed trace.")
    if trace.create_tool_requests:
        targets.append("Ensure repeated create_tool requests for the same tool_id do not fail with duplicate-state errors.")
    return _dedup_strings(targets)


def _tool_id_areas(tool_id: str) -> List[str]:
    if not tool_id:
        return []
    if tool_id.startswith("vision.") or tool_id == "ui.predict_coords":
        return ["src/model_server", "src/tool_runtime"]
    if tool_id.startswith(("ui.", "fs.", "math.", "net.", "py.", "sys.")):
        return ["src/tool_runtime"]
    if tool_id == "create_tool":
        return ["src/tool_runtime", "config"]
    return ["src/tool_runtime"]


def _trace_has_model_unavailable_error(trace: TraceObservation) -> bool:
    for failure in trace.error_events:
        evidence = failure.get("evidence")
        if isinstance(evidence, EvidenceRef) and evidence.event_type == "model_error":
            return True
        error_text = str(failure.get("error", "") or "").strip().lower()
        error_type = str(failure.get("error_type", "") or "").strip().lower()
        if "offline" in error_text or error_type in {"ollamaerror", "hferror"}:
            return True
    return False


def _traceback_paths(trace: TraceObservation) -> List[str]:
    rows: List[str] = []
    seen: set[str] = set()
    for failure in trace.error_events:
        refs = failure.get("traceback_files", [])
        if not isinstance(refs, list):
            continue
        for ref in refs:
            rel_path = str(ref).split(":", 1)[0].strip()
            if not rel_path or rel_path in seen:
                continue
            seen.add(rel_path)
            rows.append(rel_path)
    return rows


def _path_to_area(rel_path: str) -> str:
    path = Path(rel_path)
    parts = path.parts
    if not parts:
        return ""
    if parts[0] == "src" and len(parts) >= 2:
        return f"src/{parts[1]}"
    if parts[0] in {"config", "scripts", "docs"}:
        return parts[0]
    return parts[0]


def _path_to_module(rel_path: str) -> str:
    path = Path(rel_path)
    parts = list(path.parts)
    if not parts or parts[0] != "src" or path.suffix != ".py":
        return ""
    parts = parts[1:]
    if not parts:
        return ""
    stem = path.stem
    if stem == "__init__":
        parts = parts[:-1]
    else:
        parts[-1] = stem
    return ".".join(part for part in parts if part)


def _existing_paths(soul: SoulSnapshot, candidates: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    rows: List[str] = []
    for candidate in candidates:
        path = str(candidate)
        if not path or path in seen:
            continue
        if path in soul.files_by_path:
            rows.append(path)
            seen.add(path)
    return rows


def _resolve_symbol_refs(
    soul: SoulSnapshot,
    symbol_names: Sequence[str],
    target_files: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int]] = set()
    preferred = set(target_files)
    for name in symbol_names:
        for qualname, refs in soul.symbols_by_qualname.items():
            if qualname == name or qualname.endswith(f".{name}") or qualname.endswith(name):
                for ref in refs:
                    path = str(ref.get("path", ""))
                    start_line = int(ref.get("start_line", 0) or 0)
                    key = (path, qualname, start_line)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "path": path,
                            "qualname": str(ref.get("qualname", "")),
                            "kind": str(ref.get("kind", "")),
                            "start_line": start_line,
                            "end_line": int(ref.get("end_line", 0) or 0),
                            "_preferred": 0 if path in preferred else 1,
                        }
                    )
    rows.sort(key=lambda item: (int(item.pop("_preferred", 1)), str(item["path"]), int(item["start_line"])))
    return rows


def _structural_suggestions(module_name: str, path: str, inbound: int, outbound: int) -> List[str]:
    suggestions = [
        f"Split {module_name} into smaller seams organized by responsibility instead of keeping the current concentration in {path}.",
        "Add or tighten focused tests around the highest-span symbols before refactoring.",
    ]
    if outbound >= 8:
        suggestions.append("Reduce outbound dependencies by introducing narrower interfaces or helper boundaries.")
    if inbound >= 8:
        suggestions.append("Stabilize the public contract of this module and add contract-level regression tests for dependents.")
    return _dedup_strings(suggestions)


def _compact_evidence(refs: Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, str, str]] = set()
    for ref in refs:
        if ref is None:
            continue
        if isinstance(ref, EvidenceRef):
            key = (ref.source, ref.line, ref.event_type, ref.trace_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": ref.source,
                    "line": ref.line,
                    "event_type": ref.event_type,
                    "trace_id": ref.trace_id,
                    "summary": ref.summary,
                }
            )
            continue
        if isinstance(ref, dict):
            evidence = ref.get("evidence")
            if isinstance(evidence, EvidenceRef):
                key = (evidence.source, evidence.line, evidence.event_type, evidence.trace_id)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "source": evidence.source,
                        "line": evidence.line,
                        "event_type": evidence.event_type,
                        "trace_id": evidence.trace_id,
                        "summary": evidence.summary,
                    }
                )
    return rows


def _render_summary(manifest: Dict[str, Any], observations: Dict[str, Any], dossiers: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Reflection Summary",
        "",
        "## Run",
        f"- Generated at: {manifest.get('generated_at', '')}",
        f"- Dossiers: {manifest.get('dossier_count', 0)}",
        f"- Trace failures observed: {observations.get('trace_failure_count', 0)}",
        f"- Create-tool requests observed: {observations.get('create_tool_request_count', 0)}",
        "",
        "## Dossiers",
    ]
    if not dossiers:
        lines.append("- No dossiers generated.")
        return "\n".join(lines) + "\n"

    for dossier in dossiers:
        lines.append(
            f"- [{str(dossier.get('severity', '')).upper()}] {dossier.get('title')} "
            f"(score={dossier.get('score')}, confidence={dossier.get('confidence')})"
        )
        areas = dossier.get("affected_areas", [])
        if isinstance(areas, list) and areas:
            lines.append(f"  Areas: {', '.join(str(item) for item in areas)}")
        problem = dossier.get("problem_statement", "")
        if isinstance(problem, str) and problem:
            lines.append(f"  Problem: {problem}")
        evidence = dossier.get("evidence", [])
        if isinstance(evidence, list):
            for item in evidence[:3]:
                lines.append(
                    f"  Evidence: {item.get('source')}:{item.get('line', 0)} "
                    f"{item.get('event_type', '')} {item.get('summary', '')}"
                )
        actions = dossier.get("suggested_actions", [])
        if isinstance(actions, list):
            for item in actions[:3]:
                lines.append(f"  Action: {item}")
    return "\n".join(lines) + "\n"


def _count_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "") or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _slug_id(prefix: str, text: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    return f"{prefix}-{safe[:64] or 'item'}"


def _dedup_strings(values: Sequence[str]) -> List[str]:
    rows: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
