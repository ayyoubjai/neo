import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.config import load_settings


@dataclass
class PromptEntry:
    mode: str
    system: Optional[str]
    prompt: str
    turn_id: Optional[str]


@dataclass
class PromptPair:
    mode: str
    system: Optional[str]
    prompt: str
    response: str
    turn_id: Optional[str]


@dataclass
class TraceState:
    trace_id: str
    turn_id: Optional[str] = None
    user_text: Optional[str] = None
    route_mode: Optional[str] = None
    tool_intent: bool = False
    prompt_queue: List[PromptEntry] = field(default_factory=list)
    prompt_pairs: List[PromptPair] = field(default_factory=list)


def _iter_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _trace_for(traces: Dict[str, TraceState], trace_id: str) -> TraceState:
    trace = traces.get(trace_id)
    if trace is None:
        trace = TraceState(trace_id=trace_id)
        traces[trace_id] = trace
    return trace


def _mark_tool_intent(trace: TraceState) -> None:
    trace.tool_intent = True


def _handle_record(
    traces: Dict[str, TraceState],
    system_cache: Dict[str, str],
    record: Dict[str, Any],
) -> None:
    event_type = record.get("event_type")
    payload = record.get("payload", {}) if isinstance(record.get("payload"), dict) else {}
    trace_id = payload.get("trace_id")
    if not trace_id:
        return
    trace = _trace_for(traces, trace_id)
    system_ref = payload.get("system_ref")
    system_text = payload.get("system")
    if isinstance(system_ref, str) and isinstance(system_text, str) and system_text:
        system_cache[system_ref] = system_text

    if event_type == "user_query":
        trace.user_text = payload.get("text")
        trace.turn_id = payload.get("turn_id") or trace.turn_id
        return

    if event_type == "router_decision":
        trace.route_mode = payload.get("mode") or trace.route_mode
        trace.turn_id = payload.get("turn_id") or trace.turn_id
        return

    if event_type in ("tool_request", "tool_response", "tool_execute_start", "tool_execute_result"):
        _mark_tool_intent(trace)
        return

    if event_type == "model_prompt":
        mode = payload.get("mode") or ""
        prompt = payload.get("prompt") or ""
        system = payload.get("system")
        if (system is None or system == "") and isinstance(system_ref, str):
            system = system_cache.get(system_ref, system)
        turn_id = payload.get("turn_id") or trace.turn_id
        if isinstance(mode, str) and isinstance(prompt, str) and prompt:
            trace.prompt_queue.append(PromptEntry(mode=mode, system=system, prompt=prompt, turn_id=turn_id))
        return

    if event_type == "model_output":
        response = payload.get("text") or ""
        if not trace.prompt_queue or not isinstance(response, str):
            return
        prompt_entry = trace.prompt_queue.pop(0)
        trace.prompt_pairs.append(
            PromptPair(
                mode=prompt_entry.mode,
                system=prompt_entry.system,
                prompt=prompt_entry.prompt,
                response=response,
                turn_id=prompt_entry.turn_id,
            )
        )


def build_datasets(
    record_path: str,
    out_dir: str,
    min_prompt_chars: int = 10,
    min_response_chars: int = 5,
) -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    records = _iter_records(record_path)
    traces: Dict[str, TraceState] = {}
    system_cache: Dict[str, str] = {}
    for record in records:
        _handle_record(traces, system_cache, record)

    router_path = os.path.join(out_dir, "router.jsonl")
    general_path = os.path.join(out_dir, "sft_general.jsonl")
    agent_path = os.path.join(out_dir, "sft_agent.jsonl")
    complex_path = os.path.join(out_dir, "sft_complex.jsonl")

    counts = {"router": 0, "general": 0, "agent": 0, "complex": 0}

    with open(router_path, "w", encoding="utf-8") as router_file:
        for trace in traces.values():
            if not trace.user_text or not isinstance(trace.user_text, str):
                continue
            label = "AGENT" if trace.tool_intent else "GENERAL"
            entry = {
                "trace_id": trace.trace_id,
                "turn_id": trace.turn_id,
                "text": trace.user_text,
                "label": label,
                "label_tool_intent": trace.tool_intent,
                "label_router": trace.route_mode,
            }
            router_file.write(json.dumps(entry, ensure_ascii=True) + "\n")
            counts["router"] += 1

    def write_pair(path: str, entry: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    for trace in traces.values():
        for pair in trace.prompt_pairs:
            prompt = pair.prompt or ""
            response = pair.response or ""
            if len(prompt) < min_prompt_chars or len(response) < min_response_chars:
                continue
            mode = (pair.mode or "").upper()
            entry = {
                "trace_id": trace.trace_id,
                "turn_id": pair.turn_id or trace.turn_id,
                "mode": mode,
                "system": pair.system or "",
                "prompt": prompt,
                "response": response,
            }
            if mode == "GENERAL":
                write_pair(general_path, entry)
                counts["general"] += 1
            elif mode == "AGENT":
                write_pair(agent_path, entry)
                counts["agent"] += 1
            elif mode == "COMPLEX":
                write_pair(complex_path, entry)
                counts["complex"] += 1

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "record_path": record_path,
                "counts": counts,
                "files": {
                    "router": router_path,
                    "sft_general": general_path,
                    "sft_agent": agent_path,
                    "sft_complex": complex_path,
                },
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    return counts


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Build training datasets from record.log")
    parser.add_argument("--record", default=os.path.join(settings.data_dir, "record.log"))
    parser.add_argument("--out", default=os.path.join(settings.workspace_root, "evolve", "datasets"))
    parser.add_argument("--min-prompt-chars", type=int, default=10)
    parser.add_argument("--min-response-chars", type=int, default=5)
    args = parser.parse_args()

    counts = build_datasets(
        record_path=args.record,
        out_dir=args.out,
        min_prompt_chars=args.min_prompt_chars,
        min_response_chars=args.min_response_chars,
    )
    print("Dataset build complete.")
    print(json.dumps(counts, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
