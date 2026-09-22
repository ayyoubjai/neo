import hashlib
import json
import os
from typing import Any, Dict, Optional

from common.config import load_settings
from common.jsonlog import append_jsonl
from common.time_utils import utc_now_iso


_LAST_SYSTEM_HASH: Optional[str] = None
_HUMAN_TURN_OPEN = False


def _truncate_text(text: str, limit: int = 4000) -> str:
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _sanitize(value: Any, limit: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value, limit=limit)
    if isinstance(value, dict):
        return {str(k): _sanitize(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, limit) for item in value]
    return _truncate_text(str(value), limit=limit)


def _allowed_event(event_type: str, level: str) -> bool:
    if level == "training":
        return event_type in {
            "user_query",
            "router_decision",
            "model_prompt",
            "model_output",
            "model_provider",
            "tool_retrieval",
            "tool_request",
            "tool_response",
            "assistant_final",
        }
    if level == "minimal":
        return event_type in {"user_query", "router_decision", "assistant_final", "tool_response", "model_provider"}
    return True


def _append_human_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _format_human_event(event_type: str, payload: Dict[str, Any], max_chars: int) -> Optional[str]:
    if event_type == "tool_retrieval":
        selected = payload.get("selected_tool_ids", [])
        scores = ", ".join(
            f"{item['tool_id']}={item['score']:.4f}"
            for item in payload.get("top_candidates", [])
        )
        line = (
            f"TOOL_RETRIEVAL: status={payload.get('status')} "
            f"min_similarity={payload.get('min_similarity')} "
            f"selected={len(selected)}/{payload.get('max_tools')}\n"
            f"RETRIEVED_TOOLS: {', '.join(selected) or 'none'}\n"
            f"TOP_TOOL_SCORES: {scores or 'none'}"
        )
        if payload.get("error"):
            line += f"\nRETRIEVAL_ERROR: {_truncate_text(str(payload['error']), max_chars)}"
        return line
    if event_type == "user_query":
        text = payload.get("text") or ""
        if not isinstance(text, str) or not text:
            return None
        return f"USER: {_truncate_text(text, max_chars)}"
    if event_type == "router_decision":
        mode = payload.get("mode") or ""
        if not isinstance(mode, str) or not mode:
            return None
        return f"MODE: {mode}"
    if event_type == "model_prompt":
        mode = payload.get("mode") or ""
        prompt = payload.get("prompt") or ""
        if not isinstance(prompt, str) or not prompt:
            return None
        mode_label = mode if isinstance(mode, str) and mode else "UNKNOWN"
        lines = []
        # Extract from the original prompt before truncation hides the catalog.
        # Repeat per call so changes after initialization or codegen are visible.
        marker = "AVAILABLE TOOLS:\n"
        if marker in prompt:
            catalog = prompt.rsplit(marker, 1)[1].split("\n\n", 1)[0]
            tool_ids = [
                line[2:].split(":", 1)[0].strip()
                for line in catalog.splitlines()
                if line.startswith("- ") and ":" in line
            ]
            model = payload.get("model") or mode_label
            lines.append(f"TOOLS_GIVEN[{model}] ({len(tool_ids)}): {', '.join(tool_ids) or 'none'}")
        lines.append(f"PROMPT[{mode_label}]: {_truncate_text(prompt, max_chars)}")
        return "\n".join(lines)
    if event_type == "model_output":
        text = payload.get("text") or ""
        if not isinstance(text, str) or not text:
            return None
        # Intentionally avoid truncation to preserve raw model output for analysis.
        return f"LLM_RAW: {text}"
    if event_type == "tool_request":
        tool_id = payload.get("tool_id") or ""
        args = payload.get("args") or {}
        if not isinstance(tool_id, str) or not tool_id:
            return None
        try:
            args_text = json.dumps(args, ensure_ascii=True, separators=(",", ":"))
        except TypeError:
            args_text = str(args)
        return f"TOOL: {tool_id} args={_truncate_text(args_text, max_chars)}"
    if event_type == "tool_response":
        tool_id = payload.get("tool_id") or ""
        status = payload.get("status") or ""
        error = payload.get("error")
        if not isinstance(tool_id, str) or not tool_id:
            return None
        suffix = f" status={status}" if status else ""
        if isinstance(error, str) and error:
            suffix += f" error={_truncate_text(error, max_chars)}"
        return f"TOOL_RESULT: {tool_id}{suffix}"
    if event_type == "assistant_final":
        text = payload.get("text") or ""
        if not isinstance(text, str) or not text:
            return None
        return f"ASSISTANT: {_truncate_text(text, max_chars)}"
    return None


def record_event(event_type: str, payload: Dict[str, Any]) -> None:
    try:
        settings = load_settings()
        if not bool(settings.evolve.get("logging_enabled", True)):
            return
        os.makedirs(settings.data_dir, exist_ok=True)
        level = str(settings.evolve.get("record_log_level", "full")).lower()
        if not _allowed_event(event_type, level):
            return
        max_chars = int(settings.evolve.get("record_log_max_chars", 4000))
        if max_chars != 0 and max_chars < 200:
            max_chars = 200
        dedup_system = bool(settings.evolve.get("record_log_dedup_system", False))
        path = os.path.join(settings.data_dir, "record.log")
        payload = dict(payload)
        system_text = payload.get("system")
        if isinstance(system_text, str) and system_text:
            system_hash = hashlib.sha256(system_text.encode("utf-8")).hexdigest()
            payload["system_ref"] = system_hash
            if dedup_system:
                global _LAST_SYSTEM_HASH
                if _LAST_SYSTEM_HASH == system_hash:
                    payload.pop("system", None)
                else:
                    _LAST_SYSTEM_HASH = system_hash
        record = {
            "ts": utc_now_iso(),
            "event_type": event_type,
            "payload": _sanitize(payload, max_chars),
        }
        append_jsonl(path, record)

        human_enabled = bool(settings.evolve.get("human_record_log_enabled", True))
        if not human_enabled:
            return
        human_max_chars = int(settings.evolve.get("human_record_max_chars", 1200))
        if human_max_chars != 0 and human_max_chars < 200:
            human_max_chars = 200
        human_path = os.path.join(settings.data_dir, "human_record.log")
        line = _format_human_event(event_type, payload, human_max_chars)
        if line is None:
            return
        global _HUMAN_TURN_OPEN
        if event_type == "user_query":
            if _HUMAN_TURN_OPEN:
                _append_human_line(human_path, "")
            _append_human_line(human_path, "=== TURN ===")
            _HUMAN_TURN_OPEN = True
        _append_human_line(human_path, line)
    except Exception:
        pass
