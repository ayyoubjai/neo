import os
from typing import Any, Dict

from common.config import load_settings
from common.jsonlog import append_jsonl
from common.time_utils import utc_now_iso


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def record_llm_json(event_type: str, payload: Dict[str, Any]) -> None:
    try:
        settings = load_settings()
        os.makedirs(settings.data_dir, exist_ok=True)
        record = {
            "ts": utc_now_iso(),
            "event_type": event_type,
            "payload": _json_safe(payload),
        }
        path = os.path.join(settings.data_dir, "llm_json.log")
        append_jsonl(path, record)
    except Exception:
        pass
