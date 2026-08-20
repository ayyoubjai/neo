import os
import traceback
from typing import Any, Dict

from common.config import load_settings
from common.jsonlog import append_jsonl
from common.time_utils import utc_now_iso


def _truncate_text(text: str, limit: int) -> str:
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


def log_error(event_type: str, payload: Dict[str, Any], limit: int = 4000) -> None:
    try:
        settings = load_settings()
        os.makedirs(settings.data_dir, exist_ok=True)
        path = os.path.join(settings.data_dir, "error.log")
        record = {
            "ts": utc_now_iso(),
            "event_type": event_type,
            "payload": _sanitize(payload, limit),
        }
        append_jsonl(path, record)
    except Exception:
        pass


def log_exception(event_type: str, exc: Exception, payload: Dict[str, Any], limit: int = 4000) -> None:
    info = dict(payload)
    info["error_type"] = exc.__class__.__name__
    info["error"] = str(exc)
    info["traceback"] = traceback.format_exc()
    log_error(event_type, info, limit=limit)
