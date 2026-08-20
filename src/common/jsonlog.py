import json
import os
from typing import Any, Dict, Optional

_LOGGING_ENABLED: Optional[bool] = None


def _logging_enabled() -> bool:
    global _LOGGING_ENABLED
    if _LOGGING_ENABLED is not None:
        return _LOGGING_ENABLED
    try:
        from common.config import load_settings

        settings = load_settings()
        _LOGGING_ENABLED = bool(settings.evolve.get("logging_enabled", True))
    except Exception:
        _LOGGING_ENABLED = True
    return _LOGGING_ENABLED


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    if not _logging_enabled():
        return
    line = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
