import json
import os
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type


_LOCK = threading.Lock()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _trace_enabled() -> bool:
    return _env_flag("EVOLVE_TRACE_ENABLE", False)


def _trace_output_path() -> str:
    raw = os.environ.get("EVOLVE_TRACE_OUTPUT", "evolve/scenarios/traces/current.jsonl").strip()
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(raw)


def _trace_object_filter() -> List[str]:
    raw = os.environ.get("EVOLVE_TRACE_OBJECTS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _trace_max_repr_chars() -> int:
    raw = os.environ.get("EVOLVE_TRACE_MAX_REPR_CHARS", "500")
    try:
        value = int(raw)
    except ValueError:
        value = 500
    return max(64, value)


def _matches_filter(object_id: str) -> bool:
    filters = _trace_object_filter()
    if not filters:
        return True
    return object_id in filters


def _safe_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    limit = _trace_max_repr_chars()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _to_replayable(value: Any, depth: int = 0) -> Tuple[Any, bool]:
    if depth > 8:
        return None, False
    if value is None or isinstance(value, (bool, int, float, str)):
        return value, True
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            converted, ok = _to_replayable(item, depth + 1)
            if not ok:
                return None, False
            out.append(converted)
        return out, True
    if isinstance(value, tuple):
        out: List[Any] = []
        for item in value:
            converted, ok = _to_replayable(item, depth + 1)
            if not ok:
                return None, False
            out.append(converted)
        return {"__tuple__": out}, True
    if isinstance(value, dict):
        out_dict: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return None, False
            converted, ok = _to_replayable(item, depth + 1)
            if not ok:
                return None, False
            out_dict[key] = converted
        return out_dict, True
    return None, False


def _restore_replayable(value: Any) -> Any:
    if isinstance(value, dict) and "__tuple__" in value and isinstance(value["__tuple__"], list):
        return tuple(_restore_replayable(item) for item in value["__tuple__"])
    if isinstance(value, list):
        return [_restore_replayable(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_replayable(item) for key, item in value.items()}
    return value


def _append_event(event: Dict[str, Any]) -> None:
    output_path = _trace_output_path()
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(event, ensure_ascii=True)
    with _LOCK:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def trace_calls(object_id: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        source_file = os.path.abspath(getattr(func, "__code__", None).co_filename) if hasattr(func, "__code__") else ""
        source_path = source_file.replace("\\", "/")
        marker = "/src/"
        src_idx = source_path.rfind(marker)
        if src_idx != -1:
            source_path = source_path[src_idx + 1 :]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _trace_enabled() or not _matches_filter(object_id):
                return func(*args, **kwargs)
            started = time.perf_counter()
            ok = True
            error_type = ""
            error_message = ""
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                ok = False
                error_type = type(e).__name__
                error_message = _safe_repr(e)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                call_args, args_ok = _to_replayable(list(args))
                call_kwargs, kwargs_ok = _to_replayable(kwargs)
                replayable = bool(args_ok and kwargs_ok)
                event = {
                    "schema_version": 1,
                    "kind": "trace_case",
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "object_id": object_id,
                    "source_path": source_path,
                    "replayable": replayable,
                    "call": {
                        "args": call_args if replayable else [],
                        "kwargs": call_kwargs if replayable else {},
                    },
                    "preview": {
                        "args_repr": _safe_repr(args),
                        "kwargs_repr": _safe_repr(kwargs),
                    },
                    "result": {
                        "ok": ok,
                        "error_type": error_type,
                        "error_message": error_message,
                    },
                    "duration_ms": round(elapsed_ms, 6),
                }
                try:
                    _append_event(event)
                except Exception:
                    # Tracing must never break normal app execution.
                    pass

        return wrapper

    return decorator


def trace_class_calls(
    object_id_prefix: str,
    include_private: bool = False,
    include_dunder: bool = False,
) -> Callable[[Type[Any]], Type[Any]]:
    def _should_trace_name(name: str) -> bool:
        if name in {"__init__", "__call__"}:
            return True
        if name.startswith("__") and name.endswith("__"):
            return include_dunder
        if name.startswith("_"):
            return include_private
        return True

    def _wrap_function(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        return trace_calls(f"{object_id_prefix}.{name}")(func)

    def decorator(cls: Type[Any]) -> Type[Any]:
        for name, value in list(cls.__dict__.items()):
            if not _should_trace_name(name):
                continue

            if isinstance(value, staticmethod):
                wrapped = _wrap_function(name, value.__func__)
                setattr(cls, name, staticmethod(wrapped))
                continue

            if isinstance(value, classmethod):
                wrapped = _wrap_function(name, value.__func__)
                setattr(cls, name, classmethod(wrapped))
                continue

            if isinstance(value, property):
                fget = value.fget
                fset = value.fset
                fdel = value.fdel
                if fget is not None:
                    fget = _wrap_function(f"{name}.fget", fget)
                if fset is not None:
                    fset = _wrap_function(f"{name}.fset", fset)
                if fdel is not None:
                    fdel = _wrap_function(f"{name}.fdel", fdel)
                setattr(cls, name, property(fget, fset, fdel, value.__doc__))
                continue

            if callable(value):
                wrapped = _wrap_function(name, value)
                setattr(cls, name, wrapped)

        return cls

    return decorator


def restore_call_payload(payload: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    args = payload.get("args", [])
    kwargs = payload.get("kwargs", {})
    if not isinstance(args, list):
        args = []
    if not isinstance(kwargs, dict):
        kwargs = {}
    restored_args = [_restore_replayable(item) for item in args]
    restored_kwargs = {key: _restore_replayable(val) for key, val in kwargs.items()}
    return restored_args, restored_kwargs
