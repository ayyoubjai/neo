import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from common.config import load_settings


class OllamaError(Exception):
    pass


def _resolve_timeout(timeout: Optional[float]) -> Optional[float]:
    if timeout is not None:
        try:
            value = float(timeout)
        except (TypeError, ValueError):
            value = 0.0
    else:
        settings = load_settings()
        raw = settings.models.get("ollama_timeout_s", 60)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 60.0
    if value <= 0:
        return None
    return value


def _post(path: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
    settings = load_settings()
    host = settings.models.get("ollama_host", "http://127.0.0.1:11434")
    url = host.rstrip("/") + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    resolved_timeout = _resolve_timeout(timeout)
    try:
        if resolved_timeout is None:
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
        else:
            with urllib.request.urlopen(req, timeout=resolved_timeout) as resp:
                body = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout) as e:
        raise OllamaError(str(e))
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise OllamaError("Invalid JSON response")


def provider() -> str:
    settings = load_settings()
    return settings.models.get("provider", "stub")


def _extract_generate_text(resp: Dict[str, Any], use_thinking_on_empty: bool = False) -> str:
    response_text = resp.get("response", "")
    if isinstance(response_text, str) and response_text.strip():
        return response_text

    message = resp.get("message")
    if isinstance(message, dict):
        message_content = message.get("content", "")
        if isinstance(message_content, str) and message_content.strip():
            return message_content

    if use_thinking_on_empty:
        thinking_text = resp.get("thinking", "")
        if isinstance(thinking_text, str) and thinking_text.strip():
            return thinking_text

    if isinstance(response_text, str):
        return response_text
    return ""


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _split_request_controls(options: Optional[Dict[str, Any]]) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
    if not isinstance(options, dict):
        return None, None
    cleaned: Dict[str, Any] = {}
    think_value: Optional[bool] = None
    for key, value in options.items():
        if not isinstance(key, str):
            continue
        normalized = key.strip()
        if not normalized:
            continue
        if normalized in {"thinking", "enable_thinking", "think"}:
            parsed = _coerce_optional_bool(value)
            if parsed is not None:
                think_value = parsed
            continue
        cleaned[normalized] = value
    return think_value, cleaned or None


def generate(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    images=None,
    timeout: Optional[float] = None,
    response_format: Optional[str] = None,
    use_thinking_on_empty: bool = False,
) -> str:
    think_value, cleaned_options = _split_request_controls(options)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if think_value is not None:
        payload["think"] = think_value
    if cleaned_options:
        payload["options"] = cleaned_options
    if images:
        payload["images"] = images
    if response_format:
        payload["format"] = response_format
    resp = _post("/api/generate", payload, timeout=timeout)
    return _extract_generate_text(resp, use_thinking_on_empty=use_thinking_on_empty)


def generate_raw(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    images=None,
    timeout: Optional[float] = None,
    response_format: Optional[str] = None,
) -> Dict[str, Any]:
    think_value, cleaned_options = _split_request_controls(options)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if think_value is not None:
        payload["think"] = think_value
    if cleaned_options:
        payload["options"] = cleaned_options
    if images:
        payload["images"] = images
    if response_format:
        payload["format"] = response_format
    return _post("/api/generate", payload, timeout=timeout)


def embed(text: str, model: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    payload = {"model": model, "prompt": text}
    return _post("/api/embeddings", payload, timeout=timeout)
