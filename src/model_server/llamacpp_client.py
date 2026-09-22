import json
import hashlib
import os
from contextlib import nullcontext
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from common.config import load_settings
from model_server.resource_lock import router_request_lock


class LlamacppError(Exception):
    pass


def _resolve_timeout(timeout: Optional[float]) -> Optional[float]:
    if timeout is not None:
        try:
            value = float(timeout)
        except (TypeError, ValueError):
            value = 0.0
    else:
        settings = load_settings()
        raw = settings.models.get("llamacpp_timeout_s", 60)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 60.0
    if value <= 0:
        return None
    return value


def _post(path: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
    settings = load_settings()
    host = settings.models.get("llamacpp_host", "http://127.0.0.1:8080")
    embedding_host = settings.models.get("embedding_host", "")
    separate_embedding = path == "/v1/embeddings" and bool(embedding_host)
    if separate_embedding:
        host = embedding_host
    guard = nullcontext()
    if settings.models.get("serialize_model_requests", False) and not separate_embedding:
        key = hashlib.sha256(host.rstrip('/').encode()).hexdigest()[:16]
        guard = router_request_lock(
            os.path.join(settings.data_dir, 'locks', f'llama-router-{key}.lock'),
            timeout=float(settings.models.get('model_queue_timeout_s', 600)),
        )
    try:
        with guard:
            return _post_to_host(host, path, payload, timeout)
    except (OSError, TimeoutError) as exc:
        raise LlamacppError(str(exc)) from exc


def _post_to_host(host: str, path: str, payload: Dict[str, Any], timeout: Optional[float]) -> Dict[str, Any]:
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
        raise LlamacppError(str(e))
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise LlamacppError("Invalid JSON response")


def provider() -> str:
    settings = load_settings()
    return settings.models.get("provider", "stub")


def _extract_generate_text(resp: Dict[str, Any], use_thinking_on_empty: bool = False) -> str:
    """Extract generated text from a /v1/chat/completions response, handling
    <think>...</think>-style embedded reasoning tokens some models emit
    inside message.content.
    """
    import re

    choices = resp.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return ""

    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""

    response_text = message.get("content", "")
    if not isinstance(response_text, str) or not response_text.strip():
        return ""

    # Strip embedded reasoning tokens if present.
    cleaned = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL).strip()

    if cleaned:
        return cleaned

    # Nothing left after stripping -> optionally fall back to the reasoning
    # content itself rather than returning empty.
    if use_thinking_on_empty:
        thinking_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if thinking_match:
            return thinking_match.group(1).strip()

    return response_text


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
    resp = generate_raw(
        prompt,
        model,
        system=system,
        options=options,
        images=images,
        timeout=timeout,
        response_format=response_format,
    )
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
    """Generate a response from llama.cpp server via /v1/chat/completions.

    Always uses the OpenAI-compatible chat endpoint so llama-server applies
    the model's own chat template (turn markers, EOS-of-turn tokens, etc).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    if images:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = str(image or "").strip()
            if not encoded:
                continue
            image_url = encoded if encoded.startswith("data:") else f"data:image/jpeg;base64,{encoded}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    if isinstance(options, dict):
        options = {
            key.strip(): value
            for key, value in options.items()
            if isinstance(key, str) and key.strip()
        }
    if options:
        handled_options = {
            "temperature",
            "top_p",
            "top_k",
            "num_predict",
            "max_tokens",
            "stop",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "repeat_last_n",
            "thinking",
            "enable_thinking",
            "think",
            "chat_template_kwargs",
        }
        if "temperature" in options:
            payload["temperature"] = float(options["temperature"])
        if "top_p" in options:
            payload["top_p"] = float(options["top_p"])
        if "top_k" in options:
            payload["top_k"] = int(options["top_k"])
        if "num_predict" in options:
            payload["max_tokens"] = int(options["num_predict"])
        elif "max_tokens" in options:
            payload["max_tokens"] = int(options["max_tokens"])
        if "stop" in options:
            payload["stop"] = options["stop"]
        if "repeat_penalty" in options:
            payload["repeat_penalty"] = float(options["repeat_penalty"])
        if "presence_penalty" in options:
            payload["presence_penalty"] = float(options["presence_penalty"])
        if "frequency_penalty" in options:
            payload["frequency_penalty"] = float(options["frequency_penalty"])
        if "repeat_last_n" in options:
            payload["repeat_last_n"] = int(options["repeat_last_n"])

        template_kwargs = options.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict):
            payload["chat_template_kwargs"] = dict(template_kwargs)
        for thinking_key in ("thinking", "enable_thinking", "think"):
            if thinking_key not in options:
                continue
            thinking_value = _coerce_optional_bool(options[thinking_key])
            if thinking_value is not None:
                payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = thinking_value
                break

        # Preserve additional llama.cpp/OpenAI-compatible request controls
        # instead of silently dropping normalized model options. Controls
        # handled above retain their existing aliases/type conversions.
        for key, value in options.items():
            if isinstance(key, str) and key.strip() and key not in handled_options:
                payload[key.strip()] = value

    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}

    return _post("/v1/chat/completions", payload, timeout=timeout)


def embed(text: str, model: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    payload = {
        "model": model,
        "input": text,
    }
    resp = _post("/v1/embeddings", payload, timeout=timeout)
    data = resp.get("data") or []
    if data and isinstance(data[0], dict):
        return {"embedding": data[0].get("embedding", [])}
    return {"embedding": []}
