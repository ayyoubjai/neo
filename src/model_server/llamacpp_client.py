import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from common.config import load_settings


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
    except json.JSONDecodeError as e:
        raise LlamacppError("Invalid JSON response")


def provider() -> str:
    settings = load_settings()
    return settings.models.get("provider", "stub")


def _extract_generate_text(resp: Dict[str, Any], use_thinking_on_empty: bool = False) -> str:
    """Extract generated text from llama.cpp response, handling:
    - Thinking tokens (<think>...</think>)
    - Chat message format responses
    - Raw completion responses
    """
    import re
    
    # Try chat message format first (if using messages API)
    if "choices" in resp:
        choices = resp.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                response_text = message.get("content", "")
                if response_text and isinstance(response_text, str):
                    # Clean thinking tokens from message content
                    cleaned = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL).strip()
                    if cleaned:
                        return cleaned
                    # Fallback to thinking if nothing left
                    if use_thinking_on_empty:
                        thinking_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
                        if thinking_match:
                            return thinking_match.group(1).strip()
                    return response_text
    
    # Try raw completion format (default)
    response_text = resp.get("content", "")
    
    if not isinstance(response_text, str):
        return ""
    
    if not response_text.strip():
        return ""
    
    # Remove thinking tokens if present (format: <think>...</think>)
    # Models like Qwen3.5-UD and Qwen3.6-UD embed thinking in the response
    cleaned = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL).strip()
    
    if cleaned:
        return cleaned
    
    # If nothing left after removing thinking, return thinking if requested
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
    """Generate raw response from llama.cpp server.

    For chat models (Qwen, Llama, etc.), uses proper message format via /v1/chat/completions.
    For raw models, uses simple text completion via /completion.
    """
    full_prompt = prompt

    is_chat_model = any(
        indicator in model.lower()
        for indicator in [
            'qwen', 'llama2-chat', 'mistral-instruct', 'neural-chat',
            'chat', 'instruct', 'UD'
        ]
    )

    if is_chat_model:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            # llama.cpp implements the OpenAI multimodal chat schema. The old
            # `image_data` extension belongs to /completion and is ignored by
            # recent /v1/chat/completions servers.
            content = [{"type": "text", "text": full_prompt}]
            for image in images:
                encoded = str(image or "").strip()
                if not encoded:
                    continue
                image_url = encoded if encoded.startswith("data:") else f"data:image/jpeg;base64,{encoded}"
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": full_prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        endpoint = "/v1/chat/completions"
    else:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        endpoint = "/completion"

    if options:
        if "temperature" in options:
            payload["temperature"] = float(options["temperature"])
        if "top_p" in options:
            payload["top_p"] = float(options["top_p"])
        if "top_k" in options:
            payload["top_k"] = int(options["top_k"])
        if "num_predict" in options:
            payload["n_predict" if endpoint == "/completion" else "max_tokens"] = int(options["num_predict"])
        elif "max_tokens" in options:
            payload["n_predict" if endpoint == "/completion" else "max_tokens"] = int(options["max_tokens"])
        if "stop" in options:
            payload["stop"] = options["stop"]

    if images and endpoint == "/completion":
        image_data = []
        for img in images:
            image_data.append({"data": img, "id": len(image_data)})
        payload["image_data"] = image_data

    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}

    return _post(endpoint, payload, timeout=timeout)

def embed(text: str, model: str, timeout: Optional[float] = None) -> Dict[str, Any]:
    payload = {
        "model": model,
        "content": text
    }
    return _post("/embedding", payload, timeout=timeout)
