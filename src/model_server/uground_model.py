import base64
import json
import os
import re
from typing import Any, Dict, Tuple

from common.config import load_settings
from common.error_log import log_error, log_exception
from common.llm_json_log import record_llm_json
from common.record_log import record_event
from model_server.hf_client import HfError, generate_vision as hf_generate_vision
from model_server.ollama_client import OllamaError, generate_raw as ollama_generate_raw, provider
from model_server.llamacpp_client import LlamacppError, generate_raw as llamacpp_generate_raw
from tool_runtime.sandbox import SandboxViolation, resolve_workspace_path


def _load_image_bytes(image_ref: str) -> bytes:
    if image_ref.startswith("workspace:/"):
        settings = load_settings()
        path = resolve_workspace_path(settings.workspace_root, image_ref)
    elif image_ref.startswith("file:"):
        path = image_ref[len("file:") :]
    else:
        path = image_ref
    if not os.path.exists(path):
        raise FileNotFoundError("Image not found")
    with open(path, "rb") as f:
        return f.read()


def _image_size(image_bytes: bytes) -> Tuple[int, int]:
    try:
        from PIL import Image
        import io
    except Exception as e:
        raise RuntimeError(f"PIL not available: {e}")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image.width, image.height


def _extract_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


def _error_output(image_ref: str, reason: str, raw_output: str = "") -> Dict[str, Any]:
    return {
        "ok": False,
        "image_ref": image_ref,
        "error": reason,
        "raw_output": raw_output,
    }


def _build_prompt(query: str, image_w: int, image_h: int) -> str:
    return (
        "Return strict JSON only.\n"
        "Schema: {\"x\": 0-1000, \"y\": 0-1000, \"confidence\": 0-1}.\n"
        "confidence is independent and must be in [0,1] (not 0-1000).\n"
        f"Target: {query}\n"
    )


def _analyze_ollama(image_ref: str, query: str) -> Dict[str, Any]:
    settings = load_settings()
    model = settings.models.get("ui_grounding_model", "")
    if not model:
        return _error_output(image_ref, "models.ui_grounding_model is empty in settings.")
    timeout_s = settings.models.get("vision_timeout_s", settings.models.get("ollama_timeout_s", 60))
    enforce_json = bool(settings.models.get("ui_grounding_enforce_json", False))
    disable_thinking = bool(settings.models.get("ui_grounding_disable_thinking", False))
    use_thinking_on_empty = bool(settings.models.get("ui_grounding_use_thinking_on_empty", False))

    try:
        raw = _load_image_bytes(image_ref)
    except FileNotFoundError:
        return _error_output(image_ref, "Image file not found.")
    except SandboxViolation as e:
        return _error_output(image_ref, str(e))

    try:
        image_w, image_h = _image_size(raw)
    except Exception as e:
        return _error_output(image_ref, str(e))

    prompt = _build_prompt(query, image_w, image_h)
    if disable_thinking:
        prompt += "Do not include any analysis or thinking. Return JSON only.\n"
    image_b64 = base64.b64encode(raw).decode("ascii")
    model_options = {"temperature": 0.1}
    raw_model_options = settings.models.get("ui_grounding_model_options", {})
    if isinstance(raw_model_options, dict):
        for key, value in raw_model_options.items():
            if not isinstance(key, str) or not key.strip():
                continue
            model_options[key.strip()] = value
    try:
        resp = ollama_generate_raw(
            prompt,
            model,
            images=[image_b64],
            options=model_options,
            timeout=timeout_s,
            response_format="json" if enforce_json else None,
        )
    except OllamaError as e:
        log_exception(
            "ui_grounding_error",
            e,
            {"provider": "ollama", "model": model, "image_ref": image_ref, "query": query},
        )
        return _error_output(image_ref, f"Ollama UI grounding failed: {e}")

    response_text = resp.get("response", "")
    if not isinstance(response_text, str):
        response_text = ""
    if not response_text.strip() and use_thinking_on_empty:
        thinking_text = resp.get("thinking", "")
        if isinstance(thinking_text, str) and thinking_text.strip():
            response_text = thinking_text
    raw_output = response_text
    payload = _extract_json(raw_output)
    if not payload:
        preview = raw_output.strip().replace("\n", " ")[:300] if isinstance(raw_output, str) else ""
        log_error(
            "ui_grounding_parse_error",
            {"provider": "ollama", "model": model, "image_ref": image_ref, "query": query, "preview": preview},
        )
        if not enforce_json:
            return {
                "ok": True,
                "image_ref": image_ref,
                "json": {},
                "raw_text": raw_output,
                "raw_output": raw_output,
                "raw_payload": {},
            }
        return _error_output(image_ref, f"UGround returned non-JSON output: {preview}", raw_output)

    record_llm_json("ui_grounding_output", {"image_ref": image_ref, "query": query, "json": payload})
    record_event("model_provider", {"provider": "ollama", "model": model})
    record_llm_json("model_provider", {"provider": "ollama", "model": model})

    return {
        "ok": True,
        "image_ref": image_ref,
        "json": payload,
        "raw_output": raw_output,
        "raw_payload": payload,
    }


def _analyze_llamacpp(image_ref: str, query: str) -> Dict[str, Any]:
    settings = load_settings()
    model = settings.models.get("ui_grounding_model", "")
    if not model:
        return _error_output(image_ref, "models.ui_grounding_model is empty in settings.")
    timeout_s = settings.models.get("vision_timeout_s", settings.models.get("llamacpp_timeout_s", 60))
    enforce_json = bool(settings.models.get("ui_grounding_enforce_json", False))
    disable_thinking = bool(settings.models.get("ui_grounding_disable_thinking", False))
    use_thinking_on_empty = bool(settings.models.get("ui_grounding_use_thinking_on_empty", False))

    try:
        raw = _load_image_bytes(image_ref)
    except FileNotFoundError:
        return _error_output(image_ref, "Image file not found.")
    except SandboxViolation as e:
        return _error_output(image_ref, str(e))

    try:
        image_w, image_h = _image_size(raw)
    except Exception as e:
        return _error_output(image_ref, str(e))

    prompt = _build_prompt(query, image_w, image_h)
    if disable_thinking:
        prompt += "Do not include any analysis or thinking. Return JSON only.\n"
    image_b64 = base64.b64encode(raw).decode("ascii")
    model_options = {"temperature": 0.1}
    raw_model_options = settings.models.get("ui_grounding_model_options", {})
    if isinstance(raw_model_options, dict):
        for key, value in raw_model_options.items():
            if not isinstance(key, str) or not key.strip():
                continue
            model_options[key.strip()] = value
    try:
        resp = llamacpp_generate_raw(
            prompt,
            model,
            images=[image_b64],
            options=model_options,
            timeout=timeout_s,
            response_format="json" if enforce_json else None,
        )
    except LlamacppError as e:
        log_exception(
            "ui_grounding_error",
            e,
            {"provider": "llamacpp", "model": model, "image_ref": image_ref, "query": query},
        )
        return _error_output(image_ref, f"llamacpp UI grounding failed: {e}")

    response_text = resp.get("content", "")
    if not isinstance(response_text, str):
        response_text = ""
    raw_output = response_text
    payload = _extract_json(raw_output)
    if not payload:
        preview = raw_output.strip().replace("\n", " ")[:300] if isinstance(raw_output, str) else ""
        log_error(
            "ui_grounding_parse_error",
            {"provider": "llamacpp", "model": model, "image_ref": image_ref, "query": query, "preview": preview},
        )
        if not enforce_json:
            return {
                "ok": True,
                "image_ref": image_ref,
                "json": {},
                "raw_text": raw_output,
                "raw_output": raw_output,
                "raw_payload": {},
            }
        return _error_output(image_ref, f"UGround returned non-JSON output: {preview}", raw_output)

    record_llm_json("ui_grounding_output", {"image_ref": image_ref, "query": query, "json": payload})
    record_event("model_provider", {"provider": "llamacpp", "model": model})
    record_llm_json("model_provider", {"provider": "llamacpp", "model": model})

    return {
        "ok": True,
        "image_ref": image_ref,
        "json": payload,
        "raw_output": raw_output,
        "raw_payload": payload,
    }


def _analyze_hf(image_ref: str, query: str) -> Dict[str, Any]:
    settings = load_settings()
    model = settings.models.get("ui_grounding_model", "")
    if not model:
        return _error_output(image_ref, "models.ui_grounding_model is empty in settings.")
    enforce_json = bool(settings.models.get("ui_grounding_enforce_json", False))
    disable_thinking = bool(settings.models.get("ui_grounding_disable_thinking", False))

    try:
        raw = _load_image_bytes(image_ref)
    except FileNotFoundError:
        return _error_output(image_ref, "Image file not found.")
    except SandboxViolation as e:
        return _error_output(image_ref, str(e))

    try:
        image_w, image_h = _image_size(raw)
    except Exception as e:
        return _error_output(image_ref, str(e))

    prompt = _build_prompt(query, image_w, image_h)
    if disable_thinking:
        prompt += "Do not include any analysis or thinking. Return JSON only.\n"
    try:
        raw_output = hf_generate_vision(prompt, model, raw)
    except HfError as e:
        log_exception(
            "ui_grounding_error",
            e,
            {"provider": "hf", "model": model, "image_ref": image_ref, "query": query},
        )
        return _error_output(image_ref, f"HF UI grounding failed: {e}")

    payload = _extract_json(raw_output)
    if not payload:
        preview = raw_output.strip().replace("\n", " ")[:300] if isinstance(raw_output, str) else ""
        log_error(
            "ui_grounding_parse_error",
            {"provider": "hf", "model": model, "image_ref": image_ref, "query": query, "preview": preview},
        )
        if not enforce_json:
            return {
                "ok": True,
                "image_ref": image_ref,
                "json": {},
                "raw_text": raw_output,
                "raw_output": raw_output,
                "raw_payload": {},
            }
        return _error_output(image_ref, f"UGround returned non-JSON output: {preview}", raw_output)

    record_llm_json("ui_grounding_output", {"image_ref": image_ref, "query": query, "json": payload})
    record_event("model_provider", {"provider": "hf", "model": model})
    record_llm_json("model_provider", {"provider": "hf", "model": model})

    return {
        "ok": True,
        "image_ref": image_ref,
        "json": payload,
        "raw_output": raw_output,
        "raw_payload": payload,
    }


def predict(image_ref: str, query: str) -> Dict[str, Any]:
    current_provider = provider()
    if current_provider == "hf":
        try:
            result = _analyze_hf(image_ref, query=query)
            if isinstance(result, dict) and result.get("ok") is False:
                return _analyze_ollama(image_ref, query=query)
            return result
        except Exception as e:
            model = load_settings().models.get("ui_grounding_model", "")
            log_exception(
                "ui_grounding_error",
                e,
                {"provider": "hf", "model": model, "image_ref": image_ref, "query": query},
            )
            return _analyze_ollama(image_ref, query=query)
    if current_provider == "ollama":
        return _analyze_ollama(image_ref, query=query)
    if current_provider == "llamacpp":
        return _analyze_llamacpp(image_ref, query=query)
    return _error_output(
        image_ref,
        f"Unsupported UI grounding provider '{current_provider}'. Expected 'hf', 'ollama' or 'llamacpp'.",
    )
