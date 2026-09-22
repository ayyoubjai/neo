import base64
import json
import os
import re
from typing import Any, Dict, Optional

from common.config import load_settings
from common.error_log import log_error, log_exception
from common.llm_json_log import record_llm_json
from common.record_log import record_event
from model_server.hf_client import HfError, generate_vision as hf_generate_vision
from model_server.ollama_client import OllamaError, generate_raw as ollama_generate_raw, provider
from model_server.llamacpp_client import LlamacppError, generate as llamacpp_generate
from tool_runtime.sandbox import SandboxViolation, resolve_workspace_path


def _load_image_bytes(image_ref: str) -> bytes:
    if image_ref.startswith("workspace:/"):
        settings = load_settings()
        path = resolve_workspace_path(settings.workspace_root, image_ref)
    elif image_ref.startswith("file:"):
        path = image_ref[len("file:"):]
    else:
        path = image_ref
    if not os.path.exists(path):
        raise FileNotFoundError("Image not found")
    with open(path, "rb") as f:
        return f.read()


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


def _coerce_output(
    image_ref: str,
    payload: Dict[str, Any],
    fallback_summary: str,
    question_text: str,
    raw_output: str,
) -> Dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), str) else fallback_summary
    answer = payload.get("answer") if isinstance(payload.get("answer"), str) else ""
    if not answer:
        answer = summary if summary else ("No answer available." if question_text else "")
    text = payload.get("text") if isinstance(payload.get("text"), str) else ""
    objects = payload.get("objects")
    if not isinstance(objects, list):
        objects = []
    return {
        "ok": True,
        "image_ref": image_ref,
        "summary": summary,
        "answer": answer,
        "text": text,
        "objects": objects,
        "raw_output": raw_output,
        "raw_payload": payload,
    }


def _error_output(image_ref: str, reason: str, raw_output: str = "", *, code: str = 'UNKNOWN') -> Dict[str, Any]:
    return {
        "ok": False,
        "image_ref": image_ref,
        "summary": f"Vision analysis failed: {reason}",
        "answer": "",
        "text": "",
        "objects": [],
        "raw_output": raw_output,
        "error": reason,
        "error_code": code,
    }


def _analyze_ollama(image_ref: str, question: Optional[str] = None) -> Dict[str, Any]:
    settings = load_settings()
    model = settings.models.get("vision_model", "")
    if not model:
        return _error_output(image_ref, "models.vision_model is empty in settings.", code="MISSING_DEPENDENCY")
    vision_timeout = settings.models.get("vision_timeout_s", settings.models.get("ollama_timeout_s", 60))
    enforce_json = bool(settings.models.get("vision_enforce_json", True))
    disable_thinking = bool(settings.models.get("vision_disable_thinking", False))
    use_thinking_on_empty = bool(settings.models.get("vision_use_thinking_on_empty", False))

    try:
        raw = _load_image_bytes(image_ref)
    except FileNotFoundError:
        return _error_output(image_ref, "Image file not found.", code="MISSING_INPUT")
    except SandboxViolation as e:
        return _error_output(image_ref, str(e), code="PERMISSION_DENIED")

    image_b64 = base64.b64encode(raw).decode("ascii")
    question_text = (question or "").strip()
    prompt = (
        "Analyze the image and respond with strict JSON only.\n"
        "Return this schema: "
        "{\"summary\":\"...\",\"answer\":\"...\",\"text\":\"...\",\"objects\":[{\"name\":\"...\",\"count\":1}]}\n"
        "Rules: keep summary brief; text is visible OCR; objects is a list (may be empty); "
        "answer must directly answer the user question when provided.\n"
    )
    if disable_thinking:
        prompt += "Do not include any analysis or thinking. Return JSON only.\n"
    if question_text:
        prompt += f"User question: {question_text}\n"
        prompt += "Use the question to focus both summary and answer."
    else:
        prompt += "No user question was provided; set answer to a short key takeaway."
    model_options = {"temperature": 0.2}
    raw_model_options = settings.models.get("vision_model_options", {})
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
            timeout=vision_timeout,
            response_format="json" if enforce_json else None,
        )
    except OllamaError as e:
        log_exception(
            "vision_error",
            e,
            {
                "provider": "ollama",
                "model": model,
                "image_ref": image_ref,
                "question": question_text,
            },
        )
        return _error_output(image_ref, f"Ollama vision request failed: {e}")
    print(resp)
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
        if not enforce_json:
            fallback_text = raw_output.strip() if isinstance(raw_output, str) else ""
            return _coerce_output(
                image_ref,
                {"summary": fallback_text, "answer": fallback_text, "text": "", "objects": []},
                fallback_text or "Vision analysis completed.",
                question_text,
                raw_output,
            )
        preview = raw_output.strip().replace("\n", " ")[:300] if isinstance(raw_output, str) else ""
        log_error(
            "vision_parse_error",
            {
                "provider": "ollama",
                "model": model,
                "image_ref": image_ref,
                "question": question_text,
                "preview": preview,
            },
        )
        return _error_output(image_ref, f"Vision model returned non-JSON output: {preview}", raw_output, code="INVALID_MODEL_OUTPUT")
    record_llm_json(
        "vision_output",
        {"image_ref": image_ref, "question": question_text, "json": payload},
    )
    record_event(
        "model_provider",
        {"provider": "ollama", "model": model},
    )
    record_llm_json(
        "model_provider",
        {"provider": "ollama", "model": model},
    )
    return _coerce_output(image_ref, payload, "Vision analysis completed.", question_text, raw_output)


def _analyze_llamacpp(image_ref: str, question: Optional[str] = None) -> Dict[str, Any]:
    settings = load_settings()
    model = settings.models.get("vision_model", "")
    if not model:
        return _error_output(image_ref, "models.vision_model is empty in settings.", code="MISSING_DEPENDENCY")
    vision_timeout = settings.models.get("vision_timeout_s", settings.models.get("llamacpp_timeout_s", 60))
    enforce_json = bool(settings.models.get("vision_enforce_json", True))
    disable_thinking = bool(settings.models.get("vision_disable_thinking", False))
    use_thinking_on_empty = bool(settings.models.get("vision_use_thinking_on_empty", False))

    try:
        raw = _load_image_bytes(image_ref)
    except FileNotFoundError:
        return _error_output(image_ref, "Image file not found.", code="MISSING_INPUT")
    except SandboxViolation as e:
        return _error_output(image_ref, str(e), code="PERMISSION_DENIED")

    image_b64 = base64.b64encode(raw).decode("ascii")
    question_text = (question or "").strip()
    prompt = (
        "Analyze the image and respond with strict JSON only.\n"
        "Return this schema: "
        "{\"summary\":\"...\",\"answer\":\"...\",\"text\":\"...\",\"objects\":[{\"name\":\"...\",\"count\":1}]}\n"
        "Rules: keep summary brief; text is visible OCR; objects is a list (may be empty); "
        "answer must directly answer the user question when provided.\n"
    )
    if disable_thinking:
        prompt += "Do not include any analysis or thinking. Return JSON only.\n"
    if question_text:
        prompt += f"User question: {question_text}\n"
        prompt += "Use the question to focus both summary and answer."
    else:
        prompt += "No user question was provided; set answer to a short key takeaway."
    model_options = {"temperature": 0.2}
    raw_model_options = settings.models.get("vision_model_options", {})
    if isinstance(raw_model_options, dict):
        for key, value in raw_model_options.items():
            if not isinstance(key, str) or not key.strip():
                continue
            model_options[key.strip()] = value

    try:
        raw_output = llamacpp_generate(
            prompt,
            model,
            images=[image_b64],
            options=model_options,
            timeout=vision_timeout,
            response_format="json" if enforce_json else None,
            use_thinking_on_empty=use_thinking_on_empty,
        )
    except LlamacppError as e:
        log_exception(
            "vision_error",
            e,
            {
                "provider": "llamacpp",
                "model": model,
                "image_ref": image_ref,
                "question": question_text,
            },
        )
        return _error_output(image_ref, f"llamacpp vision request failed: {e}")

    payload = _extract_json(raw_output)
    if not payload:
        if not enforce_json:
            fallback_text = raw_output.strip() if isinstance(raw_output, str) else ""
            return _coerce_output(
                image_ref,
                {"summary": fallback_text, "answer": fallback_text, "text": "", "objects": []},
                fallback_text or "Vision analysis completed.",
                question_text,
                raw_output,
            )
        preview = raw_output.strip().replace("\n", " ")[:300] if isinstance(raw_output, str) else ""
        log_error(
            "vision_parse_error",
            {
                "provider": "llamacpp",
                "model": model,
                "image_ref": image_ref,
                "question": question_text,
                "preview": preview,
            },
        )
        return _error_output(image_ref, f"Vision model returned non-JSON output: {preview}", raw_output, code="INVALID_MODEL_OUTPUT")
    record_llm_json(
        "vision_output",
        {"image_ref": image_ref, "question": question_text, "json": payload},
    )
    record_event(
        "model_provider",
        {"provider": "llamacpp", "model": model},
    )
    record_llm_json(
        "model_provider",
        {"provider": "llamacpp", "model": model},
    )
    return _coerce_output(image_ref, payload, "Vision analysis completed.", question_text, raw_output)


def _analyze_hf(image_ref: str, question: Optional[str] = None) -> Dict[str, Any]:
    settings = load_settings()
    model = settings.models.get("vision_model", "")
    if not model:
        return _error_output(image_ref, "models.vision_model is empty in settings.", code="MISSING_DEPENDENCY")

    try:
        raw = _load_image_bytes(image_ref)
    except FileNotFoundError:
        return _error_output(image_ref, "Image file not found.", code="MISSING_INPUT")
    except SandboxViolation as e:
        return _error_output(image_ref, str(e), code="PERMISSION_DENIED")

    question_text = (question or "").strip()
    prompt = (
        "Analyze the image and respond with strict JSON only.\n"
        "Return this schema: "
        "{\"summary\":\"...\",\"answer\":\"...\",\"text\":\"...\",\"objects\":[{\"name\":\"...\",\"count\":1}]}\n"
        "Rules: keep summary brief; text is visible OCR; objects is a list (may be empty); "
        "answer must directly answer the user question when provided.\n"
    )
    if question_text:
        prompt += f"User question: {question_text}\n"
        prompt += "Use the question to focus both summary and answer."
    else:
        prompt += "No user question was provided; set answer to a short key takeaway."

    enforce_json = bool(settings.models.get("vision_enforce_json", True))
    try:
        raw_output = hf_generate_vision(prompt, model, raw)
    except HfError as e:
        log_exception(
            "vision_error",
            e,
            {
                "provider": "hf",
                "model": model,
                "image_ref": image_ref,
                "question": question_text,
            },
        )
        return _error_output(image_ref, f"HF vision request failed: {e}")

    payload = _extract_json(raw_output)
    if not payload:
        if not enforce_json:
            fallback_text = raw_output.strip() if isinstance(raw_output, str) else ""
            return _coerce_output(
                image_ref,
                {"summary": fallback_text, "answer": fallback_text, "text": "", "objects": []},
                fallback_text or "Vision analysis completed.",
                question_text,
                raw_output,
            )
        preview = raw_output.strip().replace("\n", " ")[:300] if isinstance(raw_output, str) else ""
        log_error(
            "vision_parse_error",
            {
                "provider": "hf",
                "model": model,
                "image_ref": image_ref,
                "question": question_text,
                "preview": preview,
            },
        )
        return _error_output(image_ref, f"Vision model returned non-JSON output: {preview}", raw_output, code="INVALID_MODEL_OUTPUT")
    record_llm_json(
        "vision_output",
        {"image_ref": image_ref, "question": question_text, "json": payload},
    )
    record_event(
        "model_provider",
        {"provider": provider(), "model": model},
    )
    record_llm_json(
        "model_provider",
        {"provider": provider(), "model": model},
    )
    return _coerce_output(image_ref, payload, "Vision analysis completed.", question_text, raw_output)


def analyze(image_ref: str, question: Optional[str] = None) -> Dict[str, Any]:
    current_provider = provider()
    if current_provider == "hf":
        try:
            result = _analyze_hf(image_ref, question=question)
            if isinstance(result, dict) and result.get("ok") is False:
                return _analyze_ollama(image_ref, question=question)
            return result
        except Exception as e:
            model = load_settings().models.get("vision_model", "")
            log_exception(
                "vision_error",
                e,
                {"provider": "hf", "model": model, "image_ref": image_ref},
            )
            return _analyze_ollama(image_ref, question=question)
    if current_provider == "ollama":
        return _analyze_ollama(image_ref, question=question)
    if current_provider == "llamacpp":
        return _analyze_llamacpp(image_ref, question=question)
    return _error_output(
        image_ref,
        f"Unsupported vision provider '{current_provider}'. Expected 'hf', 'ollama' or 'llamacpp'.",
    )
