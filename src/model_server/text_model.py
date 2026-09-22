import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config import load_settings
from common.error_log import log_exception
from common.evolve_trace import trace_calls
from common.llm_json_log import record_llm_json
from common.record_log import record_event
from common.system_entity import DEFAULT_PATH, load_system_entity
from model_server.hf_client import HfError, generate_text as hf_generate_text
from model_server.ollama_client import OllamaError, generate as ollama_generate, provider
from model_server.llamacpp_client import LlamacppError, generate as llamacpp_generate


# Path to personality configuration
PERSONALITY_PATH = Path(__file__).parent.parent.parent / "config" / "personality.json"


def _format_memory(memory: List[Dict[str, Any]]) -> str:
    lines = []
    for item in memory:
        if isinstance(item, dict):
            data = item.get("data")
            if isinstance(data, dict):
                summary = data.get("summary")
                if isinstance(summary, str) and summary:
                    lines.append(f"- {summary}")
                    continue
                value = data.get("value")
                if isinstance(value, str) and value:
                    lines.append(f"- {value}")
                    continue
        if isinstance(item, dict) and "embedding" in item:
            item = dict(item)
            item.pop("embedding", None)
        try:
            lines.append(json.dumps(item, ensure_ascii=False))
        except TypeError:
            lines.append(str(item))
    return "\n".join(lines)


_SYSTEM_PROMPT_CACHE: Dict[str, Optional[object]] = {
    "entity_mtime": None, 
    "personality_mtime": None,
    "prompt": None
}


def _load_personality() -> Optional[str]:
    """Load expanded personality from personality.json if it exists."""
    try:
        if PERSONALITY_PATH.exists():
            with open(PERSONALITY_PATH, "r") as f:
                data = json.load(f)
                return data.get("expanded_personality", "").strip()
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _build_system_prompt() -> str:
    """Build system prompt with entity identity and personality."""
    # Get modification times for cache invalidation
    try:
        entity_mtime = os.path.getmtime(DEFAULT_PATH)
    except OSError:
        entity_mtime = None
    
    try:
        personality_mtime = PERSONALITY_PATH.stat().st_mtime if PERSONALITY_PATH.exists() else None
    except OSError:
        personality_mtime = None
    
    # Check cache validity
    cached_entity_mtime = _SYSTEM_PROMPT_CACHE.get("entity_mtime")
    cached_personality_mtime = _SYSTEM_PROMPT_CACHE.get("personality_mtime")
    cached_prompt = _SYSTEM_PROMPT_CACHE.get("prompt")
    
    # Return cached version if both files unchanged
    if (cached_prompt is not None and 
        (cached_entity_mtime == entity_mtime or entity_mtime is None) and
        (cached_personality_mtime == personality_mtime or personality_mtime is None)):
        return cached_prompt

    # Load system entity
    entity = load_system_entity()
    aliases = ", ".join(entity.aliases) if entity.aliases else "None"
    
    # Build base prompt
    prompt_lines = [
        "You are the assistant. Be concise and helpful.",
        "Always respond in the same language the user writes in.",
        f"Your name: {entity.name}",
        f"Your aliases: {aliases}",
    ]
    
    # Load and append personality if available
    personality = _load_personality()
    if personality:
        prompt_lines.append("")
        prompt_lines.append("=" * 60)
        prompt_lines.append("PERSONALITY PROFILE")
        prompt_lines.append("=" * 60)
        prompt_lines.append(personality)
        prompt_lines.append("=" * 60)
    
    prompt = "\n".join(prompt_lines)
    
    # Update cache
    _SYSTEM_PROMPT_CACHE["entity_mtime"] = entity_mtime
    _SYSTEM_PROMPT_CACHE["personality_mtime"] = personality_mtime
    _SYSTEM_PROMPT_CACHE["prompt"] = prompt
    
    return prompt


def _requires_structured_json(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return "STRICT JSON ONLY" in text.upper()


def _should_request_json_response(mode: str, text: str, enforce_json: bool) -> bool:
    return bool(enforce_json) or _requires_structured_json(text)


def _format_prompt_user_turn(mode: Any, text: str) -> str:
    return text


def _mode_uses_raw_prompt(mode: Any) -> bool:
    return True


def _normalize_model_options(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        name = key.strip()
        if not name:
            continue
        if name == "reasoning_effort":
            text = str(item).strip()
            if text:
                normalized[name] = text
            continue
        if name in {"thinking", "enable_thinking", "think"}:
            if isinstance(item, bool):
                normalized["thinking"] = item
                continue
            if isinstance(item, (int, float)):
                normalized["thinking"] = bool(item)
                continue
            if isinstance(item, str):
                lowered = item.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    normalized["thinking"] = True
                    continue
                if lowered in {"0", "false", "no", "off"}:
                    normalized["thinking"] = False
                    continue
            continue
        normalized[name] = item
    return normalized


def generate(
    text: str,
    context: Dict[str, Any],
    model_override: Optional[Any] = None,
    options_override: Optional[Any] = None,
) -> str:
    mode = "COGNITION"
    summary = context.get("summary", "")
    memory = context.get("memory", [])
    trace_id = context.get("trace_id")
    turn_id = context.get("turn_id")
    current_time = str(context.get("current_time") or "").strip()
    suppress_summary_memory = bool(context.get("suppress_summary_memory", False))
    settings = load_settings()
    override_model = str(model_override or "").strip()
    model = override_model or settings.models.get("text_model", "")
    enforce_json = bool(settings.models.get("text_enforce_json", False))
    use_thinking_on_empty = bool(settings.models.get("text_use_thinking_on_empty", True))
    resolved_options: Dict[str, Any] = {"temperature": 0.2}
    resolved_options.update(_normalize_model_options(settings.models.get("text_model_options", {})))
    if isinstance(options_override, dict):
        resolved_options.update(_normalize_model_options(options_override))
    try:
        temperature = float(resolved_options.get("temperature", 0.2))
    except (TypeError, ValueError):
        temperature = 0.2
        resolved_options["temperature"] = temperature
    structured_json_required = _should_request_json_response(mode, text, enforce_json)
    system = _build_system_prompt()
    memory_block = _format_memory(memory) if memory else "None"
    if _mode_uses_raw_prompt(mode):
        semantic_summary = str(context.get("semantic_summary") or "").strip()
        conversation_context = semantic_summary or str(summary or "").strip()
        prompt_parts = []
        if current_time:
            prompt_parts.extend([f"Current UTC time: {current_time}", ""])
        if not suppress_summary_memory and conversation_context:
            prompt_parts.extend(["Conversation context:", conversation_context, ""])
        if not suppress_summary_memory and memory:
            prompt_parts.extend(["Relevant memory:", memory_block, ""])
        prompt_parts.append(text)
        prompt = "\n".join(prompt_parts)
    elif suppress_summary_memory:
        prompt_parts = [f"Mode: {mode}", ""]
        prompt_parts.extend([_format_prompt_user_turn(mode, text), "Assistant:"])
        prompt = "\n".join(prompt_parts)
    else:
        prompt_parts = [
            f"Mode: {mode}",
            f"Summary: {summary}",
            f"Memory:\n{memory_block}",
            "",
        ]
        prompt_parts.extend([_format_prompt_user_turn(mode, text), "Assistant:"])
        prompt = "\n".join(prompt_parts)
    current_provider = provider()
    fallback_error_type = ""
    fallback_error = ""
    if current_provider == "hf" and model:
        try:
            record_event(
                "model_prompt",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "system": system,
                    "prompt": prompt,
                    "memory_count": len(memory),
                    "model": model,
                },
            )
            output = hf_generate_text(prompt, model, temperature=temperature)
            record_event(
                "model_output",
                {"turn_id": turn_id, "trace_id": trace_id, "mode": mode, "text": output},
            )
            record_event(
                "model_provider",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "hf",
                    "model": model,
                },
            )
            record_llm_json(
                "model_provider",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "hf",
                    "model": model,
                },
            )
            return output
        except HfError as e:
            fallback_error_type = e.__class__.__name__
            fallback_error = str(e)
            log_exception(
                "model_error",
                e,
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "hf",
                    "model": model,
                },
            )

    if current_provider == "llamacpp" and model:
        try:
            record_event(
                "model_prompt",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "system": system,
                    "prompt": prompt,
                    "memory_count": len(memory),
                    "model": model,
                },
            )
            response_format = "json" if structured_json_required else None
            output = llamacpp_generate(
                prompt,
                model,
                system=system,
                options=resolved_options,
                response_format=response_format,
                use_thinking_on_empty=use_thinking_on_empty,
            )
            record_event(
                "model_output",
                {"turn_id": turn_id, "trace_id": trace_id, "mode": mode, "text": output},
            )
            record_event(
                "model_provider",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "llamacpp",
                    "model": model,
                },
            )
            record_llm_json(
                "model_provider",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "llamacpp",
                    "model": model,
                },
            )
            return output
        except LlamacppError as e:
            fallback_error_type = e.__class__.__name__
            fallback_error = str(e)
            log_exception(
                "model_error",
                e,
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "llamacpp",
                    "model": model,
                },
            )

    if model:
        try:
            record_event(
                "model_prompt",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "system": system,
                    "prompt": prompt,
                    "memory_count": len(memory),
                    "model": model,
                },
            )
            response_format = "json" if structured_json_required else None
            output = ollama_generate(
                prompt,
                model,
                system=system,
                options=resolved_options,
                response_format=response_format,
                use_thinking_on_empty=use_thinking_on_empty,
            )
            record_event(
                "model_output",
                {"turn_id": turn_id, "trace_id": trace_id, "mode": mode, "text": output},
            )
            record_event(
                "model_provider",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "ollama",
                    "model": model,
                },
            )
            record_llm_json(
                "model_provider",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "ollama",
                    "model": model,
                },
            )
            return output
        except OllamaError as e:
            fallback_error_type = e.__class__.__name__
            fallback_error = str(e)
            log_exception(
                "model_error",
                e,
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "provider": "ollama",
                    "model": model,
                },
            )

    if structured_json_required:
        record_event(
            "model_output",
            {"turn_id": turn_id, "trace_id": trace_id, "mode": mode, "text": ""},
        )
        record_event(
            "model_unavailable",
            {
                "turn_id": turn_id,
                "trace_id": trace_id,
                "mode": mode,
                "structured_json_required": True,
                "provider": current_provider,
                "model": model,
                "error_type": fallback_error_type,
                "error": fallback_error,
            },
        )
        return ""

    if context.get("require_model_output"):
        raise RuntimeError(f"Model generation failed ({model}): {fallback_error or 'no output'}")

    fallback = (
        "I hit a model/runtime issue while processing your request, so I cannot give a reliable answer right now. "
        "Please try again or switch to a smaller/local model."
    )
    record_event(
        "model_fallback",
        {
            "turn_id": turn_id,
            "trace_id": trace_id,
            "mode": mode,
            "text": fallback,
            "provider": current_provider,
            "model": model,
            "error_type": fallback_error_type,
            "error": fallback_error,
        },
    )
    record_event("model_output", {"turn_id": turn_id, "trace_id": trace_id, "mode": mode, "text": fallback})
    return fallback
