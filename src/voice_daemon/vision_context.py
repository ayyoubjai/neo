from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from runtime_core.runtime import RuntimeContext


VISION_LATEST_SCENE_KEY = "vision.latest_scene"
VISION_STABLE_SCENE_KEY = "vision.stable_scene"
VISION_TRACKED_ENTITIES_KEY = "vision.entities"

_DEFAULT_QUERY_TRIGGERS = [
    "what do you see",
    "what can you see",
    "look around",
    "what is in front of you",
    "what's in front of you",
    "describe what you see",
    "describe the scene",
    "read this",
    "scan this",
]


def vision_query_triggers(value: Optional[Iterable[str]] = None) -> List[str]:
    if value is None:
        return list(_DEFAULT_QUERY_TRIGGERS)
    cleaned = []
    for item in value:
        trigger = str(item or "").strip().lower()
        if trigger:
            cleaned.append(trigger)
    return cleaned or list(_DEFAULT_QUERY_TRIGGERS)


def wants_vision_context(text: str, triggers: Optional[Iterable[str]] = None) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return False
    for trigger in vision_query_triggers(triggers):
        if trigger in normalized:
            return True
    return False


def format_scene_context(scene: Dict[str, Any]) -> str:
    if not isinstance(scene, dict):
        return ""
    summary = str(scene.get("summary") or "").strip()
    objects = scene.get("objects") or []
    entities = scene.get("entities") or []
    lines: List[str] = []
    if summary:
        lines.append(f"Summary: {summary}")
    if isinstance(objects, list) and objects:
        parts: List[str] = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            count = item.get("count")
            if not label or count is None:
                continue
            parts.append(f"{count}x {label}")
        if parts:
            lines.append("Objects: " + ", ".join(parts))
    if isinstance(entities, list) and entities:
        tracked_parts: List[str] = []
        for item in entities:
            if not isinstance(item, dict):
                continue
            track_id = item.get("track_id")
            label = str(item.get("object_name") or item.get("label") or "").strip()
            if not label:
                continue
            prefix = f"#{track_id} " if track_id not in (None, "") else ""
            tracked_parts.append(prefix + label)
        if tracked_parts:
            lines.append("Tracked: " + ", ".join(tracked_parts))
    return "\n".join(lines).strip()


async def maybe_augment_turn_with_vision_context(
    context: RuntimeContext,
    text: str,
    *,
    triggers: Optional[Iterable[str]] = None,
    state_key: str = VISION_STABLE_SCENE_KEY,
) -> str:
    if not wants_vision_context(text, triggers):
        return text
    scene = await context.get_state(state_key)
    if not isinstance(scene, dict):
        return text
    scene_context = format_scene_context(scene)
    if not scene_context:
        return text
    return f"{text}\n\n[Local camera scene]\n{scene_context}"
