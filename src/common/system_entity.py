import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SystemEntity:
    mem_id: str
    name: str
    aliases: List[str]
    properties: Dict[str, str]
    pins: List[str]


DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "system_entity.json",
)


def load_system_entity(path: str = DEFAULT_PATH) -> SystemEntity:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return SystemEntity(
        mem_id=raw.get("mem_id", "system.object"),
        name=raw.get("name", "Assistant"),
        aliases=raw.get("aliases", []),
        properties=raw.get("properties", {}),
        pins=raw.get("pins", []),
    )


def _strip_alias_anywhere(text: str, alias: str) -> Optional[str]:
    text_clean = text.strip()
    alias_clean = alias.strip()
    if not alias_clean:
        return None
    if text_clean.lower() == alias_clean.lower():
        return ""
    parts = [re.escape(part) for part in alias_clean.split()]
    if not parts:
        return None
    pattern = r"\b" + r"\s+".join(parts) + r"\b"
    match = re.search(pattern, text_clean, flags=re.IGNORECASE)
    if not match:
        return None
    start, end = match.span()
    before = text_clean[:start]
    after = text_clean[end:]
    combined = (before + " " + after).strip()
    combined = re.sub(r"\s+", " ", combined)
    combined = re.sub(r"\s+([,.:;!?])", r"\1", combined)
    combined = combined.strip(" ,:;-\t")
    return combined


def detect_wake_word(text: str, entity: SystemEntity) -> Tuple[bool, str]:
    aliases = [entity.name] + entity.aliases
    aliases = sorted(aliases, key=lambda item: len(item or ""), reverse=True)
    for alias in aliases:
        remainder = _strip_alias_anywhere(text, alias)
        if remainder is not None:
            return True, remainder
    return False, text


def strip_wake_word(text: str, entity: SystemEntity, strip: bool = True) -> Tuple[bool, str]:
    is_wake, remainder = detect_wake_word(text, entity)
    if is_wake and not strip:
        return True, text
    return is_wake, remainder
