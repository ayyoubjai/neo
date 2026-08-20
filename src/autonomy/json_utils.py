from __future__ import annotations

import json
import re
from typing import Any, Optional


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json_candidate(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        fenced = fence_match.group(1).strip()
        if fenced.startswith("{") and fenced.endswith("}"):
            return fenced
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_json_object(raw: str) -> dict[str, Any]:
    candidate = extract_json_candidate(raw)
    if not candidate:
        return {}
    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            payload = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}
