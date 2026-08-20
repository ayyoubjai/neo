from __future__ import annotations

from typing import Iterable


PURPOSES = ["assistant", "coding", "research", "automation", "phone_companion"]


def infer_purpose(text: str, allowed: Iterable[str] = PURPOSES) -> str:
    allowed_set = set(allowed)
    normalized = (text or "").lower()
    scores = {
        "coding": _count_keywords(
            normalized,
            ["code", "coding", "program", "debug", "repo", "python", "javascript", "software"],
        ),
        "research": _count_keywords(
            normalized,
            ["research", "study", "learn", "paper", "summarize", "knowledge", "web", "search"],
        ),
        "automation": _count_keywords(
            normalized,
            ["automate", "automation", "execute", "workflow", "script", "task", "operate"],
        ),
        "phone_companion": _count_keywords(
            normalized,
            ["phone", "android", "termux", "mobile", "pocket", "call", "sms"],
        ),
        "assistant": 1,
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for purpose, score in ranked:
        if purpose in allowed_set and score > 0:
            return purpose
    return "assistant"


def _count_keywords(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)

