from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from model_server.text_model import generate as generate_text


def build_frame_prompt(question: Optional[str], *, source_label: str = "frame") -> str:
    base = f"Describe this {source_label} briefly. Include visible text, main objects, and any obvious action."
    question_text = str(question or "").strip()
    if question_text:
        return f"{base} Keep the user's overall question in mind: {question_text}"
    return base


def aggregate_objects(keyframes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for item in keyframes:
        for obj in item.get("objects") or []:
            if isinstance(obj, dict):
                name = str(obj.get("name") or obj.get("label") or "").strip()
                count = obj.get("count", 1)
            else:
                name = str(obj or "").strip()
                count = 1
            if not name:
                continue
            try:
                count_value = max(1, int(count))
            except (TypeError, ValueError):
                count_value = 1
            bucket = buckets.setdefault(name, {"name": name, "count": 0, "mentions": 0})
            bucket["count"] = max(int(bucket["count"]), count_value)
            bucket["mentions"] = int(bucket["mentions"]) + 1
    objects = list(buckets.values())
    objects.sort(key=lambda item: (-int(item["mentions"]), -int(item["count"]), str(item["name"])))
    return objects[:16]


def collect_visible_text(keyframes: Sequence[Mapping[str, Any]]) -> str:
    seen = set()
    parts: List[str] = []
    for item in keyframes:
        text = str(item.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return "\n".join(parts[:8])


def summary_from_keyframes(keyframes: Sequence[Mapping[str, Any]]) -> str:
    seen = set()
    parts: List[str] = []
    for item in keyframes:
        summary = str(item.get("summary") or "").strip()
        if not summary or summary in seen:
            continue
        seen.add(summary)
        parts.append(summary)
    if not parts:
        return "Visual analysis completed, but no clear summary was extracted."
    return "; ".join(parts[:5])


def answer_from_keyframes(
    question: str,
    summary: str,
    keyframes: Sequence[Mapping[str, Any]],
    *,
    evidence_label: str,
    duration_s: Optional[float] = None,
    transcript: str = "",
) -> str:
    if not question:
        return summary
    timeline_lines = []
    for item in keyframes[:12]:
        timeline_lines.append(f"- {item.get('timestamp_s', 0):.2f}s: {item.get('summary', '')}")
    duration_line = ""
    if duration_s is not None:
        duration_line = f"{evidence_label.title()} duration: {duration_s} seconds\n"
    transcript_block = ""
    transcript_text = transcript.strip()
    if transcript_text:
        transcript_block = f"\nTranscript excerpt:\n{transcript_text[:4000]}\n"
    prompt = (
        f"Use the following sampled {evidence_label} evidence to answer the user's question.\n"
        f"{duration_line}"
        f"Overall summary: {summary}\n"
        "Timeline:\n"
        + "\n".join(timeline_lines)
        + transcript_block
        + f"\n\nUser question: {question}\n"
        "Answer directly and concisely. If the evidence is insufficient, say so."
    )
    return generate_text(prompt, {"mode": "COGNITION", "summary": "", "memory": [], "suppress_summary_memory": True}).strip()
