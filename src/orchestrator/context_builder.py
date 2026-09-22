from typing import Any, Dict, List


class ContextBuilder:
    def __init__(self, max_turns: int = 10):
        self._max_turns = max_turns

    def build(
        self,
        history: List[Dict[str, Any]],
        summary: str,
        memory_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        recent = history[-self._max_turns :] if history else []
        return {
            "summary": summary,
            "recent_turns": recent,
            "memory": memory_hits,
        }
