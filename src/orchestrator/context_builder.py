import time
from typing import Any, Dict, List, Optional

from embodiment.manager import EmbodimentManager


class ContextBuilder:
    def __init__(
        self,
        max_turns: int = 10,
        embodiment_manager: Optional[EmbodimentManager] = None,
        embodiment_cache_ttl_s: float = 5.0,
    ):
        self._max_turns = max_turns
        self._embodiment_manager = embodiment_manager or EmbodimentManager()
        self._embodiment_cache_ttl_s = max(0.0, float(embodiment_cache_ttl_s))
        self._embodiment_cache: Dict[str, Any] = {}
        self._embodiment_cache_ts = 0.0

    def _embodiment_context(self) -> Dict[str, Any]:
        now = time.monotonic()
        if self._embodiment_cache and now - self._embodiment_cache_ts <= self._embodiment_cache_ttl_s:
            return dict(self._embodiment_cache)
        try:
            snapshot = self._embodiment_manager.summarize_context()
        except Exception:
            snapshot = {
                "summary": "Embodiment context unavailable.",
                "available_capabilities": [],
                "configured_capabilities": [],
            }
        self._embodiment_cache = dict(snapshot)
        self._embodiment_cache_ts = now
        return dict(snapshot)

    def build(self, history: List[Dict[str, Any]], summary: str, memory_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        recent = history[-self._max_turns :] if history else []
        return {
            "summary": summary,
            "recent_turns": recent,
            "memory": memory_hits,
            "embodiment": self._embodiment_context(),
        }
