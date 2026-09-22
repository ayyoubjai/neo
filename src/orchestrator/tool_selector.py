import json
import time
from typing import Any, Dict, List, Optional

from common.config import load_settings


_PROFILE_BUCKETS = {
    "default": {"base", "google", "ui.specialized"},
    "initial": {"base", "google", "ui.specialized"},
    "full": {"*"},
}


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ToolRegistry:
    def __init__(self, registry_path: Optional[str] = None):
        settings = load_settings()
        self._settings = settings
        if registry_path is None:
            registry_path = settings.workspace_root + "/config/tool_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._planner_profile = self._normalize_profile(settings.tool.get("planner_tool_profile", "default"))
        self._planner_activation_ttl_s = max(
            0.0,
            _coerce_float(settings.tool.get("planner_activation_ttl_s", 5.0), 5.0),
        )
        self._active_tool_limit = max(1, int(settings.tool.get("active_tool_limit", 100)))
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._tool_order: List[Dict[str, Any]] = []
        for tool in raw.get("tools", []):
            if not isinstance(tool, dict):
                continue
            tool_id = tool.get("tool_id")
            if not isinstance(tool_id, str) or not tool_id:
                continue
            item = self._decorate_tool(tool)
            self._tools[tool_id] = item
            self._tool_order.append(item)
        self._active_cache: List[Dict[str, Any]] = []
        self._active_cache_ts = 0.0

    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        tool = self._tools.get(tool_id)
        if tool is None:
            return None
        return dict(tool)

    def list_active(self) -> List[Dict[str, Any]]:
        self._refresh_active()
        return [dict(tool) for tool in self._active_cache]

    def search_active(self, capability: str) -> List[Dict[str, Any]]:
        return [tool for tool in self.list_active() if capability in tool.get("capabilities", [])]

    def list_available(self) -> List[Dict[str, Any]]:
        """Recovery searches the full permitted catalog, beyond the active cap."""
        return [dict(tool) for tool in self._tool_order if self._is_planner_visible(tool)]

    def _refresh_active(self, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and self._active_cache_ts > 0.0
            and now - self._active_cache_ts <= self._planner_activation_ttl_s
        ):
            return

        active: List[Dict[str, Any]] = []
        for tool in self._tool_order:
            if not self._is_planner_visible(tool):
                continue
            active.append(tool)
            if len(active) >= self._active_tool_limit:
                break
        self._active_cache = active
        self._active_cache_ts = now

    def _is_planner_visible(self, tool: Dict[str, Any]) -> bool:
        bucket = str(tool.get("tool_bucket") or "").strip()
        tool_id = str(tool.get("tool_id") or "").strip()
        if not tool_id or not self._profile_allows_bucket(bucket):
            return False
        return True

    def _profile_allows_bucket(self, bucket: str) -> bool:
        allowed = _PROFILE_BUCKETS.get(self._planner_profile, _PROFILE_BUCKETS["default"])
        return "*" in allowed or bucket in allowed

    def _decorate_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(tool)
        item["tool_bucket"] = self._bucket_for_tool(item)
        item["tool_profile"] = self._planner_profile
        return item

    @staticmethod
    def _normalize_profile(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"default", "initial", "full"}:
            return text
        if text in {"base", "base_only", "legacy", "legacy_ui"}:
            return "initial"
        return "default"

    @staticmethod
    def _bucket_for_tool(tool: Dict[str, Any]) -> str:
        tool_id = str(tool.get("tool_id") or "").strip()
        if tool_id.startswith("google."):
            return "google"
        if tool_id.startswith("ui."):
            return "ui.specialized"
        return "base"
