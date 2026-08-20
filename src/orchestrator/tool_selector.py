import json
import time
from typing import Any, Dict, List, Optional, Set

from common.config import load_settings


_DUPLICATE_UI_TOOL_IDS = {
    "ui.screenshot",
    "ui.click",
    "ui.type",
    "ui.focus_window",
}

_PROFILE_BUCKETS = {
    "default": {
        "base",
        "google",
        "embodiment.observe",
        "embodiment.desktop",
        "embodiment.camera",
        "embodiment.phone",
        "embodiment.robot",
        "ui.specialized",
    },
    "initial": {
        "base",
        "google",
        "embodiment.observe",
        "ui.specialized",
        "ui.duplicate",
    },
    "embodiment": {
        "google",
        "embodiment.observe",
        "embodiment.desktop",
        "embodiment.camera",
        "embodiment.phone",
        "embodiment.robot",
    },
    "full": {"*"},
}

_ALWAYS_VISIBLE_BUCKETS = {"base", "google", "embodiment.observe"}
_DETECTED_DEVICE_BUCKETS = {"embodiment.phone", "embodiment.robot"}
_DYNAMIC_BUCKETS = {
    "embodiment.desktop",
    "embodiment.camera",
    "embodiment.phone",
    "embodiment.robot",
    "ui.specialized",
    "ui.duplicate",
}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ToolRegistry:
    def __init__(self, registry_path: Optional[str] = None, embodiment_manager: Any = None):
        settings = load_settings()
        self._settings = settings
        if registry_path is None:
            registry_path = settings.workspace_root + "/config/tool_registry.json"
        with open(registry_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._embodiment_manager = embodiment_manager
        self._planner_profile = self._normalize_profile(settings.tool.get("planner_tool_profile", "default"))
        self._planner_hide_duplicate_ui_tools = _coerce_bool(
            settings.tool.get("planner_hide_duplicate_ui_tools", True),
            True,
        )
        self._planner_dynamic_activation_enabled = _coerce_bool(
            settings.tool.get("planner_dynamic_activation_enabled", True),
            True,
        )
        self._planner_activation_ttl_s = max(
            0.0,
            _coerce_float(settings.tool.get("planner_activation_ttl_s", 5.0), 5.0),
        )
        self._planner_show_embodiment_fallback_tools = _coerce_bool(
            settings.tool.get("planner_show_embodiment_fallback_tools", False),
            False,
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

    def _refresh_active(self, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and self._active_cache_ts > 0.0
            and now - self._active_cache_ts <= self._planner_activation_ttl_s
        ):
            return

        capability_statuses = self._load_capability_statuses() if self._planner_dynamic_activation_enabled else {}
        fallback_count = 0
        if self._planner_show_embodiment_fallback_tools:
            fallback_count = self._load_fallback_count()

        active: List[Dict[str, Any]] = []
        for tool in self._tool_order:
            if not self._is_planner_visible(tool, capability_statuses, fallback_count):
                continue
            active.append(tool)
            if len(active) >= self._active_tool_limit:
                break
        self._active_cache = active
        self._active_cache_ts = now

    def _is_planner_visible(
        self,
        tool: Dict[str, Any],
        capability_statuses: Dict[str, Set[str]],
        fallback_count: int,
    ) -> bool:
        bucket = str(tool.get("tool_bucket") or "").strip()
        tool_id = str(tool.get("tool_id") or "").strip()
        if not tool_id or not self._profile_allows_bucket(bucket):
            return False

        if (
            self._planner_profile not in {"initial", "full"}
            and self._planner_hide_duplicate_ui_tools
            and tool_id in _DUPLICATE_UI_TOOL_IDS
        ):
            return False

        if bucket == "embodiment.fallback":
            return self._planner_show_embodiment_fallback_tools and fallback_count > 0

        if bucket in _ALWAYS_VISIBLE_BUCKETS:
            return True

        if bucket == "embodiment.generated":
            return self._planner_profile == "full"

        if bucket not in _DYNAMIC_BUCKETS:
            return True

        statuses = capability_statuses.get(tool_id, set())
        if not statuses:
            return False
        if bucket in _DETECTED_DEVICE_BUCKETS:
            return bool(statuses.intersection({"available", "configured"}))
        return "available" in statuses

    def _profile_allows_bucket(self, bucket: str) -> bool:
        allowed = _PROFILE_BUCKETS.get(self._planner_profile, _PROFILE_BUCKETS["default"])
        return "*" in allowed or bucket in allowed

    def _load_capability_statuses(self) -> Dict[str, Set[str]]:
        manager = self._manager()
        if manager is None:
            return {}
        try:
            payload = manager.list_capabilities()
        except Exception:
            return {}
        statuses: Dict[str, Set[str]] = {}
        capability_status_by_id: Dict[str, str] = {}
        capability_status_by_abstract: Dict[str, Set[str]] = {}
        for item in payload.get("capabilities", []):
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip()
            if not status:
                continue
            capability_id = str(item.get("capability_id") or "").strip()
            abstract_capability = str(item.get("abstract_capability") or "").strip()
            if capability_id:
                capability_status_by_id[capability_id] = status
            if abstract_capability:
                capability_status_by_abstract.setdefault(abstract_capability, set()).add(status)
            tool_ids = item.get("tool_ids", [])
            if not isinstance(tool_ids, list):
                continue
            for tool_id in tool_ids:
                if not isinstance(tool_id, str) or not tool_id:
                    continue
                statuses.setdefault(tool_id, set()).add(status)
        for tool_id, tool in self._tools.items():
            capability_id = str(tool.get("implements_capability_id") or "").strip()
            if capability_id and capability_id in capability_status_by_id:
                statuses.setdefault(tool_id, set()).add(capability_status_by_id[capability_id])
            abstract_capability = str(tool.get("implements_abstract_capability") or "").strip()
            if abstract_capability:
                for status in capability_status_by_abstract.get(abstract_capability, set()):
                    statuses.setdefault(tool_id, set()).add(status)
        return statuses

    def _load_fallback_count(self) -> int:
        manager = self._manager()
        if manager is None:
            return 0
        try:
            payload = manager.list_fallbacks()
        except Exception:
            return 0
        fallbacks = payload.get("fallbacks", [])
        if not isinstance(fallbacks, list):
            return 0
        return len(fallbacks)

    def _manager(self) -> Any:
        if self._embodiment_manager is not None:
            return self._embodiment_manager
        try:
            from embodiment.manager import EmbodimentManager
        except Exception:
            return None
        self._embodiment_manager = EmbodimentManager(settings=self._settings)
        return self._embodiment_manager

    def _decorate_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(tool)
        item["tool_bucket"] = self._bucket_for_tool(item)
        item["tool_profile"] = self._planner_profile
        return item

    @staticmethod
    def _normalize_profile(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"default", "initial", "embodiment", "full"}:
            return text
        if text in {"base", "base_only", "legacy", "legacy_ui"}:
            return "initial"
        return "default"

    @staticmethod
    def _bucket_for_tool(tool: Dict[str, Any]) -> str:
        tool_id = str(tool.get("tool_id") or "").strip()
        capabilities = tool.get("capabilities", [])
        capability_values = {
            str(item).strip().lower()
            for item in capabilities
            if isinstance(item, str) and str(item).strip()
        }
        if tool_id.startswith("google."):
            return "google"
        if tool_id in {
            "embodiment.describe_host",
            "embodiment.list_capabilities",
            "embodiment.get_state",
        }:
            return "embodiment.observe"
        if tool_id == "embodiment.list_fallbacks":
            return "embodiment.fallback"
        if tool_id.startswith("embodiment.generated."):
            if "phone" in capability_values or ".phone_" in tool_id:
                return "embodiment.phone"
            if "robot" in capability_values or ".robot_" in tool_id:
                return "embodiment.robot"
            return "embodiment.generated"
        if tool_id.startswith("embodiment.phone_") or "phone" in capability_values:
            return "embodiment.phone"
        if tool_id.startswith("embodiment.robot_") or "robot" in capability_values:
            return "embodiment.robot"
        if tool_id == "embodiment.camera_capture":
            return "embodiment.camera"
        if tool_id.startswith("embodiment."):
            return "embodiment.desktop"
        if tool_id in _DUPLICATE_UI_TOOL_IDS:
            return "ui.duplicate"
        if tool_id.startswith("ui."):
            return "ui.specialized"
        return "base"
