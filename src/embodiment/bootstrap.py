from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

from common.config import Settings, load_settings
from common.time_utils import utc_now_iso
from embodiment.manager import EmbodimentManager
from orchestrator.tool_selector import ToolRegistry


CreateToolFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class EmbodimentBootstrap:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embodiment_manager: Optional[EmbodimentManager] = None,
        tool_registry_cls: Type[ToolRegistry] = ToolRegistry,
    ):
        self._settings = settings or load_settings()
        self._manager = embodiment_manager or EmbodimentManager(settings=self._settings)
        self._tool_registry_cls = tool_registry_cls

    def manifest_path(self, path: Optional[str] = None) -> str:
        if isinstance(path, str) and path.strip():
            value = path.strip()
            if os.path.isabs(value):
                return value
            return os.path.abspath(os.path.join(self._settings.workspace_root, value))
        return os.path.join(self._settings.data_dir, "embodiment_manifest.json")

    def plan(self) -> Dict[str, Any]:
        host_payload = self._manager.describe_host()
        planner_registry = self._tool_registry_cls(embodiment_manager=self._manager)
        planner_tools = planner_registry.list_active()
        queue = self._tool_generation_queue(host_payload.get("fallbacks", []), planner_registry)
        summary = dict(host_payload.get("summary", {}))
        summary.update(
            {
                "planner_tool_count": len(planner_tools),
                "tool_generation_queue_count": len(queue),
            }
        )
        return {
            "ts": utc_now_iso(),
            "host": host_payload.get("host", {}),
            "devices": list(host_payload.get("devices", [])),
            "capabilities": list(host_payload.get("capabilities", [])),
            "planner_tools": [self._planner_tool_view(tool) for tool in planner_tools],
            "tool_generation_queue": queue,
            "summary": summary,
        }

    async def apply(
        self,
        create_tool: CreateToolFn,
        *,
        manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        plan = self.plan()
        created: List[Dict[str, Any]] = []
        existing: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for item in plan.get("tool_generation_queue", []):
            if not isinstance(item, dict):
                continue
            tool_spec = item.get("tool_spec")
            if not isinstance(tool_spec, dict):
                continue
            tool_id = str(tool_spec.get("tool_id") or "").strip()
            if not tool_id:
                continue
            current_registry = self._tool_registry_cls(embodiment_manager=self._manager)
            if current_registry.get_tool(tool_id):
                existing.append(
                    {
                        "tool_id": tool_id,
                        "capability_id": item.get("capability_id"),
                        "status": "EXISTS",
                    }
                )
                continue
            result = await create_tool(dict(tool_spec))
            if str(result.get("status") or "").strip().upper() == "APPROVED":
                created.append(
                    {
                        "tool_id": tool_id,
                        "capability_id": item.get("capability_id"),
                        "status": "APPROVED",
                        "result": result.get("result", {}),
                    }
                )
            else:
                failed.append(
                    {
                        "tool_id": tool_id,
                        "capability_id": item.get("capability_id"),
                        "status": str(result.get("status") or "ERROR"),
                        "error": str(result.get("error") or ""),
                    }
                )

        refreshed_plan = self.plan()
        manifest = {
            "ts": utc_now_iso(),
            "host": refreshed_plan.get("host", {}),
            "devices": refreshed_plan.get("devices", []),
            "capabilities": refreshed_plan.get("capabilities", []),
            "planner_tools": refreshed_plan.get("planner_tools", []),
            "tool_generation_queue": refreshed_plan.get("tool_generation_queue", []),
            "tool_generation_results": {
                "created": created,
                "existing": existing,
                "failed": failed,
            },
            "summary": self._apply_summary(refreshed_plan.get("summary", {}), created, existing, failed),
        }
        self.write_manifest(manifest, path=manifest_path)
        return manifest

    def write_manifest(self, manifest: Dict[str, Any], *, path: Optional[str] = None) -> str:
        target = self.manifest_path(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)
        return target

    def _tool_generation_queue(
        self,
        fallbacks: Any,
        planner_registry: ToolRegistry,
    ) -> List[Dict[str, Any]]:
        if not isinstance(fallbacks, list):
            return []
        queue: List[Dict[str, Any]] = []
        seen = set()
        for item in fallbacks:
            if not isinstance(item, dict):
                continue
            tool_spec = item.get("tool_spec")
            if not isinstance(tool_spec, dict):
                continue
            tool_id = str(tool_spec.get("tool_id") or "").strip()
            if not tool_id or tool_id in seen:
                continue
            if planner_registry.get_tool(tool_id):
                continue
            seen.add(tool_id)
            queue.append(
                {
                    "capability_id": item.get("capability_id"),
                    "abstract_capability": item.get("abstract_capability"),
                    "reason": item.get("reason", ""),
                    "tool_exists": False,
                    "tool_spec": dict(tool_spec),
                }
            )
        return queue

    @staticmethod
    def _planner_tool_view(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tool_id": tool.get("tool_id"),
            "name": tool.get("name"),
            "description": tool.get("description"),
            "tool_bucket": tool.get("tool_bucket"),
        }

    @staticmethod
    def _apply_summary(
        summary: Dict[str, Any],
        created: List[Dict[str, Any]],
        existing: List[Dict[str, Any]],
        failed: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(summary or {})
        merged["created_tool_count"] = len(created)
        merged["existing_tool_count"] = len(existing)
        merged["failed_tool_count"] = len(failed)
        return merged
