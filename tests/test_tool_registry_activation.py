from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import Settings
from orchestrator.tool_selector import ToolRegistry


class _FakeEmbodimentManager:
    def __init__(self, capabilities=None, fallbacks=None) -> None:
        self._capabilities = list(capabilities or [])
        self._fallbacks = list(fallbacks or [])

    def list_capabilities(self):
        return {
            "host_id": "host:test",
            "capabilities": list(self._capabilities),
            "summary": {"capability_count": len(self._capabilities)},
        }

    def list_fallbacks(self):
        return {
            "host_id": "host:test",
            "fallbacks": list(self._fallbacks),
            "summary": {"fallback_count": len(self._fallbacks)},
        }


def _settings(workspace_root: str, tool_overrides=None) -> Settings:
    tool = {
        "active_tool_limit": 100,
        "planner_tool_profile": "default",
        "planner_dynamic_activation_enabled": True,
        "planner_activation_ttl_s": 0.0,
        "planner_hide_duplicate_ui_tools": True,
        "planner_show_embodiment_fallback_tools": False,
    }
    if isinstance(tool_overrides, dict):
        tool.update(tool_overrides)
    return Settings(
        workspace_root=workspace_root,
        data_dir=str(Path(workspace_root) / "data"),
        rpc={},
        interface={"mode": "local"},
        voice={},
        vision={},
        video={},
        telegram={},
        google={},
        tool=tool,
        search={},
        orchestrator={},
        evolve={"candidate_dir": str(Path(workspace_root) / "evolve")},
        models={},
    )


class ToolRegistryActivationTests(unittest.TestCase):
    def _write_registry(self, tmpdir: str, tools) -> str:
        registry_path = Path(tmpdir) / "tool_registry.json"
        registry_path.write_text(json.dumps({"tools": list(tools)}, ensure_ascii=True), encoding="utf-8")
        return str(registry_path)

    def test_default_profile_hides_duplicate_ui_and_undetected_phone_robot_tools(self) -> None:
        tools = [
            {"tool_id": "sys.time", "name": "Time", "description": "Time"},
            {"tool_id": "google.status", "name": "Google", "description": "Google"},
            {"tool_id": "embodiment.describe_host", "name": "Host", "description": "Host"},
            {"tool_id": "embodiment.screen_capture", "name": "Emb Screen", "description": "Screen"},
            {"tool_id": "ui.screenshot", "name": "UI Screen", "description": "Screen"},
            {"tool_id": "ui.drag", "name": "UI Drag", "description": "Drag"},
            {"tool_id": "embodiment.phone_tap", "name": "Phone Tap", "description": "Phone"},
            {"tool_id": "embodiment.robot_get_state", "name": "Robot State", "description": "Robot"},
            {"tool_id": "embodiment.list_fallbacks", "name": "Fallbacks", "description": "Fallbacks"},
        ]
        capabilities = [
            {
                "capability_id": "screen.capture",
                "status": "available",
                "tool_ids": ["embodiment.screen_capture", "ui.screenshot"],
            },
            {
                "capability_id": "pointer.drag",
                "status": "available",
                "tool_ids": ["ui.drag"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = self._write_registry(tmpdir, tools)
            with patch("orchestrator.tool_selector.load_settings", return_value=_settings(tmpdir)):
                registry = ToolRegistry(
                    registry_path=registry_path,
                    embodiment_manager=_FakeEmbodimentManager(capabilities=capabilities),
                )

        active_ids = [item["tool_id"] for item in registry.list_active()]
        self.assertIn("sys.time", active_ids)
        self.assertIn("google.status", active_ids)
        self.assertIn("embodiment.describe_host", active_ids)
        self.assertIn("embodiment.screen_capture", active_ids)
        self.assertIn("ui.drag", active_ids)
        self.assertNotIn("ui.screenshot", active_ids)
        self.assertNotIn("embodiment.phone_tap", active_ids)
        self.assertNotIn("embodiment.robot_get_state", active_ids)
        self.assertNotIn("embodiment.list_fallbacks", active_ids)
        self.assertIsNotNone(registry.get_tool("ui.screenshot"))
        self.assertIsNotNone(registry.get_tool("embodiment.phone_tap"))

    def test_phone_and_robot_tools_become_active_when_detected_or_configured(self) -> None:
        tools = [
            {"tool_id": "embodiment.phone_tap", "name": "Phone Tap", "description": "Phone"},
            {"tool_id": "embodiment.robot_get_state", "name": "Robot State", "description": "Robot"},
        ]
        capabilities = [
            {
                "capability_id": "phone.pointer.tap",
                "status": "configured",
                "tool_ids": ["embodiment.phone_tap"],
            },
            {
                "capability_id": "robot.get_state",
                "status": "configured",
                "tool_ids": ["embodiment.robot_get_state"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = self._write_registry(tmpdir, tools)
            with patch("orchestrator.tool_selector.load_settings", return_value=_settings(tmpdir)):
                registry = ToolRegistry(
                    registry_path=registry_path,
                    embodiment_manager=_FakeEmbodimentManager(capabilities=capabilities),
                )

        active_ids = [item["tool_id"] for item in registry.list_active()]
        self.assertEqual(active_ids, ["embodiment.phone_tap", "embodiment.robot_get_state"])

    def test_initial_profile_prefers_legacy_ui_surface_over_embodiment_duplicates(self) -> None:
        tools = [
            {"tool_id": "embodiment.describe_host", "name": "Host", "description": "Host"},
            {"tool_id": "embodiment.screen_capture", "name": "Emb Screen", "description": "Screen"},
            {"tool_id": "ui.screenshot", "name": "UI Screen", "description": "Screen"},
            {"tool_id": "ui.drag", "name": "UI Drag", "description": "Drag"},
        ]
        capabilities = [
            {
                "capability_id": "screen.capture",
                "status": "available",
                "tool_ids": ["embodiment.screen_capture", "ui.screenshot"],
            },
            {
                "capability_id": "pointer.drag",
                "status": "available",
                "tool_ids": ["ui.drag"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = self._write_registry(tmpdir, tools)
            settings = _settings(tmpdir, {"planner_tool_profile": "initial"})
            with patch("orchestrator.tool_selector.load_settings", return_value=settings):
                registry = ToolRegistry(
                    registry_path=registry_path,
                    embodiment_manager=_FakeEmbodimentManager(capabilities=capabilities),
                )

        active_ids = [item["tool_id"] for item in registry.list_active()]
        self.assertIn("embodiment.describe_host", active_ids)
        self.assertIn("ui.screenshot", active_ids)
        self.assertIn("ui.drag", active_ids)
        self.assertNotIn("embodiment.screen_capture", active_ids)

    def test_initial_profile_shows_application_tools_when_launch_capabilities_are_available(self) -> None:
        tools = [
            {"tool_id": "ui.list_applications", "name": "List Apps", "description": "List apps"},
            {"tool_id": "ui.open_application", "name": "Open App", "description": "Open app"},
        ]
        capabilities = [
            {
                "capability_id": "app.catalog",
                "status": "available",
                "tool_ids": ["ui.list_applications"],
            },
            {
                "capability_id": "app.launch",
                "status": "available",
                "tool_ids": ["ui.open_application"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = self._write_registry(tmpdir, tools)
            settings = _settings(tmpdir, {"planner_tool_profile": "initial"})
            with patch("orchestrator.tool_selector.load_settings", return_value=settings):
                registry = ToolRegistry(
                    registry_path=registry_path,
                    embodiment_manager=_FakeEmbodimentManager(capabilities=capabilities),
                )

        active_ids = [item["tool_id"] for item in registry.list_active()]
        self.assertEqual(active_ids, ["ui.list_applications", "ui.open_application"])

    def test_generated_embodiment_tool_activates_from_implemented_capability_metadata(self) -> None:
        tools = [
            {
                "tool_id": "embodiment.generated.phone_tap",
                "name": "Generated Phone Tap",
                "description": "Generated phone tap.",
                "capabilities": ["write", "embodiment", "phone", "ui"],
                "implements_capability_id": "phone.pointer.tap",
                "implements_abstract_capability": "pointer.click",
            }
        ]
        capabilities = [
            {
                "capability_id": "phone.pointer.tap",
                "abstract_capability": "pointer.click",
                "status": "configured",
                "tool_ids": ["embodiment.phone_tap"],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = self._write_registry(tmpdir, tools)
            with patch("orchestrator.tool_selector.load_settings", return_value=_settings(tmpdir)):
                registry = ToolRegistry(
                    registry_path=registry_path,
                    embodiment_manager=_FakeEmbodimentManager(capabilities=capabilities),
                )

        active_ids = [item["tool_id"] for item in registry.list_active()]
        self.assertEqual(active_ids, ["embodiment.generated.phone_tap"])


if __name__ == "__main__":
    unittest.main()
