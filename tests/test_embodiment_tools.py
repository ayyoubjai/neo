from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestrator.tool_selector import ToolRegistry
from tool_runtime.tools import (
    embodiment_describe_host,
    embodiment_get_state,
    embodiment_list_capabilities,
    embodiment_list_fallbacks,
    tool_registry,
    ui_list_applications,
    ui_open_application,
)


class EmbodimentToolTests(unittest.TestCase):
    def test_runtime_tool_registry_contains_embodiment_tools(self) -> None:
        registry = tool_registry()
        self.assertIn("embodiment.describe_host", registry)
        self.assertIn("embodiment.list_capabilities", registry)
        self.assertIn("embodiment.list_fallbacks", registry)
        self.assertIn("embodiment.get_state", registry)
        self.assertIn("ui.list_applications", registry)
        self.assertIn("ui.open_application", registry)
        self.assertEqual(registry["embodiment.describe_host"]["tier"], 0)

    def test_metadata_registry_contains_embodiment_tools(self) -> None:
        registry = ToolRegistry()
        tool = registry.get_tool("embodiment.describe_host")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.get("required_permissions"), ["tier0"])
        self.assertEqual(registry.get_tool("ui.list_applications")["required_permissions"], ["tier0"])
        self.assertEqual(registry.get_tool("ui.open_application")["required_permissions"], ["tier2"])

    @patch("tool_runtime.tools.EmbodimentManager")
    def test_describe_host_tool_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.describe_host.return_value = {
            "host": {"host_id": "host:test"},
            "devices": [],
            "capabilities": [],
            "summary": {},
        }
        result, io = embodiment_describe_host({}, str(REPO_ROOT))
        self.assertEqual(result["host"]["host_id"], "host:test")
        self.assertEqual(io, {})
        manager_cls.return_value.describe_host.assert_called_once()

    @patch("tool_runtime.tools.EmbodimentManager")
    def test_list_capabilities_tool_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.list_capabilities.return_value = {
            "host_id": "host:test",
            "capabilities": [{"capability_id": "screen.capture"}],
            "summary": {"capability_count": 1},
        }
        result, io = embodiment_list_capabilities({}, str(REPO_ROOT))
        self.assertEqual(result["capabilities"][0]["capability_id"], "screen.capture")
        self.assertEqual(io, {})
        manager_cls.return_value.list_capabilities.assert_called_once()

    @patch("tool_runtime.tools.EmbodimentManager")
    def test_list_fallbacks_tool_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.list_fallbacks.return_value = {
            "host_id": "host:test",
            "fallbacks": [{"capability_id": "phone.screen.capture"}],
            "summary": {"fallback_count": 1},
        }
        result, io = embodiment_list_fallbacks({}, str(REPO_ROOT))
        self.assertEqual(result["fallbacks"][0]["capability_id"], "phone.screen.capture")
        self.assertEqual(io, {})
        manager_cls.return_value.list_fallbacks.assert_called_once()

    @patch("tool_runtime.tools.EmbodimentManager")
    def test_get_state_tool_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.get_state.return_value = {
            "host_id": "host:test",
            "ts": "2026-03-18T00:00:00Z",
            "interface_mode": "local",
            "headless": False,
            "screen": {},
            "audio": {},
            "vision": {},
            "power": {},
            "network": {},
            "robot": {"detected": False},
            "capability_summary": {"capability_count": 0},
        }
        result, io = embodiment_get_state({}, str(REPO_ROOT))
        self.assertEqual(result["interface_mode"], "local")
        self.assertEqual(io, {})
        manager_cls.return_value.get_state.assert_called_once()

    @patch("tool_runtime.tools.EmbodimentManager")
    def test_ui_list_applications_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.list_launchable_applications.return_value = {
            "platform": "wsl",
            "catalog_path": "/tmp/apps.json",
            "catalog_exists": True,
            "count": 1,
            "applications": [{"app_id": "notepad", "name": "Notepad", "description": "", "aliases": []}],
        }
        result, io = ui_list_applications({}, str(REPO_ROOT))
        self.assertEqual(result["applications"][0]["app_id"], "notepad")
        self.assertEqual(io, {})
        manager_cls.return_value.list_launchable_applications.assert_called_once()

    @patch("tool_runtime.tools.EmbodimentManager")
    def test_ui_open_application_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.launch_desktop_application.return_value = {
            "app_id": "notepad",
            "name": "Notepad",
            "platform": "wsl",
            "launched": True,
            "pid": 4242,
            "command": ["/mnt/c/Windows/System32/notepad.exe"],
        }
        result, io = ui_open_application({"app_id": "notepad"}, str(REPO_ROOT))
        self.assertEqual(result["app_id"], "notepad")
        self.assertEqual(io, {})
        manager_cls.return_value.launch_desktop_application.assert_called_once_with("notepad")


if __name__ == "__main__":
    unittest.main()
