from __future__ import annotations

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
from embodiment.adapters.desktop import DesktopAdapterError
from embodiment.manager import EmbodimentManager


def _settings(tool_overrides=None) -> Settings:
    tool = {"active_tool_limit": 100}
    if isinstance(tool_overrides, dict):
        tool.update(tool_overrides)
    return Settings(
        workspace_root=str(REPO_ROOT),
        data_dir=str(REPO_ROOT / "data"),
        rpc={},
        interface={"mode": "local", "senses": ["audio", "vision"]},
        voice={"input_device": "USB Mic", "tts_enabled": True, "sound_output_device": "Speakers"},
        vision={"camera_index": 1, "preview_enabled": True, "track_enabled": True, "model_path": "yolov8n.pt"},
        video={},
        telegram={},
        google={},
        tool=tool,
        search={"provider": "searxng", "searxng_url": "http://localhost:8080"},
        orchestrator={},
        evolve={"candidate_dir": str(REPO_ROOT / "evolve" / "candidates")},
        models={"ui_grounding_model": "qwen3-vl:8b"},
    )


class EmbodimentManagerTests(unittest.TestCase):
    def test_describe_host_reports_devices_and_capabilities(self) -> None:
        manager = EmbodimentManager(settings=_settings())
        available_modules = {"pyautogui", "pygetwindow", "sounddevice", "cv2", "ultralytics", "pyttsx3"}

        with patch.object(manager, "_is_headless", return_value=False), patch.object(
            manager, "_module_available", side_effect=lambda name: name in available_modules
        ), patch.object(
            manager,
            "_probe_audio_devices",
            return_value={"input_count": 2, "output_count": 3, "error": None},
        ), patch.object(
            manager, "_probe_screen_size", return_value={"width": 1920, "height": 1080}
        ), patch.object(
            manager,
            "_load_desktop_application_catalog",
            return_value={
                "platform": "wsl",
                "catalog_path": str(REPO_ROOT / "config" / "app_launchers.json"),
                "catalog_exists": True,
                "applications": [
                    {"app_id": "notepad", "name": "Notepad", "description": "", "aliases": [], "command": ["C:\\Windows\\System32\\notepad.exe"], "platform": "wsl"}
                ],
            },
        ), patch.object(
            manager,
            "_detect_ros_controller",
            return_value={"detected": True, "transport": "ros", "ros_version": "2"},
        ):
            payload = manager.describe_host()

        self.assertEqual(payload["host"]["interface_mode"], "local")
        self.assertFalse(payload["host"]["headless"])
        device_ids = {item["device_id"] for item in payload["devices"]}
        self.assertIn("display.main", device_ids)
        self.assertIn("camera.primary", device_ids)
        self.assertIn("robot.controller.1", device_ids)

        capabilities = {item["capability_id"]: item for item in payload["capabilities"]}
        self.assertEqual(capabilities["screen.capture"]["status"], "available")
        self.assertIn("embodiment.screen_capture", capabilities["screen.capture"]["tool_ids"])
        self.assertIn("ui.screenshot", capabilities["screen.capture"]["tool_ids"])
        self.assertIn("embodiment.pointer_click", capabilities["pointer.click"]["tool_ids"])
        self.assertIn("embodiment.keyboard_type", capabilities["keyboard.type"]["tool_ids"])
        self.assertIn("embodiment.window_focus", capabilities["window.focus"]["tool_ids"])
        self.assertEqual(capabilities["app.catalog"]["status"], "available")
        self.assertIn("ui.list_applications", capabilities["app.catalog"]["tool_ids"])
        self.assertEqual(capabilities["app.launch"]["status"], "available")
        self.assertIn("ui.open_application", capabilities["app.launch"]["tool_ids"])
        self.assertIn("embodiment.camera_capture", capabilities["camera.capture"]["tool_ids"])
        self.assertEqual(capabilities["audio.listen"]["status"], "available")
        self.assertEqual(capabilities["vision.detect"]["status"], "available")
        self.assertEqual(capabilities["robot.control"]["status"], "available")
        self.assertGreaterEqual(payload["summary"]["available_capability_count"], 1)

    def test_get_state_reports_lightweight_runtime_state(self) -> None:
        manager = EmbodimentManager(settings=_settings())

        with patch.object(manager, "_is_headless", return_value=False), patch.object(
            manager, "_module_available", return_value=False
        ), patch.object(
            manager,
            "_probe_audio_devices",
            return_value={"input_count": 1, "output_count": 1, "error": None},
        ), patch.object(
            manager, "_probe_screen_size", return_value={"width": 1366, "height": 768}
        ), patch.object(
            manager, "_probe_active_window", return_value="Editor"
        ), patch.object(
            manager,
            "_load_desktop_application_catalog",
            return_value={
                "platform": "wsl",
                "catalog_path": str(REPO_ROOT / "config" / "app_launchers.json"),
                "catalog_exists": True,
                "applications": [
                    {"app_id": "notepad", "name": "Notepad", "description": "", "aliases": [], "command": ["C:\\Windows\\System32\\notepad.exe"], "platform": "wsl"}
                ],
            },
        ), patch.object(
            manager,
            "_probe_battery",
            return_value={"available": True, "percent": 87, "power_plugged": False, "secsleft": 3600},
        ), patch.object(
            manager, "_detect_ros_controller", return_value={"detected": False}
        ):
            state = manager.get_state()

        self.assertEqual(state["screen"]["width"], 1366)
        self.assertEqual(state["screen"]["active_window_title"], "Editor")
        self.assertEqual(state["audio"]["input_device_count"], 1)
        self.assertEqual(state["power"]["percent"], 87)
        self.assertEqual(state["robot"]["detected"], False)
        self.assertGreaterEqual(state["applications"]["count"], 1)
        self.assertIn("capability_summary", state)

    def test_describe_host_includes_phone_capabilities_and_fallbacks_when_phone_is_configured(self) -> None:
        manager = EmbodimentManager(settings=_settings())

        with patch.object(manager, "_is_headless", return_value=False), patch.object(
            manager, "_module_available", return_value=False
        ), patch.object(
            manager,
            "_probe_audio_devices",
            return_value={"input_count": 0, "output_count": 0, "error": None},
        ), patch.object(
            manager, "_probe_screen_size", return_value={"width": None, "height": None}
        ), patch.object(
            manager, "_detect_phone_device", return_value={"detected": True, "transport": "local", "adapter_available": False}
        ), patch.object(
            manager,
            "_load_desktop_application_catalog",
            return_value={"platform": "wsl", "catalog_path": "/tmp/apps.json", "catalog_exists": False, "applications": []},
        ), patch.object(
            manager, "_detect_ros_controller", return_value={"detected": False}
        ), patch.object(
            manager, "_host_class", return_value="phone"
        ):
            payload = manager.describe_host()

        capabilities = {item["capability_id"]: item for item in payload["capabilities"]}
        self.assertEqual(capabilities["phone.screen.capture"]["status"], "configured")
        self.assertEqual(capabilities["phone.screen.capture"]["abstract_capability"], "screen.capture")
        self.assertIn("embodiment.phone_screen_capture", capabilities["phone.screen.capture"]["tool_ids"])
        self.assertGreaterEqual(payload["summary"]["fallback_count"], 1)
        self.assertTrue(
            any(
                item["tool_spec"]["tool_id"] == "embodiment.generated.phone_screen_capture"
                for item in payload["fallbacks"]
            )
        )

    def test_list_and_launch_applications_respect_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "app_launchers.json"
            catalog_path.write_text(
                '{"applications":[{"app_id":"notepad","name":"Notepad","description":"Text editor","aliases":["text_editor"],"commands":{"wsl":["C:\\\\Windows\\\\System32\\\\notepad.exe"]}}]}',
                encoding="utf-8",
            )
            settings = _settings({"application_catalog_path": str(catalog_path)})
            manager = EmbodimentManager(settings=settings)

            with patch("embodiment.manager.DesktopAdapter.runtime_platform", return_value="wsl"):
                apps = manager.list_launchable_applications()
                self.assertEqual(apps["count"], 1)
                self.assertEqual(apps["applications"][0]["app_id"], "notepad")

                with patch("embodiment.manager.DesktopAdapter.launch_supported", return_value=True), patch(
                    "embodiment.manager.DesktopAdapter.launch_application",
                    return_value={"launched": True, "pid": 4242, "command": ["/mnt/c/Windows/System32/notepad.exe"]},
                ):
                    launched = manager.launch_desktop_application("text_editor")

            self.assertEqual(launched["app_id"], "notepad")
            self.assertEqual(launched["pid"], 4242)

    def test_launch_desktop_application_rejects_unknown_app_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "app_launchers.json"
            catalog_path.write_text('{"applications":[]}', encoding="utf-8")
            settings = _settings({"application_catalog_path": str(catalog_path)})
            manager = EmbodimentManager(settings=settings)

            with patch("embodiment.manager.DesktopAdapter.runtime_platform", return_value="wsl"):
                with self.assertRaises(DesktopAdapterError):
                    manager.launch_desktop_application("unknown")


if __name__ == "__main__":
    unittest.main()
