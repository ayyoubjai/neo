from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tool_runtime.tools import (
    embodiment_camera_capture,
    embodiment_keyboard_type,
    embodiment_pointer_click,
    embodiment_phone_screen_capture,
    embodiment_phone_tap,
    embodiment_robot_move_joint,
    embodiment_screen_capture,
    embodiment_window_focus,
)


class EmbodimentActionToolTests(unittest.TestCase):
    @patch("tool_runtime.tools.time.time", return_value=1710000000.123)
    @patch("tool_runtime.tools.DesktopAdapter")
    def test_screen_capture_uses_desktop_adapter_and_default_path(self, adapter_cls, _time_mock) -> None:
        adapter_cls.return_value.capture_screen.return_value = {"width": 1280, "height": 720}

        result, io = embodiment_screen_capture({}, str(REPO_ROOT))

        self.assertEqual(
            result,
            {
                "image_ref": "workspace:/screenshots/screenshot_1710000000123.png",
                "width": 1280,
                "height": 720,
            },
        )
        self.assertEqual(io, {})
        adapter_cls.return_value.capture_screen.assert_called_once_with(
            str(REPO_ROOT / "screenshots" / "screenshot_1710000000123.png"),
            region=None,
        )

    @patch("tool_runtime.tools.DesktopAdapter")
    def test_pointer_click_delegates_to_desktop_adapter(self, adapter_cls) -> None:
        adapter_cls.return_value.click.return_value = {"clicked": True, "x": 10, "y": 20}

        result, io = embodiment_pointer_click({"x": 10, "y": 20}, str(REPO_ROOT))

        self.assertEqual(result["clicked"], True)
        self.assertEqual(io, {})
        adapter_cls.return_value.click.assert_called_once_with(
            10,
            20,
            button="left",
            clicks=1,
            interval=0.0,
        )

    @patch("tool_runtime.tools.DesktopAdapter")
    def test_keyboard_type_delegates_to_desktop_adapter(self, adapter_cls) -> None:
        adapter_cls.return_value.type_text.return_value = {"typed": True, "chars": 5}

        result, io = embodiment_keyboard_type({"text": "hello"}, str(REPO_ROOT))

        self.assertEqual(result["typed"], True)
        self.assertEqual(io, {})
        adapter_cls.return_value.type_text.assert_called_once_with("hello", interval=0.0)

    @patch("tool_runtime.tools.DesktopAdapter")
    def test_window_focus_delegates_to_desktop_adapter(self, adapter_cls) -> None:
        adapter_cls.return_value.focus_window.return_value = {"focused": True, "title": "Editor"}

        result, io = embodiment_window_focus({"title": "Edit"}, str(REPO_ROOT))

        self.assertEqual(result["focused"], True)
        self.assertEqual(io, {})
        adapter_cls.return_value.focus_window.assert_called_once_with("Edit")

    @patch("tool_runtime.tools.time.time", return_value=1710000000.5)
    @patch("tool_runtime.tools.OpenCvCameraAdapter")
    def test_camera_capture_uses_camera_adapter_and_returns_workspace_ref(self, adapter_cls, _time_mock) -> None:
        adapter_cls.return_value.capture_frame.return_value = {
            "width": 640,
            "height": 480,
            "camera_index": 2,
        }

        result, io = embodiment_camera_capture(
            {"camera_index": 2, "width": 640, "height": 480},
            str(REPO_ROOT),
        )

        self.assertEqual(
            result,
            {
                "image_ref": "workspace:/camera/frame_1710000000500.jpg",
                "width": 640,
                "height": 480,
                "camera_index": 2,
            },
        )
        self.assertEqual(io, {})
        adapter_cls.assert_called_once_with(camera_index=2)
        adapter_cls.return_value.capture_frame.assert_called_once_with(
            str(REPO_ROOT / "camera" / "frame_1710000000500.jpg"),
            width=640,
            height=480,
        )

    @patch("tool_runtime.tools.time.time", return_value=1710000001.0)
    @patch("tool_runtime.tools.AdbPhoneAdapter")
    def test_phone_screen_capture_uses_phone_adapter(self, adapter_cls, _time_mock) -> None:
        adapter_cls.return_value.capture_screen.return_value = {
            "width": 1080,
            "height": 2400,
            "serial": "device-1",
        }

        result, io = embodiment_phone_screen_capture({"serial": "device-1"}, str(REPO_ROOT))

        self.assertEqual(
            result,
            {
                "image_ref": "workspace:/phone/screenshot_1710000001000.png",
                "width": 1080,
                "height": 2400,
                "serial": "device-1",
            },
        )
        self.assertEqual(io, {})
        adapter_cls.assert_called_once_with(serial="device-1")
        adapter_cls.return_value.capture_screen.assert_called_once_with(
            str(REPO_ROOT / "phone" / "screenshot_1710000001000.png")
        )

    @patch("tool_runtime.tools.AdbPhoneAdapter")
    def test_phone_tap_delegates_to_phone_adapter(self, adapter_cls) -> None:
        adapter_cls.return_value.tap.return_value = {"tapped": True, "x": 10, "y": 20}

        result, io = embodiment_phone_tap({"x": 10, "y": 20, "serial": "device-1"}, str(REPO_ROOT))

        self.assertEqual(result["tapped"], True)
        self.assertEqual(io, {})
        adapter_cls.assert_called_once_with(serial="device-1")
        adapter_cls.return_value.tap.assert_called_once_with(10, 20)

    @patch("tool_runtime.tools.RosCliRobotAdapter")
    def test_robot_move_joint_delegates_to_robot_adapter(self, adapter_cls) -> None:
        adapter_cls.return_value.move_joint.return_value = {"queued": True}

        result, io = embodiment_robot_move_joint(
            {"joint": "shoulder", "position": 1.5, "velocity": 0.2},
            str(REPO_ROOT),
        )

        self.assertEqual(result["queued"], True)
        self.assertEqual(io, {})
        adapter_cls.assert_called_once_with()
        adapter_cls.return_value.move_joint.assert_called_once_with("shoulder", 1.5, velocity=0.2)


if __name__ == "__main__":
    unittest.main()
