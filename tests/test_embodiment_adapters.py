from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodiment.adapters.phone import AdbPhoneAdapter
from embodiment.adapters.desktop import DesktopAdapter
from embodiment.adapters.robot import RosCliRobotAdapter


class _Runner:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), dict(kwargs)))
        key = tuple(cmd)
        response = self._responses[key]
        return response() if callable(response) else response


class EmbodimentAdapterTests(unittest.TestCase):
    def test_desktop_adapter_normalizes_windows_launcher_path_under_wsl(self) -> None:
        launched = []

        def _launcher(cmd, **kwargs):
            launched.append((list(cmd), dict(kwargs)))
            return SimpleNamespace(pid=4242)

        with patch("embodiment.adapters.desktop.sys.platform", "linux"), patch(
            "embodiment.adapters.desktop.platform.release",
            return_value="6.6.87.2-microsoft-standard-WSL2",
        ):
            adapter = DesktopAdapter(launcher=_launcher)
            result = adapter.launch_application(["C:\\Windows\\System32\\notepad.exe"])

        self.assertEqual(result["pid"], 4242)
        self.assertEqual(result["command"][0], "/mnt/c/Windows/System32/notepad.exe")
        self.assertEqual(launched[0][0][0], "/mnt/c/Windows/System32/notepad.exe")

    def test_adb_phone_adapter_describes_device_and_captures_screen(self) -> None:
        png = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x048"
            b"\x00\x00\t`"
            b"\x08\x02\x00\x00\x00"
            b"\x00\x00\x00\x00"
        )
        runner = _Runner(
            {
                ("adb", "-s", "device-1", "shell", "getprop", "ro.product.model"): SimpleNamespace(stdout="Pixel 8\n", stderr=""),
                ("adb", "-s", "device-1", "shell", "getprop", "ro.build.version.release"): SimpleNamespace(stdout="14\n", stderr=""),
                ("adb", "-s", "device-1", "shell", "getprop", "ro.build.version.sdk"): SimpleNamespace(stdout="34\n", stderr=""),
                ("adb", "-s", "device-1", "shell", "dumpsys", "window", "windows"): SimpleNamespace(
                    stdout="mCurrentFocus=Window{42 u0 com.android.settings/.Settings}\n",
                    stderr="",
                ),
                ("adb", "-s", "device-1", "exec-out", "screencap", "-p"): SimpleNamespace(stdout=png, stderr=b""),
            }
        )
        adapter = AdbPhoneAdapter(serial="device-1", adb_bin="adb", runner=runner)

        desc = adapter.describe_device()
        with tempfile.TemporaryDirectory() as tmp:
            screenshot = Path(tmp) / "phone.png"
            capture = adapter.capture_screen(str(screenshot))
            self.assertTrue(screenshot.exists())

        self.assertEqual(desc["model"], "Pixel 8")
        self.assertEqual(desc["android_version"], "14")
        self.assertEqual(desc["current_app"], "com.android.settings/.Settings")
        self.assertEqual(capture["width"], 1080)
        self.assertEqual(capture["height"], 2400)

    def test_ros_robot_adapter_publishes_joint_command_via_ros2_cli(self) -> None:
        runner = _Runner(
            {
                (
                    "ros2",
                    "topic",
                    "pub",
                    "--once",
                    "/embodiment/command",
                    "std_msgs/msg/String",
                    '{data: "{\\"op\\":\\"move_joint\\",\\"joint\\":\\"shoulder_pan\\",\\"position\\":1.25,\\"velocity\\":0.5}"}',
                ): SimpleNamespace(stdout="", stderr=""),
            }
        )
        adapter = RosCliRobotAdapter(
            ros_version="2",
            ros2_bin="ros2",
            rostopic_bin=None,
            runner=runner,
        )

        result = adapter.move_joint("shoulder_pan", 1.25, velocity=0.5)

        self.assertEqual(result["queued"], True)
        self.assertEqual(result["joint"], "shoulder_pan")
        self.assertEqual(len(runner.calls), 1)


if __name__ == "__main__":
    unittest.main()
