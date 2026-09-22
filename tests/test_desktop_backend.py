import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from tool_runtime.desktop import capture_screen, load_input_backend
from tool_runtime.tools import computer_focus_window, ToolError


class DesktopBackendTests(unittest.TestCase):
    def test_failed_window_activation_is_not_reported_as_success(self):
        def run(command, **kwargs):
            if command[:2] == ["xdotool", "search"]:
                return subprocess.CompletedProcess(command, 0, "123\n", "")
            if command[:2] == ["wmctrl", "-l"]:
                return subprocess.CompletedProcess(command, 0, "0x123 0 host Editor\n", "")
            return subprocess.CompletedProcess(command, 1, "", "activation denied")

        with patch("tool_runtime.tools.is_wayland", return_value=False), patch("tool_runtime.tools.os.name", "posix"), patch.dict("sys.modules", {"pygetwindow": None}), patch("tool_runtime.tools._run_subprocess", side_effect=run):
            with self.assertRaisesRegex(ToolError, "activation denied"):
                computer_focus_window({"title": "Editor"}, ".")

    def test_windows_capture_uses_pillow_and_xywh_conversion(self):
        with patch("tool_runtime.desktop.sys.platform", "win32"), patch("PIL.ImageGrab.grab") as grab:
            capture_screen((-10, 20, 100, 50))
            grab.assert_called_once_with(bbox=(-10, 20, 90, 70))

    def test_x11_capture(self):
        with patch("tool_runtime.desktop.is_wayland", return_value=False), patch("PIL.ImageGrab.grab") as grab:
            capture_screen()
            grab.assert_called_once_with(bbox=None)

    def test_wayland_fallback_crop_and_temp_cleanup(self):
        paths = []

        def run(command, **kwargs):
            self.assertEqual(kwargs["timeout"], 20)
            if "spectacle" in command[0]:
                raise subprocess.TimeoutExpired(command, 20)
            path = command[-1]
            paths.append(path)
            Image.new("RGB", (100, 80), "red").save(path)

        with patch("tool_runtime.desktop.is_wayland", return_value=True), patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "KDE"}), patch("tool_runtime.desktop.shutil.which", side_effect=lambda name: "/usr/bin/" + name), patch("tool_runtime.desktop.subprocess.run", side_effect=run), patch("PIL.ImageGrab.grab") as grab:
            image = capture_screen((10, 20, 30, 40))
            self.assertEqual(image.size, (30, 40))
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            image.close()
            grab.assert_not_called()
        self.assertTrue(paths)
        self.assertFalse(Path(paths[0]).exists())

    def test_wayland_missing_capture_backend(self):
        with patch("tool_runtime.desktop.is_wayland", return_value=True), patch("tool_runtime.desktop.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "No compatible capture executable"):
                capture_screen()

    def test_wayland_input_is_not_silently_sent_to_xwayland(self):
        with patch("tool_runtime.desktop.is_wayland", return_value=True), \
                patch("tool_runtime.desktop._wayland_input", None), \
                patch("tool_runtime.wayland_input.WaylandInput") as portal:
            self.assertIs(load_input_backend(), portal.return_value)
            self.assertIs(load_input_backend(), portal.return_value)
            portal.assert_called_once()
