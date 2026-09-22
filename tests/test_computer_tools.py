import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tool_runtime.tools import (
    ToolError,
    computer_click,
    computer_screenshot,
    computer_type,
    tool_registry,
)


class ComputerToolTests(unittest.TestCase):
    def test_computer_tools_are_registered(self) -> None:
        registry = tool_registry()
        expected = {
            "computer.screenshot",
            "computer.click",
            "computer.move",
            "computer.drag",
            "computer.type",
            "computer.hotkey",
            "computer.scroll",
            "computer.focus_window",
            "computer.launch_application",
        }
        self.assertTrue(expected.issubset(registry))
        self.assertEqual(registry["computer.click"]["tier"], 2)

    def test_screenshot_saves_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Mock(width=640, height=480)
            with patch("tool_runtime.tools.capture_screen", return_value=image):
                result, io = computer_screenshot({}, tmp)

            self.assertEqual(result["width"], 640)
            self.assertEqual(result["height"], 480)
            self.assertEqual(io, {})
            image.save.assert_called_once()
            self.assertTrue(str(image.save.call_args.args[0]).startswith(tmp))

    def test_click_and_type_delegate_to_backend(self) -> None:
        backend = Mock()
        with patch("tool_runtime.tools._load_computer_control_backend", return_value=backend):
            click_result, _ = computer_click(
                {"x": 12, "y": 34, "button": "left", "clicks": 2, "interval": 0.1}, "."
            )
            type_result, _ = computer_type({"text": "hello", "interval": 0.02}, ".")

        backend.click.assert_called_once_with(x=12, y=34, clicks=2, interval=0.1, button="left")
        backend.write.assert_called_once_with("hello", interval=0.02)
        self.assertEqual(click_result["clicks"], 2)
        self.assertEqual(type_result["character_count"], 5)

    def test_invalid_click_is_rejected_before_backend_call(self) -> None:
        backend = Mock()
        with patch("tool_runtime.tools._load_computer_control_backend", return_value=backend):
            with self.assertRaises(ToolError):
                computer_click({"x": 1, "y": 2, "clicks": 0}, ".")
        backend.click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
