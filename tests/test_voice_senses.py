from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_daemon.senses import TextSense


class _FakeLineSource:
    def __init__(self, lines: List[str]):
        self._lines = list(lines)

    async def read_line(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)


class _FakeContext:
    def __init__(self) -> None:
        self.last_turn_id = None
        self.turns: List[str] = []
        self.wakes: List[dict] = []
        self.patches: List[Tuple[str, str]] = []

    async def submit_turn(self, text: str) -> str:
        self.turns.append(text)
        self.last_turn_id = f"turn-{len(self.turns)}"
        return self.last_turn_id

    async def send_wake(self, payload=None) -> None:
        self.wakes.append(dict(payload or {}))

    async def patch_turn(self, turn_id: str, appended_text: str) -> None:
        self.patches.append((turn_id, appended_text))


class _FakeDelegate:
    def __init__(self) -> None:
        self.approval_lines: List[str] = []

    def wake_parts(self, text: str):
        if text.lower().startswith("atlas "):
            remainder = text.split(" ", 1)[1]
            return True, remainder, remainder
        if text.lower() == "atlas":
            return True, "", text
        return False, "", text

    async def handle_console_approval(self, line: str, context) -> bool:
        if line.startswith("/approve") or line.startswith("/deny"):
            self.approval_lines.append(line)
            return True
        return False


class TextSenseTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_sense_routes_turns_wake_patch_and_approvals(self) -> None:
        context = _FakeContext()
        delegate = _FakeDelegate()
        sense = TextSense(
            "Atlas",
            ["atlas"],
            delegate,
            line_source=_FakeLineSource(
                [
                    "/approve 123\n",
                    "atlas\n",
                    "atlas hello there\n",
                    "+ extra\n",
                    "plain text\n",
                    "",
                ]
            ),
        )

        await sense.run(context)

        self.assertEqual(delegate.approval_lines, ["/approve 123"])
        self.assertEqual(context.wakes, [{"wake_word": "atlas"}, {"wake_word": "atlas hello there"}])
        self.assertEqual(context.turns, ["hello there", "plain text"])
        self.assertEqual(context.patches, [("turn-1", "extra")])


if __name__ == "__main__":
    unittest.main()
