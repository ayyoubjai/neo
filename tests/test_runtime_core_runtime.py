from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from runtime_core.runtime import LocalRuntime, RuntimeContext, Sense


class _FakeSession:
    def __init__(self) -> None:
        self.events: List[str] = []
        self.read_started = asyncio.Event()
        self.release_read = asyncio.Event()

    async def connect(self) -> None:
        self.events.append("connect")

    async def read_events(self) -> None:
        self.events.append("read_start")
        self.read_started.set()
        await self.release_read.wait()
        self.events.append("read_done")

    async def close(self) -> None:
        self.events.append("close")


class _FakeSense(Sense):
    name = "fake"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.completed = asyncio.Event()
        self.context: RuntimeContext | None = None

    async def run(self, context: RuntimeContext) -> None:
        self.context = context
        self.started.set()
        self.completed.set()


class LocalRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_session_and_sense_and_closes_session(self) -> None:
        session = _FakeSession()
        sense = _FakeSense()
        runtime = LocalRuntime(session, [sense])

        await runtime.run()

        self.assertEqual(session.events[0], "connect")
        self.assertIn("read_start", session.events)
        self.assertEqual(session.events[-1], "close")
        self.assertTrue(sense.started.is_set())
        self.assertTrue(sense.completed.is_set())
        self.assertIsNotNone(sense.context)


if __name__ == "__main__":
    unittest.main()
