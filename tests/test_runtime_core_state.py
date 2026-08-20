from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from runtime_core.runtime import RuntimeContext, RuntimeState


class _FakeSession:
    last_turn_id = None
    pending_permissions = {}

    def resolve_permission_id(self, token=None):
        raise AssertionError("not used")


class RuntimeStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_context_exposes_shared_state_operations(self) -> None:
        context = RuntimeContext(_FakeSession(), RuntimeState())

        await context.set_state("vision.scene", {"summary": "laptop"})
        self.assertEqual(await context.get_state("vision.scene"), {"summary": "laptop"})

        snapshot = await context.state_snapshot()
        self.assertEqual(snapshot["vision.scene"]["summary"], "laptop")

        await context.delete_state("vision.scene")
        self.assertIsNone(await context.get_state("vision.scene"))


if __name__ == "__main__":
    unittest.main()
