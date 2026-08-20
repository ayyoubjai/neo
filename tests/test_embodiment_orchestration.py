from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestrator.main import Orchestrator


class _FakeTools:
    def __init__(self) -> None:
        self._tools = {
            "embodiment.phone_screen_capture": {
                "tool_id": "embodiment.phone_screen_capture",
                "required_permissions": [],
                "capabilities": ["embodiment", "phone"],
            }
        }

    def add_tool(self, tool_id: str) -> None:
        self._tools[tool_id] = {
            "tool_id": tool_id,
            "required_permissions": [],
            "capabilities": ["embodiment", "phone"],
        }

    def get_tool(self, tool_id: str):
        return self._tools.get(tool_id)

    def list_active(self):
        return list(self._tools.values())


class EmbodimentOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_tool_action_auto_creates_and_uses_embodiment_fallback(self) -> None:
        orchestrator = Orchestrator()
        fake_tools = _FakeTools()
        orchestrator._tools = fake_tools

        async def _fake_call_tool(tool_id, args, approval_token, trace_id, required_artifacts):
            if tool_id == "embodiment.list_capabilities":
                return {
                    "status": "APPROVED",
                    "result": {
                        "capabilities": [
                            {
                                "capability_id": "phone.screen.capture",
                                "abstract_capability": "screen.capture",
                                "status": "configured",
                                "tool_ids": ["embodiment.phone_screen_capture"],
                            }
                        ]
                    },
                }
            if tool_id == "embodiment.list_fallbacks":
                return {
                    "status": "APPROVED",
                    "result": {
                        "fallbacks": [
                            {
                                "capability_id": "phone.screen.capture",
                                "abstract_capability": "screen.capture",
                                "tool_spec": {
                                    "tool_id": "embodiment.generated.phone_screen_capture",
                                    "description": "generated phone screenshot",
                                    "capabilities": ["embodiment", "phone"],
                                },
                            }
                        ]
                    },
                }
            if tool_id == "embodiment.generated.phone_screen_capture":
                return {
                    "status": "APPROVED",
                    "result": {"image_ref": "workspace:/phone/generated.png"},
                    "error": "",
                }
            return {"status": "ERROR", "error": f"Unexpected tool call: {tool_id}"}

        create_calls = []

        async def _fake_handle_create_tool(spec, trace_id):
            create_calls.append((dict(spec), trace_id))
            fake_tools.add_tool(spec["tool_id"])
            return {"status": "APPROVED", "result": {"tool_id": spec["tool_id"]}}

        orchestrator._call_tool = _fake_call_tool  # type: ignore[method-assign]
        orchestrator._handle_create_tool = _fake_handle_create_tool  # type: ignore[method-assign]

        result = await orchestrator._execute_tool_action(
            "embodiment.phone_screen_capture",
            {"path": "workspace:/phone/requested.png"},
            "trace-1",
            allow_embodiment_fallback=True,
        )

        self.assertEqual(len(create_calls), 1)
        self.assertEqual(create_calls[0][0]["tool_id"], "embodiment.generated.phone_screen_capture")
        self.assertEqual(result["action"]["tool_id"], "embodiment.generated.phone_screen_capture")
        self.assertEqual(result["action"]["requested_tool_id"], "embodiment.phone_screen_capture")
        self.assertEqual(result["action"]["auto_fallback"]["capability_id"], "phone.screen.capture")
        self.assertEqual(result["observation"]["status"], "APPROVED")
        self.assertEqual(result["observation"]["result"]["image_ref"], "workspace:/phone/generated.png")


if __name__ == "__main__":
    unittest.main()
