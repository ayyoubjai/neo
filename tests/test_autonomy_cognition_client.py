from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autonomy.cognition_client import CognitionClient
from common.config import Settings


class _FakeSession:
    responses = []
    instances = []

    def __init__(
        self,
        host: str,
        port: int,
        *,
        source_id=None,
        on_assistant_final=None,
        on_assistant_draft=None,
        on_permission_request=None,
    ) -> None:
        self.host = host
        self.port = port
        self.source_id = source_id
        self._on_assistant_final = on_assistant_final
        self._on_permission_request = on_permission_request
        self.closed = False
        self.submissions = []
        self.decisions = []
        self.payload = dict(self.responses.pop(0))
        self.instances.append(self)

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    async def submit_turn(self, text: str, *, requested_mode=None, turn_id=None, skip_distillation=False, autonomy_constraints=None) -> str:
        self.constraints = autonomy_constraints
        self.skip_distillation = skip_distillation
        self.submissions.append((text, requested_mode))
        return "turn-1"

    async def read_events(self) -> None:
        await asyncio.sleep(0)
        permission = self.payload.get("permission")
        if permission and self._on_permission_request is not None:
            await self._on_permission_request(permission)
        await asyncio.sleep(0)
        if self._on_assistant_final is not None:
            await self._on_assistant_final({"text": self.payload["text"], "turn_id": "turn-1", "tool_evidence": self.payload.get("tool_evidence", [])})

    async def send_permission_decision(self, request_id: str, approved: bool) -> bool:
        self.decisions.append((request_id, approved))
        return True


def _settings() -> Settings:
    return Settings(
        workspace_root=str(REPO_ROOT),
        data_dir=str(REPO_ROOT / "data"),
        rpc={"orch_host": "127.0.0.1", "orch_port": 50051},
        interface={},
        voice={},
        vision={},
        video={},
        telegram={},
        google={},
        tool={"active_tool_limit": 100},
        search={},
        orchestrator={},
        evolve={},
        models={"text_model": "test-model"},
        autonomy={},
    )


class CognitionClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _FakeSession.responses = []
        _FakeSession.instances = []

    async def test_generate_json_submits_cognition_turn_and_parses_payload(self) -> None:
        _FakeSession.responses = [{"text": "{\"match\": true, \"reasoning\": \"ok\"}"}]

        with patch("autonomy.cognition_client.OrchestratorSession", _FakeSession):
            client = CognitionClient(settings=_settings())
            payload = await client.generate_json("Return a match result.")

        self.assertTrue(payload["match"])
        self.assertEqual(payload["reasoning"], "ok")
        self.assertEqual(_FakeSession.instances[0].submissions[0][1], "COGNITION")
        self.assertTrue(_FakeSession.instances[0].closed)

    async def test_permission_policy_deny_rejects_permission_requests(self) -> None:
        _FakeSession.responses = [
            {
                "permission": {"request_id": "req-1", "tool_id": "shell.exec"},
                "text": "{\"goal_text\": \"fallback\", \"reasoning\": \"denied\"}",
            }
        ]

        with patch("autonomy.cognition_client.OrchestratorSession", _FakeSession):
            client = CognitionClient(settings=_settings(), permission_policy="deny")
            payload = await client.generate_json("Return a goal.")

        self.assertEqual(payload["goal_text"], "fallback")
        self.assertEqual(_FakeSession.instances[0].decisions, [("req-1", False)])

    async def test_requested_mode_can_force_system1(self) -> None:
        _FakeSession.responses = [{"text": "{\"match\": true, \"reasoning\": \"ok\"}"}]

        with patch("autonomy.cognition_client.OrchestratorSession", _FakeSession):
            client = CognitionClient(settings=_settings(), requested_mode="system1")
            await client.generate_json("Return a match result.")

        self.assertEqual(_FakeSession.instances[0].submissions[0][1], "SYSTEM1")

    async def test_wait_until_available_reports_unreachable_orchestrator(self) -> None:
        async def _fail_connect(host, port):
            raise ConnectionRefusedError("refused")

        with patch("asyncio.open_connection", _fail_connect):
            client = CognitionClient(settings=_settings())
            with self.assertRaisesRegex(Exception, "COGNITION orchestrator is not reachable"):
                await client.wait_until_available(timeout_s=0.01, interval_s=0.01)


if __name__ == "__main__":
    unittest.main()
