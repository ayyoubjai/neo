from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.types import (
    EVENT_ASSISTANT_DRAFT,
    EVENT_ASSISTANT_FINAL,
    EVENT_PERMISSION_REQUEST,
    EVENT_PERMISSION_RESPONSE,
    EVENT_STT_FINAL,
    EVENT_TURN_PATCH,
    EVENT_WAKE,
    make_event,
)
from runtime_core.session import OrchestratorSession


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class OrchestratorSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_turn_preserves_caller_acceptance_checks(self):
        session = OrchestratorSession("127.0.0.1", 50051)
        session.send_event = AsyncMock()
        checks = [{"id": "value", "tool_id": "math.eval", "path": ["result", "value"], "expected": 42}]
        await session.submit_turn("Calculate", acceptance_checks=checks)
        self.assertEqual(session.send_event.await_args.args[0]["payload"]["acceptance_checks"], checks)

    async def test_reads_events_tracks_permissions_and_sends_decisions(self) -> None:
        drafts: List[Dict[str, Any]] = []
        finals: List[Dict[str, Any]] = []
        permissions: List[Dict[str, Any]] = []
        outbound: List[Dict[str, Any]] = []
        permission_seen = asyncio.Event()
        reader = object()
        writer = _FakeWriter()
        read_jsonl = AsyncMock(
            side_effect=[
                make_event(EVENT_ASSISTANT_DRAFT, {"text": "draft"}),
                make_event(EVENT_ASSISTANT_FINAL, {"text": "final", "turn_id": "turn-1"}),
                make_event(
                    EVENT_PERMISSION_REQUEST,
                    {
                        "request_id": "req-1",
                        "tool_id": "files.write",
                        "justification": "Need to write a file.",
                        "turn_id": "turn-1",
                    },
                ),
                None,
            ]
        )

        async def capture_write(_writer: Any, event: Dict[str, Any]) -> None:
            outbound.append(event)

        async def on_draft(payload: Dict[str, Any]) -> None:
            drafts.append(payload)

        async def on_final(payload: Dict[str, Any]) -> None:
            finals.append(payload)

        async def on_permission(payload: Dict[str, Any]) -> None:
            permissions.append(payload)
            permission_seen.set()

        session = OrchestratorSession(
            "127.0.0.1",
            50051,
            on_assistant_draft=on_draft,
            on_assistant_final=on_final,
            on_permission_request=on_permission,
        )
        with patch("runtime_core.session.asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
            with patch("runtime_core.session.read_jsonl", read_jsonl):
                with patch("runtime_core.session.write_jsonl", capture_write):
                    await session.connect()
                    read_task = asyncio.create_task(session.read_events())

                    await asyncio.wait_for(permission_seen.wait(), timeout=1.0)
                    self.assertEqual(drafts, [{"text": "draft"}])
                    self.assertEqual(finals, [{"text": "final", "turn_id": "turn-1"}])
                    self.assertEqual(len(permissions), 1)
                    self.assertIn("req-1", session.pending_permissions)

                    sent = await session.send_permission_decision("req-1", approved=True)
                    self.assertTrue(sent)
                    await asyncio.wait_for(read_task, timeout=1.0)

                    self.assertEqual(len(outbound), 1)
                    self.assertEqual(outbound[0]["event_type"], EVENT_PERMISSION_RESPONSE)
                    self.assertEqual(outbound[0]["payload"]["request_id"], "req-1")
                    self.assertTrue(outbound[0]["payload"]["approved"])
                    self.assertIn("token", outbound[0]["payload"])
                    self.assertEqual(session.pending_permissions, {})

                    await session.close()
                    self.assertTrue(writer.closed)

    async def test_submits_turns_patches_and_wake_events(self) -> None:
        outbound: List[Dict[str, Any]] = []
        reader = object()
        writer = _FakeWriter()

        async def capture_write(_writer: Any, event: Dict[str, Any]) -> None:
            outbound.append(event)

        session = OrchestratorSession("127.0.0.1", 50051)
        with patch("runtime_core.session.asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
            with patch("runtime_core.session.write_jsonl", capture_write):
                await session.connect()

                turn_id = await session.submit_turn("hello world", requested_mode="cognition")
                await session.patch_turn(turn_id, "plus more")
                await session.send_wake({"ts": 123.0, "wake_word": "atlas"})

                self.assertEqual(session.last_turn_id, turn_id)
                self.assertEqual(outbound[0]["event_type"], EVENT_STT_FINAL)
                self.assertEqual(outbound[0]["payload"]["turn_id"], turn_id)
                self.assertEqual(outbound[0]["payload"]["text"], "hello world")
                self.assertEqual(outbound[0]["payload"]["requested_mode"], "COGNITION")

                self.assertEqual(outbound[1]["event_type"], EVENT_TURN_PATCH)
                self.assertEqual(outbound[1]["payload"], {"turn_id": turn_id, "appended_text": "plus more"})

                self.assertEqual(outbound[2]["event_type"], EVENT_WAKE)
                self.assertEqual(outbound[2]["payload"]["wake_word"], "atlas")
                self.assertEqual(outbound[2]["payload"]["ts"], 123.0)

                await session.close()
                self.assertTrue(writer.closed)

    async def test_submit_turn_can_set_skip_distillation(self) -> None:
        outbound: List[Dict[str, Any]] = []
        reader = object()
        writer = _FakeWriter()

        async def capture_write(_writer: Any, event: Dict[str, Any]) -> None:
            outbound.append(event)

        session = OrchestratorSession("127.0.0.1", 50051)
        with patch("runtime_core.session.asyncio.open_connection", AsyncMock(return_value=(reader, writer))):
            with patch("runtime_core.session.write_jsonl", capture_write):
                await session.connect()
                await session.submit_turn("hello world", requested_mode="system0", skip_distillation=True)
                self.assertTrue(outbound[0]["payload"]["skip_distillation"])
                self.assertEqual(outbound[0]["payload"]["requested_mode"], "SYSTEM0")
                await session.close()

    def test_normalizes_cognition_requested_mode(self) -> None:
        self.assertEqual(OrchestratorSession._normalize_requested_mode("cognition"), "COGNITION")
        self.assertEqual(OrchestratorSession._normalize_requested_mode("system0"), "SYSTEM0")

if __name__ == "__main__":
    unittest.main()
