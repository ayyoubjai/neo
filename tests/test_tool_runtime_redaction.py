from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tool_runtime.main import ToolRuntime


class ToolRuntimeRedactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_sensitive_tool_logs_are_redacted(self) -> None:
        runtime = ToolRuntime(str(REPO_ROOT), "127.0.0.1", 50053, str(REPO_ROOT / "data"))
        runtime._tools = {
            "secret.tool": {
                "fn": lambda args, workspace_root: ({"secret": "result"}, {"trace": "internal"}),
                "tier": 0,
                "log_policy": {
                    "redact_args": True,
                    "redact_result": True,
                    "redact_io": True,
                },
            }
        }

        jsonl_records = []
        events = []

        def capture_jsonl(_path, obj):
            jsonl_records.append(obj)

        def capture_event(event_type, payload):
            events.append((event_type, payload))

        with patch("tool_runtime.main.append_jsonl", capture_jsonl), patch(
            "tool_runtime.main.record_event", capture_event
        ):
            resp = await runtime.Execute({"tool_id": "secret.tool", "args": {"token": "abc123"}, "trace_id": "t1"})

        self.assertEqual(resp["status"], "APPROVED")
        self.assertGreaterEqual(len(jsonl_records), 2)
        self.assertEqual(jsonl_records[0]["args"], {"_redacted": True, "kind": "args"})
        self.assertEqual(jsonl_records[-1]["result"], {"_redacted": True, "kind": "result"})
        self.assertEqual(jsonl_records[-1]["io"], {"_redacted": True, "kind": "io"})

        tool_start = next(payload for event_type, payload in events if event_type == "tool_execute_start")
        tool_done = next(payload for event_type, payload in events if event_type == "tool_execute_result")
        self.assertEqual(tool_start["args"], {"_redacted": True, "kind": "args"})
        self.assertEqual(tool_done["result"], {"_redacted": True, "kind": "result"})

    async def test_non_sensitive_tool_logs_are_preserved(self) -> None:
        runtime = ToolRuntime(str(REPO_ROOT), "127.0.0.1", 50053, str(REPO_ROOT / "data"))
        runtime._tools = {
            "plain.tool": {
                "fn": lambda args, workspace_root: ({"echo": args["value"]}, {}),
                "tier": 0,
            }
        }

        jsonl_records = []
        events = []

        def capture_jsonl(_path, obj):
            jsonl_records.append(obj)

        def capture_event(event_type, payload):
            events.append((event_type, payload))

        with patch("tool_runtime.main.append_jsonl", capture_jsonl), patch(
            "tool_runtime.main.record_event", capture_event
        ):
            resp = await runtime.Execute({"tool_id": "plain.tool", "args": {"value": "hello"}, "trace_id": "t2"})

        self.assertEqual(resp["status"], "APPROVED")
        self.assertEqual(jsonl_records[0]["args"], {"value": "hello"})
        self.assertEqual(jsonl_records[-1]["result"], {"echo": "hello"})

        tool_start = next(payload for event_type, payload in events if event_type == "tool_execute_start")
        tool_done = next(payload for event_type, payload in events if event_type == "tool_execute_result")
        self.assertEqual(tool_start["args"], {"value": "hello"})
        self.assertEqual(tool_done["result"], {"echo": "hello"})


if __name__ == "__main__":
    unittest.main()
