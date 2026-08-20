from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestrator.tool_selector import ToolRegistry
from tool_runtime.tools import (
    google_authorize,
    google_gmail_send_message,
    google_status,
    tool_registry,
)


class GoogleToolTests(unittest.TestCase):
    def test_runtime_tool_registry_contains_google_tools(self) -> None:
        registry = tool_registry()
        self.assertIn("google.status", registry)
        self.assertIn("google.authorize", registry)
        self.assertIn("google.gmail.send_message", registry)
        self.assertIn("google.calendar.create_event", registry)
        self.assertIn("google.docs.get_document", registry)
        self.assertEqual(registry["google.status"]["tier"], 0)
        self.assertEqual(registry["google.gmail.send_message"]["tier"], 2)

    def test_metadata_registry_contains_google_redaction_policy(self) -> None:
        registry = ToolRegistry()
        tool = registry.get_tool("google.gmail.send_message")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.get("required_permissions"), ["tier2"])
        self.assertEqual(
            tool.get("log_policy"),
            {"redact_args": True, "redact_result": True, "redact_io": True},
        )

    @patch("tool_runtime.tools.GoogleWorkspaceManager")
    def test_google_status_tool_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.status.return_value = {
            "configured": True,
            "account_name": "default",
            "bundles": [],
        }
        result, io = google_status({"account_name": "default"}, str(REPO_ROOT))
        self.assertTrue(result["configured"])
        self.assertEqual(io, {})
        manager_cls.return_value.status.assert_called_once_with(account_name="default")

    @patch("tool_runtime.tools.GoogleWorkspaceManager")
    def test_google_authorize_tool_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.authorize.return_value = {
            "authorized": True,
            "account_name": "default",
            "bundle": "gmail_send",
            "scopes": ["scope:a"],
        }
        result, io = google_authorize(
            {"bundle": "gmail_send", "account_name": "default", "force_reconsent": True},
            str(REPO_ROOT),
        )
        self.assertTrue(result["authorized"])
        self.assertEqual(io, {})
        manager_cls.return_value.authorize.assert_called_once_with(
            bundle="gmail_send",
            account_name="default",
            force_reconsent=True,
        )

    @patch("tool_runtime.tools.GoogleWorkspaceManager")
    def test_google_gmail_send_message_tool_delegates_to_manager(self, manager_cls) -> None:
        manager_cls.return_value.gmail_send_message.return_value = {
            "message_id": "msg-1",
            "thread_id": "thr-1",
            "label_ids": ["SENT"],
        }
        result, io = google_gmail_send_message(
            {
                "account_name": "default",
                "to": ["a@example.com"],
                "subject": "Hello",
                "body_text": "Test body",
            },
            str(REPO_ROOT),
        )
        self.assertEqual(result["message_id"], "msg-1")
        self.assertEqual(io, {})
        manager_cls.return_value.gmail_send_message.assert_called_once_with(
            subject="Hello",
            to=["a@example.com"],
            body_text="Test body",
            account_name="default",
            cc=None,
            bcc=None,
            body_html="",
            reply_to_message_id="",
        )


if __name__ == "__main__":
    unittest.main()
