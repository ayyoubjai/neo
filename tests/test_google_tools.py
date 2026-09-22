from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestrator.tool_selector import ToolRegistry
from integrations.google_workspace import GoogleWorkspaceManager
from tool_runtime.tools import (
    google_authorize,
    google_gmail_send_message,
    google_status,
    tool_registry,
)


class GoogleToolTests(unittest.TestCase):
    def test_oauth_rejects_partial_grants_and_uses_pkce(self) -> None:
        from integrations.google_workspace import GoogleAuthError
        with tempfile.TemporaryDirectory() as tmp:
            client = Path(tmp) / "client.json"
            client.write_text(json.dumps({"installed": {
                "client_id": "test", "client_secret": "test",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"}}))
            manager = GoogleWorkspaceManager()
            manager._cfg = {"enabled": True, "client_secrets_path": str(client)}
            flow_cls = MagicMock()
            creds = flow_cls.from_client_secrets_file.return_value.run_local_server.return_value
            creds.granted_scopes = ["unrelated"]
            creds.has_scopes.return_value = True
            with patch.object(manager, "_google_auth_modules", return_value=(None, None, flow_cls)), patch.object(manager, "_store_token_json") as store:
                with self.assertRaises(GoogleAuthError):
                    manager._authorize_interactive("default", "gmail_send", ["required"])
                store.assert_not_called()
                self.assertTrue(flow_cls.from_client_secrets_file.call_args.kwargs["autogenerate_code_verifier"])
                creds.granted_scopes = ["required"]
                manager._authorize_interactive("default", "gmail_send", ["required"])
                store.assert_called_once()
                manager._cfg["oauth_bind_host"] = "0.0.0.0"
                with self.assertRaises(GoogleAuthError):
                    manager._authorize_interactive("default", "gmail_send", ["required"])

    def test_disabled_google_blocks_credentials_and_authorization(self) -> None:
        manager = GoogleWorkspaceManager()
        manager._cfg = {"enabled": False}
        with patch.object(manager, "_load_keyring") as keyring:
            from integrations.google_workspace import GoogleAuthError
            with self.assertRaises(GoogleAuthError):
                manager._load_credentials("default", "gmail_send")
            with self.assertRaises(GoogleAuthError):
                manager.authorize("gmail_send")
            self.assertFalse(manager.status()["enabled"])
            keyring.assert_not_called()

    def test_failed_reconsent_preserves_existing_credentials(self) -> None:
        manager = GoogleWorkspaceManager()
        manager._cfg = {"enabled": True}
        with patch.object(manager, "_authorize_interactive", side_effect=RuntimeError("cancelled")), patch.object(manager, "_delete_token_json") as delete:
            with self.assertRaises(RuntimeError):
                manager.authorize("gmail_send", force_reconsent=True)
            delete.assert_not_called()

    def test_drafts_use_compose_scope_without_broadening_send_scope(self) -> None:
        manager = GoogleWorkspaceManager()
        self.assertEqual(manager._scopes_for_bundle("gmail_send"), ("https://www.googleapis.com/auth/gmail.send",))
        self.assertEqual(manager._scopes_for_bundle("gmail_compose"), ("https://www.googleapis.com/auth/gmail.compose",))
        with patch.object(manager, "_build_service") as build:
            service = build.return_value
            service.users.return_value.drafts.return_value.create.return_value.execute.return_value = {
                "id": "draft-1", "message": {"id": "message-1"}
            }
            result = manager.gmail_create_draft("Subject", ["recipient@example.com"], "Body")
            build.assert_called_once_with(None, "gmail_compose", "gmail", "v1")
            self.assertEqual(result["draft_id"], "draft-1")
            service.users.return_value.messages.assert_not_called()

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
