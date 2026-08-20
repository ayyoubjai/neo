from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tool_runtime.main import DEFAULT_GOOGLE_AUTO_BUNDLES, _auto_google_authorize


class GoogleAutoAuthorizeTests(unittest.TestCase):
    @patch("tool_runtime.main.GoogleWorkspaceManager")
    def test_skips_when_disabled(self, manager_cls) -> None:
        settings = SimpleNamespace(google={"auto_google_authorize": False})
        _auto_google_authorize(settings)
        manager_cls.assert_not_called()

    @patch("builtins.print")
    @patch("tool_runtime.main.GoogleWorkspaceManager")
    def test_uses_all_bundles_when_enabled_and_list_is_empty(self, manager_cls, print_mock) -> None:
        settings = SimpleNamespace(
            google={
                "auto_google_authorize": True,
                "auto_google_authorize_bundles": [],
                "default_account_name": "default",
            }
        )
        _auto_google_authorize(settings)
        called_bundles = [call.kwargs["bundle"] for call in manager_cls.return_value.ensure_authorized.call_args_list]
        self.assertEqual(called_bundles, list(DEFAULT_GOOGLE_AUTO_BUNDLES))

    @patch("builtins.print")
    @patch("tool_runtime.main.GoogleWorkspaceManager")
    def test_uses_explicit_bundle_subset(self, manager_cls, print_mock) -> None:
        settings = SimpleNamespace(
            google={
                "auto_google_authorize": True,
                "auto_google_authorize_bundles": ["gmail_send", "docs_edit"],
                "default_account_name": "default",
            }
        )
        _auto_google_authorize(settings)
        called_bundles = [call.kwargs["bundle"] for call in manager_cls.return_value.ensure_authorized.call_args_list]
        self.assertEqual(called_bundles, ["gmail_send", "docs_edit"])


if __name__ == "__main__":
    unittest.main()
