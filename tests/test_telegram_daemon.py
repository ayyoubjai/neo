from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from telegram_daemon.main import TelegramDaemon


class TelegramDaemonChatIdTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_chat_ids_accepts_multiple_formats(self) -> None:
        self.assertEqual(TelegramDaemon._parse_chat_ids("123,456"), {123, 456})
        self.assertEqual(TelegramDaemon._parse_chat_ids(["123", 456]), {123, 456})
        self.assertEqual(TelegramDaemon._parse_chat_ids("[123, 456]"), {123, 456})

    def test_parse_chat_id_returns_scalar_for_incoming_messages(self) -> None:
        self.assertEqual(TelegramDaemon._parse_chat_id("123"), 123)
        self.assertEqual(TelegramDaemon._parse_chat_id(456), 456)
        self.assertIsNone(TelegramDaemon._parse_chat_id(""))

    async def test_handle_message_routes_allowed_chat_as_scalar(self) -> None:
        daemon = TelegramDaemon.__new__(TelegramDaemon)
        daemon._allowed_chat_ids = {123, 456}
        daemon._active_chat_id = None
        daemon._unsupported_files_to_agent = False
        daemon._handle_text_line = AsyncMock()
        daemon._send_message = AsyncMock()
        daemon._handle_audio_attachment = AsyncMock()
        daemon._handle_image_attachment = AsyncMock()
        daemon._handle_video_attachment = AsyncMock()
        daemon._handle_document_attachment = AsyncMock()
        daemon._has_audio_attachment = lambda message: False
        daemon._has_image_attachment = lambda message: False
        daemon._has_video_attachment = lambda message: False
        daemon._handle_other_attachment_with_complex = AsyncMock(return_value=False)

        await daemon._handle_message({"chat": {"id": "123"}, "text": "hello"})

        daemon._handle_text_line.assert_awaited_once_with("hello", 123)
        daemon._send_message.assert_not_awaited()
        self.assertEqual(daemon._active_chat_id, 123)

    def test_format_telegram_network_error_explains_dns_failures(self) -> None:
        daemon = TelegramDaemon.__new__(TelegramDaemon)
        daemon._api_base = "https://api.telegram.org"

        message = daemon._format_telegram_network_error(socket.gaierror(11001, "getaddrinfo failed"))

        self.assertIn("Telegram network error: [Errno 11001] getaddrinfo failed", message)
        self.assertIn("DNS lookup failed for Telegram host 'api.telegram.org'", message)
        self.assertIn("Check TELEGRAM_API_BASE_URL", message)


if __name__ == "__main__":
    unittest.main()
