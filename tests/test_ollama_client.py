from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model_server.ollama_client import generate


class OllamaClientTests(unittest.TestCase):
    @patch("model_server.ollama_client._post")
    def test_generate_lifts_thinking_control_out_of_options(self, post_mock) -> None:
        post_mock.return_value = {"response": "ok"}

        result = generate(
            "prompt",
            "model",
            options={"thinking": False, "reasoning_effort": "medium", "temperature": 0.2},
        )

        self.assertEqual(result, "ok")
        payload = post_mock.call_args.args[1]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["reasoning_effort"], "medium")
        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertNotIn("thinking", payload["options"])

    @patch("model_server.ollama_client._post")
    def test_generate_uses_response_when_present(self, post_mock) -> None:
        post_mock.return_value = {
            "response": "{\"type\":\"final\",\"text\":\"from-response\"}",
            "thinking": "{\"type\":\"final\",\"text\":\"from-thinking\"}",
        }

        result = generate("prompt", "model", use_thinking_on_empty=True)

        self.assertEqual(result, "{\"type\":\"final\",\"text\":\"from-response\"}")

    @patch("model_server.ollama_client._post")
    def test_generate_falls_back_to_thinking_when_response_is_empty(self, post_mock) -> None:
        post_mock.return_value = {
            "response": "",
            "thinking": "{\"type\":\"final\",\"text\":\"from-thinking\"}",
        }

        result = generate("prompt", "model", use_thinking_on_empty=True)

        self.assertEqual(result, "{\"type\":\"final\",\"text\":\"from-thinking\"}")

    @patch("model_server.ollama_client._post")
    def test_generate_uses_message_content_when_present(self, post_mock) -> None:
        post_mock.return_value = {
            "response": "",
            "message": {"content": "{\"type\":\"final\",\"text\":\"from-message\"}"},
        }

        result = generate("prompt", "model")

        self.assertEqual(result, "{\"type\":\"final\",\"text\":\"from-message\"}")
