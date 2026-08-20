import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model_server import llamacpp_client


class LlamacppClientTests(unittest.TestCase):
    @patch("model_server.llamacpp_client._post")
    def test_vision_chat_uses_openai_image_url_content_parts(self, post):
        post.return_value = {"choices": [{"message": {"content": "ok"}}]}

        llamacpp_client.generate_raw(
            "Describe this image.",
            "qwen-vision-instruct.gguf",
            images=["aGVsbG8="],
        )

        path, payload = post.call_args.args
        self.assertEqual(path, "/v1/chat/completions")
        content = payload["messages"][-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Describe this image."})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64,aGVsbG8=")
        self.assertNotIn("image_data", payload)
