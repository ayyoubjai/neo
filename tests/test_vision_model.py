import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_server import vision_model


class VisionModelTests(unittest.TestCase):
    def analyze_response(self, response):
        settings = SimpleNamespace(models={"vision_model": "vision.gguf"})
        with patch.object(vision_model, "load_settings", return_value=settings), \
                patch.object(vision_model, "_load_image_bytes", return_value=b"image"), \
                patch.object(vision_model, "record_event"), \
                patch.object(vision_model, "record_llm_json"), \
                patch.object(vision_model, "log_error"), \
                patch("model_server.llamacpp_client._post", return_value=response) as post:
            result = vision_model._analyze_llamacpp("workspace:/screen.png", "What is visible?")
            self.assertEqual(post.call_args.args[0], "/v1/chat/completions")
            self.assertEqual(post.call_args.args[1]["response_format"], {"type": "json_object"})
            return result

    def test_extracts_chat_content_before_parsing_vision_json(self):
        payload = {"summary": "A terminal", "answer": "A terminal window is open.", "text": "hello", "objects": []}
        for prefix in ("", "<think>Inspecting the image.</think>\n"):
            with self.subTest(prefix=prefix):
                result = self.analyze_response({"choices": [{"message": {"content": prefix + json.dumps(payload)}}]})
                self.assertTrue(result["ok"])
                self.assertEqual(result["raw_payload"], payload)

    def test_empty_or_invalid_content_remains_an_error(self):
        for response in ({"choices": []}, {"choices": [{"message": {"content": "not JSON"}}]}):
            with self.subTest(response=response):
                self.assertFalse(self.analyze_response(response)["ok"])
