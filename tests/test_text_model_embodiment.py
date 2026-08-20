from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import Settings
from model_server.text_model import generate
from model_server.ollama_client import OllamaError


def _settings() -> Settings:
    return Settings(
        workspace_root=str(REPO_ROOT),
        data_dir=str(REPO_ROOT / "data"),
        rpc={},
        interface={},
        voice={},
        vision={},
        video={},
        telegram={},
        google={},
        tool={"active_tool_limit": 100},
        search={},
        orchestrator={},
        evolve={"candidate_dir": str(REPO_ROOT / "evolve" / "candidates")},
        models={"text_model": "test-model", "text_enforce_json": False, "text_model_options": {}},
    )


class TextModelEmbodimentTests(unittest.TestCase):
    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", return_value="ok")
    @patch("model_server.text_model.load_settings")
    def test_generate_uses_raw_prompt_even_for_legacy_mode_inputs(
        self,
        load_settings_mock,
        ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        load_settings_mock.return_value = _settings()

        result = generate(
            "Open the camera.",
            {
                "mode": "GENERAL",
                "summary": "conversation-summary",
                "memory": [{"data": {"summary": "memory-hit"}}],
                "embodiment": {"summary": "host=desktop; interface=local; camera=available"},
                "trace_id": "trace-1",
                "turn_id": "turn-1",
            },
        )

        self.assertEqual(result, "ok")
        prompt = ollama_generate_mock.call_args.args[0]
        self.assertEqual(prompt, "Open the camera.")

    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", return_value="ok")
    @patch("model_server.text_model.load_settings")
    def test_generate_ignores_embodiment_and_summary_wrappers_when_suppressed(
        self,
        load_settings_mock,
        ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        load_settings_mock.return_value = _settings()

        result = generate(
            "Plan the next step.",
            {
                "mode": "AGENT",
                "summary": "should-not-appear",
                "memory": [{"data": {"summary": "also-hidden"}}],
                "embodiment": {"summary": "host=phone; interface=telegram"},
                "trace_id": "trace-2",
                "turn_id": "turn-2",
                "suppress_summary_memory": True,
            },
        )

        self.assertEqual(result, "ok")
        prompt = ollama_generate_mock.call_args.args[0]
        self.assertEqual(prompt, "Plan the next step.")

    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", return_value="ok")
    @patch("model_server.text_model.load_settings")
    def test_generate_uses_raw_cognition_prompt_without_outer_context_wrapper(
        self,
        load_settings_mock,
        ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        load_settings_mock.return_value = _settings()

        result = generate(
            "COGNITION SYSTEM1\n\nAnalyze the mode.",
            {
                "mode": "COGNITION",
                "summary": "conversation-summary",
                "memory": [{"data": {"summary": "memory-hit"}}],
                "embodiment": {"summary": "host=desktop; interface=local; camera=available"},
                "trace_id": "trace-cognition-1",
                "turn_id": "turn-cognition-1",
            },
        )

        self.assertEqual(result, "ok")
        prompt = ollama_generate_mock.call_args.args[0]
        self.assertNotIn("Mode: COGNITION", prompt)
        self.assertNotIn("Summary:", prompt)
        self.assertNotIn("Memory:", prompt)
        self.assertNotIn("Embodiment:", prompt)
        self.assertIn("COGNITION SYSTEM1", prompt)
        self.assertNotIn("User: COGNITION SYSTEM1", prompt)
        self.assertIn("Analyze the mode.", prompt)
        self.assertTrue(ollama_generate_mock.call_args.kwargs.get("use_thinking_on_empty"))

    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", return_value="ok")
    @patch("model_server.text_model.load_settings")
    def test_generate_uses_raw_prompt_for_other_cognition_stages(
        self,
        load_settings_mock,
        ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        load_settings_mock.return_value = _settings()

        result = generate(
            "COGNITION FINALIZER\n\nFinalize the answer.",
            {
                "mode": "COGNITION",
                "summary": "conversation-summary",
                "memory": [{"data": {"summary": "memory-hit"}}],
                "embodiment": {"summary": "host=desktop; interface=local; camera=available"},
                "trace_id": "trace-cognition-2",
                "turn_id": "turn-cognition-2",
            },
        )

        self.assertEqual(result, "ok")
        prompt = ollama_generate_mock.call_args.args[0]
        self.assertEqual(prompt, "COGNITION FINALIZER\n\nFinalize the answer.")
        self.assertNotIn("Mode: COGNITION", prompt)
        self.assertNotIn("Summary:", prompt)
        self.assertNotIn("Memory:", prompt)
        self.assertNotIn("Embodiment:", prompt)

    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", return_value="ok")
    @patch("model_server.text_model.load_settings")
    def test_generate_uses_model_override_and_options_override(
        self,
        load_settings_mock,
        ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        load_settings_mock.return_value = _settings()

        result = generate(
            "COGNITION SYSTEM3 PROPOSAL\n\nAnalyze the mode.",
            {
                "mode": "COGNITION",
                "summary": "",
                "memory": [],
                "trace_id": "trace-override",
                "turn_id": "turn-override",
            },
            model_override="peer-model",
            options_override={"temperature": 0.4, "top_p": 0.9},
        )

        self.assertEqual(result, "ok")
        self.assertEqual(ollama_generate_mock.call_args.args[1], "peer-model")
        self.assertEqual(ollama_generate_mock.call_args.kwargs["options"]["temperature"], 0.4)
        self.assertEqual(ollama_generate_mock.call_args.kwargs["options"]["top_p"], 0.9)

    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", return_value="ok")
    @patch("model_server.text_model.load_settings")
    def test_generate_merges_text_model_runtime_options(
        self,
        load_settings_mock,
        ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        settings = _settings()
        settings.models["text_model_options"] = {"thinking": False, "reasoning_effort": "high"}
        load_settings_mock.return_value = settings

        result = generate(
            "Analyze the mode.",
            {
                "mode": "COGNITION",
                "summary": "",
                "memory": [],
                "trace_id": "trace-model-options",
                "turn_id": "turn-model-options",
            },
        )

        self.assertEqual(result, "ok")
        self.assertFalse(ollama_generate_mock.call_args.kwargs["options"]["thinking"])
        self.assertEqual(ollama_generate_mock.call_args.kwargs["options"]["reasoning_effort"], "high")

    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", side_effect=OllamaError("offline"))
    @patch("model_server.text_model.load_settings")
    def test_generate_returns_empty_for_strict_json_when_backend_is_unavailable(
        self,
        load_settings_mock,
        ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        load_settings_mock.return_value = _settings()

        result = generate(
            "Respond with STRICT JSON only:\n{\"ok\":true}",
            {
                "mode": "ASI",
                "summary": "",
                "memory": [],
                "embodiment": {},
                "trace_id": "trace-3",
                "turn_id": "turn-3",
            },
        )

        self.assertEqual(result, "")
        self.assertEqual(ollama_generate_mock.call_args.kwargs.get("response_format"), "json")

    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", side_effect=OllamaError("offline"))
    @patch("model_server.text_model.load_settings")
    def test_generate_keeps_plaintext_fallback_when_backend_is_unavailable(
        self,
        load_settings_mock,
        ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        load_settings_mock.return_value = _settings()

        result = generate(
            "Say hello.",
            {
                "mode": "GENERAL",
                "summary": "",
                "memory": [],
                "embodiment": {},
                "trace_id": "trace-4",
                "turn_id": "turn-4",
            },
        )

        self.assertIn("I hit a model/runtime issue while processing your request", result)
        self.assertIn("Please try again or switch to a smaller/local model.", result)
        self.assertIsNone(ollama_generate_mock.call_args.kwargs.get("response_format"))

    @patch("model_server.text_model._build_system_prompt", return_value="system")
    @patch("model_server.text_model.provider", return_value="ollama")
    @patch("model_server.text_model.ollama_generate", side_effect=OllamaError("offline"))
    @patch("model_server.text_model.load_settings")
    def test_generate_omits_embodiment_from_cognition_fallback(
        self,
        load_settings_mock,
        _ollama_generate_mock,
        _provider_mock,
        _system_prompt_mock,
    ) -> None:
        load_settings_mock.return_value = _settings()

        result = generate(
            "Analyze the mode.",
            {
                "mode": "COGNITION",
                "summary": "conversation-summary",
                "memory": [],
                "embodiment": {"summary": "host=desktop; interface=local; camera=available"},
                "trace_id": "trace-cognition-2",
                "turn_id": "turn-cognition-2",
            },
        )

        self.assertIn("I hit a model/runtime issue while processing your request", result)
        self.assertNotIn("Embodiment:", result)
        self.assertNotIn("Context summary:", result)


if __name__ == "__main__":
    unittest.main()
