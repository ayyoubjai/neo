import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestrator.main import Orchestrator
from model_server import text_model


class _Memory:
    def __init__(self, hits=None):
        self.calls = []
        self.hits = hits or []

    async def retrieve(self, text, top_k=3):
        self.calls.append((text, top_k))
        return list(self.hits)


class CognitionSystem0AndTimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabling_system0_sends_request_to_router(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = Orchestrator()
            orchestrator._cognition_system0_enabled = False
            orchestrator._cognition_force_system = "system1"
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._cognition_distill_enabled = False
            orchestrator._memory = _Memory()
            orchestrator._compute_semantic_context_summary = AsyncMock(return_value="previous context")
            orchestrator._call_tool = AsyncMock(
                return_value={"status": "APPROVED", "result": {"iso": "2026-09-05T12:00:00Z"}}
            )
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._cognition_episodes_path = str(Path(tmp) / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    '{"type":"route","thinking_mode":"system1","reason":"direct"}',
                    '{"type":"final","thought":"done","text":"answer"}',
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Answer this request.",
                {"summary": "previous turn"},
                "trace-system0-disabled",
                "turn-system0-disabled",
            )

            self.assertEqual(result, "answer")
            prompts = [call.args[0] for call in orchestrator._generate_cognition_response.await_args_list]
            self.assertNotIn("COGNITION SYSTEM0", prompts[0])
            self.assertIn("COGNITION THINKING ROUTER", prompts[0])
            self.assertIn("USER REQUEST:\nAnswer this request.", prompts[0])
            orchestrator._call_tool.assert_awaited_once_with(
                "sys.time", {}, None, "trace-system0-disabled", []
            )

    async def test_system0_receives_context_and_escalation_reuses_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = _Memory([{"name": "user preference", "score": 0.9, "data": {"value": "concise"}}])
            orchestrator = Orchestrator()
            orchestrator._cognition_system0_enabled = True
            orchestrator._cognition_force_system = ""
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._cognition_distill_enabled = False
            orchestrator._memory = memory
            orchestrator._compute_semantic_context_summary = AsyncMock(return_value="prior goal and constraints")
            orchestrator._call_tool = AsyncMock(
                return_value={"status": "APPROVED", "result": {"iso": "2026-09-05T12:00:00Z"}}
            )
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._cognition_episodes_path = str(Path(tmp) / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    '{"type":"escalate","thought":"needs reasoning","reason":"non-trivial"}',
                    '{"type":"route","thinking_mode":"system1","reason":"direct"}',
                    '{"type":"final","thought":"done","text":"answer"}',
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Use my preference.",
                {"summary": "previous turn"},
                "trace-system0-escalation",
                "turn-system0-escalation",
            )

            self.assertEqual(result, "answer")
            self.assertEqual(len(memory.calls), 1)
            first_prompt = orchestrator._generate_cognition_response.await_args_list[0].args[0]
            self.assertIn("COGNITION SYSTEM0", first_prompt)
            self.assertIn("prior goal and constraints", first_prompt)
            self.assertIn("user preference", first_prompt)

    def test_model_prompt_receives_time_from_context(self):
        with patch.object(text_model, "provider", return_value="llamacpp"), patch.object(
            text_model, "llamacpp_generate", return_value="ok"
        ) as generate:
            result = text_model.generate(
                "User request",
                {"current_time": "2026-09-05T12:00:00Z"},
                model_override="test-model",
            )

        self.assertEqual(result, "ok")
        self.assertIn("Current UTC time: 2026-09-05T12:00:00Z", generate.call_args.args[0])

    async def test_semantic_summary_keeps_prior_context_on_generation_failure(self):
        orchestrator = Orchestrator()
        for response in [RuntimeError("missing model"), {"status": "DEGRADED", "text": "runtime error"}]:
            with self.subTest(response=response), patch(
                "orchestrator.main.send_request", new_callable=AsyncMock
            ) as rpc, patch("orchestrator.main.record_event") as log:
                if isinstance(response, Exception):
                    rpc.side_effect = response
                else:
                    rpc.return_value = response
                result = await orchestrator._compute_semantic_context_summary(
                    {"summary": "User prefers concise answers"}, "next match?", "trace", "turn"
                )
                self.assertEqual(result, "User prefers concise answers")
                self.assertTrue(rpc.call_args.args[3]["context"]["require_model_output"])
                self.assertEqual(log.call_args.args[0], "cognition_semantic_summary_error")

    def test_required_model_output_never_returns_user_facing_error_as_summary(self):
        with patch.object(text_model, "provider", return_value="llamacpp"), patch.object(
            text_model, "llamacpp_generate", side_effect=text_model.LlamacppError("missing model")
        ), patch.object(
            text_model, "ollama_generate", side_effect=text_model.OllamaError("offline")
        ), patch.object(text_model, "record_event"), patch.object(text_model, "log_exception"):
            with self.assertRaisesRegex(RuntimeError, "Model generation failed"):
                text_model.generate(
                    "Summarize prior conversation", {"require_model_output": True},
                    model_override="missing.gguf",
                )

    def test_router_preserves_current_query_with_state_disabled_in_both_modes(self):
        orchestrator = Orchestrator()
        orchestrator._cognition_query_state_enabled = False
        orchestrator._cognition_state_of_mind_enabled = False
        for mode in ["small", "full"]:
            with self.subTest(mode=mode):
                orchestrator._cognition_mode = mode
                prompt = orchestrator._build_cognition_route_prompt(
                    "When is Real Madrid's next match?", {}, {}, [], [],
                    semantic_summary="Previously exchanged greetings",
                )
                self.assertIn("USER REQUEST:\nWhen is Real Madrid's next match?", prompt)


if __name__ == "__main__":
    unittest.main()
