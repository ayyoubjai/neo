from __future__ import annotations

import sys
import tempfile
import unittest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model_server.router_model import _coerce_route
from orchestrator.main import Orchestrator


class _FakeMemory:
    def __init__(self) -> None:
        self.facts = []
        self.entities = []

    async def get_facts(self, prefix: str = ""):
        return [item for item in self.facts if item["name"].startswith(prefix)]

    async def store_fact(self, name: str, value: str, trace_id: str, confidence: float = 0.6):
        self.facts.append(
            {
                "name": name,
                "data": {"value": value},
                "trace_id": trace_id,
                "confidence": confidence,
            }
        )
        return name

    async def store_entity(self, name: str, data, trace_id: str, confidence: float = 0.6):
        self.entities.append(
            {
                "name": name,
                "data": data,
                "trace_id": trace_id,
                "confidence": confidence,
            }
        )
        return name


class _FakeRetrievalMemory:
    def __init__(self, hits=None) -> None:
        self._hits = hits or []

    async def retrieve(self, text: str, top_k: int = 0):
        return list(self._hits)


class CognitionModeTests(unittest.IsolatedAsyncioTestCase):
    def test_normalize_model_call_options_keeps_thinking_and_reasoning_effort(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)

        normalized = orchestrator._normalize_model_call_options(
            {"thinking": "false", "reasoning_effort": "medium", "temperature": 0.2}
        )

        self.assertFalse(normalized["thinking"])
        self.assertEqual(normalized["reasoning_effort"], "medium")
        self.assertEqual(normalized["temperature"], 0.2)

    def test_merge_model_call_options_overrides_base_values(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)

        merged = orchestrator._merge_model_call_options(
            {"thinking": True, "reasoning_effort": "low"},
            {"thinking": False, "top_p": 0.9},
        )

        self.assertFalse(merged["thinking"])
        self.assertEqual(merged["reasoning_effort"], "low")
        self.assertEqual(merged["top_p"], 0.9)

    def test_merge_model_call_options_supports_system3_defaults(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)

        merged = orchestrator._merge_model_call_options(
            {"thinking": True, "reasoning_effort": "medium"},
            {"thinking": False, "temperature": 0.3},
        )

        self.assertFalse(merged["thinking"])
        self.assertEqual(merged["reasoning_effort"], "medium")
        self.assertEqual(merged["temperature"], 0.3)

    def test_resolve_cognition_model_override_uses_system0_system1_and_system2_models(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._cognition_system0_model = "system0-model"
        orchestrator._cognition_system1_model = "system1-model"
        orchestrator._cognition_system2_model = "system2-model"

        self.assertEqual(orchestrator._resolve_cognition_model_override("system0"), "system0-model")
        self.assertEqual(orchestrator._resolve_cognition_model_override("system1"), "system1-model")
        self.assertEqual(orchestrator._resolve_cognition_model_override("system2"), "system2-model")
        self.assertEqual(orchestrator._resolve_cognition_model_override("system3"), "")

    def test_resolve_cognition_model_override_has_no_legacy_fallback(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._cognition_system0_model = ""
        orchestrator._cognition_system1_model = ""
        orchestrator._cognition_system2_model = ""

        self.assertEqual(orchestrator._resolve_cognition_model_override("system0"), "")
        self.assertEqual(orchestrator._resolve_cognition_model_override("system1"), "")
        self.assertEqual(orchestrator._resolve_cognition_model_override("system2"), "")
        self.assertEqual(orchestrator._resolve_cognition_model_override("system3"), "")

    def test_resolve_cognition_model_override_keeps_peer_specific_system3_model(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._cognition_system0_model = ""
        orchestrator._cognition_system1_model = ""
        orchestrator._cognition_system2_model = ""

        self.assertEqual(
            orchestrator._resolve_cognition_model_override("system3", explicit_model="peer-model"),
            "peer-model",
        )

    def test_normalize_cognition_peer_profile_requires_model(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)

        peer = orchestrator._normalize_cognition_peer_profile(
            {"peer_id": "peer-a", "role": "planner"},
            pool_name="proposer",
            index=1,
        )

        self.assertIsNone(peer)

    def test_router_accepts_cognition_mode(self) -> None:
        route = _coerce_route({"mode": "COGNITION", "needs_memory": False})
        self.assertEqual(route["mode"], "COGNITION")
        self.assertFalse(route["needs_memory"])

    def test_router_coerces_legacy_modes_to_cognition(self) -> None:
        route = _coerce_route({"mode": "GENERAL", "needs_memory": True})
        self.assertEqual(route["mode"], "COGNITION")
        self.assertFalse(route["needs_memory"])

    def test_orchestrator_normalizes_cognition_mode(self) -> None:
        self.assertEqual(Orchestrator._normalize_requested_mode(" cognition "), "COGNITION")
        self.assertEqual(Orchestrator._normalize_requested_mode(" system0 "), "SYSTEM0")

    async def test_run_mode_once_forces_system0_execution(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._run_cognition_loop = AsyncMock(return_value="ok")

        result = await orchestrator._run_mode_once(
            "SYSTEM0",
            "describe the image",
            {},
            "trace-1",
            "turn-1",
        )

        self.assertEqual(result, "ok")
        orchestrator._run_cognition_loop.assert_awaited_once_with(
            "describe the image",
            {},
            "trace-1",
            "turn-1",
            retrieval_query=None,
            forced_execution_mode="system0",
        )

    def test_cognition_system1_prompt_omits_think_quickly_text(self) -> None:
        orchestrator = Orchestrator()

        prompt = orchestrator._build_cognition_system1_prompt(
            orchestrator._default_cognition_query_state("Analyze the mode."),
            orchestrator._default_cognition_state_of_mind(),
            [],
            [],
            cycle_index=1,
        )

        self.assertIn("COGNITION SYSTEM1", prompt)
        self.assertNotIn("Think quickly.", prompt)
        self.assertNotIn("USER REQUEST:", prompt)
        self.assertIn("CYCLE INDEX: 1", prompt)
        self.assertNotIn("REUSABLE SKILLS:", prompt)
        self.assertNotIn("PATTERNS:", prompt)
        self.assertNotIn("SAFETY RULES:", prompt)

    def test_cognition_system0_prompt_mentions_fast_safe_path(self) -> None:
        orchestrator = Orchestrator()

        prompt = orchestrator._build_cognition_system0_prompt(
            orchestrator._default_cognition_query_state("What time?"),
            orchestrator._default_cognition_state_of_mind(),
            [],
            cycle_index=1,
        )

        self.assertIn("COGNITION SYSTEM0", prompt)
        self.assertIn("fastest safe path", prompt)
        self.assertIn("CYCLE INDEX: 1", prompt)
        self.assertNotIn("\"type\":\"clarify\"", prompt)
        self.assertNotIn("\"type\":\"act\"", prompt)
        self.assertNotIn("AVAILABLE TOOLS:", prompt)
        self.assertNotIn("SPECIAL ORCHESTRATION ACTIONS:", prompt)

    def test_cognition_system2_prompt_omits_catalog_sections(self) -> None:
        orchestrator = Orchestrator()

        prompt = orchestrator._build_cognition_system2_prompt(
            "Analyze the mode.",
            orchestrator._default_cognition_query_state("Analyze the mode."),
            orchestrator._default_cognition_state_of_mind(),
            [],
            [],
            [],
            cycle_index=1,
        )

        self.assertIn("COGNITION SYSTEM2", prompt)
        self.assertNotIn("REUSABLE SKILLS:", prompt)
        self.assertNotIn("PATTERNS:", prompt)
        self.assertNotIn("SAFETY RULES:", prompt)

    def test_cognition_system1_clarify_can_be_disabled(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._cognition_system1_clarify_enabled = False

        prompt = orchestrator._build_cognition_system1_prompt({}, {}, [], [], cycle_index=1)
        schema = orchestrator._build_cognition_execution_schema_text(
            allow_step=False,
            allow_clarify=orchestrator._cognition_system1_clarify_enabled,
        )
        parsed = orchestrator._parse_cognition_thinking_response(
            "{\"type\":\"clarify\",\"thought\":\"blocked\",\"question\":\"Which file?\"}",
            allow_step=False,
            allow_clarify=orchestrator._cognition_system1_clarify_enabled,
        )

        self.assertNotIn("\"type\":\"clarify\"", prompt)
        self.assertNotIn("\"type\":\"clarify\"", schema)
        self.assertIn("Do not return clarify", prompt)
        self.assertIsNone(parsed)

    def test_cognition_system2_clarify_can_be_disabled(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._cognition_system2_clarify_enabled = False

        prompt = orchestrator._build_cognition_system2_prompt(
            "Analyze the mode.",
            {},
            {},
            [],
            [],
            [],
            cycle_index=1,
        )
        schema = orchestrator._build_cognition_execution_schema_text(
            allow_step=True,
            allow_clarify=orchestrator._cognition_system2_clarify_enabled,
        )
        parsed = orchestrator._parse_cognition_thinking_response(
            "{\"type\":\"clarify\",\"thought\":\"blocked\",\"question\":\"Which file?\"}",
            allow_step=True,
            allow_clarify=orchestrator._cognition_system2_clarify_enabled,
        )

        self.assertNotIn("\"type\":\"clarify\"", prompt)
        self.assertNotIn("\"type\":\"clarify\"", schema)
        self.assertIn("Do not return clarify", prompt)
        self.assertIsNone(parsed)

    def test_cognition_prompts_expose_tool_generation_action(self) -> None:
        orchestrator = Orchestrator()

        system1_prompt = orchestrator._build_cognition_system1_prompt({}, {}, [], [], cycle_index=1)
        system2_prompt = orchestrator._build_cognition_system2_prompt(
            "Build a missing capability.",
            {},
            {},
            [],
            [],
            [],
            cycle_index=1,
        )

        self.assertIn("SPECIAL ORCHESTRATION ACTIONS:", system1_prompt)
        self.assertIn("generate_tools_from_spec", system1_prompt)
        self.assertIn("SPECIAL ORCHESTRATION ACTIONS:", system2_prompt)
        self.assertIn("generate_tools_from_spec", system2_prompt)

    def test_cognition_route_prompt_describes_system1_as_execution_loop(self) -> None:
        orchestrator = Orchestrator()

        prompt = orchestrator._build_cognition_route_prompt(
            "Analyze the mode.",
            orchestrator._default_cognition_query_state("Analyze the mode."),
            orchestrator._default_cognition_state_of_mind(),
            [],
            [],
        )

        self.assertIn("action-oriented execution loop", prompt)
        self.assertIn("iterative tool calls and observations", prompt)
        self.assertIn("Focus only on whether the task needs the faster execution loop", prompt)
        self.assertNotIn("similar to COMPLEX mode", prompt)
        self.assertNotIn("a very small number of obvious actions", prompt)
        self.assertNotIn("REUSABLE SKILLS:", prompt)
        self.assertNotIn("PATTERNS:", prompt)
        self.assertNotIn("SAFETY RULES:", prompt)

    def test_cognition_route_prompt_mentions_system3_when_enabled(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._cognition_system3_enabled = True
        orchestrator._cognition_peer_pools = {
            "proposers": [{"peer_id": "prop_a", "model": "m1", "role": ""}],
            "critics": [{"peer_id": "crit_a", "model": "m2", "role": ""}],
        }

        prompt = orchestrator._build_cognition_route_prompt(
            "Analyze the mode.",
            orchestrator._default_cognition_query_state("Analyze the mode."),
            orchestrator._default_cognition_state_of_mind(),
            [],
            [],
        )

        self.assertIn("system1|system2|system3", prompt)
        self.assertIn("Choose system3", prompt)

    def test_cognition_route_prompt_exposes_route_time_tool_generation(self) -> None:
        orchestrator = Orchestrator()

        prompt = orchestrator._build_cognition_route_prompt(
            "Create a missing capability before execution.",
            orchestrator._default_cognition_query_state("Create a missing capability before execution."),
            orchestrator._default_cognition_state_of_mind(),
            [],
            [{"tool_id": "sys.time", "name": "Time", "description": "Read the time"}],
        )

        self.assertIn("\"generate_tools_first\":true", prompt)
        self.assertIn("\"tool_generation_spec\"", prompt)
        self.assertIn("AVAILABLE TOOLS:", prompt)
        self.assertIn("stable reusable capability", prompt)

    def test_cognition_pattern_router_prompt_contains_catalog_sections(self) -> None:
        orchestrator = Orchestrator()

        prompt = orchestrator._build_cognition_pattern_router_prompt(
            "What time is it?",
            orchestrator._default_cognition_query_state("What time is it?"),
            orchestrator._default_cognition_state_of_mind(),
            [
                {
                    "name": "Check Time Skill",
                    "description": "Use the clock tool for time questions.",
                    "when_to_use": ["When the user asks for the current time."],
                }
            ],
            [
                {
                    "pattern_id": "answer_time",
                    "name": "Answer Time",
                    "description": "Answer time requests.",
                    "when_to_use": ["When the task is to answer with the current time."],
                    "source": "def answer_time():\n    now = sys.time()\n    return now.value\n",
                }
            ],
            [
                {
                    "rule": "Do not guess the current time.",
                    "rationale": "Read it from a grounded source.",
                }
            ],
        )

        self.assertIn("COGNITION PATTERN ROUTER", prompt)
        self.assertIn("REUSABLE SKILLS:", prompt)
        self.assertIn("PATTERNS:", prompt)
        self.assertIn("SAFETY RULES:", prompt)

    def test_parse_cognition_pattern_route_accepts_call_syntax(self) -> None:
        orchestrator = Orchestrator()

        parsed = orchestrator._parse_cognition_pattern_route_response(
            "{\"type\":\"pattern_route\",\"decision\":\"use\",\"pattern_id\":\"answer_time()\",\"reason\":\"fits\"}"
        )

        self.assertEqual(
            parsed,
            {
                "type": "pattern_route",
                "decision": "use",
                "pattern_id": "answer_time",
                "reason": "fits",
            },
        )

    def test_cognition_prompts_omit_state_sections_when_disabled(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._cognition_query_state_enabled = False
        orchestrator._cognition_state_of_mind_enabled = False

        system1_prompt = orchestrator._build_cognition_system1_prompt({}, {}, [], [], cycle_index=1)
        system2_prompt = orchestrator._build_cognition_system2_prompt(
            "Analyze the mode.",
            {},
            {},
            [],
            [],
            [],
            cycle_index=1,
        )
        final_prompt = orchestrator._build_cognition_final_prompt("Analyze the mode.", {}, {}, [], [])

        self.assertNotIn("QUERY STATE:", system1_prompt)
        self.assertNotIn("STATE OF MIND:", system1_prompt)
        self.assertNotIn("query_state_delta", system1_prompt)
        self.assertNotIn("state_of_mind_delta", system1_prompt)
        self.assertNotIn("QUERY STATE:", system2_prompt)
        self.assertNotIn("STATE OF MIND:", system2_prompt)
        self.assertNotIn("todo_updates", system2_prompt)
        self.assertNotIn("QUERY STATE:", final_prompt)
        self.assertNotIn("STATE OF MIND:", final_prompt)

    def test_parse_cognition_tool_selection_response_filters_invalid_tool_ids(self) -> None:
        orchestrator = Orchestrator()

        parsed = orchestrator._parse_cognition_tool_selection_response(
            (
                "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"sys.time\",\"missing.tool\"],"
                "\"reason\":\"time lookup needs only the clock tool\"}"
            ),
            {"sys.time", "sys.weather"},
        )

        self.assertEqual(
            parsed,
            {
                "type": "tool_selection",
                "selected_tool_ids": ["sys.time"],
                "reason": "time lookup needs only the clock tool",
            },
        )

        invalid = orchestrator._parse_cognition_tool_selection_response(
            (
                "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"missing.tool\"],"
                "\"reason\":\"invalid\"}"
            ),
            {"sys.time", "sys.weather"},
        )
        self.assertIsNone(invalid)

    def test_parse_cognition_tool_selection_response_accepts_categories(self) -> None:
        orchestrator = Orchestrator()

        parsed = orchestrator._parse_cognition_tool_selection_response(
            (
                "{\"type\":\"tool_selection\",\"selected_category_ids\":[\"ui\",\"missing\"],"
                "\"selected_tool_ids\":[\"sys.time\"],\"reason\":\"ui workflow also needs time\"}"
            ),
            {"sys.time", "sys.weather"},
            {"ui", "sys"},
        )

        self.assertEqual(
            parsed,
            {
                "type": "tool_selection",
                "selected_tool_ids": ["sys.time"],
                "selected_category_ids": ["ui"],
                "reason": "ui workflow also needs time",
            },
        )

        invalid = orchestrator._parse_cognition_tool_selection_response(
            (
                "{\"type\":\"tool_selection\",\"selected_category_ids\":[\"missing\"],"
                "\"selected_tool_ids\":[],\"reason\":\"invalid\"}"
            ),
            {"sys.time", "sys.weather"},
            {"ui", "sys"},
        )
        self.assertIsNone(invalid)

    def test_parse_cognition_final_response_accepts_answer_alias(self) -> None:
        orchestrator = Orchestrator()

        parsed = orchestrator._parse_cognition_thinking_response(
            "{\"type\":\"final\",\"thought\":\"done\",\"answer\":\"Saved at workspace:/screenshot.png\"}",
            allow_step=False,
        )

        self.assertEqual(parsed["text"], "Saved at workspace:/screenshot.png")

    def test_cognition_result_payload_text_preserves_empty_final_schema(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)

        text = orchestrator._cognition_result_payload_text({"type": "final", "text": ""})

        self.assertEqual(text, "{\"type\":\"final\",\"text\":\"\"}")
        self.assertNotIn("couldn't produce a cognition result", text)

    async def test_final_response_critic_parse_failure_is_not_marked_fulfilled(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._generate_response = AsyncMock(side_effect=["not json", "still not json"])

        result = await orchestrator._evaluate_final_response_critic(
            "What happened?",
            "Fallback answer.",
            "COGNITION",
            "trace-critic",
            "turn-critic",
        )

        self.assertFalse(result["fulfilled"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("final response critic unavailable or parse failed", result["issues"])

    def test_normalize_cognition_actions_accepts_tool_generation_action(self) -> None:
        orchestrator = Orchestrator()

        actions = orchestrator._normalize_cognition_actions(
            [
                {
                    "action_type": "generate_tools_from_spec",
                    "spec": {"request": "Create an echo tool.", "namespace_prefix": "demo"},
                    "label": "bootstrap missing tool",
                },
                {"tool_id": "sys.time", "args": {}, "label": "check time"},
            ],
            limit=4,
        )

        self.assertEqual(
            actions,
            [
                {
                    "action_type": "generate_tools_from_spec",
                    "spec": {"request": "Create an echo tool.", "namespace_prefix": "demo"},
                    "label": "bootstrap missing tool",
                },
                {"action_type": "tool", "tool_id": "sys.time", "args": {}, "label": "check time"},
            ],
        )

    def test_parse_cognition_route_response_accepts_tool_generation_request(self) -> None:
        orchestrator = Orchestrator()

        parsed = orchestrator._parse_cognition_route_response(
            (
                "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"bootstrap first\","
                "\"generate_tools_first\":true,"
                "\"tool_generation_spec\":{\"request\":\"Create a demo echo tool.\","
                "\"namespace_prefix\":\"demo\",\"max_tools\":1}}"
            )
        )

        self.assertEqual(
            parsed,
            {
                "type": "route",
                "thinking_mode": "system1",
                "reason": "bootstrap first",
                "generate_tools_first": True,
                "tool_generation_spec": {
                    "request": "Create a demo echo tool.",
                    "namespace_prefix": "demo",
                    "max_tools": 1,
                },
            },
        )

    def test_cognition_distill_prompt_requests_source_backed_patterns(self) -> None:
        orchestrator = Orchestrator()

        prompt = orchestrator._build_cognition_distill_prompt(
            "What time is it?",
            {"type": "final", "text": "12:00"},
            orchestrator._default_cognition_query_state("What time is it?"),
            orchestrator._default_cognition_state_of_mind(),
            [],
            [],
            [{"tool_id": "sys.time", "name": "Time", "description": "Read the time"}],
            [
                {
                    "pattern_id": "child_time",
                    "name": "Child Time",
                    "when_to_use": ["Read the current time."],
                    "source": "def child_time():\n    now = sys.time()\n    return now.value",
                    "inputs_needed": [],
                }
            ],
        )

        self.assertIn("\"source\":\"def pattern_name", prompt)
        self.assertIn("patterns must be reusable source-backed functions", prompt)
        self.assertIn("pat.<pattern_id>(...)", prompt)
        self.assertIn("CALLABLE PATTERNS:", prompt)
        self.assertIn("The top-level type must be exactly \"distill\"", prompt)
        self.assertIn("Do not return final/clarify/act/step JSON", prompt)
        self.assertNotIn("\"trigger\":\"...\"", prompt)

    def test_json_repair_prompt_rejects_wrong_schema_wrappers(self) -> None:
        orchestrator = Orchestrator.__new__(Orchestrator)

        prompt = orchestrator._build_json_repair_prompt(
            "{\"type\":\"distill\",\"reusable_skills\":[],\"memory_facts\":[],\"patterns\":[],"
            "\"failure_lessons\":[],\"safety_rules\":[]}",
            "{\"type\":\"final\",\"text\":\"done\"}",
        )

        self.assertIn("wrong schema", prompt)
        self.assertIn("top-level type/value must match", prompt)
        self.assertIn("discard that wrapper", prompt)

    async def test_distillation_timeout_fails_open_without_repair_loop(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._cognition_distill_timeout_s = 1
        orchestrator._cognition_repair_timeout_s = 1
        orchestrator._cognition_distill_model = ""
        orchestrator._cognition_distill_options = {}
        orchestrator._cognition_repair_model = "repair-model"
        orchestrator._cognition_repair_options = {}

        async def _slow_generate(*args, **kwargs):
            await asyncio.sleep(5)
            return "{\"type\":\"distill\"}"

        orchestrator._generate_cognition_response = _slow_generate
        logged_events = []
        orchestrator._log_mode_event = lambda mode, event_type, payload: logged_events.append(event_type)

        result = await orchestrator._distill_cognition_episode(
            {"type": "final", "text": "done"},
            orchestrator._default_cognition_query_state("take a screenshot"),
            orchestrator._default_cognition_state_of_mind(),
            [],
            [],
            [],
            [],
            "trace-timeout",
            "turn-timeout",
        )

        self.assertEqual(
            result,
            {
                "reusable_skills": [],
                "memory_facts": [],
                "patterns": [],
                "failure_lessons": [],
                "safety_rules": [],
            },
        )
        self.assertIn("distill_timeout", logged_events)
        self.assertIn("distill_repair_timeout", logged_events)

    async def test_persist_cognition_distillation_dedupes_catalogs_and_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            fake_memory = _FakeMemory()
            orchestrator._memory = fake_memory
            orchestrator._cognition_skills_path = str(tmp_path / "cognition_skills.json")
            orchestrator._cognition_patterns_path = str(tmp_path / "cognition_patterns.json")
            orchestrator._cognition_lessons_path = str(tmp_path / "cognition_lessons.json")
            orchestrator._cognition_safety_rules_path = str(tmp_path / "cognition_safety_rules.json")

            distillation = {
                "reusable_skills": [
                    {
                        "name": "Map mode architecture",
                        "description": "Trace router, dispatcher, and execution loops before adding a new mode.",
                        "when_to_use": ["When changing orchestrator mode behavior"],
                        "confidence": 0.8,
                    }
                ],
                "memory_facts": [
                    {
                        "name": "project.mode.cognition.enabled",
                        "value": "true",
                        "confidence": 0.9,
                    }
                ],
                "patterns": [
                    {
                        "name": "Think then act then distill",
                        "description": "Separate thinking, acting, and post-episode distillation.",
                        "trigger": "Tasks that need reflection plus execution",
                        "confidence": 0.75,
                    }
                ],
                "failure_lessons": [
                    {
                        "lesson": "Do not add a mode only in dispatch; router and session overrides must agree.",
                        "when": "Introducing a new mode",
                        "confidence": 0.7,
                    }
                ],
                "safety_rules": [
                    {
                        "rule": "Do not treat internal state_of_mind as user-visible by default.",
                        "rationale": "Inner cognition should stay internal unless explicitly surfaced.",
                        "confidence": 0.95,
                    }
                ],
            }

            first = await orchestrator._persist_cognition_distillation(distillation, "trace-1")
            second = await orchestrator._persist_cognition_distillation(distillation, "trace-2")

            self.assertEqual(first, {"skills": 1, "facts": 1, "patterns": 1, "lessons": 1, "safety_rules": 1})
            self.assertEqual(second, {"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0})
            self.assertEqual(len(fake_memory.facts), 1)
            self.assertEqual(len(fake_memory.entities), 4)

            self.assertTrue((tmp_path / "cognition_skills.json").exists())
            self.assertTrue((tmp_path / "cognition_patterns.json").exists())
            self.assertTrue((tmp_path / "cognition_lessons.json").exists())
            self.assertTrue((tmp_path / "cognition_safety_rules.json").exists())

    async def test_run_cognition_loop_distills_for_system1_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 1, "facts": 1, "patterns": 1, "lessons": 1, "safety_rules": 1}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"init\",\"thought\":\"init\","
                        "\"query_state\":{\"query_rewrite\":\"analyze mode\",\"intent\":\"analyze\"},"
                        "\"state_of_mind\":{\"stance\":\"focused\",\"confidence\":0.6}}"
                    ),
                    "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"direct answer is enough\"}",
                    (
                        "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"System1 answer\","
                        "\"query_state_delta\":{},\"state_of_mind_delta\":{}}"
                    ),
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Analyze the mode.",
                {"summary": "conversation-summary"},
                "trace-system1",
                "turn-system1",
            )

            self.assertEqual(result, "System1 answer")
            orchestrator._distill_cognition_episode.assert_awaited_once()
            orchestrator._persist_cognition_distillation.assert_awaited_once()

    async def test_run_cognition_loop_executes_selected_pattern_before_thinking_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_distill_enabled = False
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(
                return_value=[{"tool_id": "sys.time", "name": "Time", "description": "Read the time"}]
            )
            orchestrator._load_cognition_skills = lambda: []
            child_pattern = orchestrator._normalize_cognition_pattern(
                {
                    "pattern_id": "child_time",
                    "name": "Child Time",
                    "description": "Read the current time.",
                    "when_to_use": ["When the user asks for the current time."],
                    "source": (
                        "def child_time():\n"
                        "    now = sys.time()\n"
                        "    return now.value\n"
                    ),
                }
            )
            parent_pattern = orchestrator._normalize_cognition_pattern(
                {
                    "pattern_id": "answer_time",
                    "name": "Answer Time",
                    "description": "Answer time requests via a child pattern.",
                    "when_to_use": ["When the task is to answer with the current time."],
                    "source": (
                        "def answer_time():\n"
                        "    reply = pat.child_time()\n"
                        "    return reply.value\n"
                    ),
                }
            )
            self.assertIsNotNone(child_pattern)
            self.assertIsNotNone(parent_pattern)
            orchestrator._load_cognition_patterns = lambda: [child_pattern, parent_pattern]
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._resolve_asi_tool_id = lambda target: "sys.time" if str(target).strip() == "sys.time" else ""
            orchestrator._execute_tool_action = AsyncMock(
                return_value={
                    "action": {"tool_id": "sys.time", "args": {}},
                    "observation": {"status": "APPROVED", "result": "12:00"},
                }
            )
            orchestrator._finalize_cognition_result = AsyncMock(
                return_value={"type": "final", "text": "finalizer answer"}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"init\",\"thought\":\"init\","
                        "\"query_state\":{\"query_rewrite\":\"what time is it\",\"intent\":\"time_lookup\"},"
                        "\"state_of_mind\":{\"stance\":\"focused\",\"confidence\":0.8}}"
                    ),
                    (
                        "{\"type\":\"pattern_route\",\"decision\":\"use\",\"pattern_id\":\"answer_time\","
                        "\"reason\":\"the callable pattern already solves it\"}"
                    ),
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "What time is it?",
                {"summary": "conversation-summary"},
                "trace-pattern",
                "turn-pattern",
            )

            self.assertEqual(result, "12:00")
            orchestrator._execute_tool_action.assert_awaited_once()
            orchestrator._finalize_cognition_result.assert_not_awaited()
            self.assertEqual(orchestrator._generate_cognition_response.await_count, 2)

    async def test_run_cognition_loop_repeats_system1_after_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_system1_max_steps = 3
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(
                return_value=[{"tool_id": "sys.time", "name": "Time", "description": "Read the time"}]
            )
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._run_cognition_actions = AsyncMock(
                return_value=[
                    {
                        "label": "check time",
                        "action": {"tool_id": "sys.time", "args": {}},
                        "observation": {"status": "OK", "result": "12:00"},
                    }
                ]
            )
            orchestrator._finalize_cognition_result = AsyncMock(
                return_value={"type": "final", "text": "finalizer answer"}
            )
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 1, "facts": 1, "patterns": 1, "lessons": 1, "safety_rules": 1}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"init\",\"thought\":\"init\","
                        "\"query_state\":{\"query_rewrite\":\"analyze mode\",\"intent\":\"analyze\"},"
                        "\"state_of_mind\":{\"stance\":\"focused\",\"confidence\":0.6}}"
                    ),
                    "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"one quick loop is enough\"}",
                    (
                        "{\"type\":\"act\",\"thought\":\"inspect first\",\"actions\":[{\"tool_id\":\"sys.time\","
                        "\"args\":{},\"label\":\"check time\"}],\"query_state_delta\":{},\"state_of_mind_delta\":{}}"
                    ),
                    (
                        "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"System1 loop answer\","
                        "\"query_state_delta\":{},\"state_of_mind_delta\":{}}"
                    ),
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Analyze the mode.",
                {"summary": "conversation-summary"},
                "trace-system1-loop",
                "turn-system1-loop",
            )

            self.assertEqual(result, "System1 loop answer")
            orchestrator._run_cognition_actions.assert_awaited_once()
            orchestrator._finalize_cognition_result.assert_not_awaited()
            orchestrator._distill_cognition_episode.assert_awaited_once()
            orchestrator._persist_cognition_distillation.assert_awaited_once()
            self.assertEqual(orchestrator._generate_cognition_response.await_count, 4)

    async def test_run_cognition_loop_makes_generated_tools_available_next_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")

            class _FakeTools:
                def get_tool(self, tool_id: str):
                    if tool_id == "demo.echo":
                        return {
                            "tool_id": "demo.echo",
                            "name": "Demo Echo",
                            "description": "Echo generated text.",
                            "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
                        }
                    return None

            orchestrator._tools = _FakeTools()  # type: ignore[assignment]
            orchestrator.generate_tools_from_spec = AsyncMock(  # type: ignore[method-assign]
                return_value={
                    "status": "APPROVED",
                    "result": {
                        "summary": "Create a demo echo tool.",
                        "planned_count": 1,
                        "created_count": 1,
                        "skipped_count": 0,
                        "failed_count": 0,
                        "tools": [{"tool_id": "demo.echo", "name": "Demo Echo", "status": "APPROVED"}],
                        "run_path": str(tmp_path / "tool_generation_run.json"),
                    },
                }
            )
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"need one generated tool first\"}",
                    (
                        "{\"type\":\"act\",\"thought\":\"create the missing helper\","
                        "\"actions\":[{\"action_type\":\"generate_tools_from_spec\","
                        "\"spec\":{\"request\":\"Create a demo echo tool.\",\"namespace_prefix\":\"demo\",\"max_tools\":1},"
                        "\"label\":\"generate echo tool\"}]}"
                    ),
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"Generated tool is now available\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Create a missing helper and continue.",
                {"summary": "conversation-summary"},
                "trace-generated-tool",
                "turn-generated-tool",
            )

            self.assertEqual(result, "Generated tool is now available")
            orchestrator.generate_tools_from_spec.assert_awaited_once()
            self.assertEqual(orchestrator._generate_cognition_response.await_count, 3)
            second_system1_prompt = orchestrator._generate_cognition_response.await_args_list[2].args[0]
            self.assertIn("demo.echo", second_system1_prompt)

    async def test_run_cognition_loop_route_generation_makes_tools_available_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")

            class _FakeTools:
                def get_tool(self, tool_id: str):
                    if tool_id == "demo.echo":
                        return {
                            "tool_id": "demo.echo",
                            "name": "Demo Echo",
                            "description": "Echo generated text.",
                            "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
                        }
                    return None

            orchestrator._tools = _FakeTools()  # type: ignore[assignment]
            orchestrator.generate_tools_from_spec = AsyncMock(  # type: ignore[method-assign]
                return_value={
                    "status": "APPROVED",
                    "result": {
                        "summary": "Create a demo echo tool.",
                        "planned_count": 1,
                        "created_count": 1,
                        "skipped_count": 0,
                        "failed_count": 0,
                        "tools": [{"tool_id": "demo.echo", "name": "Demo Echo", "status": "APPROVED"}],
                        "run_path": str(tmp_path / "tool_generation_run.json"),
                    },
                }
            )
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"bootstrap capability first\","
                        "\"generate_tools_first\":true,"
                        "\"tool_generation_spec\":{\"request\":\"Create a demo echo tool.\","
                        "\"namespace_prefix\":\"demo\",\"max_tools\":1}}"
                    ),
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"Route-time generation worked\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Create a missing helper before execution.",
                {"summary": "conversation-summary"},
                "trace-route-generated-tool",
                "turn-route-generated-tool",
            )

            self.assertEqual(result, "Route-time generation worked")
            orchestrator.generate_tools_from_spec.assert_awaited_once()
            self.assertEqual(orchestrator._generate_cognition_response.await_count, 2)
            system1_prompt = orchestrator._generate_cognition_response.await_args_list[1].args[0]
            self.assertIn("demo.echo", system1_prompt)
            self.assertIn("route-time tool generation", system1_prompt)

    async def test_run_cognition_loop_skips_init_when_query_and_state_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"direct answer is enough\"}",
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"No init answer\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Analyze the mode.",
                {"summary": "conversation-summary"},
                "trace-no-init",
                "turn-no-init",
            )

            self.assertEqual(result, "No init answer")
            self.assertEqual(orchestrator._generate_cognition_response.await_count, 2)

    async def test_run_cognition_loop_uses_route_model_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._cognition_routing_skip_trivial = False
            orchestrator._cognition_route_model = "route-model"
            orchestrator._cognition_route_options = {"thinking": False, "reasoning_effort": "low"}
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"direct answer is enough\"}",
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"Route-configured answer\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Analyze the mode.",
                {"summary": "conversation-summary"},
                "trace-route-model",
                "turn-route-model",
            )

            self.assertEqual(result, "Route-configured answer")
            route_call = orchestrator._generate_cognition_response.await_args_list[0]
            self.assertEqual(route_call.kwargs["model_override"], "route-model")
            self.assertEqual(
                route_call.kwargs["options_override"],
                {"thinking": False, "reasoning_effort": "low"},
            )

    async def test_select_relevant_tools_falls_back_to_candidates_on_invalid_selector_output(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._select_relevant_tools_enabled = True
        orchestrator._tool_selection_category_mode_enabled = False
        orchestrator._generate_cognition_response = AsyncMock(
            return_value=(
                "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"missing.tool\"],"
                "\"reason\":\"bad selector output\"}"
            )
        )
        candidate_tools = [
            {"tool_id": "sys.time", "name": "Time", "description": "Read the time"},
            {"tool_id": "sys.weather", "name": "Weather", "description": "Read the weather"},
        ]

        selected_tools, trace = await orchestrator._select_relevant_tools(
            "What time is it?",
            {"summary": "conversation-summary"},
            [],
            candidate_tools,
            "trace-tool-fallback",
            "turn-tool-fallback",
        )

        self.assertEqual(selected_tools, candidate_tools)
        self.assertIsNotNone(trace)
        self.assertEqual(trace["status"], "fallback")
        self.assertEqual(trace["selected_tool_ids"], ["sys.time", "sys.weather"])

    async def test_select_relevant_tools_expands_selected_categories_when_enabled(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._select_relevant_tools_enabled = True
        orchestrator._tool_selection_category_mode_enabled = True
        orchestrator._generate_cognition_response = AsyncMock(
            return_value=(
                "{\"type\":\"tool_selection\",\"selected_category_ids\":[\"ui\"],"
                "\"selected_tool_ids\":[\"sys.time\"],\"reason\":\"screen task needs ui and time\"}"
            )
        )
        candidate_tools = [
            {"tool_id": "ui.screenshot", "tool_bucket": "ui.specialized", "name": "Screenshot"},
            {"tool_id": "ui.click", "tool_bucket": "ui.duplicate", "name": "Click"},
            {"tool_id": "sys.time", "name": "Time", "description": "Read the time"},
            {"tool_id": "sys.weather", "name": "Weather", "description": "Read the weather"},
        ]

        selected_tools, trace = await orchestrator._select_relevant_tools(
            "Inspect the screen and report the time.",
            {"summary": "conversation-summary"},
            [],
            candidate_tools,
            "trace-category-select",
            "turn-category-select",
        )

        self.assertEqual(
            [tool["tool_id"] for tool in selected_tools],
            ["ui.screenshot", "ui.click", "sys.time"],
        )
        self.assertIsNotNone(trace)
        self.assertEqual(trace["selection_mode"], "category")
        self.assertEqual(trace["selected_category_ids"], ["ui"])
        self.assertEqual(trace["selected_tool_ids"], ["ui.screenshot", "ui.click", "sys.time"])

    async def test_category_selection_splits_base_tools_by_namespace(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._select_relevant_tools_enabled = True
        orchestrator._tool_selection_category_mode_enabled = True
        orchestrator._generate_cognition_response = AsyncMock(
            return_value=(
                "{\"type\":\"tool_selection\",\"selected_category_ids\":[\"fs\"],"
                "\"selected_tool_ids\":[],\"reason\":\"filesystem task\"}"
            )
        )
        candidate_tools = [
            {"tool_id": "sys.time", "tool_bucket": "base", "name": "Time"},
            {"tool_id": "math.eval", "tool_bucket": "base", "name": "Math"},
            {"tool_id": "fs.mkdir", "tool_bucket": "base", "name": "Make Directory"},
            {"tool_id": "fs.write_file", "tool_bucket": "base", "name": "Write File"},
            {"tool_id": "ui.screenshot", "tool_bucket": "ui.specialized", "name": "Screenshot"},
        ]

        selected_tools, trace = await orchestrator._select_relevant_tools(
            "Create a file.",
            {"summary": "conversation-summary"},
            [],
            candidate_tools,
            "trace-fs-category",
            "turn-fs-category",
        )

        self.assertEqual([tool["tool_id"] for tool in selected_tools], ["fs.mkdir", "fs.write_file"])
        self.assertEqual(trace["selected_category_ids"], ["fs"])
        selector_prompt = orchestrator._generate_cognition_response.await_args.args[0]
        self.assertIn("- fs: 2 tools", selector_prompt)
        self.assertNotIn("- base:", selector_prompt)

    async def test_init_tools_needed_false_keeps_tool_selector_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = True
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._select_relevant_tools_enabled = True
            orchestrator._tool_selection_category_mode_enabled = True
            orchestrator._cognition_init_controls_tools_needed = False
            orchestrator._cognition_force_system = "system1"
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(
                return_value=[
                    {"tool_id": "fs.mkdir", "tool_bucket": "base", "name": "Make Directory", "description": "Create directory"},
                    {"tool_id": "fs.write_file", "tool_bucket": "base", "name": "Write File", "description": "Write file"},
                    {"tool_id": "sys.time", "tool_bucket": "base", "name": "Time", "description": "Read time"},
                ]
            )
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"tool_selection\",\"selected_category_ids\":[\"fs\"],"
                        "\"selected_tool_ids\":[],\"reason\":\"filesystem task\"}"
                    ),
                    (
                        "{\"type\":\"init\",\"thought\":\"tools are available but misclassified\","
                        "\"tools_needed\":false,"
                        "\"query_state\":{\"query_rewrite\":\"Create a file\",\"intent\":\"Create a file\","
                        "\"objectives\":[\"Create file\"],\"constraints\":[],\"unknowns\":[],"
                        "\"success_criteria\":[\"File exists\"],\"working_set\":[\"fs.write_file\"],\"todo\":[]}}"
                    ),
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"done\",\"query_state_delta\":{}}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Create a file",
                {"summary": "conversation-summary"},
                "trace-init-keeps-tools",
                "turn-init-keeps-tools",
            )

            self.assertEqual(result, "done")
            init_prompt = orchestrator._generate_cognition_response.await_args_list[1].args[0]
            system1_prompt = orchestrator._generate_cognition_response.await_args_list[2].args[0]
            self.assertIn("fs.mkdir", init_prompt)
            self.assertIn("fs.write_file", init_prompt)
            self.assertNotIn("sys.time", init_prompt)
            self.assertIn("AVAILABLE TOOLS:", system1_prompt)
            self.assertIn("fs.mkdir", system1_prompt)
            self.assertIn("fs.write_file", system1_prompt)
            self.assertNotIn("sys.time", system1_prompt)

    async def test_init_tools_needed_false_can_drop_selector_results_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = True
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._select_relevant_tools_enabled = True
            orchestrator._tool_selection_category_mode_enabled = True
            orchestrator._cognition_init_controls_tools_needed = True
            orchestrator._cognition_force_system = "system1"
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(
                return_value=[
                    {"tool_id": "fs.mkdir", "tool_bucket": "base", "name": "Make Directory", "description": "Create directory"},
                    {"tool_id": "fs.write_file", "tool_bucket": "base", "name": "Write File", "description": "Write file"},
                ]
            )
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"tool_selection\",\"selected_category_ids\":[\"fs\"],"
                        "\"selected_tool_ids\":[],\"reason\":\"filesystem task\"}"
                    ),
                    (
                        "{\"type\":\"init\",\"thought\":\"no tools\","
                        "\"tools_needed\":false,"
                        "\"query_state\":{\"query_rewrite\":\"Create a file\",\"intent\":\"Create a file\","
                        "\"objectives\":[\"Create file\"],\"constraints\":[],\"unknowns\":[],"
                        "\"success_criteria\":[\"File exists\"],\"working_set\":[],\"todo\":[]}}"
                    ),
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"done\",\"query_state_delta\":{}}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Create a file",
                {"summary": "conversation-summary"},
                "trace-init-drops-tools",
                "turn-init-drops-tools",
            )

            self.assertEqual(result, "done")
            init_prompt = orchestrator._generate_cognition_response.await_args_list[1].args[0]
            system1_prompt = orchestrator._generate_cognition_response.await_args_list[2].args[0]
            self.assertIn("fs.mkdir", init_prompt)
            self.assertIn("fs.write_file", init_prompt)
            self.assertNotIn("AVAILABLE TOOLS:", system1_prompt)

    async def test_run_cognition_loop_system3_prefers_system1_on_tied_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._cognition_system3_enabled = True
            orchestrator._cognition_system3_rounds = 1
            orchestrator._cognition_peer_pools = {
                "proposers": [
                    {"peer_id": "prop_a", "model": "m1", "role": "executor"},
                    {"peer_id": "prop_b", "model": "m2", "role": "planner"},
                ],
                "critics": [
                    {"peer_id": "crit_a", "model": "m3", "role": "reviewer"},
                    {"peer_id": "crit_b", "model": "m4", "role": "reviewer"},
                ],
            }
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    "{\"type\":\"route\",\"thinking_mode\":\"system3\",\"reason\":\"needs peer review\"}",
                    "{\"type\":\"proposal\",\"peer_id\":\"prop_a\",\"recommended_mode\":\"system1\",\"strategy\":\"Fast grounded answer\",\"argument\":\"Direct execution is enough\"}",
                    "{\"type\":\"proposal\",\"peer_id\":\"prop_b\",\"recommended_mode\":\"system2\",\"strategy\":\"Reflective answer\",\"argument\":\"More deliberation is safer\"}",
                    "{\"type\":\"review\",\"critic_id\":\"crit_a\",\"proposal_peer_id\":\"prop_a\",\"score\":8,\"comment\":\"strong\"}",
                    "{\"type\":\"review\",\"critic_id\":\"crit_a\",\"proposal_peer_id\":\"prop_b\",\"score\":8,\"comment\":\"strong\"}",
                    "{\"type\":\"review\",\"critic_id\":\"crit_b\",\"proposal_peer_id\":\"prop_a\",\"score\":8,\"comment\":\"strong\"}",
                    "{\"type\":\"review\",\"critic_id\":\"crit_b\",\"proposal_peer_id\":\"prop_b\",\"score\":8,\"comment\":\"strong\"}",
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"System3 picked system1\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Analyze the mode.",
                {"summary": "conversation-summary"},
                "trace-system3-tie",
                "turn-system3-tie",
            )

            self.assertEqual(result, "System3 picked system1")
            evidence = orchestrator._get_turn_execution_evidence("turn-system3-tie")
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence["source"], "cognition_system3")
            entity = evidence["entity"]
            self.assertEqual(entity["thinking_mode"], "system3")
            self.assertEqual(entity["execution_mode"], "system1")

    async def test_run_cognition_loop_force_system1_skips_pattern_and_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._cognition_force_system = "system1"
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            forced_pattern = orchestrator._normalize_cognition_pattern(
                {
                    "pattern_id": "forced_pattern",
                    "name": "Forced Pattern",
                    "description": "Should be skipped when force mode is active.",
                    "when_to_use": ["Never in this test."],
                    "source": "def forced_pattern():\n    return 'skip'\n",
                }
            )
            orchestrator._load_cognition_patterns = lambda: [forced_pattern]
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"Forced system1 answer\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "Analyze the mode.",
                {"summary": "conversation-summary"},
                "trace-force-system1",
                "turn-force-system1",
            )

            self.assertEqual(result, "Forced system1 answer")
            self.assertEqual(orchestrator._generate_cognition_response.await_count, 1)
            evidence = orchestrator._get_turn_execution_evidence("turn-force-system1")
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence["entity"]["thinking_mode"], "system1")
            self.assertEqual(evidence["entity"]["execution_mode"], "system1")

    async def test_run_cognition_loop_trivial_query_target_system0_skips_init_and_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = True
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._cognition_routing_skip_trivial = True
            orchestrator._cognition_trivial_query_target = "system0"
            orchestrator._cognition_distill_trivial_max_chars = 60
            orchestrator._cognition_system0_max_steps = 1
            orchestrator._cognition_system0_model = "system0-model"
            orchestrator._cognition_system0_options = {"thinking": False, "reasoning_effort": "low"}
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"System0 answer\",\"query_state_delta\":{},\"state_of_mind_delta\":{}}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "What time?",
                {"summary": "conversation-summary"},
                "trace-trivial-system0",
                "turn-trivial-system0",
            )

            self.assertEqual(result, "System0 answer")
            self.assertEqual(orchestrator._generate_cognition_response.await_count, 1)
            system0_call = orchestrator._generate_cognition_response.await_args_list[0]
            self.assertIn("COGNITION SYSTEM0", system0_call.args[0])
            self.assertNotIn("AVAILABLE TOOLS:", system0_call.args[0])
            self.assertNotIn("SPECIAL ORCHESTRATION ACTIONS:", system0_call.args[0])
            self.assertNotIn("\"type\":\"clarify\"", system0_call.args[0])
            self.assertNotIn("\"type\":\"act\"", system0_call.args[0])
            self.assertEqual(system0_call.kwargs["model_override"], "system0-model")
            self.assertEqual(
                system0_call.kwargs["options_override"],
                {"thinking": False, "reasoning_effort": "low"},
            )
            evidence = orchestrator._get_turn_execution_evidence("turn-trivial-system0")
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence["source"], "cognition_system0")
            self.assertEqual(evidence["entity"]["thinking_mode"], "system0")
            self.assertEqual(evidence["entity"]["execution_mode"], "system0")

    async def test_run_cognition_loop_uses_selected_tools_when_selector_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._select_relevant_tools_enabled = True
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(
                return_value=[
                    {"tool_id": "sys.time", "name": "Time", "description": "Read the time"},
                    {"tool_id": "sys.weather", "name": "Weather", "description": "Read the weather"},
                ]
            )
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"sys.time\"],"
                        "\"reason\":\"only the clock tool is relevant\"}"
                    ),
                    "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"direct answer is enough\"}",
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"Selected tool answer\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "What time is it?",
                {"summary": "conversation-summary"},
                "trace-tool-select",
                "turn-tool-select",
            )

            self.assertEqual(result, "Selected tool answer")
            self.assertEqual(orchestrator._generate_cognition_response.await_count, 3)
            system1_prompt = orchestrator._generate_cognition_response.await_args_list[2].args[0]
            self.assertIn("sys.time", system1_prompt)
            self.assertNotIn("sys.weather", system1_prompt)

    async def test_run_cognition_loop_unions_retrieved_and_selected_tools_when_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._retrieve_relevant_tools_enabled = True
            orchestrator._select_relevant_tools_enabled = True
            orchestrator._tool_selection_independent_from_retrieval = True
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._tools.list_active = lambda: [
                {"tool_id": "sys.time", "name": "Time", "description": "Read the time"},
                {"tool_id": "sys.weather", "name": "Weather", "description": "Read the weather"},
            ]
            orchestrator._retrieve_relevant_tools = AsyncMock(
                return_value=[{"tool_id": "sys.time", "name": "Time", "description": "Read the time"}]
            )
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"sys.weather\"],"
                        "\"reason\":\"weather is also relevant\"}"
                    ),
                    "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"direct answer is enough\"}",
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"Independent tool selection answer\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "What time is it and what is the weather?",
                {"summary": "conversation-summary"},
                "trace-tool-union",
                "turn-tool-union",
            )

            self.assertEqual(result, "Independent tool selection answer")
            selector_prompt = orchestrator._generate_cognition_response.await_args_list[0].args[0]
            self.assertIn("sys.time", selector_prompt)
            self.assertIn("sys.weather", selector_prompt)
            system1_prompt = orchestrator._generate_cognition_response.await_args_list[2].args[0]
            self.assertIn("sys.time", system1_prompt)
            self.assertIn("sys.weather", system1_prompt)

    async def test_run_cognition_loop_keeps_retrieved_tools_on_independent_selector_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._retrieve_relevant_tools_enabled = True
            orchestrator._select_relevant_tools_enabled = True
            orchestrator._tool_selection_independent_from_retrieval = True
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._tools.list_active = lambda: [
                {"tool_id": "sys.time", "name": "Time", "description": "Read the time"},
                {"tool_id": "sys.weather", "name": "Weather", "description": "Read the weather"},
            ]
            orchestrator._retrieve_relevant_tools = AsyncMock(
                return_value=[{"tool_id": "sys.time", "name": "Time", "description": "Read the time"}]
            )
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock(
                side_effect=[
                    (
                        "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"missing.tool\"],"
                        "\"reason\":\"invalid\"}"
                    ),
                    (
                        "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"missing.tool\"],"
                        "\"reason\":\"still invalid\"}"
                    ),
                    "{\"type\":\"route\",\"thinking_mode\":\"system1\",\"reason\":\"direct answer is enough\"}",
                    "{\"type\":\"final\",\"thought\":\"done\",\"text\":\"Selector fallback answer\"}",
                ]
            )

            result = await orchestrator._run_cognition_loop(
                "What time is it?",
                {"summary": "conversation-summary"},
                "trace-tool-union-fallback",
                "turn-tool-union-fallback",
            )

            self.assertEqual(result, "Selector fallback answer")
            system1_prompt = orchestrator._generate_cognition_response.await_args_list[3].args[0]
            self.assertIn("sys.time", system1_prompt)
            self.assertNotIn("sys.weather", system1_prompt)

    async def test_run_cognition_loop_force_system3_returns_explicit_error_when_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            orchestrator = Orchestrator()
            orchestrator._cognition_query_state_enabled = False
            orchestrator._cognition_state_of_mind_enabled = False
            orchestrator._cognition_force_system = "system3"
            orchestrator._cognition_system3_enabled = False
            orchestrator._cognition_peer_pools = {"proposers": [], "critics": []}
            orchestrator._memory = _FakeRetrievalMemory()
            orchestrator._retrieve_relevant_tools = AsyncMock(return_value=[])
            orchestrator._load_cognition_skills = lambda: []
            orchestrator._load_cognition_patterns = lambda: []
            orchestrator._load_cognition_safety_rules = lambda: []
            orchestrator._distill_cognition_episode = AsyncMock(return_value={"type": "distill"})
            orchestrator._persist_cognition_distillation = AsyncMock(
                return_value={"skills": 0, "facts": 0, "patterns": 0, "lessons": 0, "safety_rules": 0}
            )
            orchestrator._cognition_episodes_path = str(tmp_path / "episodes.jsonl")
            orchestrator._generate_cognition_response = AsyncMock()

            result = await orchestrator._run_cognition_loop(
                "Analyze the mode.",
                {"summary": "conversation-summary"},
                "trace-force-system3",
                "turn-force-system3",
            )

            self.assertIn("System3 is forced but unavailable", result)
            orchestrator._generate_cognition_response.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
