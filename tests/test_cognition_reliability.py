import asyncio
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator.main import Orchestrator
from orchestrator.cognition_reliability import ACTIVE_RUN, RunStore, evaluate_checks, CognitionBudgetExceeded


CHECK = {"id": "file-content", "tool_id": "fs.read_file", "args": {"path": "result.txt"},
         "path": ["result", "content"], "expected": "42"}


def observation(content="42", status="APPROVED", evidence_id="tool-1"):
    return {"evidence_id": evidence_id, "action": {"tool_id": "fs.read_file", "args": {"path": "result.txt"}},
            "observation": {"status": status, "result": {"content": content}}}


class AcceptanceCheckTests(unittest.TestCase):
    def test_successful_call_does_not_imply_correct_result(self):
        self.assertFalse(evaluate_checks([CHECK], [observation("wrong")])[0]["passed"])
        self.assertTrue(evaluate_checks([CHECK], [observation()])[0]["passed"])

    def test_latest_failure_invalidates_earlier_success(self):
        self.assertFalse(evaluate_checks([CHECK], [observation(), observation(status="ERROR")])[0]["passed"])

    def test_wrong_artifact_and_missing_fields_do_not_pass(self):
        item = observation()
        item["action"]["args"]["path"] = "wrong.txt"
        self.assertFalse(evaluate_checks([CHECK], [item])[0]["passed"])
        self.assertFalse(evaluate_checks([{**CHECK, "path": ["result", "missing"]}], [observation()])[0]["passed"])

    def test_approval_envelope_cannot_hide_failed_test_command(self):
        item = observation()
        item["observation"]["result"]["exit_code"] = 1
        self.assertFalse(evaluate_checks([CHECK], [item])[0]["passed"])

    def test_budget_marker_is_not_execution_evidence(self):
        self.assertTrue(evaluate_checks([CHECK], [observation(), {"action": None, "observation": {"status": "BUDGET_EXHAUSTED"}}])[0]["passed"])

    def test_checkpoint_identifier_cannot_escape_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            store.write("runs", "../../outside", {"original_request": "request"})
            self.assertEqual(store.read("runs", "../../outside")["original_request"], "request")
            self.assertEqual(len(list(Path(tmp).rglob("*.json"))), 1)

    def test_checkpoint_has_only_one_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            with store.lease("run"):
                with self.assertRaisesRegex(RuntimeError, "already executing"):
                    with store.lease("run"):
                        self.fail("Concurrent execution was allowed")
            with store.lease("run"):
                pass


class CognitionReliabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.o = Orchestrator()
        self.o._cognition_episodes_path = str(Path(self.tmp.name) / "episodes.jsonl")
        self.o._cognition_system0_enabled = False
        self.o._cognition_distill_enabled = False
        self.o._cognition_query_state_enabled = False
        self.o._cognition_state_of_mind_enabled = False
        self.o._settings.orchestrator["cognition_max_segments"] = 3
        self.o._refresh_cognition_time_context = AsyncMock()
        self.o._compute_semantic_context_summary = AsyncMock(return_value="")
        self.o._memory = AsyncMock()
        self.o._memory.retrieve.return_value = []
        self.o._retrieve_relevant_tools = AsyncMock(return_value=[])
        self.o._load_cognition_skills = lambda: []
        self.o._load_cognition_patterns = lambda: []
        self.o._load_cognition_safety_rules = lambda: []
        self.o._final_response_critic_enabled = True
        self.o._final_response_critic_skip_modes = set()
        self.o._final_response_critic_max_retries = 0

    async def candidate(self, checks=None):
        self.o._generate_cognition_response = AsyncMock(return_value='{"type":"final","text":"Done"}')
        return await self.o._run_cognition_loop("Write 42 to result.txt and verify it", {}, "run", "turn",
                                                forced_execution_mode="system1", acceptance_checks=checks)

    async def review(self, **kwargs):
        self.o._evaluate_final_response_critic = AsyncMock(return_value={
            "fulfilled": True, "confidence": .99, "evidence_ids": ["tool-1"], "issues": [], **kwargs})
        return await self.o._apply_final_response_critic("Done", "COGNITION", {}, "run", "turn")

    async def test_rejected_final_is_never_returned_as_success(self):
        await self.candidate()
        result = await self.review(fulfilled=False, issues=["File does not exist"])
        self.assertNotEqual(result, "Done")
        self.assertIn("couldn't verify", result)
        self.assertEqual(self.o._run_store().read("runs", "run")["status"], "incomplete")

    async def test_model_approval_cannot_override_failed_acceptance_check(self):
        await self.candidate([CHECK])
        self.o._cognition_runs["turn"]["tool_evidence"] = [observation("wrong")]
        self.assertIn("couldn't verify", await self.review())

    async def test_verified_and_model_reviewed_are_distinct(self):
        await self.candidate([CHECK])
        self.o._cognition_runs["turn"]["tool_evidence"] = [observation()]
        self.assertEqual(await self.review(), "Done")
        self.assertEqual(self.o._cognition_runs["turn"]["status"], "verified")
        self.o._cognition_runs["turn"]["checks"] = []
        self.o._cognition_runs["turn"]["status"] = "candidate"
        await self.review()
        self.assertEqual(self.o._cognition_runs["turn"]["status"], "reviewed")

    async def test_fabricated_evidence_id_cannot_pass_review(self):
        await self.candidate()
        self.o._cognition_runs["turn"]["tool_evidence"] = [observation()]
        self.assertIn("couldn't verify", await self.review(evidence_ids=["invented"]))

    async def test_missing_execution_cannot_be_approved(self):
        await self.candidate()
        self.assertIn("couldn't verify", await self.review(requires_execution=True, evidence_ids=[]))

    async def test_critic_failure_is_incomplete(self):
        await self.candidate()
        self.o._evaluate_final_response_critic = AsyncMock(side_effect=RuntimeError("offline"))
        result = await self.o._apply_final_response_critic("Done", "COGNITION", {}, "run", "turn")
        self.assertIn("couldn't verify", result)

    async def test_retry_preserves_original_request_state_and_observations(self):
        await self.candidate([CHECK])
        run = self.o._cognition_runs["turn"]
        run["query_state"] = {"constraints": ["Never overwrite other files"]}
        run["observations"] = [observation("wrong")]
        self.o._checkpoint_cognition(run)
        self.o._final_response_critic_max_retries = 1
        self.o._evaluate_final_response_critic = AsyncMock(return_value={
            "fulfilled": False, "confidence": .9, "issues": ["Wrong file contents"], "fix_instructions": ["Repair file"]})
        await self.o._apply_final_response_critic("Done", "COGNITION", {}, "run", "turn")
        resumed = self.o._cognition_runs["turn"]
        self.assertEqual(resumed["original_request"], run["original_request"])
        self.assertEqual(resumed["query_state"]["constraints"], ["Never overwrite other files"])
        self.assertEqual(resumed["observations"], [observation("wrong")])
        self.assertEqual(resumed["segments"], 2)
        self.assertEqual(resumed["feedback"]["fix_instructions"], ["Repair file"])

    async def test_successful_mutation_is_not_repeated_after_restart(self):
        await self.candidate()
        run = self.o._cognition_runs["turn"]
        self.o._call_tool_unjournaled = AsyncMock(return_value={"status": "APPROVED", "result": {"ok": True}})
        token = ACTIVE_RUN.set(run)
        try:
            first = await self.o._call_tool("test.send", {"body": "hello"}, None, "run", [])
        finally:
            ACTIVE_RUN.reset(token)
        restored = self.o._run_store().read("runs", "run")
        restored["segments"] += 1
        token = ACTIVE_RUN.set(restored)
        try:
            second = await self.o._call_tool("test.send", {"body": "hello"}, None, "run", [])
        finally:
            ACTIVE_RUN.reset(token)
        self.assertEqual(first, second)
        self.o._call_tool_unjournaled.assert_awaited_once()
        self.assertEqual(restored["actions_used"], 1)

    async def test_interrupted_mutation_is_not_replayed(self):
        await self.candidate()
        self.o._call_tool_unjournaled = AsyncMock(side_effect=asyncio.CancelledError())
        token = ACTIVE_RUN.set(self.o._cognition_runs["turn"])
        try:
            with self.assertRaises(asyncio.CancelledError):
                await self.o._call_tool("test.send", {}, None, "run", [])
        finally:
            ACTIVE_RUN.reset(token)
        token = ACTIVE_RUN.set(self.o._run_store().read("runs", "run"))
        try:
            result = await self.o._call_tool("test.send", {}, None, "run", [])
        finally:
            ACTIVE_RUN.reset(token)
        self.assertEqual(result["error_details"]["code"], "UNCERTAIN_EXECUTION")
        self.o._call_tool_unjournaled.assert_awaited_once()

    async def test_system1_exhaustion_escalates_with_observations(self):
        self.o._cognition_system1_max_steps = 1
        self.o._run_cognition_actions = AsyncMock(return_value=[observation()])
        self.o._generate_cognition_response = AsyncMock(side_effect=[
            '{"type":"route","thinking_mode":"system1"}',
            '{"type":"act","actions":[{"tool_id":"fs.read_file","args":{"path":"result.txt"}}]}',
            '{"type":"final","text":"42"}',
        ])
        result = await self.o._run_cognition_loop("Read result", {}, "run", "turn")
        self.assertEqual(result, "42")
        self.assertEqual(self.o._cognition_runs["turn"]["execution_mode"], "system2")
        self.assertIn("42", self.o._generate_cognition_response.await_args_list[-1].args[0])

    async def test_continuation_is_bounded_and_preserves_progress(self):
        self.o._cognition_system2_max_steps = 1
        self.o._finalize_cognition_result = AsyncMock(return_value={"type": "final", "text": "Partial"})
        self.o._run_cognition_actions = AsyncMock(side_effect=[
            [observation(evidence_id=f"tool-{index}")] for index in range(1, 4)])
        self.o._generate_cognition_response = AsyncMock(return_value=
            '{"type":"step","actions":[{"tool_id":"fs.read_file","args":{"path":"result.txt"}}]}')
        result = await self.o._run_cognition_loop("Read result", {}, "run", "turn", forced_execution_mode="system2")
        self.assertIn("couldn't verify", result)
        run = self.o._cognition_runs["turn"]
        self.assertEqual(run["segments"], 3)
        self.assertEqual(len(run["observations"]), 3)
        calls = self.o._generate_cognition_response.await_count
        await self.o._run_cognition_loop(run["original_request"], {}, "run", "turn", resume_checkpoint="run")
        self.assertEqual(self.o._generate_cognition_response.await_count, calls)

    async def test_provisional_knowledge_is_invisible_until_review_and_retractable(self):
        await self.candidate([CHECK])
        self.o._cognition_distill_enabled = True
        self.o._distill_cognition_episode = AsyncMock(return_value={
            "memory_facts": [{"name": "result", "value": "42", "confidence": .99}],
            "patterns": [{"pattern_id": "pattern.read_result", "name": "read_result", "confidence": .99,
                          "source": "def read_result():\n    return '42'"}]})
        run = self.o._cognition_runs["turn"]
        await self.o._stage_cognition_learning(run, "run", "turn")
        candidate_id = run["learning_candidate_id"]
        self.assertEqual(self.o._reviewed_learning_facts("result"), [])
        with self.assertRaises(ValueError):
            await self.o.review_cognition_learning(candidate_id, reviewer="tester", reason="Reviewed", approve=True)
        run["tool_evidence"] = [observation()]
        await self.review()
        await self.o.review_cognition_learning(candidate_id, reviewer="tester", reason="Inspected source and verified fact", approve=True)
        self.assertEqual(self.o._reviewed_learning_facts("result")[0]["data"]["value"], "42")
        self.assertTrue(self.o._reviewed_learning_items("patterns", self.o._normalize_cognition_pattern))
        self.o.retract_cognition_learning(candidate_id, reviewer="tester", reason="No longer valid")
        self.assertEqual(self.o._reviewed_learning_facts("result"), [])
        self.assertEqual(self.o._reviewed_learning_items("patterns", self.o._normalize_cognition_pattern), [])
        self.o._memory.store_fact.assert_not_awaited()

    async def test_original_request_is_available_to_critic_with_state_disabled(self):
        await self.candidate()
        self.o._generate_response = AsyncMock(return_value='{"fulfilled":false,"confidence":1}')
        await self.o._evaluate_final_response_critic("Done", "COGNITION", "run", "turn")
        self.assertIn("Write 42 to result.txt and verify it", self.o._generate_response.await_args.args[0])

    async def test_global_model_budget_is_shared_by_reasoning_and_critic(self):
        await self.candidate()
        run = self.o._cognition_runs["turn"]
        run["model_calls"] = run["max_model_calls"]
        token = ACTIVE_RUN.set(run)
        try:
            with self.assertRaises(CognitionBudgetExceeded):
                await Orchestrator._generate_cognition_response(self.o, "prompt")
            with self.assertRaises(CognitionBudgetExceeded):
                await self.o._generate_response("critic", "COGNITION", {})
        finally:
            ACTIVE_RUN.reset(token)

    async def test_global_tool_budget_is_not_reset_by_continuation(self):
        await self.candidate()
        run = self.o._cognition_runs["turn"]
        run["actions_used"] = run["max_actions"]
        self.o._call_tool_unjournaled = AsyncMock()
        token = ACTIVE_RUN.set(run)
        try:
            result = await self.o._call_tool("test.send", {}, None, "run", [])
        finally:
            ACTIVE_RUN.reset(token)
        self.assertEqual(result["error_details"]["code"], "BUDGET_EXHAUSTED")
        self.o._call_tool_unjournaled.assert_not_awaited()

    async def test_successful_old_observation_cannot_hide_later_failure(self):
        await self.candidate()
        self.o._cognition_runs["turn"]["observations"] = [observation(), observation(status="ERROR", evidence_id="tool-2")]
        self.assertIn("couldn't verify", await self.review())

    async def test_checkpoint_cannot_replace_the_original_task_or_checks(self):
        await self.candidate([CHECK])
        with self.assertRaises(ValueError):
            await self.o._run_cognition_loop("Different request", {}, "run", "turn", resume_checkpoint="run")
        with self.assertRaises(ValueError):
            await self.o._run_cognition_loop("Write 42 to result.txt and verify it", {}, "run", "turn",
                                             resume_checkpoint="run", acceptance_checks=[])

    async def test_expired_time_budget_does_not_restart_generation(self):
        await self.candidate()
        run = self.o._cognition_runs["turn"]
        run["elapsed_s"] = run["timeout_s"] + 1
        self.o._checkpoint_cognition(run)
        before = self.o._generate_cognition_response.await_count
        result = await self.o._run_cognition_loop(run["original_request"], {}, "run", "turn", resume_checkpoint="run")
        self.assertIn("couldn't verify", result)
        self.assertEqual(before, self.o._generate_cognition_response.await_count)

    async def test_process_turn_emits_verified_completion_and_skips_legacy_distillation(self):
        self.o._pending_turns["turn"] = "Read result.txt and report its contents"
        self.o._pending_turn_modes["turn"] = "SYSTEM1"
        self.o._pending_acceptance_checks["turn"] = [CHECK]
        self.o._generate_cognition_response = AsyncMock(side_effect=[
            '{"type":"act","actions":[{"tool_id":"fs.read_file","args":{"path":"result.txt"}}]}',
            '{"type":"final","text":"42"}',
        ])
        self.o._run_cognition_actions = AsyncMock(return_value=[observation()])
        self.o._evaluate_final_response_critic = AsyncMock(return_value={
            "fulfilled": True, "confidence": .99, "evidence_ids": ["tool-1"]})
        self.o._send_event = AsyncMock()
        self.o._store_episodic_and_maybe_distill = AsyncMock()
        await self.o._process_turn("turn")
        final = self.o._send_event.await_args.args[0]["payload"]
        self.assertEqual(final["text"], "42")
        self.assertEqual(final["completion"]["status"], "verified")
        self.assertIn("checkpoint_id", final)
        self.assertTrue(self.o._store_episodic_and_maybe_distill.await_args.kwargs["skip_distillation"])

    def test_system3_blocking_review_cannot_be_averaged_away(self):
        proposals = [{"peer_id": "p", "recommended_mode": "system2"}]
        reviews = [{"critic_id": "a", "proposal_peer_id": "p", "score": 10, "blocking": True},
                   {"critic_id": "b", "proposal_peer_id": "p", "score": 10}]
        self.assertIsNone(self.o._select_cognition_system3_winner(proposals, reviews, "run"))

    async def test_explicit_system3_request_does_not_silently_run_system2(self):
        self.o._cognition_system3_enabled = False
        self.o._generate_cognition_response = AsyncMock()
        result = await self.o._run_cognition_loop("Compare approaches", {}, "run", "turn", forced_execution_mode="system3")
        self.assertIn("unavailable", result)
        self.o._generate_cognition_response.assert_not_awaited()

    def test_nonfinite_critic_confidence_is_not_success(self):
        result = self.o._parse_final_response_critic('{"fulfilled":true,"confidence":NaN}')
        self.assertEqual(result["confidence"], 0)
