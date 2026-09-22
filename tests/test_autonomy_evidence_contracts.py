"""Regression tests for evidence, belief history, and bounded autonomous execution."""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autonomy.cognition_client import CognitionClient
from epistemic.layer3_analyzer import Analyzer
from epistemic.layer4_optimizer import Optimizer
from epistemic.main_loop import EpistemicLoop
from epistemic.types import Observation, PredictionError, Theory
from power_process.layer6_executor import Executor
from power_process.layer7_evaluator import Evaluator
from power_process.power_loop import PowerProcessLoop
from power_process.types import Goal, GoalExecution
from tests.test_epistemic_power_process import _FakeWorldModel
from tests.test_autonomy_cognition_client import _FakeSession, _settings


class BeliefEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_curiosity_never_creates_a_theory(self):
        world = _FakeWorldModel()
        observation = Observation("obs", "CURIOSITY_VOID", "failed", "connection timed out", False)
        loop = EpistemicLoop(world_model=world,
            explorer=SimpleNamespace(explore=AsyncMock(return_value=observation)),
            analyzer=SimpleNamespace(analyze=AsyncMock(return_value=PredictionError(
                False, "invented", 1.0, 0.2, "An unsupported idea", verdict="supports"))),
            optimizer=Optimizer(world))
        result = await loop.run_cycle()
        self.assertEqual(world.added_theories, [])
        self.assertEqual(world.updated_theories, [])
        self.assertEqual(result.prediction_error.verdict, "inconclusive")
        self.assertEqual(len(world.recorded_epistemic_observations), 1)

    async def test_analyzer_does_not_call_model_for_failed_observation(self):
        client = SimpleNamespace(generate_model=AsyncMock())
        result = await Analyzer(client).analyze(Theory("t", "claim"), Observation("o", "t", "read", "timeout", False))
        client.generate_model.assert_not_awaited()
        self.assertEqual(result.verdict, "inconclusive")

    def test_inconclusive_data_does_not_consume_attempts(self):
        world = _FakeWorldModel()
        theory = Theory("t", "claim", .4)
        world.theories[theory.id] = theory
        result = Optimizer(world).optimize(theory, Observation("o", "t", "read", "unclear"),
            PredictionError(False, "no conclusion", 0.0, -0.9))
        self.assertFalse(result.theory_updated)
        self.assertEqual(theory.attempts, 0)
        self.assertEqual(theory.confidence_score, .4)

    def test_rejected_belief_and_contradicting_idea_keep_their_history(self):
        world = _FakeWorldModel()
        theory = Theory("old", "The room is always empty", .1)
        world.theories[theory.id] = theory
        error = PredictionError(False, "Observed a person", 1.0, -10.0,
            "People sometimes enter the room", verdict="contradicts",
            hypothesis_relation="CONTRADICTS", concepts=["room", "people"])
        result = Optimizer(world).optimize(theory, Observation("o", "old", "observe", "person present"), error)
        self.assertTrue(result.theory_retired)
        self.assertEqual(theory.status, "retired")
        self.assertEqual(world.deleted_theories, [])
        self.assertEqual(world.enrichments[0][1]["parent_id"], "old")
        self.assertEqual(world.enrichments[0][1]["relation"], "CONTRADICTS")
        self.assertEqual(world.enrichments[0][1]["observation_id"], "o")

    def test_confidence_updates_are_bounded_and_use_current_belief(self):
        world = _FakeWorldModel()
        world.theories["t"] = Theory("t", "claim", .4)
        stale_snapshot = Theory("t", "claim", .1)
        result = Optimizer(world).optimize(stale_snapshot, Observation("o", "t", "read", "result"),
            PredictionError(True, "supports", 1.0, 1e10, verdict="supports"))
        self.assertAlmostEqual(result.updated_confidence, .65)

    def test_contradiction_cannot_raise_confidence(self):
        world = _FakeWorldModel()
        theory = world.theories["t"] = Theory("t", "claim", .4)
        result = Optimizer(world).optimize(theory, Observation("o", "t", "read", "result"),
            PredictionError(False, "contradiction", 1.0, .2, verdict="contradicts"))
        self.assertEqual(result.updated_confidence, .4)

    def test_stale_exhausted_belief_can_be_selected(self):
        world = _FakeWorldModel()
        world.revalidation = Theory("old", "old claim", .9, attempts=3)
        loop = EpistemicLoop(world_model=world, explorer=None, analyzer=None, optimizer=None)
        theory, topic = loop._select_focus()
        self.assertEqual(theory.id, "old")
        self.assertIsNone(topic)
        loop._cycles_since_curiosity = 5
        self.assertIsNone(loop._select_focus()[0])

    def test_malformed_values_do_not_become_positive_evidence(self):
        error = PredictionError.from_dict({"match": "false", "confidence_in_data": float("nan"),
            "suggested_confidence_adjustment": float("inf")})
        self.assertFalse(error.match)
        self.assertEqual(error.verdict, "inconclusive")
        self.assertEqual(error.suggested_confidence_adjustment, 0)
        self.assertFalse(Observation.from_dict({"ok": "false"}).ok)


class GoalCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def evaluate(self, score, ok=True):
        world = _FakeWorldModel()
        class Client:
            async def generate_model(self, prompt, cls):
                return cls(completion_score=score, reasoning="assessment", new_theory="lesson")
        result = await Evaluator(world, cognition_client=Client()).evaluate(
            Goal("g", "Inspect room", "learn", success_criteria="Identify room contents"),
            GoalExecution("observe", "contents", ok))
        return world, result

    async def test_partial_progress_stays_pending(self):
        world, result = await self.evaluate(.7)
        self.assertEqual(result.goal_status, "pending")
        self.assertEqual(result.completion_score, .7)

    async def test_failed_execution_cannot_complete_or_inject_belief(self):
        world, result = await self.evaluate(1.0, False)
        self.assertEqual(result.goal_status, "pending")
        self.assertEqual(result.completion_score, 0)
        self.assertEqual(world.added_theories, [])

    async def test_full_success_completes(self):
        _, result = await self.evaluate(1.0)
        self.assertEqual(result.goal_status, "completed")

    async def test_retry_receives_history_and_model_does_not_hold_graph_lock(self):
        world = _FakeWorldModel()
        world.pending_goal = Goal("g", "inspect", "learn", attempts=1)
        world.get_goal_history = lambda _: [{"action_taken": "read missing file", "ok": False}]
        lock = asyncio.Lock()
        class Client:
            async def generate_model(self, prompt, cls, **kwargs):
                self.prompt = prompt
                self.constraints = kwargs
                self.locked = lock.locked()
                return cls("read different file", "result", True)
        client = Client()
        evaluator = SimpleNamespace(evaluate=AsyncMock(return_value=SimpleNamespace()))
        loop = PowerProcessLoop(world_model=world, motivator=None,
            executor=Executor(client), evaluator=evaluator, write_lock=lock)
        await loop.run_cycle()
        self.assertIn("read missing file", client.prompt)
        self.assertFalse(client.locked)
        self.assertEqual(client.constraints["max_actions"], 3)

    async def test_replayed_execution_returns_committed_outcome(self):
        world = _FakeWorldModel()
        class Client:
            score = .7
            async def generate_model(self, prompt, cls):
                return cls(completion_score=self.score, reasoning="assessment")
        client = Client()
        evaluator = Evaluator(world, cognition_client=client)
        goal = Goal("g", "inspect", "learn")
        execution = GoalExecution("read", "result")
        first = await evaluator.evaluate(goal, execution)
        client.score = 1.0
        second = await evaluator.evaluate(goal, execution)
        self.assertEqual(second.goal_status, first.goal_status)
        self.assertEqual(second.completion_score, .7)
        self.assertEqual(len(world.goal_status_updates), 1)

    async def test_persistence_failure_does_not_repeat_the_action(self):
        world = _FakeWorldModel()
        world.pending_goal = Goal("g", "inspect", "learn")
        executor = SimpleNamespace(execute_goal=AsyncMock(return_value=GoalExecution("read", "data")))
        evaluator = SimpleNamespace(evaluate=AsyncMock(side_effect=[RuntimeError("database unavailable"), SimpleNamespace()]))
        loop = PowerProcessLoop(world_model=world, motivator=None, executor=executor, evaluator=evaluator)
        with self.assertRaisesRegex(RuntimeError, "database"):
            await loop.run_cycle()
        await loop.run_cycle()
        self.assertEqual(executor.execute_goal.await_count, 1)
        self.assertEqual(evaluator.evaluate.await_count, 2)



class TransportEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_fabricated_model_evidence_is_replaced_by_transport_results(self):
        _FakeSession.instances = []
        evidence = [{"request_id": "real", "status": "APPROVED", "tool_id": "read", "result": "file"}]
        _FakeSession.responses = [
            {"text": "Read a file", "tool_evidence": evidence},
            {"text": json.dumps({"action_taken": "read", "sensor_data": "file", "ok": True,
                                  "evidence": [{"request_id": "invented"}]})},
        ]
        with patch("autonomy.cognition_client.OrchestratorSession", _FakeSession):
            result = await CognitionClient(settings=_settings()).generate_model(
                "read", GoalExecution, max_actions=3, require_evidence=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.evidence, evidence)
        self.assertEqual(_FakeSession.instances[0].constraints["max_actions"], 3)
        self.assertEqual(_FakeSession.instances[1].constraints["max_actions"], 0)
        self.assertTrue(all(session.skip_distillation for session in _FakeSession.instances))

    async def test_success_claim_without_tool_evidence_is_inconclusive(self):
        result = CognitionClient._attach_evidence({"ok": True, "sensor_data": "I verified it"}, [])
        self.assertFalse(result["ok"])


class ExecutionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from orchestrator.main import Orchestrator
        from orchestrator.policy import PolicyEngine
        self.orch = Orchestrator.__new__(Orchestrator)
        self.orch._tools = SimpleNamespace(get_tool=lambda _: {"tool_id": "test", "required_permissions": ["tier0"]})
        self.orch._policy = PolicyEngine()
        self.orch._auto_approve_all = True
        self.orch._tool_host, self.orch._tool_port = "localhost", 1
        self.orch._model_rpc_timeout_s = 1

    async def test_dispatch_budget_is_enforced_and_results_are_recorded(self):
        from orchestrator.main import _CURRENT_AUTONOMY
        context = {"remaining": 1, "permission_policy": "deny", "evidence": []}
        token = _CURRENT_AUTONOMY.set(context)
        try:
            with patch("orchestrator.main.send_request", AsyncMock(return_value={"status": "APPROVED", "result": "actual"})) as send, patch("orchestrator.main.record_event"):
                first = await self.orch._call_tool("read", {}, None, "trace", [])
                second = await self.orch._call_tool("read", {}, None, "trace", [])
            self.assertEqual(first["status"], "APPROVED")
            self.assertEqual(second["status"], "DENIED")
            self.assertEqual(send.await_count, 1)
            self.assertEqual(context["evidence"][0]["result"], "actual")
        finally:
            _CURRENT_AUTONOMY.reset(token)

    async def test_global_auto_approval_cannot_override_autonomy_deny(self):
        from orchestrator.main import _CURRENT_AUTONOMY
        self.orch._tools.get_tool = lambda _: {"required_permissions": ["tier1"]}
        token = _CURRENT_AUTONOMY.set({"remaining": 3, "permission_policy": "deny", "evidence": []})
        try:
            with patch.object(self.orch, "_call_tool", AsyncMock()) as call:
                result = await self.orch._execute_tool_action("write", {}, "trace")
            self.assertEqual(result["observation"]["status"], "DENIED")
            call.assert_not_awaited()
        finally:
            _CURRENT_AUTONOMY.reset(token)

    async def test_evaluation_turn_cannot_dispatch_tools(self):
        from orchestrator.main import _CURRENT_AUTONOMY
        token = _CURRENT_AUTONOMY.set({"remaining": 0, "permission_policy": "approve", "evidence": []})
        try:
            with patch("orchestrator.main.send_request", AsyncMock()) as send:
                result = await self.orch._call_tool("read", {}, None, "trace", [])
            self.assertEqual(result["status"], "DENIED")
            send.assert_not_awaited()
        finally:
            _CURRENT_AUTONOMY.reset(token)


if __name__ == "__main__":
    unittest.main()
