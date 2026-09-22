from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autonomy.json_utils import parse_json_object
from epistemic.layer2_exploration import Explorer
from epistemic.layer4_optimizer import Optimizer
from epistemic.main_loop import EpistemicLoop
from epistemic.types import Observation, PredictionError, Theory
from power_process.layer5_motivator import Motivator
from power_process.layer7_evaluator import Evaluator
from power_process.power_loop import PowerProcessLoop
from power_process.types import ExecutionOutcome, Goal, GoalExecution


class JsonUtilsTests(unittest.TestCase):
    def test_parse_json_object_handles_fenced_payload_with_trailing_comma(self) -> None:
        raw = """
        analysis
        ```json
        {"match": true, "reasoning": "ok",}
        ```
        """
        payload = parse_json_object(raw)
        self.assertTrue(payload["match"])
        self.assertEqual(payload["reasoning"], "ok")


class ExplorerTests(unittest.IsolatedAsyncioTestCase):
    async def test_explorer_translates_cognition_output_into_observation(self) -> None:
        class _FakeCognitionClient:
            async def generate_model(self, prompt, model_cls, *, timeout=60, **kwargs):
                self.prompt = prompt
                self.timeout = timeout
                return model_cls(
                    action_taken="Read the current directory with a shell tool.",
                    sensor_data="Observed files: README.md, src/, config/",
                    ok=True,
                )

        theory = Theory(id="theory-1", text="The workspace contains source code.")
        explorer = Explorer(cognition_client=_FakeCognitionClient())

        observation = await explorer.explore(theory)

        self.assertEqual(observation.theory_id, "theory-1")
        self.assertIn("current directory", observation.action_taken)
        self.assertIn("src/", observation.sensor_data)
        self.assertTrue(observation.ok)


class _FakeWorldModel:
    def __init__(self) -> None:
        self.updated_theories = []
        self.deleted_theories = []
        self.added_theories = []
        self.added_goals = []
        self.goal_status_updates = []
        self.pending_goal = None
        self.grounded_theories = []
        self.recorded_epistemic_observations = []
        self.recorded_goal_executions = []
        self.goal_theory_links = []
        self.theories = {}
        self.enrichments = []
        self.revalidation = None

    def has_observation(self, observation_id):
        return any(item[1].id == observation_id for item in self.recorded_epistemic_observations) or any(
            item[1].id == observation_id for item in self.recorded_goal_executions)

    def run_transaction(self, operation):
        return operation(self)

    def get_theory(self, theory_id):
        return self.theories.get(theory_id)

    def get_revalidation_theory(self):
        return self.revalidation

    def enrich_theory(self, theory_id, **kwargs):
        self.enrichments.append((theory_id, kwargs))

    def get_goal_execution_outcome(self, observation_id):
        return next((item[2] for item in self.recorded_goal_executions if item[1].id == observation_id), None)

    def get_goal_history(self, goal_id):
        return []

    def get_theory_context(self, theory_id):
        return []

    def get_untested_theory(self, *, max_attempts=None):
        return None

    def add_theory(self, text: str, *, deduplicate: bool = True, source_observation_id: str | None = None):
        theory = Theory(id=f"theory-{len(self.added_theories) + 1}", text=text)
        self.added_theories.append((text, deduplicate, source_observation_id, theory))
        return theory

    def update_theory(self, theory_id: str, new_confidence: float, success: bool) -> None:
        self.updated_theories.append((theory_id, new_confidence, success))
        if theory_id in self.theories:
            theory = self.theories[theory_id]
            theory.confidence_score = new_confidence
            theory.attempts += 1
            theory.status = "retired" if new_confidence <= 0 else "active"

    def delete_theory(self, theory_id: str) -> None:
        self.deleted_theories.append(theory_id)

    def get_grounded_theories(self, *, limit: int = 5, min_confidence: float = 0.3):
        return list(self.grounded_theories[:limit])

    def add_goal(
        self,
        text: str,
        reasoning: str,
        *,
        deduplicate_pending: bool = True,
        grounding_theory_ids=None,
        success_criteria="",
    ):
        goal = Goal(id=f"goal-{len(self.added_goals) + 1}", text=text, reasoning=reasoning)
        normalized_grounding = list(grounding_theory_ids or [])
        self.added_goals.append((text, reasoning, deduplicate_pending, normalized_grounding, goal))
        if normalized_grounding:
            self.goal_theory_links.append((goal.id, normalized_grounding))
        return goal

    def update_goal_status(self, goal_id: str, new_status: str) -> None:
        self.goal_status_updates.append((goal_id, new_status))

    def get_pending_goal(self):
        return self.pending_goal

    def record_epistemic_observation(self, theory, observation, error, *, mode: str, focus_text: str):
        self.recorded_epistemic_observations.append((theory, observation, error, mode, focus_text))
        return observation.id

    def record_goal_execution(self, goal, execution, outcome):
        observation_id = f"goal-obs-{len(self.recorded_goal_executions) + 1}"
        self.recorded_goal_executions.append((goal, execution, outcome, observation_id))
        return observation_id


class EpistemicLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_curiosity_cycle_persists_observation_and_injects_theory(self) -> None:
        world_model = _FakeWorldModel()
        optimizer = Optimizer(world_model)

        class _FakeExplorer:
            async def explore(self, theory, curiosity_topic=None):
                return Observation(
                    id="obs-1",
                    theory_id="CURIOSITY_VOID",
                    action_taken="Inspected the local environment with vision.",
                    sensor_data="The room contains a monitor and keyboard.",
                    ok=True,
                )

        class _FakeAnalyzer:
            async def analyze(self, theory, observation):
                return PredictionError(
                    match=False,
                    reasoning="This is new information.",
                    confidence_in_data=1.0,
                    suggested_confidence_adjustment=0.0,
                    new_hypothesis="The local environment includes a monitor and keyboard.",
                )

        loop = EpistemicLoop(
            world_model=world_model,
            explorer=_FakeExplorer(),
            analyzer=_FakeAnalyzer(),
            optimizer=optimizer,
            curiosity_topics=["physical surroundings"],
            curiosity_interval=1,
        )

        result = await loop.run_cycle()

        self.assertEqual(result.mode, "curiosity")
        self.assertEqual(world_model.updated_theories, [])
        self.assertEqual(world_model.deleted_theories, [])
        self.assertEqual(world_model.recorded_epistemic_observations[0][3], "curiosity")
        self.assertEqual(world_model.recorded_epistemic_observations[0][4], "physical surroundings")
        self.assertEqual(
            world_model.added_theories[0][0],
            "The local environment includes a monitor and keyboard.",
        )
        self.assertEqual(world_model.added_theories[0][2], "obs-1")
        self.assertEqual(result.optimization.injected_theory_id, "theory-1")


class MotivatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_motivator_falls_back_to_evidence_goal_when_no_grounded_theories_exist(self) -> None:
        world_model = _FakeWorldModel()

        class _UnusedCognitionClient:
            async def generate_model(self, *args, **kwargs):
                raise AssertionError("COGNITION should not be called without grounded theories.")

        motivator = Motivator(world_model, cognition_client=_UnusedCognitionClient())
        goal = await motivator.motivate_goal()

        self.assertIn("evidence", goal.text.lower())
        self.assertEqual(len(world_model.added_goals), 1)
        self.assertEqual(world_model.added_goals[0][3], [])

    async def test_motivator_links_generated_goal_to_grounding_theories(self) -> None:
        world_model = _FakeWorldModel()
        world_model.grounded_theories = [
            Theory(id="theory-1", text="The workspace has Python files.", confidence_score=0.8, attempts=2),
            Theory(id="theory-2", text="The orchestrator can run shell tools.", confidence_score=0.7, attempts=1),
        ]

        class _FakeCognitionClient:
            async def generate_model(self, prompt, model_cls, *, timeout=60, **kwargs):
                return model_cls(goal_text="Inspect the Python entrypoints.", reasoning="Grounded by current theories.")

        motivator = Motivator(world_model, cognition_client=_FakeCognitionClient())
        goal = await motivator.motivate_goal()

        self.assertEqual(goal.text, "Inspect the Python entrypoints.")
        self.assertEqual(world_model.goal_theory_links[0], ("goal-1", ["theory-1", "theory-2"]))


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluator_requeues_failed_goal_without_promoting_failure_to_belief(self) -> None:
        world_model = _FakeWorldModel()

        class _FakeCognitionClient:
            async def generate_model(self, prompt, model_cls, *, timeout=60, **kwargs):
                return model_cls(
                    completion_score=0.0,
                    reasoning="The action did not finish the task.",
                    new_theory="Writing to protected directories requires elevated permissions.",
                )

        evaluator = Evaluator(world_model, cognition_client=_FakeCognitionClient(), max_total_attempts=3)
        goal = Goal(id="goal-1", text="Write to /root", reasoning="Check permissions", attempts=1)
        execution = GoalExecution(
            action_taken="Attempted to write a file under /root.",
            sensor_data="Permission denied",
            ok=False,
        )

        outcome = await evaluator.evaluate(goal, execution)

        self.assertFalse(outcome.match)
        self.assertEqual(outcome.goal_status, "pending")
        self.assertEqual(world_model.goal_status_updates, [("goal-1", "pending")])
        self.assertEqual(world_model.recorded_goal_executions[0][3], "goal-obs-1")
        self.assertEqual(world_model.added_theories, [])


class PowerProcessLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_cycle_uses_motivator_when_no_pending_goal_exists(self) -> None:
        world_model = _FakeWorldModel()
        generated_goal = Goal(id="goal-1", text="Inspect the host OS", reasoning="Need more context")

        class _FakeMotivator:
            def __init__(self):
                self.calls = 0

            async def motivate_goal(self):
                self.calls += 1
                return generated_goal

        class _FakeExecutor:
            async def execute_goal(self, goal):
                return GoalExecution(
                    action_taken="Ran a platform inspection command.",
                    sensor_data="Linux",
                    ok=True,
                )

        class _FakeEvaluator:
            async def evaluate(self, goal, execution):
                return ExecutionOutcome(
                    match=True,
                    reasoning="The OS was identified.",
                    goal_status="completed",
                    new_theory="The host reports Linux through platform inspection.",
                )

        motivator = _FakeMotivator()
        loop = PowerProcessLoop(
            world_model=world_model,
            motivator=motivator,
            executor=_FakeExecutor(),
            evaluator=_FakeEvaluator(),
            pause_seconds=0,
        )

        result = await loop.run_cycle()

        self.assertEqual(motivator.calls, 1)
        self.assertEqual(result.goal.text, "Inspect the host OS")
        self.assertEqual(result.execution.action_taken, "Ran a platform inspection command.")
        self.assertEqual(result.outcome.goal_status, "completed")


if __name__ == "__main__":
    unittest.main()
