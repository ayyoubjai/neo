from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import asyncio
import json
from epistemic.types import _as_float

from autonomy.cognition_client import CognitionClient, CognitionJsonError
from power_process.types import ExecutionOutcome, Goal, GoalExecution

if TYPE_CHECKING:
    from epistemic.layer1_world_model import WorldModel


@dataclass
class _EvaluationResponse:
    completion_score: float = 0.0
    reasoning: str = ""
    new_theory: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict) -> "_EvaluationResponse":
        raw_theory = payload.get("new_theory")
        new_theory = None if raw_theory in (None, "", "null") else str(raw_theory)
        raw_score = payload.get("completion_score", 0)
        if isinstance(raw_score, bool):
            score = 1.0 if raw_score else 0.0
        else:
            try:
                score = max(0.0, min(1.0, _as_float(raw_score, 0)))
            except (TypeError, ValueError):
                score = 0.0
        return cls(
            completion_score=score,
            reasoning=str(payload.get("reasoning", "")),
            new_theory=new_theory,
        )


class Evaluator:
    def __init__(
        self,
        world_model: WorldModel,
        *,
        cognition_client: Optional[CognitionClient] = None,
        max_total_attempts: int = 3,
    ) -> None:
        self.world_model = world_model
        self.cognition_client = cognition_client or CognitionClient()
        self.max_total_attempts = max(1, int(max_total_attempts))

    def _build_prompt(self, goal: Goal, execution: GoalExecution) -> str:
        return f"""
You are Layer 7 of an autonomous agent.
Evaluate how much of the goal was achieved by the action taken.

[GOAL]
Text: {goal.text}
Reasoning: {goal.reasoning}
Success criteria: {goal.success_criteria or goal.text}
Previous attempts: {json.dumps(goal.history)}

[EXECUTION RESULT]
Execution OK: {execution.ok}
Evidence: {json.dumps(execution.evidence)}
Action: {execution.action_taken}
Sensor Data:
{execution.sensor_data}

Rules:
- completion_score is a float from 0.0 (total failure) to 1.0 (fully achieved).
  Use values like 0.3 for partial progress, 0.7 for mostly done, 1.0 for complete.
- Only 1.0 means all success criteria are satisfied by evidence. Partial progress is not completion.
- new_theory should be a concise, reusable lesson learned from the attempt.
- Do not call tools. Evaluate only from the goal and execution result provided here.
""".strip()

    async def evaluate(self, goal: Goal, execution: GoalExecution) -> ExecutionOutcome:
        try:
            response = await self.cognition_client.generate_model(
                self._build_prompt(goal, execution),
                _EvaluationResponse,
            )
            completion_score = max(0.0, min(1.0, _as_float(response.completion_score, 0)))
            outcome = ExecutionOutcome(
                match=execution.ok and bool(execution.sensor_data.strip()) and completion_score >= 1.0,
                completion_score=completion_score,
                reasoning=response.reasoning,
                new_theory=response.new_theory,
            )
        except CognitionJsonError as exc:
            outcome = ExecutionOutcome(
                match=False,
                completion_score=0.0,
                reasoning=f"Evaluator JSON failure: {exc}",
            )
        except Exception as exc:
            outcome = ExecutionOutcome(
                match=False,
                completion_score=0.0,
                reasoning=f"Evaluator failure: {exc}",
            )
        if outcome.match:
            new_status = "completed"
        elif goal.attempts + 1 < self.max_total_attempts:
            new_status = "pending"
        else:
            new_status = "failed"
        outcome.goal_status = new_status
        if not execution.ok:
            outcome.completion_score = 0.0
            outcome.new_theory = None
        def persist(model):
            existing = model.get_goal_execution_outcome(execution.id)
            if existing is not None:
                return existing
            observation_id = model.record_goal_execution(goal, execution, outcome)
            model.update_goal_status(goal.id, new_status)
            if execution.ok and execution.sensor_data.strip() and outcome.new_theory and outcome.new_theory.strip():
                model.add_theory(outcome.new_theory.strip(), source_observation_id=observation_id)
            return outcome
        return await asyncio.to_thread(self.world_model.run_transaction, persist)
