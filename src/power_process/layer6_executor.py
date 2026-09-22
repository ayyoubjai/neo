from __future__ import annotations

from typing import Optional
import json

from autonomy.cognition_client import CognitionClient
from power_process.types import Goal, GoalExecution


class Executor:
    def __init__(
        self,
        cognition_client: Optional[CognitionClient] = None,
        *,
        max_actions: int = 3,
    ) -> None:
        self.cognition_client = cognition_client or CognitionClient()
        self.max_actions = max(1, int(max_actions))

    def _build_prompt(self, goal: Goal) -> str:
        return f"""
You are Layer 6 of an autonomous agent.
Make concrete progress on this goal by executing the single best next action.

[GOAL]
Text: {goal.text}
Reasoning: {goal.reasoning}
Success criteria: {goal.success_criteria or goal.text}
Previous attempts (newest first): {json.dumps(goal.history)}

Rules:
- Execute the action instead of proposing a plan.
- Use previous results to choose the next action; do not repeat failed actions unchanged.
- Prefer the smallest grounded action that can still move the goal forward.
- You may execute AT MOST {self.max_actions} tool call(s) to achieve this goal.
- Describe the action that was actually executed and the concrete observation or failure.
""".strip()

    async def execute_goal(self, goal: Goal) -> GoalExecution:
        try:
            return await self.cognition_client.generate_model(
                self._build_prompt(goal),
                GoalExecution,
                max_actions=self.max_actions,
                require_evidence=True,
            )
        except Exception as exc:
            return GoalExecution(
                action_taken="COGNITION goal execution failed.",
                sensor_data=str(exc),
                ok=False,
            )
