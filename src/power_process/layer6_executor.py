from __future__ import annotations

from typing import Optional

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

Rules:
- Execute the action instead of proposing a plan.
- Prefer the smallest grounded action that can still move the goal forward.
- You may execute AT MOST {self.max_actions} tool call(s) to achieve this goal.
- Describe the action that was actually executed and the concrete observation or failure.
""".strip()

    async def execute_goal(self, goal: Goal) -> GoalExecution:
        try:
            return await self.cognition_client.generate_model(
                self._build_prompt(goal),
                GoalExecution,
            )
        except Exception as exc:
            return GoalExecution(
                action_taken="COGNITION goal execution failed.",
                sensor_data=str(exc),
                ok=False,
            )
