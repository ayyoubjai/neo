from __future__ import annotations

from dataclasses import dataclass
import asyncio
from typing import Optional, TYPE_CHECKING

from autonomy.cognition_client import CognitionClient, CognitionJsonError
from power_process.types import Goal

if TYPE_CHECKING:
    from epistemic.layer1_world_model import WorldModel


@dataclass
class _GoalProposal:
    goal_text: str
    reasoning: str
    success_criteria: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "_GoalProposal":
        return cls(
            goal_text=str(payload.get("goal_text", "")),
            reasoning=str(payload.get("reasoning", "")),
            success_criteria=str(payload.get("success_criteria", "")),
        )


class Motivator:
    def __init__(
        self,
        world_model: WorldModel,
        *,
        cognition_client: Optional[CognitionClient] = None,
        min_grounding_confidence: float = 0.5,
        objective: str = "Develop a connected understanding of the environment, its ideas, and uncertainties.",
    ) -> None:
        self.world_model = world_model
        self.cognition_client = cognition_client or CognitionClient()
        self.min_grounding_confidence = float(min_grounding_confidence)
        self.objective = objective

    async def motivate_goal(self) -> Goal:
        theories = await asyncio.to_thread(self.world_model.get_grounded_theories, limit=10, min_confidence=self.min_grounding_confidence)
        if not theories:
            return await asyncio.to_thread(self.world_model.add_goal,
                text="Gather one concrete piece of evidence about the local environment using a safe, read-only tool.",
                reasoning="No theories are grounded strongly enough yet, so the next rational goal is to collect evidence safely.",
                grounding_theory_ids=[],
            )
        theory_texts = "\n".join(
            f"- {theory.text} (confidence={theory.confidence_score:.2f}, attempts={theory.attempts})"
            for theory in theories
        )
        prompt = f"""
You are Layer 5 of an autonomous agent.
Synthesize one pragmatic goal using the provisional world model below.
Purpose: {self.objective}

[GROUNDED THEORIES]
{theory_texts}

Rules:
- The goal must be concrete, testable, and achievable via the currently available tools.
- Supply explicit success_criteria describing the evidence that would establish completion.
- Belief confidence does not authorize actions; stay within the configured purpose.
- Prefer goals that extend the current world model instead of vague aspirations.
- Do not call tools. Synthesize the goal only from the grounded theories shown here.

Alignment safety constraints (MANDATORY):
- The goal must NOT require deleting, modifying, or exfiltrating user data without explicit user instruction.
- The goal must NOT require making irreversible changes to the system state as a first action.
- Prefer goals that generate knowledge (read/observe) over goals that change system state (write/delete).
- If a goal requires elevated permissions, flag it in reasoning and propose the safest sub-goal instead.
""".strip()
        try:
            proposal = await self.cognition_client.generate_model(prompt, _GoalProposal)
        except Exception:
            return await asyncio.to_thread(self.world_model.add_goal,
                text="Probe a grounded theory with one safe, read-only action.",
                reasoning="Goal generation failed, so fall back to a simple evidence-gathering task.",
                grounding_theory_ids=[theory.id for theory in theories[:3]],
            )
        return await asyncio.to_thread(self.world_model.add_goal,
            proposal.goal_text,
            proposal.reasoning,
            grounding_theory_ids=[theory.id for theory in theories],
            success_criteria=proposal.success_criteria or proposal.goal_text,
        )
