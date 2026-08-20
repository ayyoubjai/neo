from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Optional

from autonomy.cognition_client import CognitionClient
from epistemic.types import CURIOSITY_PLACEHOLDER_ID, Observation, Theory


@dataclass
class _ExplorationResponse:
    action_taken: str
    sensor_data: str
    ok: bool = True

    @classmethod
    def from_dict(cls, payload: dict) -> "_ExplorationResponse":
        return cls(
            action_taken=str(payload.get("action_taken", "")),
            sensor_data=str(payload.get("sensor_data", "")),
            ok=bool(payload.get("ok", True)),
        )


class Explorer:
    def __init__(self, cognition_client: Optional[CognitionClient] = None) -> None:
        self.cognition_client = cognition_client or CognitionClient()

    def _build_prompt(self, context_text: str, *, is_curiosity: bool) -> str:
        if is_curiosity:
            mission = (
                "Gather one concrete, tool-grounded observation that teaches the agent something new about "
                f"'{context_text}'."
            )
        else:
            mission = (
                f"Test this theory with the smallest grounded action that can verify or falsify it: '{context_text}'."
            )
        return f"""
You are Layer 2 of an epistemic agent.
{mission}

Execute the action instead of merely proposing it.
Use tools when they are needed for grounded evidence.
Describe the concrete action that was actually executed and the relevant observed evidence or failure.
""".strip()

    async def explore(self, theory: Optional[Theory], curiosity_topic: Optional[str] = None) -> Observation:
        is_curiosity = theory is None
        context_text = curiosity_topic if is_curiosity else theory.text
        theory_id = theory.id if theory else CURIOSITY_PLACEHOLDER_ID
        if not context_text:
            return Observation(
                id=str(uuid.uuid4()),
                theory_id=theory_id,
                action_taken="No context available for exploration.",
                sensor_data="Explorer was called without a theory or curiosity topic.",
                ok=False,
            )
        try:
            result = await self.cognition_client.generate_model(
                self._build_prompt(context_text, is_curiosity=is_curiosity),
                _ExplorationResponse,
            )
        except Exception as exc:
            action_taken = "COGNITION exploration failed."
            sensor_data = str(exc)
            ok = False
        else:
            action_taken = result.action_taken or "COGNITION exploration completed."
            sensor_data = result.sensor_data
            ok = result.ok
        return Observation(
            id=str(uuid.uuid4()),
            theory_id=theory_id,
            action_taken=action_taken,
            sensor_data=sensor_data,
            ok=ok,
        )
