from __future__ import annotations

from typing import Optional

from autonomy.cognition_client import CognitionClient, CognitionJsonError
from epistemic.types import Observation, PredictionError, Theory


class Analyzer:
    def __init__(self, cognition_client: Optional[CognitionClient] = None) -> None:
        self.cognition_client = cognition_client or CognitionClient()

    def _build_prompt(self, theory: Theory, observation: Observation) -> str:
        return f"""
You are Layer 3 of an epistemic agent.
Compare the theory against the observation and estimate the prediction error.

[THEORY]
ID: {theory.id}
Text: {theory.text}

[OBSERVATION]
Action: {observation.action_taken}
Sensor Data:
{observation.sensor_data}

Guidance:
- match means the observation supports the theory.
- Use low confidence_in_data if the tool failed, the data is noisy, or the observation is weak.
- suggested_confidence_adjustment should be modest unless the evidence is strong.
- Provide new_hypothesis only when the observation suggests a clearer replacement or extension theory.
- Do not call tools. Evaluate only from the theory and observation given here.
""".strip()

    async def analyze(self, theory: Theory, observation: Observation) -> PredictionError:
        try:
            return await self.cognition_client.generate_model(
                self._build_prompt(theory, observation),
                PredictionError,
            )
        except CognitionJsonError as exc:
            error_msg = f"Analyzer JSON failure: {str(exc)[:100]}"
            print(f"[Analyzer] ⚠️  {error_msg}")
            return PredictionError(
                match=False,
                reasoning=error_msg,
                confidence_in_data=0.0,
                suggested_confidence_adjustment=0.0,
            )
        except Exception as exc:
            error_msg = f"Analyzer failure: {str(exc)[:100]}"
            print(f"[Analyzer] ⚠️  {error_msg}")
            return PredictionError(
                match=False,
                reasoning=error_msg,
                confidence_in_data=0.0,
                suggested_confidence_adjustment=0.0,
            )
