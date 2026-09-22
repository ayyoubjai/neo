from __future__ import annotations

from typing import Optional
import json

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
Execution OK: {observation.ok}
Action: {observation.action_taken}
Evidence: {json.dumps(observation.evidence)}
Sensor Data:
{observation.sensor_data}

Guidance:
- verdict must be supports, contradicts, or inconclusive. Absence of evidence is inconclusive.
- match is true only for supports. Confidence is a heuristic, not a calibrated probability.
- Bound suggested_confidence_adjustment to [-0.25, 0.25], positive for supports, negative for contradicts.
- New ideas may remain speculative: hypothesis_kind is idea, hypothesis, or belief.
- Describe their relationship to the tested theory with hypothesis_relation:
  REFINES, EXTENDS, CONTRADICTS, or RELATED_TO. These are proposed conceptual connections, not proven facts.
- concepts is up to eight short concept names linking the new idea to the wider graph.
- Do not turn operational errors into claims about the topic being studied.
- Use low confidence_in_data if the tool failed, the data is noisy, or the observation is weak.
- suggested_confidence_adjustment should be modest unless the evidence is strong.
- Provide new_hypothesis only when the observation suggests a clearer replacement or extension theory.
- Do not call tools. Evaluate only from the theory and observation given here.
""".strip()

    async def analyze(self, theory: Theory, observation: Observation) -> PredictionError:
        if not observation.ok or not observation.sensor_data.strip():
            return PredictionError(False, "No usable observation; test is inconclusive.", 0.0, 0.0)
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
