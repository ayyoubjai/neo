from __future__ import annotations

from typing import TYPE_CHECKING

from epistemic.types import CURIOSITY_PLACEHOLDER_ID, Observation, OptimizationResult, PredictionError, Theory

if TYPE_CHECKING:
    from epistemic.layer1_world_model import WorldModel


class Optimizer:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def optimize(
        self,
        theory: Theory,
        observation: Observation,
        error: PredictionError,
        *,
        mode: str = "theory",
        focus_text: str = "",
    ) -> OptimizationResult:
        result = OptimizationResult()
        self.world_model.record_epistemic_observation(
            theory,
            observation,
            error,
            mode=mode,
            focus_text=focus_text or theory.text,
        )
        adjustment = error.suggested_confidence_adjustment * error.confidence_in_data
        new_confidence = max(0.0, min(1.0, theory.confidence_score + adjustment))
        if theory.id != CURIOSITY_PLACEHOLDER_ID:
            self.world_model.update_theory(theory.id, new_confidence, success=error.match)
            result.updated_confidence = new_confidence
            result.theory_updated = True
            if new_confidence <= 0.0:
                self.world_model.delete_theory(theory.id)
                result.theory_deleted = True
        if error.new_hypothesis and error.new_hypothesis.strip():
            new_theory = self.world_model.add_theory(
                error.new_hypothesis.strip(),
                source_observation_id=observation.id,
            )
            result.injected_theory_id = new_theory.id
        return result
