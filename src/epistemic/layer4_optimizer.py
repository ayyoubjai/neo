from __future__ import annotations

from typing import TYPE_CHECKING

from epistemic.types import CURIOSITY_PLACEHOLDER_ID, Observation, OptimizationResult, PredictionError, Theory, _as_float

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
        return self.world_model.run_transaction(
            lambda model: Optimizer(model)._optimize(theory, observation, error, mode=mode, focus_text=focus_text)
        )

    def _optimize(self, theory, observation, error, *, mode, focus_text):
        result = OptimizationResult()
        if self.world_model.has_observation(observation.id):
            return result
        usable = observation.ok and bool(observation.sensor_data.strip()) and _as_float(error.confidence_in_data, 0) > 0
        error.confidence_in_data = max(0.0, min(1.0, _as_float(error.confidence_in_data, 0)))
        error.suggested_confidence_adjustment = max(-0.25, min(0.25, _as_float(error.suggested_confidence_adjustment, 0)))
        if error.verdict not in {"supports", "contradicts", "inconclusive"}:
            error.verdict = "inconclusive"
        error.match = usable and error.verdict == "supports"
        if not usable:
            error.verdict = "inconclusive"
            error.match = False
            error.confidence_in_data = 0.0
            error.suggested_confidence_adjustment = 0.0
        self.world_model.record_epistemic_observation(
            theory,
            observation,
            error,
            mode=mode,
            focus_text=focus_text or theory.text,
        )
        adjustment = max(-0.25, min(0.25, _as_float(error.suggested_confidence_adjustment, 0)))
        adjustment *= max(0.0, min(1.0, _as_float(error.confidence_in_data, 0)))
        # A model cannot increase confidence while reporting a contradiction.
        adjustment = max(0.0, adjustment) if error.verdict == "supports" else min(0.0, adjustment)
        if usable and error.verdict in {"supports", "contradicts"} and theory.id != CURIOSITY_PLACEHOLDER_ID:
            current = self.world_model.get_theory(theory.id)
            if current is not None:
                new_confidence = max(0.0, min(1.0, current.confidence_score + adjustment))
                self.world_model.update_theory(theory.id, new_confidence, success=error.verdict == "supports")
                result.updated_confidence = new_confidence
                result.theory_updated = True
                # Keep rejected beliefs and all their relationships as intellectual history.
                result.theory_retired = new_confidence <= 0.0
        if usable and error.new_hypothesis and error.new_hypothesis.strip():
            new_theory = self.world_model.add_theory(
                error.new_hypothesis.strip(),
                source_observation_id=observation.id,
            )
            self.world_model.enrich_theory(
                new_theory.id, kind=error.hypothesis_kind, concepts=error.concepts,
                parent_id=theory.id if theory.id != CURIOSITY_PLACEHOLDER_ID else None,
                relation=error.hypothesis_relation, observation_id=observation.id,
            )
            result.injected_theory_id = new_theory.id
        return result
