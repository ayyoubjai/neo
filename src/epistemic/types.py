from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


CURIOSITY_PLACEHOLDER_ID = "CURIOSITY_VOID"


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Theory:
    id: str
    text: str
    confidence_score: float = 0.1
    predictive_success_rate: float = 0.0
    attempts: int = 0

    @classmethod
    def from_dict(cls, payload: dict) -> "Theory":
        return cls(
            id=str(payload.get("id", "")),
            text=str(payload.get("text", "")),
            confidence_score=_as_float(payload.get("confidence_score", 0.1), 0.1),
            predictive_success_rate=_as_float(payload.get("predictive_success_rate", 0.0), 0.0),
            attempts=_as_int(payload.get("attempts", 0), 0),
        )


@dataclass
class Observation:
    id: str
    theory_id: str
    action_taken: str
    sensor_data: str
    ok: bool = True

    @classmethod
    def from_dict(cls, payload: dict) -> "Observation":
        return cls(
            id=str(payload.get("id", "")),
            theory_id=str(payload.get("theory_id", "")),
            action_taken=str(payload.get("action_taken", "")),
            sensor_data=str(payload.get("sensor_data", "")),
            ok=bool(payload.get("ok", True)),
        )


@dataclass
class PredictionError:
    match: bool
    reasoning: str
    confidence_in_data: float = 1.0
    suggested_confidence_adjustment: float = 0.0
    new_hypothesis: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict) -> "PredictionError":
        raw_hypothesis = payload.get("new_hypothesis")
        hypothesis = None if raw_hypothesis in (None, "", "null") else str(raw_hypothesis)
        return cls(
            match=bool(payload.get("match", False)),
            reasoning=str(payload.get("reasoning", "")),
            confidence_in_data=max(0.0, min(1.0, _as_float(payload.get("confidence_in_data", 1.0), 1.0))),
            suggested_confidence_adjustment=_as_float(payload.get("suggested_confidence_adjustment", 0.0), 0.0),
            new_hypothesis=hypothesis,
        )


@dataclass
class OptimizationResult:
    updated_confidence: Optional[float] = None
    theory_updated: bool = False
    theory_deleted: bool = False
    injected_theory_id: Optional[str] = None


@dataclass
class EpistemicCycleResult:
    mode: str
    focus_text: str
    theory_id: Optional[str] = None
    observation: Observation = field(
        default_factory=lambda: Observation(id="", theory_id="", action_taken="", sensor_data="")
    )
    prediction_error: PredictionError = field(default_factory=lambda: PredictionError(match=False, reasoning=""))
    optimization: OptimizationResult = field(default_factory=OptimizationResult)
