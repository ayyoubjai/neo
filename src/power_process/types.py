from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Goal:
    id: str
    text: str
    reasoning: str
    status: str = "pending"
    attempts: int = 0

    @classmethod
    def from_dict(cls, payload: dict) -> "Goal":
        return cls(
            id=str(payload.get("id", "")),
            text=str(payload.get("text", "")),
            reasoning=str(payload.get("reasoning", "")),
            status=str(payload.get("status", "pending") or "pending"),
            attempts=_as_int(payload.get("attempts", 0), 0),
        )


@dataclass
class ExecutionOutcome:
    match: bool
    reasoning: str
    new_theory: Optional[str] = None
    goal_status: str = "pending"
    completion_score: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict) -> "ExecutionOutcome":
        raw_theory = payload.get("new_theory")
        new_theory = None if raw_theory in (None, "", "null") else str(raw_theory)
        raw_score = payload.get("completion_score", 1.0 if payload.get("match") else 0.0)
        try:
            completion_score = max(0.0, min(1.0, float(raw_score)))
        except (TypeError, ValueError):
            completion_score = 0.0
        return cls(
            match=completion_score >= 0.6,
            completion_score=completion_score,
            reasoning=str(payload.get("reasoning", "")),
            new_theory=new_theory,
            goal_status=str(payload.get("goal_status", "pending") or "pending"),
        )


@dataclass
class GoalExecution:
    action_taken: str
    sensor_data: str
    ok: bool = True

    @classmethod
    def from_dict(cls, payload: dict) -> "GoalExecution":
        return cls(
            action_taken=str(payload.get("action_taken", "")),
            sensor_data=str(payload.get("sensor_data", "")),
            ok=bool(payload.get("ok", True)),
        )


@dataclass
class PowerCycleResult:
    goal: Goal
    execution: GoalExecution
    outcome: ExecutionOutcome
