from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from common.config import Settings, load_settings


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_permission_policy(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"deny", "approve"}:
        return normalized
    return "deny"


def _coerce_requested_mode(value: object) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"COGNITION", "SYSTEM0", "SYSTEM1", "SYSTEM2", "SYSTEM3"}:
        return normalized
    return "COGNITION"


@dataclass
class AutonomyRuntimeConfig:
    enabled: bool
    start_with_run_all: bool
    epistemic_enabled: bool
    power_process_enabled: bool
    cognition_timeout_s: int
    permission_policy: str
    requested_mode: str
    epistemic_pause_s: float
    power_process_pause_s: float
    epistemic_curiosity_interval: int
    epistemic_max_theory_attempts: int
    goal_min_grounding_confidence: float
    power_max_goal_attempts: int
    knowledge_enabled: bool = False


def build_runtime_config(settings: Settings) -> AutonomyRuntimeConfig:
    autonomy = dict(settings.autonomy or {})
    epistemic_enabled = _coerce_bool(autonomy.get("epistemic_enabled"), False)
    power_enabled = _coerce_bool(autonomy.get("power_process_enabled"), False)
    knowledge_enabled = _coerce_bool(autonomy.get("knowledge_enabled"), False)
    explicit_enabled = autonomy.get("enabled")
    if explicit_enabled is None:
        enabled = epistemic_enabled or power_enabled or knowledge_enabled
    else:
        enabled = _coerce_bool(explicit_enabled, False)
    return AutonomyRuntimeConfig(
        enabled=enabled,
        knowledge_enabled=knowledge_enabled,
        start_with_run_all=_coerce_bool(autonomy.get("start_with_run_all"), False),
        epistemic_enabled=epistemic_enabled,
        power_process_enabled=power_enabled,
        cognition_timeout_s=max(1, _coerce_int(autonomy.get("cognition_timeout_s"), 180)),
        permission_policy=_coerce_permission_policy(autonomy.get("permission_policy", "deny")),
        requested_mode=_coerce_requested_mode(autonomy.get("requested_mode", "COGNITION")),
        epistemic_pause_s=max(0.0, _coerce_float(autonomy.get("epistemic_pause_s"), 2.0)),
        power_process_pause_s=max(0.0, _coerce_float(autonomy.get("power_process_pause_s"), 2.0)),
        epistemic_curiosity_interval=max(0, _coerce_int(autonomy.get("epistemic_curiosity_interval"), 5)),
        epistemic_max_theory_attempts=max(1, _coerce_int(autonomy.get("epistemic_max_theory_attempts"), 3)),
        goal_min_grounding_confidence=max(
            0.0,
            min(1.0, _coerce_float(autonomy.get("goal_min_grounding_confidence"), 0.3)),
        ),
        power_max_goal_attempts=max(1, _coerce_int(autonomy.get("power_max_goal_attempts"), 3)),
    )


def runtime_should_start(config: AutonomyRuntimeConfig) -> bool:
    return bool(config.enabled and (config.epistemic_enabled or config.power_process_enabled or config.knowledge_enabled))


def run_all_should_start(config: AutonomyRuntimeConfig) -> bool:
    return bool(config.start_with_run_all and runtime_should_start(config))


async def serve(settings_path: Optional[str] = None) -> None:
    settings = load_settings(settings_path) if settings_path else load_settings()
    config = build_runtime_config(settings)
    if not runtime_should_start(config):
        print("[autonomy] disabled in settings; exiting.")
        return

    from autonomy.cognition_client import CognitionClient, CognitionClientError
    from epistemic.layer1_world_model import WorldModel
    from epistemic.layer2_exploration import Explorer
    from epistemic.layer3_analyzer import Analyzer
    from epistemic.layer4_optimizer import Optimizer
    from epistemic.main_loop import EpistemicLoop
    from power_process.layer5_motivator import Motivator
    from power_process.layer6_executor import Executor
    from power_process.layer7_evaluator import Evaluator
    from power_process.power_loop import PowerProcessLoop

    cognition_client = CognitionClient(
        settings=settings,
        permission_policy=config.permission_policy,
        requested_mode=config.requested_mode,
        timeout_s=config.cognition_timeout_s,
    )
    try:
        await cognition_client.wait_until_available()
    except CognitionClientError as exc:
        print(f"[autonomy] {exc}")
        return

    world_model = WorldModel() if config.epistemic_enabled or config.power_process_enabled else None
    if world_model is not None:
        world_model.bootstrap_axioms()
    write_lock = asyncio.Lock()
    tasks = []
    try:
        if config.epistemic_enabled:
            epistemic_loop = EpistemicLoop(
                world_model=world_model,
                explorer=Explorer(cognition_client=cognition_client),
                analyzer=Analyzer(cognition_client=cognition_client),
                optimizer=Optimizer(world_model),
                curiosity_interval=config.epistemic_curiosity_interval,
                max_theory_attempts=config.epistemic_max_theory_attempts,
                pause_seconds=config.epistemic_pause_s,
                write_lock=write_lock,
            )
            print("[autonomy] starting epistemic loop")
            tasks.append(asyncio.create_task(epistemic_loop.run_forever(), name="epistemic-loop"))
        if config.power_process_enabled:
            power_loop = PowerProcessLoop(
                world_model=world_model,
                motivator=Motivator(
                    world_model,
                    cognition_client=cognition_client,
                    min_grounding_confidence=config.goal_min_grounding_confidence,
                    objective=str(settings.autonomy.get("objective") or "Develop a connected understanding of the environment, its ideas, and uncertainties."),
                ),
                executor=Executor(cognition_client=cognition_client),
                evaluator=Evaluator(
                    world_model,
                    cognition_client=cognition_client,
                    max_total_attempts=config.power_max_goal_attempts,
                ),
                pause_seconds=config.power_process_pause_s,
                write_lock=write_lock,
            )
            print("[autonomy] starting power process loop")
            tasks.append(asyncio.create_task(power_loop.run_forever(), name="power-process-loop"))
        if config.knowledge_enabled:
            from knowledge.worker import run_worker
            print("[autonomy] starting knowledge investigation worker")
            tasks.append(asyncio.create_task(run_worker(settings), name="knowledge-worker"))
        if not tasks:
            print("[autonomy] no loops enabled; exiting.")
            return
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if world_model is not None:
            world_model.close()


def main() -> None:
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("[autonomy] shutting down gracefully.")


if __name__ == "__main__":
    main()
