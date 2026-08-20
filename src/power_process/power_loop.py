from __future__ import annotations

import asyncio

from autonomy.cognition_client import CognitionClient, CognitionClientError
from power_process.layer5_motivator import Motivator
from power_process.layer6_executor import Executor
from power_process.layer7_evaluator import Evaluator
from power_process.types import PowerCycleResult


class PowerProcessLoop:
    def __init__(
        self,
        *,
        world_model,
        motivator: Motivator,
        executor: Executor,
        evaluator: Evaluator,
        pause_seconds: float = 2.0,
        write_lock: asyncio.Lock | None = None,
    ) -> None:
        self.world_model = world_model
        self.motivator = motivator
        self.executor = executor
        self.evaluator = evaluator
        self.pause_seconds = max(0.0, float(pause_seconds))
        # Shared lock with the epistemic loop to prevent concurrent Neo4j writes
        # from corrupting goal/theory state. Pass the same asyncio.Lock instance
        # to both PowerProcessLoop and EpistemicLoop at construction time.
        self._write_lock = write_lock or asyncio.Lock()

    async def run_cycle(self) -> PowerCycleResult:
        async with self._write_lock:
            goal = self.world_model.get_pending_goal()
            if goal is None:
                goal = await self.motivator.motivate_goal()
        execution = await self.executor.execute_goal(goal)
        async with self._write_lock:
            outcome = await self.evaluator.evaluate(goal, execution)
        return PowerCycleResult(
            goal=goal,
            execution=execution,
            outcome=outcome,
        )

    async def run_forever(self) -> None:
        while True:
            result = await self.run_cycle()
            print(
                "[PowerProcessLoop]",
                result.goal.text[:120],
                "| status:",
                result.outcome.goal_status,
                "| score:",
                f"{result.outcome.completion_score:.2f}",
                "| match:",
                result.outcome.match,
            )
            if self.pause_seconds > 0:
                await asyncio.sleep(self.pause_seconds)


async def run_power_process() -> None:
    print("[Power Process] Starting pragmatic goal loop...")
    from epistemic.layer1_world_model import WorldModel

    cognition_client = CognitionClient()
    try:
        await cognition_client.wait_until_available()
    except CognitionClientError as exc:
        print(f"[Power Process] {exc}")
        return
    write_lock = asyncio.Lock()
    world_model = WorldModel()
    world_model.bootstrap_axioms()
    loop = PowerProcessLoop(
        world_model=world_model,
        motivator=Motivator(world_model, cognition_client=cognition_client),
        executor=Executor(cognition_client=cognition_client),
        evaluator=Evaluator(world_model, cognition_client=cognition_client),
        write_lock=write_lock,
    )
    try:
        await loop.run_forever()
    except KeyboardInterrupt:
        print("[Power Process] Shutting down gracefully.")
    finally:
        world_model.close()


if __name__ == "__main__":
    asyncio.run(run_power_process())
