from __future__ import annotations

import asyncio
import json
from typing import Optional, TYPE_CHECKING

from autonomy.cognition_client import CognitionClient, CognitionClientError
from epistemic.layer2_exploration import Explorer
from epistemic.layer3_analyzer import Analyzer
from epistemic.layer4_optimizer import Optimizer
from epistemic.types import CURIOSITY_PLACEHOLDER_ID, EpistemicCycleResult, Theory

if TYPE_CHECKING:
    from epistemic.layer1_world_model import WorldModel


DEFAULT_CURIOSITY_TOPICS = (
    "advanced physics theories",
    "artificial intelligence architecture",
    "local software environment",
    "physical surroundings",
)


class EpistemicLoop:
    def __init__(
        self,
        *,
        world_model: WorldModel,
        explorer: Explorer,
        analyzer: Analyzer,
        optimizer: Optimizer,
        curiosity_topics: Optional[list[str]] = None,
        curiosity_interval: int = 5,
        max_theory_attempts: int = 3,
        pause_seconds: float = 2.0,
        write_lock: asyncio.Lock | None = None,
    ) -> None:
        self.world_model = world_model
        self.explorer = explorer
        self.analyzer = analyzer
        self.optimizer = optimizer
        self.curiosity_topics = curiosity_topics or list(DEFAULT_CURIOSITY_TOPICS)
        self.curiosity_interval = max(0, int(curiosity_interval))
        self.max_theory_attempts = max(1, int(max_theory_attempts))
        self.pause_seconds = max(0.0, float(pause_seconds))
        self._cycles_since_curiosity = 0
        # Per-topic exploration attempts balance attention, including failures.
        self._topic_observation_counts: dict[str, int] = {
            topic: 0 for topic in self.curiosity_topics
        }
        # Serialize local focus/commit operations; persistence is transactional.
        self._write_lock = write_lock or asyncio.Lock()

    def _next_curiosity_topic(self) -> str:
        """Return the least-attempted topic, resolving ties by configured order."""
        if not self.curiosity_topics:
            return "unknown aspects of the environment"
        # Priority: pick the topic with the lowest observation count
        least_explored = min(
            self.curiosity_topics,
            key=lambda t: self._topic_observation_counts.get(t, 0),
        )
        self._topic_observation_counts[least_explored] = (
            self._topic_observation_counts.get(least_explored, 0) + 1
        )
        return least_explored

    def _select_focus(self) -> tuple[Optional[Theory], Optional[str]]:
        if self.curiosity_interval > 0 and self._cycles_since_curiosity >= self.curiosity_interval:
            self._cycles_since_curiosity = 0
            return None, self._next_curiosity_topic()
        theory = self.world_model.get_revalidation_theory()
        if theory is None:
            theory = self.world_model.get_untested_theory(max_attempts=self.max_theory_attempts)
        if theory is None:
            self._cycles_since_curiosity = 0
            return None, self._next_curiosity_topic()
        self._cycles_since_curiosity += 1
        return theory, None

    async def run_cycle(self) -> EpistemicCycleResult:
        async with self._write_lock:
            theory, curiosity_topic = await asyncio.to_thread(self._select_focus)
        if theory is None:
            observation = await self.explorer.explore(None, curiosity_topic=curiosity_topic)
            faux_theory = Theory(
                id=CURIOSITY_PLACEHOLDER_ID,
                text=f"We currently lack a reliable model of {curiosity_topic}.",
            )
            prediction_error = await self.analyzer.analyze(faux_theory, observation)
            async with self._write_lock:
                optimization = await asyncio.to_thread(
                    self.optimizer.optimize,
                    faux_theory,
                    observation,
                    prediction_error,
                    mode="curiosity",
                    focus_text=curiosity_topic or "",
                )
            return EpistemicCycleResult(
                mode="curiosity",
                focus_text=curiosity_topic or "",
                theory_id=None,
                observation=observation,
                prediction_error=prediction_error,
                optimization=optimization,
            )
        context = await asyncio.to_thread(self.world_model.get_theory_context, theory.id)
        observation = await self.explorer.explore(theory, context=json.dumps(context))
        prediction_error = await self.analyzer.analyze(theory, observation)
        async with self._write_lock:
            optimization = await asyncio.to_thread(
                self.optimizer.optimize,
                theory,
                observation,
                prediction_error,
                mode="theory",
                focus_text=theory.text,
            )
        return EpistemicCycleResult(
            mode="theory",
            focus_text=theory.text,
            theory_id=theory.id,
            observation=observation,
            prediction_error=prediction_error,
            optimization=optimization,
        )

    async def run_forever(self) -> None:
        while True:
            try:
                result = await self.run_cycle()
            except Exception as exc:
                print(f"[EpistemicLoop] cycle failed: {exc}")
                await asyncio.sleep(max(2.0, self.pause_seconds))
                continue
            # Enhanced logging with more diagnostics
            log_msg = (
                f"[EpistemicLoop] {result.mode:8} | "
                f"focus: {result.focus_text[:60]:60} | "
                f"match: {result.prediction_error.match} | "
                f"ok: {result.observation.ok}"
            )
            print(log_msg)
            
            # Log if observation failed
            if not result.observation.ok:
                print(f"  ⚠️  Observation failed: {result.observation.sensor_data[:100]}")
            
            # Inconclusive evidence is not a prediction mismatch.
            if result.prediction_error.verdict == "contradicts":
                print(f"  ⚠️  Prediction mismatch: {result.prediction_error.reasoning[:100]}")
            
            if self.pause_seconds > 0:
                await asyncio.sleep(self.pause_seconds)


async def run_epistemic_loop() -> None:
    print("[Epistemic Pipeline] Starting epistemic loop...")
    from epistemic.layer1_world_model import WorldModel

    cognition_client = CognitionClient()
    try:
        await cognition_client.wait_until_available()
    except CognitionClientError as exc:
        print(f"[Epistemic Pipeline] {exc}")
        return
    write_lock = asyncio.Lock()
    world_model = WorldModel()
    world_model.bootstrap_axioms()
    loop = EpistemicLoop(
        world_model=world_model,
        explorer=Explorer(cognition_client=cognition_client),
        analyzer=Analyzer(cognition_client=cognition_client),
        optimizer=Optimizer(world_model),
        write_lock=write_lock,
    )
    try:
        await loop.run_forever()
    except KeyboardInterrupt:
        print("[Epistemic Pipeline] Shutting down gracefully.")
    finally:
        world_model.close()


if __name__ == "__main__":
    asyncio.run(run_epistemic_loop())
