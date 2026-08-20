from __future__ import annotations

import asyncio
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
        self._topic_index = 0
        self._cycles_since_curiosity = 0
        # Per-topic observation count for priority-based curiosity selection.
        # Topics with fewer observations are preferred to avoid round-robin bias.
        self._topic_observation_counts: dict[str, int] = {
            topic: 0 for topic in self.curiosity_topics
        }
        # Shared lock with the power loop to prevent concurrent Neo4j writes.
        self._write_lock = write_lock or asyncio.Lock()

    def _next_curiosity_topic(self) -> str:
        """Return the least-explored curiosity topic (fewest observations so far).

        Falls back to round-robin if counts are all equal or topic list is empty.
        """
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
        theory = self.world_model.get_untested_theory(max_attempts=self.max_theory_attempts)
        should_use_curiosity = theory is None or (
            self.curiosity_interval > 0 and self._cycles_since_curiosity >= self.curiosity_interval
        )
        if should_use_curiosity:
            self._cycles_since_curiosity = 0
            return None, self._next_curiosity_topic()
        self._cycles_since_curiosity += 1
        if theory is None:
            # Fallback: if no untested theories but curiosity not yet due, pick any theory
            # This prevents getting stuck in curiosity mode when all theories are exhausted
            all_theories = self.world_model.get_all_theories()
            if all_theories:
                # Pick the lowest-confidence theory for re-validation
                theory = min(all_theories, key=lambda t: (t.confidence_score, t.predictive_success_rate))
        return theory, None

    async def run_cycle(self) -> EpistemicCycleResult:
        async with self._write_lock:
            theory, curiosity_topic = self._select_focus()
        if theory is None:
            observation = await self.explorer.explore(None, curiosity_topic=curiosity_topic)
            faux_theory = Theory(
                id=CURIOSITY_PLACEHOLDER_ID,
                text=f"We currently lack a reliable model of {curiosity_topic}.",
            )
            prediction_error = await self.analyzer.analyze(faux_theory, observation)
            if not prediction_error.new_hypothesis and observation.sensor_data.strip():
                prediction_error.new_hypothesis = (
                    f"Observation about {curiosity_topic}: {observation.sensor_data[:240].strip()}"
                )
            async with self._write_lock:
                optimization = self.optimizer.optimize(
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
        observation = await self.explorer.explore(theory)
        prediction_error = await self.analyzer.analyze(theory, observation)
        async with self._write_lock:
            optimization = self.optimizer.optimize(
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
            result = await self.run_cycle()
            # Enhanced logging with more diagnostics
            log_msg = (
                f"[EpistemicLoop] {result.mode:8} | "
                f"focus: {result.focus_text[:60]:60} | "
                f"match: {result.prediction_error.match} | "
                f"ok: {result.observation.ok}"
            )
            # Add confidence info if theory mode
            if result.mode == "theory" and result.theory_id:
                theory = self.world_model.get_theory(result.theory_id)
                if theory:
                    log_msg += f" | conf: {theory.confidence_score:.2f} | attempts: {theory.attempts}"
            print(log_msg)
            
            # Log if observation failed
            if not result.observation.ok:
                print(f"  ⚠️  Observation failed: {result.observation.sensor_data[:100]}")
            
            # Log if match failed
            if not result.prediction_error.match:
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
