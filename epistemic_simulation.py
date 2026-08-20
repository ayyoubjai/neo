"""
EPISTEMIC PIPELINE SIMULATION
==============================

This script simulates the epistemic pipeline WITHOUT requiring Neo4j or LLM API calls.
It demonstrates the 4-layer feedback loop with deterministic (mocked) responses.

Run: python3 epistemic_simulation.py
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ============================================================================
# TYPE DEFINITIONS (matching src/epistemic/types.py)
# ============================================================================

@dataclass
class Theory:
    id: str
    text: str
    confidence_score: float = 0.1
    predictive_success_rate: float = 0.0
    attempts: int = 0
    updated_at: float = field(default_factory=time.time)

    def __repr__(self):
        return (
            f"Theory(id={self.id[:8]}..., confidence={self.confidence_score:.2f}, "
            f"attempts={self.attempts}, psr={self.predictive_success_rate:.2f})"
        )


@dataclass
class Observation:
    id: str
    theory_id: str
    action_taken: str
    sensor_data: str
    ok: bool = True

    def __repr__(self):
        return (
            f"Observation(id={self.id[:8]}..., theory_id={self.theory_id[:8]}..., "
            f"action_taken='{self.action_taken[:30]}...', ok={self.ok})"
        )


@dataclass
class PredictionError:
    match: bool
    reasoning: str
    confidence_in_data: float = 1.0
    suggested_confidence_adjustment: float = 0.0
    new_hypothesis: Optional[str] = None

    def __repr__(self):
        return (
            f"PredictionError(match={self.match}, "
            f"adjustment={self.suggested_confidence_adjustment:+.2f}, "
            f"confidence_in_data={self.confidence_in_data:.2f})"
        )


@dataclass
class OptimizationResult:
    updated_confidence: Optional[float] = None
    theory_updated: bool = False
    theory_deleted: bool = False
    injected_theory_id: Optional[str] = None

    def __repr__(self):
        return (
            f"OptimizationResult(confidence={self.updated_confidence}, "
            f"updated={self.theory_updated}, deleted={self.theory_deleted}, "
            f"injected={self.injected_theory_id is not None})"
        )


class ExecutionMode(Enum):
    THEORY_TESTING = "theory"
    CURIOSITY = "curiosity"


# ============================================================================
# LAYER 1: WORLD MODEL (Simulated, no Neo4j)
# ============================================================================

class SimulatedWorldModel:
    """In-memory simulation of Neo4j world model."""

    DEFAULT_AXIOMS = (
        "Autonomous agents that verify assumptions before acting make fewer catastrophic errors.",
        "Information retrieved from the internet can be outdated, biased, or fabricated.",
        "Local filesystem state changes are observable by subsequent actions.",
        "Memory retrieval quality degrades with redundancy; deduplication helps.",
        "Tool failures carry diagnostic information revealing correct next actions.",
    )

    def __init__(self):
        self.theories: dict[str, Theory] = {}
        self.observations: dict[str, Observation] = {}
        self.bootstrap_axioms()

    def bootstrap_axioms(self):
        """Initialize with default axioms."""
        for axiom_text in self.DEFAULT_AXIOMS:
            theory = self.add_theory(axiom_text)
            print(f"  [Bootstrap] Added axiom: {axiom_text[:50]}...")

    def add_theory(self, text: str, source_observation_id: Optional[str] = None) -> Theory:
        """Add a new theory to the knowledge base."""
        theory = Theory(
            id=str(uuid.uuid4()),
            text=text,
            confidence_score=0.1,
            predictive_success_rate=0.0,
            attempts=0,
        )
        self.theories[theory.id] = theory
        return theory

    def get_untested_theory(self, max_attempts: int = 3) -> Optional[Theory]:
        """Select theory with lowest attempt count."""
        candidates = [t for t in self.theories.values() if t.attempts < max_attempts]
        if not candidates:
            return None
        # Sort by: attempts (asc), confidence (asc), success_rate (asc)
        candidates.sort(
            key=lambda t: (t.attempts, t.confidence_score, t.predictive_success_rate)
        )
        return candidates[0]

    def update_theory(self, theory_id: str, new_confidence: float, success: bool):
        """Update theory confidence based on observation."""
        if theory_id not in self.theories:
            return

        theory = self.theories[theory_id]

        # Apply temporal decay
        age_days = (time.time() - theory.updated_at) / 86400
        recency_weight = max(0.5, 1.0 - (age_days / 120.0))
        decay_adjusted_confidence = (
            new_confidence * recency_weight + (1.0 - recency_weight) * 0.5
        )

        theory.confidence_score = max(0.0, min(1.0, decay_adjusted_confidence))
        theory.attempts += 1
        theory.predictive_success_rate = (
            (theory.predictive_success_rate * (theory.attempts - 1)) + (1.0 if success else 0.0)
        ) / theory.attempts
        theory.updated_at = time.time()

    def delete_theory(self, theory_id: str):
        """Remove theory from knowledge base."""
        if theory_id in self.theories:
            del self.theories[theory_id]

    def record_observation(self, observation: Observation):
        """Store observation in history."""
        self.observations[observation.id] = observation

    def theory_count(self) -> int:
        return len(self.theories)

    def get_all_theories(self) -> list[Theory]:
        return list(self.theories.values())


# ============================================================================
# LAYER 2: EXPLORER (Simulated)
# ============================================================================

class SimulatedExplorer:
    """Simulates execution layer with deterministic responses."""

    # Deterministic responses based on theory
    THEORY_RESPONSES = {
        # Filesystem persistence
        "local filesystem": (
            "Created /tmp/test.txt with marker content, then listed /tmp/",
            "Found /tmp/test.txt with marker content intact",
            True,
        ),
        # Assumption verification
        "assumptions": (
            "Attempted action with and without permission check",
            "Permission check prevented error; unverified action failed",
            True,
        ),
        # Internet reliability
        "internet": (
            "Fetched same information from 3 different APIs",
            "2 sources consistent, 1 source contradictory - unreliable",
            True,
        ),
        # Memory quality
        "memory": (
            "Stored redundant facts then queried memory system",
            "Memory recall was slow and ambiguous with duplicates",
            True,
        ),
        # Tool failures
        "tool failures": (
            "Intentionally triggered tool failure and analyzed error",
            "Error message revealed missing dependency - diagnostic information present",
            True,
        ),
    }

    # Curiosity responses
    CURIOSITY_RESPONSES = {
        "advanced physics theories": (
            "Queried arXiv API for latest quantum computing papers",
            "Found: Topological quantum error correction enables scalable quantum computers",
            True,
        ),
        "artificial intelligence architecture": (
            "Searched for latest AGI architecture papers",
            "Found: Multi-layer epistemic systems improve reasoning reliability",
            True,
        ),
        "local software environment": (
            "Scanned installed packages and system configuration",
            "Found: Python 3.11, PyTorch 2.0, Neo4j driver installed",
            True,
        ),
        "physical surroundings": (
            "Polled sensors for environmental data",
            "Found: Temperature 22°C, humidity 45%, ambient light 300lux",
            True,
        ),
    }

    async def explore(
        self, theory: Optional[Theory] = None, curiosity_topic: Optional[str] = None
    ) -> Observation:
        """Execute exploration and return observation."""
        if theory:
            # Theory testing mode
            theory_key = None
            for key in self.THEORY_RESPONSES.keys():
                if key.lower() in theory.text.lower():
                    theory_key = key
                    break

            if theory_key:
                action, sensor_data, ok = self.THEORY_RESPONSES[theory_key]
            else:
                action = "Theory testing executed"
                sensor_data = "Observation recorded but no specific response configured"
                ok = True

            theory_id = theory.id
        else:
            # Curiosity mode
            if curiosity_topic and curiosity_topic in self.CURIOSITY_RESPONSES:
                action, sensor_data, ok = self.CURIOSITY_RESPONSES[curiosity_topic]
            else:
                action = "Curiosity exploration executed"
                sensor_data = "General observation about environment"
                ok = True

            theory_id = "CURIOSITY_VOID"

        observation = Observation(
            id=str(uuid.uuid4()),
            theory_id=theory_id,
            action_taken=action,
            sensor_data=sensor_data,
            ok=ok,
        )

        print(f"  [Explorer] Action: {action}")
        print(f"  [Explorer] Sensor: {sensor_data}")
        return observation


# ============================================================================
# LAYER 3: ANALYZER (Simulated)
# ============================================================================

class SimulatedAnalyzer:
    """Simulates prediction error analysis with deterministic responses."""

    async def analyze(self, theory: Theory, observation: Observation) -> PredictionError:
        """Analyze observation against theory."""

        # Deterministic scoring based on content
        if observation.ok and "consistent" in observation.sensor_data.lower():
            match = True
            adjustment = 0.3
            confidence_in_data = 0.9
            reasoning = "Observation strongly validates theory"
        elif observation.ok and observation.sensor_data:
            match = True
            adjustment = 0.15
            confidence_in_data = 0.7
            reasoning = "Observation partially validates theory"
        elif not observation.ok:
            match = False
            adjustment = -0.2
            confidence_in_data = 0.4
            reasoning = "Tool failure reduces observation reliability"
        else:
            match = False
            adjustment = -0.1
            confidence_in_data = 0.5
            reasoning = "Observation contradicts theory"

        # Occasionally suggest new hypothesis
        new_hypothesis = None
        if match and "enable" in observation.sensor_data.lower():
            new_hypothesis = f"Refined understanding: {observation.sensor_data[:60]}"

        error = PredictionError(
            match=match,
            reasoning=reasoning,
            confidence_in_data=confidence_in_data,
            suggested_confidence_adjustment=adjustment,
            new_hypothesis=new_hypothesis,
        )

        print(f"  [Analyzer] Match: {error.match}")
        print(f"  [Analyzer] Reasoning: {error.reasoning}")
        print(f"  [Analyzer] Adjustment: {error.suggested_confidence_adjustment:+.2f}")
        if new_hypothesis:
            print(f"  [Analyzer] New hypothesis: {new_hypothesis[:50]}...")

        return error


# ============================================================================
# LAYER 4: OPTIMIZER (Feedback Control)
# ============================================================================

class SimulatedOptimizer:
    """Applies feedback to update world model."""

    def __init__(self, world_model: SimulatedWorldModel):
        self.world_model = world_model

    def optimize(
        self,
        theory: Theory,
        observation: Observation,
        error: PredictionError,
        mode: ExecutionMode,
        focus_text: str,
    ) -> OptimizationResult:
        """Apply analysis results to update theory."""

        result = OptimizationResult()

        # Record observation
        self.world_model.record_observation(observation)

        # Compute confidence adjustment
        adjustment = error.suggested_confidence_adjustment * error.confidence_in_data
        new_confidence = max(0.0, min(1.0, theory.confidence_score + adjustment))

        # Update theory (unless it's curiosity void)
        if theory.id != "CURIOSITY_VOID":
            self.world_model.update_theory(theory.id, new_confidence, success=error.match)
            result.updated_confidence = new_confidence
            result.theory_updated = True

            if new_confidence <= 0.0:
                self.world_model.delete_theory(theory.id)
                result.theory_deleted = True
                print(f"  [Optimizer] Theory deleted (confidence ≤ 0.0)")

        # Inject new hypothesis as theory
        if error.new_hypothesis and error.new_hypothesis.strip():
            new_theory = self.world_model.add_theory(error.new_hypothesis.strip())
            result.injected_theory_id = new_theory.id
            print(f"  [Optimizer] Injected new theory: {error.new_hypothesis[:50]}...")

        print(f"  [Optimizer] Theory confidence: {theory.confidence_score:.2f} → {new_confidence:.2f}")

        return result


# ============================================================================
# EPISTEMIC LOOP (Main Orchestrator)
# ============================================================================

class SimulatedEpistemicLoop:
    """Main epistemic pipeline loop."""

    DEFAULT_CURIOSITY_TOPICS = (
        "advanced physics theories",
        "artificial intelligence architecture",
        "local software environment",
        "physical surroundings",
    )

    def __init__(
        self,
        world_model: SimulatedWorldModel,
        explorer: SimulatedExplorer,
        analyzer: SimulatedAnalyzer,
        optimizer: SimulatedOptimizer,
        curiosity_interval: int = 5,
        max_theory_attempts: int = 3,
    ):
        self.world_model = world_model
        self.explorer = explorer
        self.analyzer = analyzer
        self.optimizer = optimizer
        self.curiosity_topics = list(self.DEFAULT_CURIOSITY_TOPICS)
        self.curiosity_interval = curiosity_interval
        self.max_theory_attempts = max_theory_attempts

        self._cycles_since_curiosity = 0
        self._topic_counts = {topic: 0 for topic in self.curiosity_topics}

    def _next_curiosity_topic(self) -> str:
        """Select least-explored curiosity topic."""
        least_explored = min(
            self.curiosity_topics,
            key=lambda t: self._topic_counts.get(t, 0),
        )
        self._topic_counts[least_explored] += 1
        return least_explored

    def _select_focus(self) -> tuple[Optional[Theory], Optional[str], ExecutionMode]:
        """Choose theory testing or curiosity exploration."""
        theory = self.world_model.get_untested_theory(self.max_theory_attempts)
        should_use_curiosity = theory is None or (
            self.curiosity_interval > 0
            and self._cycles_since_curiosity >= self.curiosity_interval
        )

        if should_use_curiosity:
            self._cycles_since_curiosity = 0
            return None, self._next_curiosity_topic(), ExecutionMode.CURIOSITY
        else:
            self._cycles_since_curiosity += 1
            return theory, None, ExecutionMode.THEORY_TESTING

    async def run_cycle(self, cycle_num: int):
        """Execute one complete epistemic cycle."""
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle_num}")
        print(f"{'='*80}")

        # 1. SELECT FOCUS
        theory, curiosity_topic, mode = self._select_focus()
        print(f"\n[SELECT] Mode: {mode.value}")
        if mode == ExecutionMode.THEORY_TESTING:
            print(f"[SELECT] Theory: {theory.text[:60]}...")
        else:
            print(f"[SELECT] Curiosity topic: {curiosity_topic}")

        # 2. EXPLORE
        print("\n[LAYER 2] EXPLORER")
        if mode == ExecutionMode.THEORY_TESTING:
            observation = await self.explorer.explore(theory)
        else:
            faux_theory = Theory(
                id="CURIOSITY_VOID",
                text=f"We currently lack a reliable model of {curiosity_topic}.",
            )
            observation = await self.explorer.explore(None, curiosity_topic)

        # 3. ANALYZE
        print("\n[LAYER 3] ANALYZER")
        if mode == ExecutionMode.THEORY_TESTING:
            prediction_error = await self.analyzer.analyze(theory, observation)
        else:
            prediction_error = await self.analyzer.analyze(faux_theory, observation)
            if not prediction_error.new_hypothesis and observation.sensor_data:
                prediction_error.new_hypothesis = (
                    f"Observation about {curiosity_topic}: {observation.sensor_data[:50]}"
                )

        # 4. OPTIMIZE
        print("\n[LAYER 4] OPTIMIZER")
        test_theory = theory if mode == ExecutionMode.THEORY_TESTING else faux_theory
        optimization = self.optimizer.optimize(
            test_theory,
            observation,
            prediction_error,
            mode,
            curiosity_topic if mode == ExecutionMode.CURIOSITY else theory.text,
        )

        # Summary
        print(f"\n[SUMMARY]")
        print(f"  Mode: {mode.value}")
        print(f"  Prediction match: {prediction_error.match}")
        print(f"  Theory updated: {optimization.theory_updated}")
        print(f"  Theory deleted: {optimization.theory_deleted}")
        print(f"  New theory injected: {optimization.injected_theory_id is not None}")

    async def run_n_cycles(self, n: int):
        """Run N cycles of the epistemic loop."""
        print(f"\n\n{'#'*80}")
        print(f"# EPISTEMIC PIPELINE SIMULATION - {n} CYCLES")
        print(f"{'#'*80}\n")

        for cycle_num in range(1, n + 1):
            await self.run_cycle(cycle_num)

        # Final statistics
        print(f"\n\n{'='*80}")
        print(f"FINAL STATE")
        print(f"{'='*80}\n")

        theories = sorted(
            self.world_model.get_all_theories(),
            key=lambda t: t.confidence_score,
            reverse=True,
        )

        print(f"Total theories: {len(theories)}")
        print(f"Total observations: {len(self.world_model.observations)}\n")

        print("Top 10 theories by confidence:")
        for i, theory in enumerate(theories[:10], 1):
            status = "✓" if theory.confidence_score > 0.5 else "?"
            print(
                f"  {i}. [{status}] {theory.text[:50]:50} | "
                f"Confidence: {theory.confidence_score:.2f} | "
                f"Attempts: {theory.attempts} | "
                f"PSR: {theory.predictive_success_rate:.2f}"
            )

        if len(theories) > 10:
            print(f"  ... and {len(theories) - 10} more theories\n")


# ============================================================================
# MAIN SIMULATION
# ============================================================================

async def main():
    """Run the epistemic pipeline simulation."""
    # Initialize components
    world_model = SimulatedWorldModel()
    explorer = SimulatedExplorer()
    analyzer = SimulatedAnalyzer()
    optimizer = SimulatedOptimizer(world_model)

    loop = SimulatedEpistemicLoop(
        world_model=world_model,
        explorer=explorer,
        analyzer=analyzer,
        optimizer=optimizer,
        curiosity_interval=5,
        max_theory_attempts=3,
    )

    # Run simulation
    await loop.run_n_cycles(10)

    print(f"\n{'#'*80}")
    print(f"# SIMULATION COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
