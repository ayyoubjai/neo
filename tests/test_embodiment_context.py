from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestrator.context_builder import ContextBuilder
from orchestrator.main import Orchestrator


class _FakeEmbodimentManager:
    def __init__(self, snapshot=None, exc: Exception | None = None) -> None:
        self._snapshot = snapshot or {}
        self._exc = exc
        self.calls = 0

    def summarize_context(self):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return dict(self._snapshot)


class EmbodimentContextTests(unittest.TestCase):
    def test_context_builder_injects_and_caches_embodiment_summary(self) -> None:
        manager = _FakeEmbodimentManager(
            snapshot={
                "summary": "host=desktop; interface=local; available_capabilities=screen.capture, camera.capture",
                "available_capabilities": ["screen.capture", "camera.capture"],
                "configured_capabilities": ["vision.detect"],
            }
        )
        builder = ContextBuilder(
            max_turns=2,
            embodiment_manager=manager,
            embodiment_cache_ttl_s=60.0,
        )
        history = [
            {"role": "user", "text": "one"},
            {"role": "assistant", "text": "two"},
            {"role": "user", "text": "three"},
        ]

        first = builder.build(history, "summary-1", [{"id": 1}])
        second = builder.build(history, "summary-2", [])

        self.assertEqual(manager.calls, 1)
        self.assertEqual(first["recent_turns"], history[-2:])
        self.assertEqual(first["embodiment"]["summary"], manager._snapshot["summary"])
        self.assertEqual(
            second["embodiment"]["available_capabilities"],
            ["screen.capture", "camera.capture"],
        )

    def test_context_builder_falls_back_when_embodiment_probe_fails(self) -> None:
        builder = ContextBuilder(
            embodiment_manager=_FakeEmbodimentManager(exc=RuntimeError("probe failed")),
            embodiment_cache_ttl_s=0.0,
        )

        packet = builder.build([], "", [])

        self.assertEqual(packet["embodiment"]["summary"], "Embodiment context unavailable.")
        self.assertEqual(packet["embodiment"]["available_capabilities"], [])
        self.assertEqual(packet["embodiment"]["configured_capabilities"], [])

    def test_orchestrator_build_llm_context_normalizes_to_cognition(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._turn_context_packet["turn-1"] = {
            "summary": "conversation-summary",
            "memory": [{"data": {"summary": "memory-hit"}}],
            "embodiment": {
                "summary": "host=laptop; interface=local; screen=1920x1080",
                "available_capabilities": ["screen.capture"],
            },
        }

        ctx = orchestrator._build_llm_context("AGENT", "trace-1", "turn-1")

        self.assertEqual(ctx["mode"], "COGNITION")
        self.assertEqual(ctx["summary"], "conversation-summary")
        self.assertEqual(ctx["memory"][0]["data"]["summary"], "memory-hit")
        self.assertNotIn("embodiment", ctx)

    def test_orchestrator_build_llm_context_strips_embodiment_for_cognition(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._turn_context_packet["turn-cognition"] = {
            "summary": "conversation-summary",
            "memory": [{"data": {"summary": "memory-hit"}}],
            "embodiment": {
                "summary": "host=laptop; interface=local; screen=1920x1080",
                "available_capabilities": ["screen.capture"],
            },
        }

        ctx = orchestrator._build_llm_context("COGNITION", "trace-cognition", "turn-cognition")

        self.assertEqual(ctx["mode"], "COGNITION")
        self.assertEqual(ctx["summary"], "conversation-summary")
        self.assertEqual(ctx["memory"][0]["data"]["summary"], "memory-hit")
        self.assertNotIn("embodiment", ctx)

    def test_orchestrator_build_llm_context_keeps_summary_and_memory_for_legacy_mode_inputs(self) -> None:
        orchestrator = Orchestrator()
        orchestrator._turn_context_packet["turn-asi"] = {
            "summary": "conversation-summary",
            "memory": [{"data": {"summary": "memory-hit"}}],
            "embodiment": {
                "summary": "host=laptop; interface=local; screen=1920x1080",
                "available_capabilities": ["screen.capture"],
            },
        }

        ctx = orchestrator._build_llm_context("ASI", "trace-asi", "turn-asi")

        self.assertEqual(ctx["mode"], "COGNITION")
        self.assertEqual(ctx["summary"], "conversation-summary")
        self.assertEqual(ctx["memory"][0]["data"]["summary"], "memory-hit")
        self.assertNotIn("embodiment", ctx)


if __name__ == "__main__":
    unittest.main()
