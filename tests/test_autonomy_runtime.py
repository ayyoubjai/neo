from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autonomy.cognition_client import CognitionClientError
from autonomy.runtime import build_runtime_config, run_all_should_start, runtime_should_start, serve
from common.config import Settings, load_settings


class _FakeCognitionClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def wait_until_available(self, **kwargs):
        return None


class _UnavailableCognitionClient(_FakeCognitionClient):
    async def wait_until_available(self, **kwargs):
        raise CognitionClientError("COGNITION orchestrator is not reachable at 127.0.0.1:50051.")


def _settings(repo_root: Path, autonomy_overrides: dict | None = None) -> Settings:
    autonomy = {
        "enabled": False,
        "start_with_run_all": False,
        "epistemic_enabled": False,
        "power_process_enabled": False,
        "cognition_timeout_s": 180,
        "permission_policy": "deny",
        "requested_mode": "COGNITION",
        "epistemic_pause_s": 2.0,
        "power_process_pause_s": 2.0,
        "epistemic_curiosity_interval": 5,
        "epistemic_max_theory_attempts": 3,
        "goal_min_grounding_confidence": 0.3,
        "power_max_goal_attempts": 3,
    }
    if autonomy_overrides:
        autonomy.update(autonomy_overrides)
    return Settings(
        workspace_root=str(repo_root),
        data_dir=str(repo_root / "data"),
        rpc={},
        interface={},
        voice={},
        vision={},
        video={},
        telegram={},
        google={},
        tool={"active_tool_limit": 100},
        search={},
        orchestrator={},
        evolve={},
        models={"text_model": "test-model"},
        autonomy=autonomy,
    )


class AutonomyRuntimeConfigTests(unittest.TestCase):
    def test_runtime_config_defaults_to_disabled(self) -> None:
        config = build_runtime_config(_settings(REPO_ROOT))
        self.assertFalse(runtime_should_start(config))
        self.assertFalse(run_all_should_start(config))
        self.assertEqual(config.cognition_timeout_s, 180)
        self.assertEqual(config.permission_policy, "deny")
        self.assertEqual(config.requested_mode, "COGNITION")

    def test_runtime_config_supports_enabled_loops_and_cognition_settings(self) -> None:
        config = build_runtime_config(
            _settings(
                REPO_ROOT,
                {
                    "enabled": True,
                    "start_with_run_all": True,
                    "epistemic_enabled": True,
                    "power_process_enabled": True,
                    "cognition_timeout_s": "240",
                    "permission_policy": "approve",
                    "requested_mode": "system0",
                    "epistemic_pause_s": "1.5",
                    "goal_min_grounding_confidence": "0.6",
                    "power_max_goal_attempts": "4",
                },
            )
        )
        self.assertTrue(runtime_should_start(config))
        self.assertTrue(run_all_should_start(config))
        self.assertEqual(config.cognition_timeout_s, 240)
        self.assertEqual(config.permission_policy, "approve")
        self.assertEqual(config.requested_mode, "SYSTEM0")
        self.assertEqual(config.epistemic_pause_s, 1.5)
        self.assertEqual(config.goal_min_grounding_confidence, 0.6)
        self.assertEqual(config.power_max_goal_attempts, 4)

    def test_runtime_config_infers_enabled_when_loop_flags_are_set(self) -> None:
        config = build_runtime_config(
            _settings(
                REPO_ROOT,
                {
                    "enabled": None,
                    "epistemic_enabled": True,
                    "start_with_run_all": True,
                },
            )
        )
        self.assertTrue(runtime_should_start(config))
        self.assertTrue(run_all_should_start(config))

    def test_load_settings_preserves_autonomy_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "config").mkdir()
            payload = {
                "workspace_root": ".",
                "data_dir": "./data",
                "rpc": {},
                "interface": {},
                "voice": {},
                "vision": {},
                "video": {},
                "telegram": {},
                "google": {},
                "tool": {"active_tool_limit": 100},
                "search": {},
                "orchestrator": {},
                "evolve": {},
                "models": {"text_model": "test-model"},
                "autonomy": {
                    "enabled": True,
                    "start_with_run_all": True,
                    "epistemic_enabled": True,
                    "power_process_enabled": False,
                    "cognition_timeout_s": 300,
                    "permission_policy": "approve",
                },
            }
            settings_path = repo_root / "config" / "settings.json"
            settings_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

            settings = load_settings(str(settings_path))

            self.assertTrue(settings.autonomy["enabled"])
            self.assertTrue(settings.autonomy["start_with_run_all"])
            self.assertTrue(settings.autonomy["epistemic_enabled"])
            self.assertFalse(settings.autonomy["power_process_enabled"])
            self.assertEqual(settings.autonomy["cognition_timeout_s"], 300)
            self.assertEqual(settings.autonomy["permission_policy"], "approve")


class AutonomyRuntimeServeTests(unittest.IsolatedAsyncioTestCase):
    async def test_serve_exits_before_pipelines_when_cognition_is_unavailable(self) -> None:
        created: list[str] = []

        class _FakeWorldModel:
            def __init__(self):
                created.append("world_model")

            def bootstrap_axioms(self):
                created.append("bootstrap")

            def close(self):
                created.append("close")

        class _FakeEpistemicLoop:
            def __init__(self, **kwargs):
                created.append("epistemic")

        settings = _settings(REPO_ROOT, {"enabled": True, "epistemic_enabled": True})
        with (
            patch("autonomy.runtime.load_settings", return_value=settings),
            patch("epistemic.layer1_world_model.WorldModel", _FakeWorldModel),
            patch("autonomy.cognition_client.CognitionClient", _UnavailableCognitionClient),
            patch("epistemic.main_loop.EpistemicLoop", _FakeEpistemicLoop),
            patch("builtins.print") as print_mock,
        ):
            await serve()

        self.assertEqual(created, [])
        printed = " ".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("COGNITION orchestrator is not reachable", printed)

    async def test_serve_can_run_epistemic_pipeline_only(self) -> None:
        created: list[str] = []
        ran: list[str] = []

        class _FakeWorldModel:
            def bootstrap_axioms(self):
                created.append("bootstrap")

            def close(self):
                created.append("close")

        class _FakeEpistemicLoop:
            def __init__(self, **kwargs):
                created.append("epistemic")
                self.write_lock = kwargs["write_lock"]

            async def run_forever(self):
                ran.append("epistemic")

        class _FakePowerLoop:
            def __init__(self, **kwargs):
                created.append("power")

            async def run_forever(self):
                ran.append("power")

        settings = _settings(REPO_ROOT, {"enabled": True, "epistemic_enabled": True})
        with (
            patch("autonomy.runtime.load_settings", return_value=settings),
            patch("epistemic.layer1_world_model.WorldModel", _FakeWorldModel),
            patch("autonomy.cognition_client.CognitionClient", _FakeCognitionClient),
            patch("epistemic.layer2_exploration.Explorer", lambda **kwargs: object()),
            patch("epistemic.layer3_analyzer.Analyzer", lambda **kwargs: object()),
            patch("epistemic.layer4_optimizer.Optimizer", lambda world_model: object()),
            patch("epistemic.main_loop.EpistemicLoop", _FakeEpistemicLoop),
            patch("power_process.power_loop.PowerProcessLoop", _FakePowerLoop),
        ):
            await serve()

        self.assertIn("bootstrap", created)
        self.assertIn("close", created)
        self.assertEqual(ran, ["epistemic"])
        self.assertNotIn("power", created)

    async def test_serve_can_run_power_pipeline_only(self) -> None:
        created: list[str] = []
        ran: list[str] = []

        class _FakeWorldModel:
            def bootstrap_axioms(self):
                created.append("bootstrap")

            def close(self):
                created.append("close")

        class _FakeEpistemicLoop:
            def __init__(self, **kwargs):
                created.append("epistemic")

            async def run_forever(self):
                ran.append("epistemic")

        class _FakePowerLoop:
            def __init__(self, **kwargs):
                created.append("power")
                self.write_lock = kwargs["write_lock"]

            async def run_forever(self):
                ran.append("power")

        settings = _settings(REPO_ROOT, {"enabled": True, "power_process_enabled": True})
        with (
            patch("autonomy.runtime.load_settings", return_value=settings),
            patch("epistemic.layer1_world_model.WorldModel", _FakeWorldModel),
            patch("autonomy.cognition_client.CognitionClient", _FakeCognitionClient),
            patch("power_process.layer5_motivator.Motivator", lambda *args, **kwargs: object()),
            patch("power_process.layer6_executor.Executor", lambda **kwargs: object()),
            patch("power_process.layer7_evaluator.Evaluator", lambda *args, **kwargs: object()),
            patch("epistemic.main_loop.EpistemicLoop", _FakeEpistemicLoop),
            patch("power_process.power_loop.PowerProcessLoop", _FakePowerLoop),
        ):
            await serve()

        self.assertIn("bootstrap", created)
        self.assertIn("close", created)
        self.assertEqual(ran, ["power"])
        self.assertNotIn("epistemic", created)

    async def test_serve_runs_both_pipelines_with_shared_write_lock(self) -> None:
        locks: dict[str, object] = {}
        ran: list[str] = []

        class _FakeWorldModel:
            def bootstrap_axioms(self):
                return 0

            def close(self):
                return None

        class _FakeEpistemicLoop:
            def __init__(self, **kwargs):
                locks["epistemic"] = kwargs["write_lock"]

            async def run_forever(self):
                ran.append("epistemic")

        class _FakePowerLoop:
            def __init__(self, **kwargs):
                locks["power"] = kwargs["write_lock"]

            async def run_forever(self):
                ran.append("power")

        settings = _settings(
            REPO_ROOT,
            {"enabled": True, "epistemic_enabled": True, "power_process_enabled": True},
        )
        with (
            patch("autonomy.runtime.load_settings", return_value=settings),
            patch("epistemic.layer1_world_model.WorldModel", _FakeWorldModel),
            patch("autonomy.cognition_client.CognitionClient", _FakeCognitionClient),
            patch("epistemic.layer2_exploration.Explorer", lambda **kwargs: object()),
            patch("epistemic.layer3_analyzer.Analyzer", lambda **kwargs: object()),
            patch("epistemic.layer4_optimizer.Optimizer", lambda world_model: object()),
            patch("power_process.layer5_motivator.Motivator", lambda *args, **kwargs: object()),
            patch("power_process.layer6_executor.Executor", lambda **kwargs: object()),
            patch("power_process.layer7_evaluator.Evaluator", lambda *args, **kwargs: object()),
            patch("epistemic.main_loop.EpistemicLoop", _FakeEpistemicLoop),
            patch("power_process.power_loop.PowerProcessLoop", _FakePowerLoop),
        ):
            await serve()

        self.assertEqual(sorted(ran), ["epistemic", "power"])
        self.assertIs(locks["epistemic"], locks["power"])


if __name__ == "__main__":
    unittest.main()
