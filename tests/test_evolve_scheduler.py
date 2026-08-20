from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import Settings, load_settings
from runtime_core.evolve_scheduler import (
    ResourceSnapshot,
    _calendar_due_slot,
    _hold_pid_lock,
    _read_lock_payload as _read_scheduler_lock_payload,
    _release_pid_lock,
    build_scheduler_config,
    evaluate_resource_gate,
    scheduler_should_start,
)
from scripts.evolve_improve import _hold_evolution_lock, _read_lock_payload, _release_evolution_lock


def _settings(repo_root: Path, **evolve_overrides: object) -> Settings:
    evolve = {
        "candidate_dir": str(repo_root / "evolve" / "candidates"),
        "auto_run_enabled": False,
        "auto_run_mode": "interval",
        "auto_run_on_startup": False,
        "auto_run_interval_s": 0,
        "auto_run_initial_delay_s": 300,
        "auto_run_calendar_every_days": 3,
        "auto_run_calendar_hour_local": 3,
        "auto_run_calendar_minute_local": 0,
        "auto_run_calendar_timezone": "UTC",
        "auto_run_calendar_catch_up_if_missed": True,
        "auto_run_calendar_anchor_date_local": "",
        "auto_run_config_path": str(repo_root / "evolve" / "improve_config.json"),
        "auto_run_log_path": str(repo_root / "data" / "evolve_scheduler.log"),
        "scheduler_state_path": str(repo_root / "data" / "evolve_scheduler_state.json"),
        "scheduler_lock_path": str(repo_root / "data" / "evolve_scheduler.lock"),
        "run_lock_path": str(repo_root / "data" / "evolve_improve.lock"),
        "resource_gate_enabled": False,
        "resource_check_interval_s": 300,
        "resource_required_consecutive_passes": 1,
        "resource_min_free_ram_mb": 2048,
        "resource_min_free_disk_gb": 10,
        "resource_max_cpu_percent": 70,
        "resource_max_load_avg_1m": 0,
        "resource_disk_path": str(repo_root),
    }
    evolve.update(evolve_overrides)
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
        evolve=evolve,
        models={"text_model": "test-model", "text_enforce_json": False},
    )


class EvolveSchedulerTests(unittest.TestCase):
    def test_build_scheduler_config_reads_evolve_settings(self) -> None:
        settings = _settings(
            REPO_ROOT,
            auto_run_enabled=True,
            auto_run_mode="calendar",
            auto_run_on_startup=True,
            auto_run_interval_s="900",
            auto_run_initial_delay_s="45",
            auto_run_calendar_every_days="5",
            auto_run_calendar_hour_local="2",
            auto_run_calendar_minute_local="30",
            resource_gate_enabled=True,
            resource_required_consecutive_passes="2",
            resource_min_free_disk_gb="12.5",
        )

        config = build_scheduler_config(settings)

        self.assertTrue(config.enabled)
        self.assertEqual(config.mode, "calendar")
        self.assertTrue(config.run_on_startup)
        self.assertEqual(config.interval_s, 900.0)
        self.assertEqual(config.initial_delay_s, 45.0)
        self.assertEqual(config.calendar_every_days, 5)
        self.assertEqual(config.calendar_hour_local, 2)
        self.assertEqual(config.calendar_minute_local, 30)
        self.assertTrue(config.resource_gate_enabled)
        self.assertEqual(config.resource_required_consecutive_passes, 2)
        self.assertEqual(config.resource_min_free_disk_gb, 12.5)
        self.assertTrue(config.config_path.endswith("evolve/improve_config.json"))
        self.assertTrue(config.log_path.endswith("data/evolve_scheduler.log"))
        self.assertTrue(config.state_path.endswith("data/evolve_scheduler_state.json"))
        self.assertTrue(config.scheduler_lock_path.endswith("data/evolve_scheduler.lock"))

    def test_scheduler_should_start_requires_enabled_and_trigger(self) -> None:
        disabled = build_scheduler_config(_settings(REPO_ROOT, auto_run_enabled=False, auto_run_on_startup=True))
        enabled_startup = build_scheduler_config(_settings(REPO_ROOT, auto_run_enabled=True, auto_run_on_startup=True))
        enabled_interval = build_scheduler_config(_settings(REPO_ROOT, auto_run_enabled=True, auto_run_interval_s=60))
        enabled_calendar = build_scheduler_config(
            _settings(
                REPO_ROOT,
                auto_run_enabled=True,
                auto_run_mode="calendar",
                auto_run_interval_s=0,
                auto_run_calendar_every_days=3,
            )
        )

        self.assertFalse(scheduler_should_start(disabled))
        self.assertTrue(scheduler_should_start(enabled_startup))
        self.assertTrue(scheduler_should_start(enabled_interval))
        self.assertTrue(scheduler_should_start(enabled_calendar))

    def test_load_settings_remaps_evolve_scheduler_paths(self) -> None:
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
                "evolve": {
                    "candidate_dir": "./evolve/candidates",
                    "auto_run_config_path": "./evolve/improve_config.json",
                    "auto_run_log_path": "./data/evolve_scheduler.log",
                    "scheduler_state_path": "./data/evolve_scheduler_state.json",
                    "scheduler_lock_path": "./data/evolve_scheduler.lock",
                    "run_lock_path": "./data/evolve_improve.lock",
                    "resource_disk_path": ".",
                },
                "models": {"text_model": "test-model"},
            }
            settings_path = repo_root / "config" / "settings.json"
            settings_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

            settings = load_settings(str(settings_path))

            self.assertEqual(settings.evolve["auto_run_config_path"], str(repo_root / "evolve" / "improve_config.json"))
            self.assertEqual(settings.evolve["auto_run_log_path"], str(repo_root / "data" / "evolve_scheduler.log"))
            self.assertEqual(settings.evolve["scheduler_state_path"], str(repo_root / "data" / "evolve_scheduler_state.json"))
            self.assertEqual(settings.evolve["scheduler_lock_path"], str(repo_root / "data" / "evolve_scheduler.lock"))
            self.assertEqual(settings.evolve["run_lock_path"], str(repo_root / "data" / "evolve_improve.lock"))
            self.assertEqual(settings.evolve["resource_disk_path"], str(repo_root))

    def test_calendar_due_slot_catches_up_latest_missed_slot(self) -> None:
        config = build_scheduler_config(
            _settings(
                REPO_ROOT,
                auto_run_enabled=True,
                auto_run_mode="calendar",
                auto_run_calendar_every_days=3,
                auto_run_calendar_hour_local=3,
                auto_run_calendar_minute_local=0,
                auto_run_calendar_timezone="UTC",
                auto_run_calendar_catch_up_if_missed=True,
                auto_run_calendar_anchor_date_local="2026-03-20",
            )
        )

        state, due_slot = _calendar_due_slot(config, {}, datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc))

        self.assertIsNotNone(due_slot)
        self.assertEqual(due_slot.isoformat(), "2026-03-29T03:00:00+00:00")
        self.assertEqual(state["next_calendar_slot_at"], "2026-03-29T03:00:00+00:00")

    def test_calendar_due_slot_skips_missed_slot_without_catch_up(self) -> None:
        config = build_scheduler_config(
            _settings(
                REPO_ROOT,
                auto_run_enabled=True,
                auto_run_mode="calendar",
                auto_run_calendar_every_days=3,
                auto_run_calendar_hour_local=3,
                auto_run_calendar_minute_local=0,
                auto_run_calendar_timezone="UTC",
                auto_run_calendar_catch_up_if_missed=False,
                auto_run_calendar_anchor_date_local="2026-03-20",
            )
        )

        state, due_slot = _calendar_due_slot(config, {}, datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc))

        self.assertIsNone(due_slot)
        self.assertEqual(state["next_calendar_slot_at"], "2026-04-01T03:00:00+00:00")

    def test_evaluate_resource_gate_blocks_threshold_breaches(self) -> None:
        config = build_scheduler_config(
            _settings(
                REPO_ROOT,
                resource_gate_enabled=True,
                resource_min_free_ram_mb=1024,
                resource_min_free_disk_gb=10,
                resource_max_cpu_percent=70,
                resource_max_load_avg_1m=2,
            )
        )
        snapshot = ResourceSnapshot(
            free_ram_bytes=512 * 1024 * 1024,
            free_disk_bytes=2 * (1024 ** 3),
            cpu_percent=85.0,
            load_avg_1m=4.0,
            cpu_count=8,
        )

        allowed, reasons, metrics = evaluate_resource_gate(config, snapshot)

        self.assertFalse(allowed)
        self.assertIn("free_ram_mb=512.0<1024.0", reasons)
        self.assertIn("free_disk_gb=2.0<10.0", reasons)
        self.assertIn("cpu_percent=85.0>70.0", reasons)
        self.assertIn("load_avg_1m=4.0>2.0", reasons)
        self.assertEqual(metrics["cpu_count"], 8)

    def test_hold_evolution_lock_replaces_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "evolve.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                json.dump({"pid": 999999, "created_at": "2026-03-29T00:00:00Z"}, f)

            _hold_evolution_lock(lock_path)
            payload = _read_lock_payload(lock_path)
            self.assertEqual(int(payload.get("pid", 0) or 0), os.getpid())
            _release_evolution_lock(lock_path, os.getpid())
            self.assertFalse(os.path.exists(lock_path))

    def test_hold_evolution_lock_rejects_active_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "evolve.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "created_at": "2026-03-29T00:00:00Z"}, f)

            with self.assertRaises(RuntimeError):
                _hold_evolution_lock(lock_path)

    def test_hold_scheduler_lock_replaces_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "scheduler.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                json.dump({"pid": 999999, "created_at": "2026-03-29T00:00:00Z"}, f)

            _hold_pid_lock(lock_path)
            payload = _read_scheduler_lock_payload(lock_path)
            self.assertEqual(int(payload.get("pid", 0) or 0), os.getpid())
            _release_pid_lock(lock_path, os.getpid())
            self.assertFalse(os.path.exists(lock_path))


if __name__ == "__main__":
    unittest.main()
