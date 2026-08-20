#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import load_settings
from common.soul_mirror import load_config as load_soul_config
from common.soul_mirror import sync_soul
from reflection_engine import build_reflection_run, collect_runtime_observations, load_config, load_soul_snapshot, write_reflection_run


DEFAULT_REFLECTION_CONFIG = REPO_ROOT / "config" / "reflection.json"
DEFAULT_SOUL_CONFIG = REPO_ROOT / "config" / "soul.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ranked self-reflection dossiers from soul + runtime evidence.")
    parser.add_argument("--config", default=str(DEFAULT_REFLECTION_CONFIG), help="Path to reflection config JSON.")
    parser.add_argument("--soul-config", default=str(DEFAULT_SOUL_CONFIG), help="Path to soul config JSON.")
    parser.add_argument("--no-refresh-soul", action="store_true", help="Use the existing soul mirror without refreshing it first.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    reflection_config = load_config(Path(args.config).resolve())
    soul_config = load_soul_config(Path(args.soul_config).resolve())
    if not args.no_refresh_soul:
        sync_soul(REPO_ROOT, soul_config)

    soul_root = REPO_ROOT / soul_config.output_dir
    soul_snapshot = load_soul_snapshot(soul_root)
    settings = load_settings()
    runtime = collect_runtime_observations(
        repo_root=REPO_ROOT,
        data_dir=Path(settings.data_dir),
        max_record_events=reflection_config.max_record_events,
        max_complex_events=reflection_config.max_complex_events,
        max_error_events=reflection_config.max_error_events,
        max_audit_events=reflection_config.max_audit_events,
    )
    run = build_reflection_run(REPO_ROOT, reflection_config, soul_snapshot, runtime)
    run_dir = write_reflection_run(REPO_ROOT, reflection_config, run)
    print(
        "[reflection] "
        f"wrote={run_dir.relative_to(REPO_ROOT).as_posix()} "
        f"dossiers={run.manifest['dossier_count']} "
        f"trace_failures={run.observations['trace_failure_count']} "
        f"create_tool_requests={run.observations['create_tool_request_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
