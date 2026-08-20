#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.soul_mirror import SoulConfig, load_config, sync_soul, watch_soul


DEFAULT_CONFIG = REPO_ROOT / "config" / "soul.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and maintain the soul analysis mirror.")
    subparsers = parser.add_subparsers(dest="command")

    sync_parser = subparsers.add_parser("sync", help="Run one sync pass.")
    sync_parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to soul config JSON.")

    watch_parser = subparsers.add_parser("watch", help="Continuously sync the mirror.")
    watch_parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to soul config JSON.")
    watch_parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds.")

    parser.set_defaults(command="sync")
    return parser.parse_args()


def _load_runtime_config(config_path: str) -> SoulConfig:
    return load_config(Path(config_path).resolve())


def main() -> int:
    args = _parse_args()
    config = _load_runtime_config(args.config)
    if args.command == "watch":
        watch_soul(REPO_ROOT, config=config, interval_s=args.interval)
        return 0

    result = sync_soul(REPO_ROOT, config=config)
    status = "updated" if result.changed else "no changes"
    print(
        "[soul] "
        f"{status}: files={result.manifest['file_count']} "
        f"symbols={result.manifest['symbol_count']} "
        f"edges={result.manifest['internal_dependency_edge_count']} "
        f"copied={result.copied_files} removed={result.removed_files}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
