from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from embodiment.bootstrap import EmbodimentBootstrap
from orchestrator.main import Orchestrator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or apply embodiment provisioning before runtime use.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Generate and register missing embodiment tools before writing the manifest.",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Optional manifest output path. Defaults to data/embodiment_manifest.json.",
    )
    return parser.parse_args()


async def _run_apply(manifest_path: str) -> dict:
    orchestrator = Orchestrator()
    return await orchestrator.bootstrap_embodiment(manifest_path=manifest_path or None)


def main() -> int:
    args = _parse_args()
    manifest_path = str(args.manifest_path or "").strip()
    if args.apply:
        payload = asyncio.run(_run_apply(manifest_path))
    else:
        bootstrap = EmbodimentBootstrap()
        payload = bootstrap.plan()
        bootstrap.write_manifest(payload, path=manifest_path or None)
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
