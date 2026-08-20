#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reflection_engine import (
    find_latest_reflection_run,
    load_improve_config,
    load_reflection_run,
    prepare_candidate,
    select_dossier,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "reflection_improve.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an isolated improvement candidate from a reflection dossier.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to reflection improvement config JSON.")
    parser.add_argument("--run-dir", default="", help="Specific reflection run directory. Default: latest run.")
    parser.add_argument("--dossier-id", default="", help="Specific dossier id to prepare.")
    parser.add_argument("--kind", default="", help="Select the first dossier of the given kind.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_improve_config(Path(args.config).resolve())
    runs_root = REPO_ROOT / config.runs_dir
    run_dir = Path(args.run_dir).resolve() if args.run_dir else find_latest_reflection_run(runs_root)
    run = load_reflection_run(run_dir)
    dossier = select_dossier(run, dossier_id=args.dossier_id, kind=args.kind)
    prepared = prepare_candidate(REPO_ROOT, config, run, dossier)
    print(
        "[reflection-candidate] "
        f"candidate={prepared.candidate_id} "
        f"dossier={prepared.manifest['dossier_id']} "
        f"kind={prepared.manifest['dossier_kind']} "
        f"dir={prepared.candidate_dir.relative_to(REPO_ROOT).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
