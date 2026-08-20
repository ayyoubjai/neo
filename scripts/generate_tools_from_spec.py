#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.ids import new_id
from orchestrator.main import Orchestrator


def _load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("spec file must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate runtime tools from a high-level JSON spec.")
    parser.add_argument("--spec", required=True, help="Path to a JSON spec file.")
    parser.add_argument("--trace-id", default="", help="Optional trace id override.")
    args = parser.parse_args()

    spec_path = Path(args.spec).expanduser().resolve()
    spec = _load_spec(spec_path)
    orchestrator = Orchestrator()
    trace_id = args.trace_id.strip() or new_id()
    result = asyncio.run(orchestrator.generate_tools_from_spec(spec, trace_id=trace_id))
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get("status") == "APPROVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
