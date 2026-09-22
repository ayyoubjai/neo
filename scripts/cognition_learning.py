#!/usr/bin/env python3
"""Inspect, explicitly approve/reject, or retract provisional cognition learning."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orchestrator.main import Orchestrator


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "show", "approve", "reject", "retract"])
    parser.add_argument("candidate_id", nargs="?")
    parser.add_argument("--reviewer")
    parser.add_argument("--reason")
    args = parser.parse_args()
    orchestrator = Orchestrator()
    store = orchestrator._run_store()
    if args.action == "list":
        rows = []
        for path in sorted((store.root / "learning").glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            rows.append({key: item.get(key) for key in ("candidate_id", "status", "original_request")})
        print(json.dumps(rows, indent=2))
        return
    if not args.candidate_id:
        parser.error("candidate_id is required")
    if args.action == "show":
        result = store.read("learning", args.candidate_id)
    else:
        if not args.reviewer or not args.reason:
            parser.error("reviewer and reason are required for review or retraction")
        if args.action == "retract":
            result = orchestrator.retract_cognition_learning(
                args.candidate_id, reviewer=args.reviewer, reason=args.reason)
        else:
            result = await orchestrator.review_cognition_learning(
                args.candidate_id, reviewer=args.reviewer, reason=args.reason, approve=args.action == "approve")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
