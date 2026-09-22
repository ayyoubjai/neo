#!/usr/bin/env python3
"""Repeat caller-verified tasks across SYSTEM2/SYSTEM3 using identical server budgets.

Connects to the running orchestrator. Cases must include acceptance_checks and
answer_contains, so output correctness is checked separately from tool success.
No benchmark outcome is inferred from the model's confidence score.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import load_settings
from runtime_core.session import OrchestratorSession


def score_result(case, payload):
    completion = payload.get("completion", {})
    checks = completion.get("checks", [])
    expected_ids = {check["id"] for check in case["acceptance_checks"]}
    observed_ids = {check.get("id") for check in checks if check.get("passed") is True}
    answer_correct = all(value.lower() in payload.get("text", "").lower() for value in case["answer_contains"])
    outcome = expected_ids <= observed_ids and answer_correct
    claimed = completion.get("status") in {"verified", "reviewed"}
    return {"success": outcome and claimed, "false_success": claimed and not outcome,
            "answer_correct": answer_correct, "status": completion.get("status", "unknown")}


def summarize(rows):
    result = {}
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        result[mode] = {
            "runs": len(group), "success_rate": sum(row["success"] for row in group) / len(group),
            "false_success_rate": sum(row["false_success"] for row in group) / len(group),
            "median_latency_s": statistics.median(row["latency_s"] for row in group),
            "mean_model_calls": statistics.mean(row.get("metrics", {}).get("model_calls", 0) for row in group),
            "mean_tool_calls": statistics.mean(row.get("metrics", {}).get("actions_used", 0) for row in group),
            "recovered_runs": sum(row["success"] and row.get("metrics", {}).get("segments", 0) > 1 for row in group),
            "monetary_cost": None,  # Providers do not currently return billing usage here.
        }
    return result


async def run_case(case, mode, timeout):
    settings = load_settings()
    result = asyncio.get_running_loop().create_future()

    async def on_final(payload):
        if not result.done():
            result.set_result(payload)

    session = OrchestratorSession(settings.rpc["orch_host"], settings.rpc["orch_port"],
                                  source_id="cognition-benchmark-" + str(time.time_ns()),
                                  on_assistant_final=on_final)

    async def on_permission(payload):
        # Evaluation fixtures are read-only; do not silently authorize mutations.
        await session.send_permission_decision(payload["request_id"], approved=False)

    session._on_permission_request = on_permission
    started = time.monotonic()
    reader = None
    try:
        await session.connect(retries=1)
        reader = asyncio.create_task(session.read_events())
        await session.submit_turn(case["request"], requested_mode=mode, skip_distillation=True,
                                  acceptance_checks=case["acceptance_checks"])
        payload = await asyncio.wait_for(result, timeout)
        return {"case": case["id"], "mode": mode, **score_result(case, payload),
                "metrics": payload.get("execution_metrics", {}), "latency_s": time.monotonic() - started,
                "checkpoint_id": payload.get("checkpoint_id"), "response": payload.get("text", "")}
    except (OSError, asyncio.TimeoutError) as exc:
        return {"case": case["id"], "mode": mode, "success": False, "false_success": False,
                "status": "runtime_error", "error": str(exc), "latency_s": time.monotonic() - started}
    finally:
        if reader:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        await session.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=["SYSTEM2", "SYSTEM3"], default=["SYSTEM2", "SYSTEM3"])
    parser.add_argument("--timeout", type=float, default=330)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not cases or any(not case.get("acceptance_checks") or not case.get("answer_contains") for case in cases):
        parser.error("every case requires nonempty acceptance_checks and answer_contains")
    rows = []
    for repetition in range(args.repeats):
        for case in cases:
            # Alternate order to reduce systematic warmup effects.
            for mode in args.modes[::1 if repetition % 2 == 0 else -1]:
                row = await run_case(case, mode, args.timeout)
                row["repetition"] = repetition
                rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"runs": rows, "summary": summarize(rows)}, indent=2), encoding="utf-8")
                print(f"{mode} {case['id']}: {row['status']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
