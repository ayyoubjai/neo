import argparse
import json
import os
import subprocess
import sys
import time
from typing import List


def _default_output_path() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join("evolve", "scenarios", "traces", f"{stamp}.jsonl")


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return count


def _normalize_command(parts: List[str]) -> List[str]:
    if parts and parts[0] == "--":
        return parts[1:]
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command with trace capture enabled and write trace JSONL."
    )
    parser.add_argument(
        "--output",
        default="",
        help="Trace output path. Default: evolve/scenarios/traces/<timestamp>.jsonl",
    )
    parser.add_argument(
        "--object-ids",
        default="",
        help="Comma-separated object ids to trace. Empty means all traced objects.",
    )
    parser.add_argument(
        "--max-repr-chars",
        type=int,
        default=500,
        help="Max repr chars stored in trace previews.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of truncating.",
    )
    parser.add_argument(
        "--cwd",
        default="",
        help="Working directory for the command. Default: current directory.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    args = parser.parse_args()

    command = _normalize_command(args.command)
    if not command:
        sys.stderr.write("error: missing command. Use -- <command ...>\n")
        return 2

    output_rel = args.output.strip() or _default_output_path()
    output_path = output_rel if os.path.isabs(output_rel) else os.path.abspath(output_rel)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not args.append and os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8"):
            pass

    env = os.environ.copy()
    env["EVOLVE_TRACE_ENABLE"] = "1"
    env["EVOLVE_TRACE_OUTPUT"] = output_path
    env["EVOLVE_TRACE_MAX_REPR_CHARS"] = str(max(64, int(args.max_repr_chars)))
    if args.object_ids.strip():
        env["EVOLVE_TRACE_OBJECTS"] = args.object_ids.strip()

    cwd = args.cwd.strip() or os.getcwd()
    proc = subprocess.run(command, cwd=cwd, env=env)
    total_events = _count_lines(output_path)

    summary = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "trace_file": output_path,
        "events": total_events,
        "command": command,
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=True) + "\n")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
