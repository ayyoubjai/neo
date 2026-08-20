from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import List

from adaptive_agent.runner import build_model_server_command, explain_profile, load_active_profile
from adaptive_agent.settings_apply import apply_profile_settings
from adaptive_agent.wizard import DEFAULT_OUTPUT_PATH


REPO_ROOT = Path(__file__).resolve().parents[2]


def default_run_all_command() -> List[str]:
    if os.name == "nt":
        return ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts/run_all.ps1"]
    return ["bash", "scripts/run_all.sh"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the adaptive model backend, then run the AGI system.")
    parser.add_argument("--profile", default=str(DEFAULT_OUTPUT_PATH), help="Path to active adaptive profile JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching anything.")
    parser.add_argument("--no-run-all", action="store_true", help="Start only the selected model backend.")
    parser.add_argument("--run-all-command", help="Override run_all command, parsed with shell-like quoting.")
    args = parser.parse_args(argv)

    profile = load_active_profile(args.profile)
    print(explain_profile(profile))
    if not args.dry_run:
        applied = apply_profile_settings(profile_path=args.profile)
        print(f"applied cognition mode: {applied.get('cognition_mode', '')}")
    try:
        model_command = build_model_server_command(profile)
    except ValueError as e:
        print(str(e))
        return 2

    run_all_command = shlex.split(args.run_all_command) if args.run_all_command else default_run_all_command()
    print(f"model command: {shlex.join(model_command)}")
    if not args.no_run_all:
        print(f"system command: {shlex.join(run_all_command)}")
    if args.dry_run:
        return 0

    model_proc = subprocess.Popen(model_command, cwd=str(REPO_ROOT))
    if args.no_run_all:
        return model_proc.wait()

    try:
        return subprocess.call(run_all_command, cwd=str(REPO_ROOT))
    finally:
        if model_proc.poll() is None:
            if os.name == "nt":
                model_proc.terminate()
            else:
                model_proc.send_signal(signal.SIGTERM)
            try:
                model_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                model_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
