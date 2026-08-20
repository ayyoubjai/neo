#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = SCRIPT_DIR.parent
LOCAL_AGENT = SCRIPT_DIR / "local_code_agent.py"


def _has_repo_root_arg(args: Sequence[str]) -> bool:
    for arg in args:
        if arg == "--repo-root" or arg.startswith("--repo-root="):
            return True
    return False


def build_agent_argv(args: Sequence[str], cwd: Path) -> List[str]:
    argv = [sys.executable, str(LOCAL_AGENT)]
    if not _has_repo_root_arg(args):
        argv.extend(["--repo-root", str(cwd.resolve())])
    argv.extend(args)
    return argv


def main() -> int:
    if not LOCAL_AGENT.exists():
        print(f"[agi-code] local agent not found: {LOCAL_AGENT}", file=sys.stderr)
        return 1
    argv = build_agent_argv(sys.argv[1:], Path.cwd())
    env = os.environ.copy()
    src_root = str(SYSTEM_ROOT / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root if not current_pythonpath else src_root + os.pathsep + current_pythonpath
    proc = subprocess.run(argv, cwd=str(SYSTEM_ROOT), env=env, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
