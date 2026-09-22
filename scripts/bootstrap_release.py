#!/usr/bin/env python3
"""Cross-platform dependency preflight for the AGI core release.

Without flags this script only reports what is available. `--install` creates
the project's `.venv` and installs Python requirements; it uses `npm ci` when
Node is available, but never installs Docker, llama.cpp, llama-router, drivers,
or operating-system packages on the user's behalf.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"


def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _command_status(name: str) -> str:
    location = shutil.which(name)
    return location if location else "missing"


def _node_major() -> int | None:
    node = shutil.which("node")
    if not node:
        return None
    try:
        output = subprocess.check_output([node, "--version"], text=True).strip().lstrip("v")
        return int(output.split(".", 1)[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _print_status() -> None:
    print(f"Repository: {ROOT}")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print(f"Virtual environment: {_venv_python() if _venv_python().exists() else 'not created'}")
    for command in ("git", "node", "npm", "docker", "llama-server", "llama-router", "ffmpeg"):
        print(f"{command}: {_command_status(command)}")
    node_major = _node_major()
    if node_major is not None:
        print(f"Node compatibility: {'OK' if node_major >= 20 else 'requires Node 20+'} (detected {node_major})")
    print("\nOptional capabilities:")
    print("  Telegram: add a BotFather token in .env during interactive setup.")
    print("  WhatsApp: requires Node 20+ and npm ci; pairing happens by QR code.")
    print("  SearXNG: requires Docker or Docker Desktop.")
    print("  Voice/video: ffmpeg is recommended; microphone/camera drivers are OS-managed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or install project-local release dependencies.")
    parser.add_argument("--install", action="store_true", help="Create .venv and install Python requirements; install Node dependencies when npm is present")
    parser.add_argument("--skip-node", action="store_true", help="Do not run npm ci during --install")
    args = parser.parse_args()

    _print_status()
    if not args.install:
        print("\nNext: run `python scripts/bootstrap_release.py --install` when ready.")
        return 0

    python = _venv_python()
    if not python.exists():
        print(f"\nCreating {VENV_DIR.relative_to(ROOT)}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python), "-m", "pip", "install", "-r", "requirements.txt"])
    if not args.skip_node:
        npm = shutil.which("npm")
        node_major = _node_major()
        if npm and node_major is not None and node_major >= 20:
            _run([npm, "ci"])
        elif npm:
            print("Node 20+ is required for WhatsApp; skipping npm ci.")
        else:
            print("npm is not installed; skipping WhatsApp dependencies.")
    print("\nDependencies are ready. Next run the interactive configuration command shown in docs/release-setup.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
