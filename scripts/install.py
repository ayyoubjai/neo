#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Agent – Self-contained Installer
==========================================
Place this file alongside adaptive_agent_dna.zip in your distribution folder.
Run once on any new device (PC or Android/Termux):

    python install.py

Optional flags:
    --purpose   coding | assistant | research | automation | phone_companion
    --personality "short description"
    --allow-network
    --allow-background
    --target    /path/to/install/dir   (default: ./agi_runtime next to this script)
    --source    /path/to/dna.zip       (default: ./adaptive_agent_dna.zip)
    --dry-run   show what would happen without doing it

What it does:
  1. Extracts the full repo structure from the DNA zip into --target
  2. Runs the wizard (device probe + gene resolution) to write active_profile.json
  3. Decodes only the needed bundles for this device into target/runtime/adaptive_cell
  4. Prints how to start the system
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# Make stdout/stderr UTF-8 safe on Windows (CP1252 consoles crash on special chars)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
DEFAULT_DNA = HERE / "adaptive_agent_dna.zip"
DEFAULT_TARGET = HERE / "agi_runtime"


# ---------------------------------------------------------------------------
# Step 1 – Bootstrap: extract the full repo from the DNA zip
# ---------------------------------------------------------------------------

def extract_dna(source: Path, target: Path) -> None:
    if not source.exists():
        _die(
            f"DNA zip not found: {source}\n"
            "Make sure adaptive_agent_dna.zip is in the same folder as install.py"
        )
    print(f"[1/4] Extracting DNA archive -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as zf:
        members = zf.namelist()
        for i, name in enumerate(members, 1):
            dest = target / name
            if name.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            _progress("extracting", i, len(members))
    print()  # newline after progress bar


# ---------------------------------------------------------------------------
# Step 2 – Run the wizard (inside the extracted repo)
# ---------------------------------------------------------------------------

def run_wizard(
    target: Path,
    purpose: str,
    personality: str,
    allow_network: bool,
    allow_background: bool,
    cognition_mode: str,
    dry_run: bool,
) -> Path:
    wizard_script = target / "scripts" / "adaptive_agent_wizard.py"
    if not wizard_script.exists():
        _die(
            f"Wizard script not found at {wizard_script}\n"
            "The DNA zip may be incomplete — re-pack it with:\n"
            "  python scripts/adaptive_agent_dna.py pack --output dist/adaptive_agent_dna.zip"
        )

    profile_path = target / "config" / "adaptive_agent" / "active_profile.json"
    cmd = [
        sys.executable,
        str(wizard_script),
        "--purpose", purpose,
        "--personality", personality,
        "--cognition-mode", cognition_mode,
        "--output", str(profile_path),
    ]
    if allow_network:
        cmd.append("--allow-network")
    if allow_background:
        cmd.append("--allow-background")

    print(f"[2/4] Running wizard: purpose={purpose}")
    if dry_run:
        print(f"  (dry-run) would run: {' '.join(cmd)}")
        return profile_path

    env = _env_with_src(target)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        _die(f"Wizard failed (exit {result.returncode})")

    if not profile_path.exists():
        _die(
            f"Wizard ran but {profile_path} was not created.\n"
            "Check for errors above."
        )
    return profile_path


# ---------------------------------------------------------------------------
# Step 3 – Decode only needed bundles into runtime/adaptive_cell
# ---------------------------------------------------------------------------

def decode_bundles(
    target: Path,
    profile_path: Path,
    source: Path,
    dry_run: bool,
) -> Path:
    dna_script = target / "scripts" / "adaptive_agent_dna.py"
    cell_path = target / "runtime" / "adaptive_cell"
    bundles_path = target / "config" / "adaptive_agent" / "bundles.json"

    cmd = [
        sys.executable,
        str(dna_script),
        "decode",
        "--profile", str(profile_path),
        "--source", str(source),
        "--target", str(cell_path),
        "--bundles", str(bundles_path),
    ]

    print(f"[3/4] Decoding bundles -> {cell_path}")
    if dry_run:
        print(f"  (dry-run) would run: {' '.join(cmd)}")
        return cell_path

    env = _env_with_src(target)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        _die(f"Bundle decode failed (exit {result.returncode})")
    return cell_path


# ---------------------------------------------------------------------------
# Step 4 – Print summary and next-step instructions
# ---------------------------------------------------------------------------

def print_summary(target: Path, profile_path: Path, dry_run: bool) -> None:
    print("[4/4] Installation summary")
    print(f"  Repo root  : {target}")

    if dry_run or not profile_path.exists():
        print("  Profile    : (dry-run, not written)")
        _print_next_steps(target, dry_run=True)
        return

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        device_class = profile.get("device", {}).get("device_class", "unknown")
        purpose = profile.get("purpose", {}).get("purpose", "unknown")
        model_id = (profile.get("model") or {}).get("id", "none")
        cognition_id = (profile.get("cognition") or {}).get("id", "none")
        tool_count = len(profile.get("tools", []))
        service_count = len(profile.get("services", []))
        disabled_count = len(profile.get("disabled", []))
        print(f"  Device     : {device_class}")
        print(f"  Purpose    : {purpose}")
        print(f"  Model gene : {model_id}")
        print(f"  Cognition  : {cognition_id}")
        print(f"  Tools      : {tool_count} enabled, {disabled_count} disabled (device/purpose filtered)")
        print(f"  Services   : {service_count}")
    except Exception as exc:
        print(f"  (could not read profile: {exc})")

    _print_next_steps(target, dry_run=False)


def _print_next_steps(target: Path, dry_run: bool) -> None:
    entrypoint = target / "scripts" / "adaptive_agent_entrypoint.py"
    start_model = target / "scripts" / "adaptive_agent_start_model.py"
    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("1. Place your GGUF model file in:")
    print(f"     {target / 'models' / '<your-model>.gguf'}")
    print("   Edit config/adaptive_agent/genes.json -> payload.model_path if needed.")
    print()
    print("2. (Termux/Android only) Build llama.cpp:")
    print("     pkg update && pkg install -y git cmake clang")
    print("     git clone https://github.com/ggml-org/llama.cpp")
    print("     cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF")
    print("     cmake --build llama.cpp/build --config Release -j")
    print("     export LLAMA_CPP_SERVER=\"$HOME/llama.cpp/build/bin/llama-server\"")
    print()
    print("3. Dry-run to confirm the launch command:")
    print(f"     python {start_model} --dry-run")
    print()
    print("4. Start the agent:")
    print(f"     python {entrypoint}")
    print()
    if not dry_run:
        print("   Or re-run the wizard anytime to change purpose/personality:")
        print(f"     python {target / 'scripts' / 'adaptive_agent_wizard.py'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_with_src(target: Path) -> dict:
    """Return an environment where target/src is on PYTHONPATH."""
    env = os.environ.copy()
    src_dir = str(target / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")
    return env


def _progress(label: str, current: int, total: int) -> None:
    step = max(1, total // 40) if total else 1
    if current != total and current % step != 0:
        return
    width = 28
    ratio = current / total if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label}: [{bar}] {current}/{total}", end="", flush=True)


def _die(msg: str) -> None:
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the adaptive agent on a new device from the DNA zip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--purpose",
        default="assistant",
        choices=["coding", "assistant", "research", "automation", "phone_companion"],
        help="Agent purpose (default: assistant)",
    )
    parser.add_argument(
        "--personality",
        default="pragmatic, concise, local-first",
        help="Short personality description",
    )
    parser.add_argument(
        "--cognition-mode",
        default="auto",
        choices=["auto", "system0", "system1", "system2", "system3", "full"],
        help="Cognition mode override (default: auto)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Enable genes that require network access",
    )
    parser.add_argument(
        "--allow-background",
        action="store_true",
        help="Enable background service genes",
    )
    parser.add_argument(
        "--target",
        default=str(DEFAULT_TARGET),
        help=f"Install directory (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_DNA),
        help=f"Path to adaptive_agent_dna.zip (default: {DEFAULT_DNA})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without doing anything",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction step (if target already extracted)",
    )
    args = parser.parse_args(argv)

    target = Path(args.target).resolve()
    source = Path(args.source).resolve()

    print()
    print("  Adaptive Agent Installer")
    print(f"  Source : {source}")
    print(f"  Target : {target}")
    print(f"  Purpose: {args.purpose}")
    print()

    # Step 1: Extract
    if args.skip_extract:
        print("[1/4] Skipping extraction (--skip-extract)")
    else:
        extract_dna(source, target)

    # Step 2: Wizard
    profile_path = run_wizard(
        target=target,
        purpose=args.purpose,
        personality=args.personality,
        allow_network=args.allow_network,
        allow_background=args.allow_background,
        cognition_mode=args.cognition_mode,
        dry_run=args.dry_run,
    )

    # Step 3: Decode bundles
    decode_bundles(
        target=target,
        profile_path=profile_path,
        source=source,
        dry_run=args.dry_run,
    )

    # Step 4: Summary
    print_summary(target, profile_path, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
