#!/bin/bash
# system_bootstrap.sh - Wrapper for system bootstrap script on Linux/macOS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python3 scripts/system_bootstrap.py
