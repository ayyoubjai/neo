#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXE="${PYTHON:-python3}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
exec "${PYTHON_EXE}" -m runtime_core.evolve_scheduler "$@"
