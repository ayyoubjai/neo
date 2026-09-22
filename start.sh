#!/usr/bin/env bash
# start.sh — Start the full AGI assistant stack in one command.
#
# What this script does (in order):
#   1. SearXNG private web search  — requires Docker; skipped gracefully if absent
#   2. llama-router model server   — local router for llama.cpp
#                                    skipped (with a warning) when provider = "ollama"
#   3. The main assistant          — runs scripts/run_all.sh in the foreground
#
# Stop everything cleanly with Ctrl+C.
#
# First-time setup:
#   python3 scripts/bootstrap_release.py --install
#   python scripts/setup_release.py --interactive

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"

# ── Activate virtual environment ───────────────────────────────────────────────
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
fi

# ── Load .env ─────────────────────────────────────────────────────────────────
if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

# ── Pick Python ────────────────────────────────────────────────────────────────
PYTHON_EXE=""
for _candidate in python python3; do
    if command -v "$_candidate" &>/dev/null && "$_candidate" -c "print('ok')" &>/dev/null; then
        PYTHON_EXE="$_candidate"
        break
    fi
done
if [[ -z "$PYTHON_EXE" ]]; then
    echo "[start] ERROR: Python not found." >&2
    echo "[start]        Install Python 3.10+ or activate a virtual environment first." >&2
    exit 1
fi

# ── Read a dotted key from settings (local → tracked → example fallback) ──────
_read_setting() {
    local key="$1" default="$2"
    "$PYTHON_EXE" -c "
import json, os, pathlib, sys
key, default = '$key', '$default'
keys = key.split('.')
paths = [os.environ['AGI_SETTINGS_PATH']] if os.environ.get('AGI_SETTINGS_PATH') else ['config/settings.local.json', 'config/settings.json', 'config/settings.example.json']
for f in paths:
    p = pathlib.Path(f)
    if not p.exists():
        continue
    try:
        d = json.loads(p.read_text())
        for k in keys:
            d = d[k]
        if d:
            print(d)
            sys.exit(0)
    except (KeyError, TypeError, json.JSONDecodeError):
        pass
print(default)
" 2>/dev/null
}

PROVIDER=$(_read_setting "models.provider" "llamacpp")
LLAMACPP_HOST=$(_read_setting "models.llamacpp_host" "http://127.0.0.1:8080")
LLAMACPP_HOST="${LLAMACPP_HOST%/}"   # strip any trailing slash

# ── Graceful cleanup on exit / Ctrl+C ─────────────────────────────────────────
LLAMA_SERVER_PID=""

_cleanup() {
    echo ""
    echo "[start] Shutting down..."
    if [[ -n "${LLAMA_SERVER_PID:-}" ]] && kill -0 "$LLAMA_SERVER_PID" 2>/dev/null; then
        echo "[start] Stopping llama-server (PID $LLAMA_SERVER_PID)..."
        kill "$LLAMA_SERVER_PID" 2>/dev/null || true
        wait "$LLAMA_SERVER_PID" 2>/dev/null || true
    fi
    echo "[start] Done."
}
trap _cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
echo "[start] ═══ Step 1 — SearXNG (private web search) ═══════════════"
_start_searxng() {
    local -a compose_cmd=()

    if ! command -v docker &>/dev/null; then
        echo "[start] Docker not found — skipping SearXNG."
        echo "[start] Web search (net.search tool) will be unavailable."
        echo "[start] Install Docker to enable it: https://docs.docker.com/get-docker/"
        return 0
    fi

    if docker compose version &>/dev/null; then
        compose_cmd=(docker compose)
    elif command -v docker-compose &>/dev/null && docker-compose version &>/dev/null; then
        compose_cmd=(docker-compose)
    else
        echo "[start] Docker Compose not found — skipping SearXNG."
        echo "[start] Web search (net.search tool) will be unavailable."
        echo "[start] Install the Docker Compose v2 plugin, then re-run ./start.sh."
        return 0
    fi

    local docker_error
    if ! docker_error=$(docker info 2>&1); then
        echo "[start] Docker daemon is unavailable — skipping SearXNG."
        if [[ "$docker_error" == *"permission denied"* ]]; then
            echo "[start] Docker socket access denied for user $(id -un)."
            echo "[start] For a standard Docker Engine install, grant this user docker-group access, then log out and back in."
            echo "[start] Instructions: https://docs.docker.com/engine/install/linux-postinstall/"
        fi
        echo "[start] Start Docker and ensure this user can access it; 'docker ps' should succeed."
        echo "[start] Web search (net.search tool) will be unavailable."
        return 0
    fi

    echo "[start] Starting SearXNG..."
    if ! "${compose_cmd[@]}" -f docker-compose.searxng.yml up -d; then
        echo "[start] WARNING: SearXNG failed to start; continuing without web search." >&2
        return 0
    fi
    echo "[start] SearXNG is running →  http://127.0.0.1:8081"
}

_start_searxng

# ─────────────────────────────────────────────────────────────────────────────
echo "[start] ═══ Step 2 — Model server ══════════════════════════════"
if [[ "$PROVIDER" == "llamacpp" ]]; then
    # Read configured models dir (fallback to ./models)
    MODELS_DIR=$(_read_setting "models.models_dir" "./models")
    _port="${LLAMACPP_HOST##*:}"
    if "$PYTHON_EXE" -c "
import socket, sys
try:
    s = socket.create_connection(('127.0.0.1', int('$_port')), timeout=1)
    s.close()
    sys.exit(0)   # port already open
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "[start] llama-server appears to already be running on port $_port — skipping start."
    else
        LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$(_read_setting "models.llama_server" "llama-server")}"
        if ! command -v "$LLAMA_SERVER_BIN" &>/dev/null; then
            echo "[start] ERROR: llama-server executable not found: $LLAMA_SERVER_BIN" >&2
            echo "[start] Build or install llama.cpp, then enter its executable path during setup." >&2
            echo "[start] Setup saves it as models.llama_server in config/settings.local.json." >&2
            echo "[start] For existing settings, edit that key directly. See docs/release-setup.md." >&2
            exit 1
        fi
        echo "[start] Starting llama-server (router mode)..."
        # Generate models.ini from settings so llama-server receives model presets
        MODELS_PRESET_FILE="$ROOT/config/models.ini"
        if [ -n "${PYTHON_EXE:-}" ] && [ -f "$ROOT/scripts/generate_models_ini.py" ]; then
            mkdir -p "$ROOT/config"
            echo "[start] Generating models preset -> $MODELS_PRESET_FILE"
            "$PYTHON_EXE" "$ROOT/scripts/generate_models_ini.py" --models-dir "$MODELS_DIR" --output "$MODELS_PRESET_FILE"
        fi

        LLAMA_SERVER_LOG="$ROOT/data/llama-server.log"
        mkdir -p "$(dirname "$LLAMA_SERVER_LOG")"
        "$PYTHON_EXE" -m model_server.managed_router --server-bin "$LLAMA_SERVER_BIN" --models-preset "$MODELS_PRESET_FILE" --host 127.0.0.1 --port "$_port" >"$LLAMA_SERVER_LOG" 2>&1 &
        LLAMA_SERVER_PID=$!

        echo "[start] Waiting for llama-server to be ready (up to 60 s)..."
        _ready=0
        for _i in $(seq 1 60); do
            if ! kill -0 "$LLAMA_SERVER_PID" 2>/dev/null; then
                echo "[start] ERROR: llama-server exited unexpectedly. See $LLAMA_SERVER_LOG" >&2
                exit 1
            fi
            if "$PYTHON_EXE" -c "
import urllib.request, urllib.error, sys
try:
    urllib.request.urlopen('${LLAMACPP_HOST}/health', timeout=2)
    sys.exit(0)
except urllib.error.HTTPError:
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
                _ready=1
                break
            fi
            sleep 1
        done

        if [[ "$_ready" -eq 0 ]]; then
            echo "[start] ERROR: llama-server did not respond within 60 seconds. See $LLAMA_SERVER_LOG" >&2
            exit 1
        fi
        echo "[start] llama-server is ready →  ${LLAMACPP_HOST}"
    fi

elif [[ "$PROVIDER" == "ollama" ]]; then
    echo "[start] Provider is Ollama — checking connectivity..."
    if "$PYTHON_EXE" -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=3)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "[start] Ollama is reachable →  http://127.0.0.1:11434"
    else
        echo "[start] WARNING: Ollama does not appear to be running."
        echo "[start]          Start Ollama, then re-run this script."
        echo "[start]          Download: https://ollama.com"
    fi
else
    echo "[start] Provider '$PROVIDER' — no model server to start."
fi

# ─────────────────────────────────────────────────────────────────────────────
echo "[start] ═══ Step 3 — Assistant ════════════════════════════════"
echo "[start] Starting assistant...  (Ctrl+C stops everything)"
echo ""

"$ROOT/scripts/run_all.sh"
