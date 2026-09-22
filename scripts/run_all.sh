#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src"

pick_python() {
  local candidates=()
  if [[ -n "${PYTHON:-}" ]]; then
    candidates+=("${PYTHON}")
  fi
  candidates+=("python" "python3")
  local candidate
  for candidate in "${candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" -c "print('ok')" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON_EXE="$(pick_python || true)"
if [[ -z "${PYTHON_EXE}" ]]; then
  echo "[run_all] Python not found. Set PYTHON or install Python." >&2
  exit 1
fi

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.env"
  set +a
fi

SETTINGS_INTERFACE_MODES="$("${PYTHON_EXE}" -c "from common.config import load_settings; i=load_settings().interface; raw=i.get('mode', '') if isinstance(i, dict) else ''; values=raw if isinstance(raw, list) else [raw]; print(','.join(str(x).strip() for x in values if str(x).strip()))")"
SETTINGS_INTERFACE_SENSES="$("${PYTHON_EXE}" -c "from common.config import load_settings; i=load_settings().interface; s=i.get('senses', []) if isinstance(i, dict) else []; print(','.join(str(x) for x in s if str(x).strip()))")"
SETTINGS_AUTONOMY_RUN_ALL="$("${PYTHON_EXE}" -c "from common.config import load_settings; a=load_settings().autonomy; a=a if isinstance(a, dict) else {}; b=lambda v: v if isinstance(v, bool) else str(v).strip().lower() in ('1','true','yes','on'); ep=b(a.get('epistemic_enabled', False)); pp=b(a.get('power_process_enabled', False)); kg=b(a.get('knowledge_enabled', False)); enabled=a.get('enabled', None); enabled=(ep or pp or kg) if enabled is None else b(enabled); start=b(a.get('start_with_run_all', False)); print('1' if start and enabled and (ep or pp or kg) else '')")"
RAW_INTERFACE_MODES="${SETTINGS_INTERFACE_MODES}"
if [[ -z "${RAW_INTERFACE_MODES}" ]]; then
  RAW_INTERFACE_MODES="text"
fi

IFS=',' read -r -a INTERFACE_MODES_RAW <<< "${RAW_INTERFACE_MODES}"
INTERFACE_MODES_LC=()
for raw_mode in "${INTERFACE_MODES_RAW[@]}"; do
  normalized="$(printf '%s' "${raw_mode}" | tr '[:upper:]' '[:lower:]')"
  if [[ -n "${normalized}" ]]; then
    INTERFACE_MODES_LC+=("${normalized}")
  fi
done
if [[ ${#INTERFACE_MODES_LC[@]} -eq 0 ]]; then
  INTERFACE_MODES_LC=("text")
fi

INTERFACE_PIDS=()
AUTONOMY_PID=""

"${PYTHON_EXE}" -m storage.main &
STORAGE_PID=$!

"${PYTHON_EXE}" -m model_server.main &
MODEL_PID=$!

"${PYTHON_EXE}" -m tool_runtime.main &
TOOL_PID=$!

"${PYTHON_EXE}" -m orchestrator.main &
ORCH_PID=$!

"${PYTHON_EXE}" -m runtime_core.evolve_scheduler &
EVOLVE_SCHEDULER_PID=$!

cleanup() {
  local pids=(${ORCH_PID} ${TOOL_PID} ${MODEL_PID} ${STORAGE_PID} ${EVOLVE_SCHEDULER_PID})
  if [[ -n "${AUTONOMY_PID:-}" ]]; then
    pids+=("${AUTONOMY_PID}")
  fi
  if [[ ${#INTERFACE_PIDS[@]} -gt 0 ]]; then
    pids+=("${INTERFACE_PIDS[@]}")
  fi
  kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT

sleep 0.5

if [[ -n "${SETTINGS_AUTONOMY_RUN_ALL}" ]]; then
  echo "[start] autonomy-runtime"
  "${PYTHON_EXE}" -m autonomy.runtime &
  AUTONOMY_PID=$!
fi

if [[ ${#INTERFACE_MODES_LC[@]} -gt 1 ]]; then
  for interface_mode in "${INTERFACE_MODES_LC[@]}"; do
    if [[ "${interface_mode}" != "telegram" && "${interface_mode}" != "whatsapp" ]]; then
      echo "[run_all] Multiple interface modes currently support only 'telegram' and 'whatsapp'." >&2
      exit 1
    fi
  done

  for interface_mode in "${INTERFACE_MODES_LC[@]}"; do
    if [[ "${interface_mode}" == "telegram" ]]; then
      echo "[start] telegram"
      "${PYTHON_EXE}" -m telegram_daemon.main &
      INTERFACE_PIDS+=($!)
    elif [[ "${interface_mode}" == "whatsapp" ]]; then
      if ! command -v node >/dev/null 2>&1; then
        echo "[run_all] WhatsApp mode requires Node.js 20+ and npm install." >&2
        exit 1
      fi
      echo "[start] whatsapp"
      node ./src/whatsapp_daemon/main.mjs &
      INTERFACE_PIDS+=($!)
    fi
  done

  wait -n "${INTERFACE_PIDS[@]}"
else
  INTERFACE_MODE_LC="${INTERFACE_MODES_LC[0]}"
  if [[ "${INTERFACE_MODE_LC}" == "telegram" ]]; then
    "${PYTHON_EXE}" -m telegram_daemon.main
  elif [[ "${INTERFACE_MODE_LC}" == "whatsapp" ]]; then
    if ! command -v node >/dev/null 2>&1; then
      echo "[run_all] WhatsApp mode requires Node.js 20+ and npm install." >&2
      exit 1
    fi
    node ./src/whatsapp_daemon/main.mjs
  elif [[ "${INTERFACE_MODE_LC}" == "text" ]]; then
    "${PYTHON_EXE}" -m voice_daemon.main --mode text
  elif [[ "${INTERFACE_MODE_LC}" == "audio" ]]; then
    "${PYTHON_EXE}" -m voice_daemon.main --mode audio
  elif [[ "${INTERFACE_MODE_LC}" == "vision" ]]; then
    "${PYTHON_EXE}" -m voice_daemon.main --mode vision
  elif [[ "${INTERFACE_MODE_LC}" == "local" ]]; then
    if [[ -n "${SETTINGS_INTERFACE_SENSES}" ]]; then
      "${PYTHON_EXE}" -m voice_daemon.main --mode local --senses "${SETTINGS_INTERFACE_SENSES}"
    else
      "${PYTHON_EXE}" -m voice_daemon.main --mode local
    fi
  else
    echo "[run_all] Unknown interface mode '${INTERFACE_MODES_RAW[0]}'. Use 'text', 'audio', 'vision', 'local', 'telegram', or 'whatsapp'." >&2
    exit 1
  fi
fi
