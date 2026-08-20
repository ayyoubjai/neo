from __future__ import annotations

import atexit
import asyncio
import ctypes
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from common.config import DEFAULT_SETTINGS_PATH, Settings, load_settings

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - older runtimes
    ZoneInfo = None  # type: ignore[assignment]


_ACTIVE_SCHEDULER_LOCKS: Dict[str, int] = {}


@dataclass(frozen=True)
class EvolveSchedulerConfig:
    enabled: bool
    mode: str
    run_on_startup: bool
    interval_s: float
    initial_delay_s: float
    config_path: str
    log_path: str
    state_path: str
    scheduler_lock_path: str
    repo_root: str
    calendar_every_days: int
    calendar_hour_local: int
    calendar_minute_local: int
    calendar_timezone: str
    calendar_catch_up_if_missed: bool
    calendar_anchor_date_local: str
    resource_gate_enabled: bool
    resource_check_interval_s: float
    resource_required_consecutive_passes: int
    resource_min_free_ram_mb: float
    resource_min_free_disk_gb: float
    resource_max_cpu_percent: float
    resource_max_load_avg_1m: float
    resource_disk_path: str


@dataclass(frozen=True)
class ResourceSnapshot:
    free_ram_bytes: Optional[int]
    free_disk_bytes: Optional[int]
    cpu_percent: Optional[float]
    load_avg_1m: Optional[float]
    cpu_count: Optional[int]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def build_scheduler_config(settings: Settings) -> EvolveSchedulerConfig:
    evolve = dict(settings.evolve or {})
    repo_root = os.path.abspath(settings.workspace_root or os.getcwd())
    config_path = str(evolve.get("auto_run_config_path", "") or "").strip()
    if not config_path:
        config_path = os.path.join(repo_root, "evolve", "improve_config.json")
    log_path = str(evolve.get("auto_run_log_path", "") or "").strip()
    if not log_path:
        log_path = os.path.join(repo_root, "data", "evolve_scheduler.log")
    state_path = str(evolve.get("scheduler_state_path", "") or "").strip()
    if not state_path:
        state_path = os.path.join(repo_root, "data", "evolve_scheduler_state.json")
    scheduler_lock_path = str(evolve.get("scheduler_lock_path", "") or "").strip()
    if not scheduler_lock_path:
        scheduler_lock_path = os.path.join(repo_root, "data", "evolve_scheduler.lock")
    mode = str(evolve.get("auto_run_mode", "interval") or "interval").strip().lower()
    if mode not in {"interval", "calendar"}:
        mode = "interval"
    resource_disk_path = str(evolve.get("resource_disk_path", "") or "").strip()
    if not resource_disk_path:
        resource_disk_path = repo_root
    return EvolveSchedulerConfig(
        enabled=_coerce_bool(evolve.get("auto_run_enabled"), False),
        mode=mode,
        run_on_startup=_coerce_bool(evolve.get("auto_run_on_startup"), False),
        interval_s=max(0.0, _coerce_float(evolve.get("auto_run_interval_s"), 0.0)),
        initial_delay_s=max(0.0, _coerce_float(evolve.get("auto_run_initial_delay_s"), 300.0)),
        config_path=config_path,
        log_path=log_path,
        state_path=state_path,
        scheduler_lock_path=scheduler_lock_path,
        repo_root=repo_root,
        calendar_every_days=max(1, _coerce_int(evolve.get("auto_run_calendar_every_days"), 3)),
        calendar_hour_local=min(23, max(0, _coerce_int(evolve.get("auto_run_calendar_hour_local"), 3))),
        calendar_minute_local=min(59, max(0, _coerce_int(evolve.get("auto_run_calendar_minute_local"), 0))),
        calendar_timezone=str(evolve.get("auto_run_calendar_timezone", "") or "").strip(),
        calendar_catch_up_if_missed=_coerce_bool(evolve.get("auto_run_calendar_catch_up_if_missed"), True),
        calendar_anchor_date_local=str(evolve.get("auto_run_calendar_anchor_date_local", "") or "").strip(),
        resource_gate_enabled=_coerce_bool(evolve.get("resource_gate_enabled"), False),
        resource_check_interval_s=max(5.0, _coerce_float(evolve.get("resource_check_interval_s"), 300.0)),
        resource_required_consecutive_passes=max(
            1,
            _coerce_int(evolve.get("resource_required_consecutive_passes"), 1),
        ),
        resource_min_free_ram_mb=max(0.0, _coerce_float(evolve.get("resource_min_free_ram_mb"), 2048.0)),
        resource_min_free_disk_gb=max(0.0, _coerce_float(evolve.get("resource_min_free_disk_gb"), 10.0)),
        resource_max_cpu_percent=max(0.0, _coerce_float(evolve.get("resource_max_cpu_percent"), 70.0)),
        resource_max_load_avg_1m=max(0.0, _coerce_float(evolve.get("resource_max_load_avg_1m"), 0.0)),
        resource_disk_path=resource_disk_path,
    )


def scheduler_should_start(config: EvolveSchedulerConfig) -> bool:
    if not config.enabled:
        return False
    if config.run_on_startup:
        return True
    if config.mode == "calendar":
        return config.calendar_every_days > 0
    return config.interval_s > 0.0


def _append_scheduler_log(path: str, message: str) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


def _scheduler_log(config: EvolveSchedulerConfig, message: str) -> None:
    print(f"[evolve-scheduler] {message}", flush=True)
    _append_scheduler_log(config.log_path, message)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_datetime(text: str) -> Optional[datetime]:
    value = str(text or "").strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_date(text: str) -> Optional[date]:
    value = str(text or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _resolve_scheduler_timezone(config: EvolveSchedulerConfig):
    tz_name = str(config.calendar_timezone or "").strip()
    if tz_name and ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def _scheduler_signature(config: EvolveSchedulerConfig) -> str:
    return "|".join(
        [
            config.mode,
            str(config.calendar_every_days),
            str(config.calendar_hour_local),
            str(config.calendar_minute_local),
            str(config.calendar_timezone or "local"),
            str(config.calendar_anchor_date_local or "auto"),
            "catchup" if config.calendar_catch_up_if_missed else "skipmissed",
        ]
    )


def _load_scheduler_state(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _write_scheduler_state(path: str, state: Dict[str, Any]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2, sort_keys=True)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock_payload(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _release_pid_lock(path: str, owner_pid: int) -> None:
    if not path:
        return
    payload = _read_lock_payload(path)
    if int(payload.get("pid", 0) or 0) != owner_pid:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    finally:
        _ACTIVE_SCHEDULER_LOCKS.pop(path, None)


def _hold_pid_lock(lock_path: str) -> None:
    owner_pid = os.getpid()
    if _ACTIVE_SCHEDULER_LOCKS.get(lock_path) == owner_pid:
        return
    parent = os.path.dirname(lock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "pid": owner_pid,
        "created_at": _format_utc(datetime.now(timezone.utc)),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_lock_payload(lock_path)
            existing_pid = int(existing.get("pid", 0) or 0)
            if existing_pid <= 0 or not _pid_is_running(existing_pid):
                try:
                    os.unlink(lock_path)
                    continue
                except FileNotFoundError:
                    continue
            raise RuntimeError(f"scheduler already in progress: {lock_path}")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
        _ACTIVE_SCHEDULER_LOCKS[lock_path] = owner_pid
        atexit.register(_release_pid_lock, lock_path, owner_pid)
        return


def _calendar_slot_for_date(config: EvolveSchedulerConfig, slot_date: date, tzinfo) -> datetime:
    return datetime(
        slot_date.year,
        slot_date.month,
        slot_date.day,
        config.calendar_hour_local,
        config.calendar_minute_local,
        tzinfo=tzinfo,
    )


def _initialize_calendar_state(
    config: EvolveSchedulerConfig,
    state: Dict[str, Any],
    now_local: datetime,
) -> Tuple[Dict[str, Any], datetime]:
    tzinfo = now_local.tzinfo or _resolve_scheduler_timezone(config)
    anchor_date = _parse_date(config.calendar_anchor_date_local)
    if anchor_date is None:
        anchor_date = _parse_date(str(state.get("calendar_anchor_date_local", "") or ""))
    if anchor_date is None:
        anchor_date = now_local.date()
    step = timedelta(days=max(1, config.calendar_every_days))
    next_slot = _calendar_slot_for_date(config, anchor_date, tzinfo)
    if config.calendar_catch_up_if_missed:
        while (next_slot + step) <= now_local:
            next_slot += step
    else:
        while next_slot <= now_local:
            next_slot += step
    state = dict(state)
    state["calendar_anchor_date_local"] = anchor_date.isoformat()
    state["calendar_signature"] = _scheduler_signature(config)
    state["next_calendar_slot_at"] = next_slot.isoformat()
    return state, next_slot


def _calendar_due_slot(
    config: EvolveSchedulerConfig,
    state: Dict[str, Any],
    now_local: datetime,
) -> Tuple[Dict[str, Any], Optional[datetime]]:
    state = dict(state)
    signature = _scheduler_signature(config)
    next_slot = _parse_datetime(str(state.get("next_calendar_slot_at", "") or ""))
    if state.get("calendar_signature") != signature or next_slot is None:
        state, next_slot = _initialize_calendar_state(config, state, now_local)
    if next_slot is None:
        return state, None
    next_slot = next_slot.astimezone(now_local.tzinfo or _resolve_scheduler_timezone(config))
    step = timedelta(days=max(1, config.calendar_every_days))
    if not config.calendar_catch_up_if_missed:
        while next_slot <= now_local:
            next_slot += step
        state["next_calendar_slot_at"] = next_slot.isoformat()
        return state, None
    state["next_calendar_slot_at"] = next_slot.isoformat()
    if next_slot <= now_local:
        return state, next_slot
    return state, None


def _advance_calendar_state_after_run(
    config: EvolveSchedulerConfig,
    state: Dict[str, Any],
    slot_at: datetime,
    now_local: datetime,
) -> Dict[str, Any]:
    state = dict(state)
    step = timedelta(days=max(1, config.calendar_every_days))
    next_slot = slot_at + step
    while next_slot <= now_local:
        next_slot += step
    state["last_calendar_slot_at"] = slot_at.isoformat()
    state["next_calendar_slot_at"] = next_slot.isoformat()
    return state


def _probe_available_ram_bytes() -> Optional[int]:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except Exception:
            pass
    if hasattr(os, "sysconf"):
        try:
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if available_pages > 0 and page_size > 0:
                return available_pages * page_size
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    if os.name == "nt":
        try:
            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            if ok:
                return int(status.ullAvailPhys)
        except Exception:
            pass
    return None


def _probe_free_disk_bytes(path: str) -> Optional[int]:
    target = path or os.getcwd()
    if os.path.isfile(target):
        target = os.path.dirname(target) or os.getcwd()
    try:
        usage = shutil.disk_usage(target)
    except Exception:
        return None
    return int(usage.free)


def _probe_cpu_percent() -> Optional[float]:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=0.2))
    except Exception:
        return None


def _probe_load_avg_1m() -> Optional[float]:
    if hasattr(os, "getloadavg"):
        try:
            return float(os.getloadavg()[0])
        except (OSError, ValueError):
            return None
    return None


def capture_resource_snapshot(config: EvolveSchedulerConfig) -> ResourceSnapshot:
    return ResourceSnapshot(
        free_ram_bytes=_probe_available_ram_bytes(),
        free_disk_bytes=_probe_free_disk_bytes(config.resource_disk_path or config.repo_root),
        cpu_percent=_probe_cpu_percent(),
        load_avg_1m=_probe_load_avg_1m(),
        cpu_count=os.cpu_count(),
    )


def evaluate_resource_gate(
    config: EvolveSchedulerConfig,
    snapshot: ResourceSnapshot,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    metrics: Dict[str, Any] = {
        "free_ram_mb": round((snapshot.free_ram_bytes or 0) / (1024 * 1024), 1) if snapshot.free_ram_bytes is not None else None,
        "free_disk_gb": round((snapshot.free_disk_bytes or 0) / (1024 ** 3), 2) if snapshot.free_disk_bytes is not None else None,
        "cpu_percent": round(snapshot.cpu_percent, 1) if snapshot.cpu_percent is not None else None,
        "load_avg_1m": round(snapshot.load_avg_1m, 2) if snapshot.load_avg_1m is not None else None,
        "cpu_count": snapshot.cpu_count,
    }
    reasons: List[str] = []
    if config.resource_min_free_ram_mb > 0.0:
        if snapshot.free_ram_bytes is None:
            reasons.append("free_ram_unavailable")
        elif (snapshot.free_ram_bytes / (1024 * 1024)) < config.resource_min_free_ram_mb:
            reasons.append(
                f"free_ram_mb={metrics['free_ram_mb']}<{round(config.resource_min_free_ram_mb, 1)}"
            )
    if config.resource_min_free_disk_gb > 0.0:
        if snapshot.free_disk_bytes is None:
            reasons.append("free_disk_unavailable")
        elif (snapshot.free_disk_bytes / (1024 ** 3)) < config.resource_min_free_disk_gb:
            reasons.append(
                f"free_disk_gb={metrics['free_disk_gb']}<{round(config.resource_min_free_disk_gb, 2)}"
            )
    if config.resource_max_cpu_percent > 0.0:
        if snapshot.cpu_percent is None:
            reasons.append("cpu_percent_unavailable")
        elif snapshot.cpu_percent > config.resource_max_cpu_percent:
            reasons.append(
                f"cpu_percent={metrics['cpu_percent']}>{round(config.resource_max_cpu_percent, 1)}"
            )
    if config.resource_max_load_avg_1m > 0.0:
        if snapshot.load_avg_1m is None:
            reasons.append("load_avg_1m_unavailable")
        elif snapshot.load_avg_1m > config.resource_max_load_avg_1m:
            reasons.append(
                f"load_avg_1m={metrics['load_avg_1m']}>{round(config.resource_max_load_avg_1m, 2)}"
            )
    return (len(reasons) == 0), reasons, metrics


def _describe_gate(metrics: Dict[str, Any], reasons: List[str]) -> str:
    parts = [f"{key}={value}" for key, value in metrics.items()]
    if reasons:
        parts.append(f"blocked_by={','.join(reasons)}")
    return " ".join(parts)


async def _run_pipeline_once(config: EvolveSchedulerConfig, reason: str) -> int:
    command = [
        sys.executable or "python3",
        "-u",
        os.path.join(config.repo_root, "scripts", "evolve_improve.py"),
        "--config",
        config.config_path,
    ]
    _scheduler_log(config, f"starting reason={reason} config={config.config_path}")
    proc = await asyncio.create_subprocess_exec(*command, cwd=config.repo_root)
    returncode = await proc.wait()
    _scheduler_log(config, f"finished reason={reason} exit_code={returncode}")
    return returncode


async def serve(settings_path: str = DEFAULT_SETTINGS_PATH) -> int:
    config = build_scheduler_config(load_settings(settings_path))
    if not scheduler_should_start(config):
        return 0
    try:
        _hold_pid_lock(config.scheduler_lock_path)
    except RuntimeError as e:
        _scheduler_log(config, f"lock_unavailable error={e}")
        return 2

    state = _load_scheduler_state(config.state_path)
    startup_pending = bool(config.run_on_startup)
    next_interval_due_at = time.time() + config.initial_delay_s
    if config.interval_s > 0.0 and config.initial_delay_s <= 0.0:
        next_interval_due_at = time.time()
    consecutive_gate_passes = 0

    while True:
        now_utc = datetime.now(timezone.utc)
        now_local = datetime.now(_resolve_scheduler_timezone(config))
        due_reason: Optional[str] = None
        due_slot: Optional[datetime] = None
        sleep_s = config.resource_check_interval_s

        if startup_pending:
            due_reason = "startup"
        elif config.mode == "calendar":
            state, due_slot = _calendar_due_slot(config, state, now_local)
            next_slot = _parse_datetime(str(state.get("next_calendar_slot_at", "") or ""))
            if next_slot is not None:
                next_slot = next_slot.astimezone(now_local.tzinfo or _resolve_scheduler_timezone(config))
                if next_slot > now_local:
                    sleep_s = max(1.0, (next_slot - now_local).total_seconds())
            if due_slot is not None:
                due_reason = f"calendar slot={due_slot.isoformat()}"
        elif config.interval_s > 0.0:
            remaining = next_interval_due_at - time.time()
            if remaining <= 0.0:
                due_reason = "interval"
            else:
                sleep_s = max(1.0, remaining)
        else:
            return 0

        if due_reason is None:
            consecutive_gate_passes = 0
            _write_scheduler_state(config.state_path, state)
            await asyncio.sleep(sleep_s)
            continue

        if config.resource_gate_enabled:
            snapshot = capture_resource_snapshot(config)
            gate_ok, reasons, metrics = evaluate_resource_gate(config, snapshot)
            if not gate_ok:
                consecutive_gate_passes = 0
                _scheduler_log(
                    config,
                    f"resource_gate blocked reason={due_reason} {_describe_gate(metrics, reasons)}",
                )
                _write_scheduler_state(config.state_path, state)
                await asyncio.sleep(config.resource_check_interval_s)
                continue
            consecutive_gate_passes += 1
            if consecutive_gate_passes < config.resource_required_consecutive_passes:
                _scheduler_log(
                    config,
                    "resource_gate warming "
                    f"reason={due_reason} pass={consecutive_gate_passes}/{config.resource_required_consecutive_passes} "
                    f"{_describe_gate(metrics, [])}",
                )
                await asyncio.sleep(config.resource_check_interval_s)
                continue
        consecutive_gate_passes = 0

        state["last_run_reason"] = due_reason
        state["last_run_started_at"] = _format_utc(now_utc)
        if due_slot is not None:
            state["pending_calendar_slot_at"] = due_slot.isoformat()
        _write_scheduler_state(config.state_path, state)
        exit_code = await _run_pipeline_once(config, due_reason)
        finished_utc = datetime.now(timezone.utc)
        finished_local = datetime.now(_resolve_scheduler_timezone(config))
        state["last_run_completed_at"] = _format_utc(finished_utc)
        state["last_run_exit_code"] = int(exit_code)
        if exit_code == 0:
            state["last_run_succeeded_at"] = _format_utc(finished_utc)
        if due_slot is not None:
            state = _advance_calendar_state_after_run(config, state, due_slot, finished_local)
            state.pop("pending_calendar_slot_at", None)
        if due_reason == "startup":
            startup_pending = False
            if config.mode == "interval" and config.interval_s > 0.0:
                next_interval_due_at = time.time() + config.interval_s
        elif config.mode == "interval" and config.interval_s > 0.0:
            next_interval_due_at = time.time() + config.interval_s
        _write_scheduler_state(config.state_path, state)


def main() -> int:
    try:
        return asyncio.run(serve())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
