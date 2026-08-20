from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class DeviceProfile:
    os_name: str
    platform_system: str
    machine: str
    cpu_count: int
    ram_mb: int
    storage_free_mb: int
    is_termux: bool
    is_android: bool
    device_class: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_meminfo_mb() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(int(parts[1]) / 1024)
    except OSError:
        pass
    return 0


def _storage_free_mb(path: str) -> int:
    try:
        usage = shutil.disk_usage(path)
        return int(usage.free / (1024 * 1024))
    except OSError:
        return 0


def _is_termux() -> bool:
    prefix = os.environ.get("PREFIX", "")
    return bool(os.environ.get("TERMUX_VERSION")) or "com.termux" in prefix


def _is_android(is_termux: bool) -> bool:
    if is_termux:
        return True
    return os.path.exists("/system/build.prop") or "ANDROID_ROOT" in os.environ


def classify_device(
    *,
    is_android: bool,
    ram_mb: int,
    cpu_count: int,
    storage_free_mb: int,
) -> str:
    if is_android:
        if ram_mb and ram_mb < 3500:
            return "phone_tiny"
        if ram_mb and ram_mb < 7000:
            return "phone_small"
        return "phone_strong"

    if ram_mb and ram_mb < 5000:
        return "pc_tiny"
    if ram_mb and ram_mb < 12000:
        return "pc_small"
    if ram_mb and ram_mb >= 24000 and cpu_count >= 8 and storage_free_mb >= 20000:
        return "pc_strong"
    return "pc_medium"


def probe_device(base_path: str | None = None) -> DeviceProfile:
    base = base_path or os.getcwd()
    termux = _is_termux()
    android = _is_android(termux)
    ram_mb = _read_meminfo_mb()
    cpu_count = os.cpu_count() or 1
    storage_free_mb = _storage_free_mb(base)
    device_class = classify_device(
        is_android=android,
        ram_mb=ram_mb,
        cpu_count=cpu_count,
        storage_free_mb=storage_free_mb,
    )
    return DeviceProfile(
        os_name=os.name,
        platform_system=platform.system().lower(),
        machine=platform.machine().lower(),
        cpu_count=cpu_count,
        ram_mb=ram_mb,
        storage_free_mb=storage_free_mb,
        is_termux=termux,
        is_android=android,
        device_class=device_class,
    )

