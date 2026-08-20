import os
import re
import shutil
import struct
import subprocess
from typing import Any, Dict, List, Optional


class PhoneAdapterError(RuntimeError):
    pass


class AdbPhoneAdapter:
    def __init__(self, serial: Optional[str] = None, adb_bin: Optional[str] = None, runner=None):
        self._serial = serial.strip() if isinstance(serial, str) and serial.strip() else None
        self._adb_bin = adb_bin or shutil.which("adb") or "adb"
        self._runner = runner or subprocess.run

    @staticmethod
    def is_available() -> bool:
        return shutil.which("adb") is not None

    def _adb_cmd(self, *parts: str) -> List[str]:
        cmd = [self._adb_bin]
        if self._serial:
            cmd.extend(["-s", self._serial])
        cmd.extend(str(part) for part in parts)
        return cmd

    def _run(
        self,
        *parts: str,
        text: bool = True,
        timeout_s: float = 10.0,
    ):
        try:
            return self._runner(
                self._adb_cmd(*parts),
                capture_output=True,
                text=text,
                timeout=timeout_s,
                check=True,
            )
        except FileNotFoundError as e:
            raise PhoneAdapterError(f"adb not available: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise PhoneAdapterError(f"adb command timed out: {' '.join(self._adb_cmd(*parts))}") from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr or "")
            stderr = stderr.strip() or str(e)
            raise PhoneAdapterError(stderr) from e

    @classmethod
    def detect(cls) -> Dict[str, Any]:
        if not cls.is_available():
            return {"detected": False, "transport": "adb", "adapter_available": False}
        adapter = cls()
        try:
            devices = adapter.list_devices()
        except PhoneAdapterError as e:
            return {
                "detected": False,
                "transport": "adb",
                "adapter_available": True,
                "error": str(e),
            }
        connected = [item for item in devices if item.get("state") == "device"]
        if not connected:
            return {
                "detected": False,
                "transport": "adb",
                "adapter_available": True,
                "devices": devices,
            }
        primary = connected[0]
        try:
            details = cls(serial=primary.get("serial")).describe_device()
        except PhoneAdapterError as e:
            return {
                "detected": True,
                "transport": "adb",
                "adapter_available": True,
                "serial": primary.get("serial"),
                "error": str(e),
                "devices": devices,
            }
        details["devices"] = devices
        return details

    def list_devices(self) -> List[Dict[str, Any]]:
        result = self._run("devices", text=True, timeout_s=5.0)
        lines = [line.strip() for line in str(result.stdout or "").splitlines()]
        devices: List[Dict[str, Any]] = []
        for line in lines[1:]:
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if not parts:
                continue
            serial = parts[0].strip()
            state = parts[1].strip() if len(parts) > 1 else "unknown"
            if serial:
                devices.append({"serial": serial, "state": state})
        return devices

    def shell(self, *parts: str, timeout_s: float = 10.0) -> str:
        result = self._run("shell", *parts, text=True, timeout_s=timeout_s)
        return str(result.stdout or "").strip()

    def exec_out(self, *parts: str, timeout_s: float = 15.0) -> bytes:
        result = self._run("exec-out", *parts, text=False, timeout_s=timeout_s)
        data = result.stdout
        if isinstance(data, bytes):
            return data
        return str(data or "").encode("utf-8")

    def getprop(self, name: str) -> str:
        return self.shell("getprop", name, timeout_s=5.0)

    def describe_device(self) -> Dict[str, Any]:
        serial = self._serial or ""
        model = self.getprop("ro.product.model")
        android_version = self.getprop("ro.build.version.release")
        sdk = self.getprop("ro.build.version.sdk")
        try:
            current_app = self.current_app()
        except PhoneAdapterError:
            current_app = None
        return {
            "detected": True,
            "transport": "adb",
            "adapter_available": True,
            "serial": serial,
            "model": model or None,
            "android_version": android_version or None,
            "sdk": sdk or None,
            "current_app": current_app,
        }

    def current_app(self) -> Optional[str]:
        text = self.shell("dumpsys", "window", "windows", timeout_s=8.0)
        match = re.search(r"mCurrentFocus=.*? ([A-Za-z0-9._]+)/([A-Za-z0-9._$]+)", text)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        text = self.shell("dumpsys", "activity", "activities", timeout_s=8.0)
        match = re.search(r"mResumedActivity:.*? ([A-Za-z0-9._]+)/([A-Za-z0-9._$]+)", text)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        return None

    def capture_screen(self, abs_path: str) -> Dict[str, Any]:
        png = self.exec_out("screencap", "-p", timeout_s=20.0)
        if not png:
            raise PhoneAdapterError("Unable to capture phone screen")
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as f:
            f.write(png)
        width, height = self._png_size(png)
        return {"width": width, "height": height, "serial": self._serial}

    def tap(self, x: int, y: int) -> Dict[str, Any]:
        self.shell("input", "tap", str(int(x)), str(int(y)), timeout_s=6.0)
        return {"tapped": True, "x": int(x), "y": int(y), "serial": self._serial}

    def swipe(self, start: Dict[str, Any], end: Dict[str, Any], duration_ms: Optional[int] = None) -> Dict[str, Any]:
        x1 = int(start.get("x", 0))
        y1 = int(start.get("y", 0))
        x2 = int(end.get("x", 0))
        y2 = int(end.get("y", 0))
        parts = ["input", "swipe", str(x1), str(y1), str(x2), str(y2)]
        if duration_ms is not None:
            parts.append(str(int(duration_ms)))
        self.shell(*parts, timeout_s=8.0)
        return {
            "swiped": True,
            "start": {"x": x1, "y": y1},
            "end": {"x": x2, "y": y2},
            "duration_ms": int(duration_ms) if duration_ms is not None else None,
            "serial": self._serial,
        }

    def type_text(self, text: str) -> Dict[str, Any]:
        escaped = self._escape_input_text(text)
        self.shell("input", "text", escaped, timeout_s=8.0)
        return {"typed": True, "chars": len(text), "serial": self._serial}

    def launch_app(self, package: str, activity: Optional[str] = None) -> Dict[str, Any]:
        if activity:
            component = f"{package}/{activity}"
            self.shell("am", "start", "-n", component, timeout_s=10.0)
            return {"launched": True, "component": component, "serial": self._serial}
        self.shell("monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1", timeout_s=10.0)
        return {"launched": True, "package": package, "serial": self._serial}

    @staticmethod
    def _escape_input_text(text: str) -> str:
        escaped = text.replace("\\", "\\\\").replace(" ", "%s")
        escaped = escaped.replace("&", "\\&").replace("|", "\\|").replace(";", "\\;")
        escaped = escaped.replace("<", "\\<").replace(">", "\\>")
        escaped = escaped.replace("(", "\\(").replace(")", "\\)")
        escaped = escaped.replace("\"", "\\\"").replace("'", "\\'")
        return escaped

    @staticmethod
    def _png_size(data: bytes) -> tuple[int, int]:
        if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", data[16:24])
            return int(width), int(height)
        return 0, 0
