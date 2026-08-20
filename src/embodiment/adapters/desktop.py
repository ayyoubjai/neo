import importlib.util
import os
import platform
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, Optional, Tuple


class DesktopAdapterError(RuntimeError):
    pass


class DesktopAdapter:
    def __init__(self, pyautogui_module=None, pygetwindow_module=None, launcher=None):
        self._pyautogui_module = pyautogui_module
        self._pygetwindow_module = pygetwindow_module
        self._launcher = launcher or subprocess.Popen

    @staticmethod
    def _module_available(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    @classmethod
    def is_available(cls) -> bool:
        return cls._module_available("pyautogui")

    @staticmethod
    def runtime_platform() -> str:
        if sys.platform == "win32":
            return "windows"
        if sys.platform == "darwin":
            return "darwin"
        if sys.platform.startswith("linux"):
            release = platform.release().lower()
            if "microsoft" in release or os.environ.get("WSL_DISTRO_NAME"):
                return "wsl"
            return "linux"
        return sys.platform

    @classmethod
    def launch_supported(cls, *, headless: bool = False) -> bool:
        runtime = cls.runtime_platform()
        if runtime in {"windows", "darwin", "wsl"}:
            return True
        if runtime == "linux":
            return not headless
        return False

    def _pyautogui(self):
        if self._pyautogui_module is not None:
            return self._pyautogui_module
        try:
            import pyautogui
        except ImportError as e:
            raise DesktopAdapterError(f"pyautogui not available: {e}") from e
        self._pyautogui_module = pyautogui
        return pyautogui

    def _pygetwindow(self):
        if self._pygetwindow_module is not None:
            return self._pygetwindow_module
        try:
            import pygetwindow as gw
        except ImportError as e:
            raise DesktopAdapterError(f"pygetwindow not available: {e}") from e
        self._pygetwindow_module = gw
        return gw

    def screen_size(self) -> Dict[str, Optional[int]]:
        pyautogui = self._pyautogui()
        try:
            size = pyautogui.size()
        except Exception as e:
            raise DesktopAdapterError(f"Unable to read screen size: {e}") from e
        return {"width": int(size.width), "height": int(size.height)}

    def capture_screen(self, abs_path: str, region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        pyautogui = self._pyautogui()
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        try:
            image = pyautogui.screenshot(region=region)
        except Exception as e:
            raise DesktopAdapterError(f"Unable to capture screen: {e}") from e
        image.save(abs_path)
        return {"width": int(image.width), "height": int(image.height)}

    def click(
        self,
        x: int,
        y: int,
        *,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.0,
    ) -> Dict[str, Any]:
        pyautogui = self._pyautogui()
        try:
            pyautogui.click(x=int(x), y=int(y), clicks=int(clicks), interval=float(interval), button=button)
        except Exception as e:
            raise DesktopAdapterError(f"Unable to click pointer: {e}") from e
        return {"clicked": True, "x": int(x), "y": int(y), "button": button, "clicks": int(clicks)}

    def move(self, x: int, y: int, *, duration: float = 0.0) -> Dict[str, Any]:
        pyautogui = self._pyautogui()
        try:
            pyautogui.moveTo(int(x), int(y), duration=float(duration))
        except Exception as e:
            raise DesktopAdapterError(f"Unable to move pointer: {e}") from e
        return {"moved": True, "x": int(x), "y": int(y)}

    def drag(
        self,
        start: Dict[str, Any],
        end: Dict[str, Any],
        *,
        duration: float = 0.0,
        button: str = "left",
    ) -> Dict[str, Any]:
        pyautogui = self._pyautogui()
        x1 = int(start.get("x", 0))
        y1 = int(start.get("y", 0))
        x2 = int(end.get("x", 0))
        y2 = int(end.get("y", 0))
        try:
            pyautogui.moveTo(x1, y1)
            pyautogui.dragTo(x2, y2, duration=float(duration), button=button)
        except Exception as e:
            raise DesktopAdapterError(f"Unable to drag pointer: {e}") from e
        return {"dragged": True, "start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}, "button": button}

    def type_text(self, text: str, *, interval: float = 0.0) -> Dict[str, Any]:
        pyautogui = self._pyautogui()
        try:
            pyautogui.typewrite(text, interval=float(interval))
        except Exception as e:
            raise DesktopAdapterError(f"Unable to type text: {e}") from e
        return {"typed": True, "chars": len(text)}

    def hotkey(self, keys: Iterable[str]) -> Dict[str, Any]:
        pyautogui = self._pyautogui()
        keys = [str(item) for item in keys]
        try:
            pyautogui.hotkey(*keys)
        except Exception as e:
            raise DesktopAdapterError(f"Unable to press hotkey: {e}") from e
        return {"hotkey": keys}

    def scroll(self, amount: int) -> Dict[str, Any]:
        pyautogui = self._pyautogui()
        try:
            pyautogui.scroll(int(amount))
        except Exception as e:
            raise DesktopAdapterError(f"Unable to scroll pointer: {e}") from e
        return {"scrolled": True, "amount": int(amount)}

    def focus_window(self, title: str) -> Dict[str, Any]:
        gw = self._pygetwindow()
        try:
            matches = [window for window in gw.getAllWindows() if title.lower() in window.title.lower()]
        except Exception as e:
            raise DesktopAdapterError(f"Unable to query windows: {e}") from e
        if not matches:
            return {"focused": False, "error": "No matching window"}
        win = matches[0]
        try:
            win.activate()
        except Exception as e:
            return {"focused": False, "error": str(e), "title": win.title}
        return {"focused": True, "title": win.title}

    def active_window_title(self) -> Optional[str]:
        gw = self._pygetwindow()
        try:
            active = gw.getActiveWindow()
        except Exception as e:
            raise DesktopAdapterError(f"Unable to query active window: {e}") from e
        title = getattr(active, "title", None)
        if not isinstance(title, str):
            return None
        title = title.strip()
        return title or None

    def launch_application(self, command: Iterable[str]) -> Dict[str, Any]:
        normalized = self._normalize_launch_command(command)
        try:
            proc = self._launcher(
                normalized,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise DesktopAdapterError(f"Application launcher not found: {e.filename or normalized[0]}") from e
        except OSError as e:
            raise DesktopAdapterError(f"Unable to launch application: {e}") from e
        pid = getattr(proc, "pid", None)
        return {"launched": True, "pid": int(pid) if isinstance(pid, int) else None, "command": normalized}

    def _normalize_launch_command(self, command: Iterable[str]) -> list[str]:
        if not isinstance(command, (list, tuple)):
            raise DesktopAdapterError("command must be a list of strings")
        normalized = [str(item).strip() for item in command if str(item).strip()]
        if not normalized:
            raise DesktopAdapterError("command must be a non-empty list of strings")
        if self.runtime_platform() == "wsl":
            normalized[0] = self._normalize_wsl_command_path(normalized[0])
        return normalized

    @staticmethod
    def _normalize_wsl_command_path(value: str) -> str:
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
        if not match:
            return value
        drive = match.group(1).lower()
        tail = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{tail}"
