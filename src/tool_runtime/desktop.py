"""Desktop capture without importing an input-control backend.

Wayland capture must go through the compositor; XWayland's root window is
not a reliable representation of the desktop.
"""
import os
import shutil
import subprocess
import sys
import tempfile


_wayland_input = None
_last_capture_size = None


def _remember_capture_size(size):
    global _last_capture_size
    _last_capture_size = tuple(size)


class DesktopError(RuntimeError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def is_wayland():
    return sys.platform.startswith("linux") and (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        or bool(os.environ.get("WAYLAND_DISPLAY"))
    )


def capture_screen(region=None):
    try:
        from PIL import Image, ImageGrab
    except ImportError as exc:
        raise DesktopError("Screenshot capture requires Pillow in the Python environment running Neo.", "MISSING_DEPENDENCY") from exc

    if not is_wayland():
        bbox = None
        if region is not None:
            x, y, width, height = region
            bbox = (x, y, x + width, y + height)
        return ImageGrab.grab(bbox=bbox)

    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    candidates = ["spectacle", "gnome-screenshot", "grim"]
    if "gnome" in desktop:
        candidates = ["gnome-screenshot", "grim", "spectacle"]
    elif "kde" not in desktop:
        candidates = ["grim", "spectacle", "gnome-screenshot"]
    failures = []
    with tempfile.TemporaryDirectory(prefix="neo-capture-") as directory:
        for backend in candidates:
            executable = shutil.which(backend)
            if not executable:
                continue
            path = os.path.join(directory, backend + ".png")
            options = {
                "spectacle": ["-b", "-n", "-f", "-o", path],
                "gnome-screenshot": ["-f", path],
                "grim": [path],
            }
            try:
                subprocess.run(
                    [executable, *options[backend]], check=True, timeout=20,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                with Image.open(path) as source:
                    source.load()
                    _remember_capture_size(source.size)
                    if region is None:
                        return source.copy()
                    x, y, width, height = region
                    return source.crop((x, y, x + width, y + height))
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(f"{backend}: {exc}")
    detail = "; ".join(failures) or "No compatible capture executable found"
    raise DesktopError(
        "Wayland screen capture failed. Install/use Spectacle for KDE, "
        "gnome-screenshot for GNOME, or grim for a compatible wlroots compositor. "
        "Run Neo inside the desktop session and allow any compositor capture prompt. "
        + detail, "UNAVAILABLE_SOURCE" if failures else "MISSING_DEPENDENCY"
    )


def load_input_backend():
    if is_wayland():
        global _wayland_input
        if _wayland_input is None:
            from tool_runtime.wayland_input import WaylandInput
            _wayland_input = WaylandInput(capture_size=lambda: _last_capture_size)
        return _wayland_input
    try:
        import pyautogui
    except Exception as exc:
        raise DesktopError(
            f"Cannot initialize desktop input with {sys.executable}: {exc}. "
            "Install the project requirements in this Python environment and run Neo "
            "inside an interactive desktop session (with DISPLAY/Xauthority on Linux).",
            "MISSING_DEPENDENCY" if isinstance(exc, ImportError) else "UNAVAILABLE_SOURCE"
        ) from exc
    return pyautogui
