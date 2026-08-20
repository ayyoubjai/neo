import json
import importlib.util
import os
import platform
import shutil
import socket
import sys
from typing import Any, Dict, Optional

from common.config import Settings, load_settings
from common.time_utils import utc_now_iso
from embodiment.adapters import (
    AdbPhoneAdapter,
    DesktopAdapter,
    DesktopAdapterError,
    RosCliRobotAdapter,
)
from embodiment.fallback import EmbodimentFallbackPlanner
from embodiment.models import EmbodimentCapability, EmbodimentDevice, EmbodimentHostProfile
from embodiment.registry import EmbodimentRegistry


class EmbodimentManager:
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or load_settings()

    def describe_host(self) -> Dict[str, Any]:
        registry = self._build_registry()
        payload = registry.describe()
        fallbacks = self._build_fallbacks(registry)
        payload["fallbacks"] = fallbacks
        payload["summary"]["fallback_count"] = len(fallbacks)
        return payload

    def list_capabilities(self) -> Dict[str, Any]:
        registry = self._build_registry()
        payload = registry.capabilities_payload()
        payload["summary"]["fallback_count"] = len(self._build_fallbacks(registry))
        return payload

    def list_fallbacks(self) -> Dict[str, Any]:
        registry = self._build_registry()
        fallbacks = self._build_fallbacks(registry)
        return {
            "host_id": registry.host.host_id,
            "fallbacks": fallbacks,
            "summary": {"fallback_count": len(fallbacks)},
        }

    def list_launchable_applications(self) -> Dict[str, Any]:
        catalog = self._load_desktop_application_catalog()
        applications = [
            {
                "app_id": item["app_id"],
                "name": item["name"],
                "description": item["description"],
                "aliases": list(item["aliases"]),
            }
            for item in catalog["applications"]
        ]
        return {
            "platform": catalog["platform"],
            "catalog_path": catalog["catalog_path"],
            "catalog_exists": catalog["catalog_exists"],
            "count": len(applications),
            "applications": applications,
            "error": catalog.get("error"),
        }

    def launch_desktop_application(self, app_id: str) -> Dict[str, Any]:
        resolved = self._resolve_launchable_application(app_id)
        if not DesktopAdapter.launch_supported(headless=self._is_headless()):
            raise DesktopAdapterError("Desktop application launch is not available on this host")
        result = DesktopAdapter().launch_application(resolved["command"])
        return {
            "app_id": resolved["app_id"],
            "name": resolved["name"],
            "platform": resolved["platform"],
            "launched": True,
            "pid": result.get("pid"),
            "command": result.get("command"),
        }

    def summarize_context(self, max_capabilities: int = 8) -> Dict[str, Any]:
        registry = self._build_registry()
        capabilities = registry.capabilities()
        fallbacks = self._build_fallbacks(registry)
        app_catalog = self._load_desktop_application_catalog()
        available = [
            item["capability_id"]
            for item in capabilities
            if item.get("status") == "available"
        ]
        configured = [
            item["capability_id"]
            for item in capabilities
            if item.get("status") == "configured"
        ]
        available_abstract = self._unique_capabilities(
            item.get("abstract_capability") or item.get("capability_id")
            for item in capabilities
            if item.get("status") == "available"
        )
        configured_abstract = self._unique_capabilities(
            item.get("abstract_capability") or item.get("capability_id")
            for item in capabilities
            if item.get("status") == "configured"
        )
        screen = self._probe_screen_size()
        active_window = self._probe_active_window()
        summary_parts = [
            f"host={registry.host.host_class} {registry.host.platform}",
            f"interface={registry.host.interface_mode}",
            "headless=yes" if registry.host.headless else "headless=no",
        ]
        if screen.get("width") and screen.get("height"):
            summary_parts.append(f"screen={screen['width']}x{screen['height']}")
        if active_window:
            summary_parts.append(f"active_window={active_window}")
        if available:
            summary_parts.append(
                "available_capabilities=" + ", ".join(available[: max(1, max_capabilities)])
            )
        if configured:
            summary_parts.append(
                "configured_capabilities=" + ", ".join(configured[: max(1, max_capabilities // 2 or 1)])
            )
        if available_abstract:
            summary_parts.append(
                "available_abstract_capabilities="
                + ", ".join(available_abstract[: max(1, max_capabilities)])
            )
        if app_catalog["applications"]:
            app_ids = [item["app_id"] for item in app_catalog["applications"][: max(1, max_capabilities // 2 or 1)]]
            summary_parts.append("launchable_apps=" + ", ".join(app_ids))
        if fallbacks:
            summary_parts.append(f"fallback_count={len(fallbacks)}")
        return {
            "summary": "; ".join(summary_parts),
            "host": registry.host.to_dict(),
            "available_capabilities": available[: max(1, max_capabilities)],
            "configured_capabilities": configured[: max(1, max_capabilities)],
            "available_abstract_capabilities": available_abstract[: max(1, max_capabilities)],
            "configured_abstract_capabilities": configured_abstract[: max(1, max_capabilities)],
            "active_window_title": active_window,
            "fallback_count": len(fallbacks),
        }

    def get_state(self) -> Dict[str, Any]:
        registry = self._build_registry()
        screen = self._probe_screen_size()
        battery = self._probe_battery()
        audio = self._probe_audio_devices()
        phone = self._detect_phone_device()
        robot = self._detect_ros_controller()
        fallbacks = self._build_fallbacks(registry)
        app_catalog = self._load_desktop_application_catalog()
        return {
            "host_id": registry.host.host_id,
            "ts": utc_now_iso(),
            "interface_mode": registry.host.interface_mode,
            "headless": registry.host.headless,
            "screen": {
                "width": screen.get("width"),
                "height": screen.get("height"),
                "active_window_title": self._probe_active_window(),
            },
            "audio": {
                "configured_input_device": self._settings.voice.get("input_device"),
                "configured_output_device": self._settings.voice.get("sound_output_device"),
                "input_device_count": audio.get("input_count"),
                "output_device_count": audio.get("output_count"),
                "tts_enabled": bool(self._settings.voice.get("tts_enabled", False)),
                "error": audio.get("error"),
            },
            "vision": {
                "camera_index": self._settings.vision.get("camera_index", 0),
                "preview_enabled": bool(self._settings.vision.get("preview_enabled", False)),
                "track_enabled": bool(self._settings.vision.get("track_enabled", False)),
                "model_path": self._resolve_local_path(str(self._settings.vision.get("model_path", ""))),
            },
            "power": battery,
            "network": {
                "hostname": self._host_name(),
                "search_provider": self._settings.search.get("provider"),
                "search_endpoint": self._settings.search.get("searxng_url"),
            },
            "applications": {
                "platform": app_catalog["platform"],
                "catalog_path": app_catalog["catalog_path"],
                "catalog_exists": app_catalog["catalog_exists"],
                "count": len(app_catalog["applications"]),
                "app_ids": [item["app_id"] for item in app_catalog["applications"]],
                "error": app_catalog.get("error"),
            },
            "phone": phone,
            "robot": robot,
            "capability_summary": registry.summary(),
            "fallback_count": len(fallbacks),
        }

    def _build_registry(self) -> EmbodimentRegistry:
        headless = self._is_headless()
        host_name = self._host_name()
        host = EmbodimentHostProfile(
            host_id=f"host:{host_name}",
            name=host_name,
            host_class=self._host_class(),
            platform=sys.platform,
            platform_release=platform.release(),
            architecture=platform.machine(),
            python_version=platform.python_version(),
            interface_mode=str(self._settings.interface.get("mode", "text")),
            headless=headless,
        )
        registry = EmbodimentRegistry(host)

        ui_adapter_available = self._module_available("pyautogui")
        ui_grounding_available = bool(str(self._settings.models.get("ui_grounding_model", "")).strip())
        window_adapter_available = self._module_available("pygetwindow")
        audio_probe = self._probe_audio_devices()
        tts_available = self._tts_available()
        camera_adapter_available = self._module_available("cv2")
        vision_adapter_available = camera_adapter_available and self._module_available("ultralytics")
        search_available = bool(str(self._settings.search.get("provider", "")).strip())
        phone_state = self._detect_phone_device()
        ros_state = self._detect_ros_controller()
        screen = self._probe_screen_size()
        app_catalog = self._load_desktop_application_catalog()
        vision_model_path = self._resolve_local_path(str(self._settings.vision.get("model_path", "")))
        vision_model_exists = bool(vision_model_path and os.path.exists(vision_model_path))
        launch_supported = DesktopAdapter.launch_supported(headless=headless)
        if launch_supported and app_catalog["applications"]:
            app_status = "available"
        elif app_catalog["catalog_exists"]:
            app_status = "configured"
        else:
            app_status = "unavailable"

        display_status = "available" if (ui_adapter_available and not headless) else "unavailable"
        display_capabilities = [
            "screen.capture",
            "screen.inspect",
            "pointer.move",
            "pointer.click",
            "pointer.drag",
            "pointer.scroll",
            "keyboard.type",
            "window.focus",
        ]
        registry.add_device(
            EmbodimentDevice(
                device_id="display.main",
                kind="display",
                name="Primary display and desktop UI",
                status=display_status,
                capabilities=display_capabilities,
                details={
                    "headless": headless,
                    "width": screen.get("width"),
                    "height": screen.get("height"),
                    "ui_adapter_available": ui_adapter_available,
                    "window_adapter_available": window_adapter_available,
                },
            )
        )
        registry.add_device(
            EmbodimentDevice(
                device_id="desktop.apps",
                kind="application_runtime",
                name="Desktop application launcher",
                status=app_status,
                capabilities=["app.catalog", "app.launch"],
                details={
                    "platform": app_catalog["platform"],
                    "catalog_path": app_catalog["catalog_path"],
                    "catalog_exists": app_catalog["catalog_exists"],
                    "launch_supported": launch_supported,
                    "application_count": len(app_catalog["applications"]),
                    "app_ids": [item["app_id"] for item in app_catalog["applications"]],
                    "error": app_catalog.get("error"),
                },
            )
        )
        registry.add_capability(
            self._capability(
                "app.catalog",
                "observe",
                "LOW",
                "desktop.apps",
                app_status,
                ["ui.list_applications"],
                {
                    "platform": app_catalog["platform"],
                    "application_count": len(app_catalog["applications"]),
                    "app_ids": [item["app_id"] for item in app_catalog["applications"]],
                },
                abstract_capability="app.catalog",
            )
        )
        registry.add_capability(
            self._capability(
                "app.launch",
                "interact",
                "HIGH",
                "desktop.apps",
                app_status,
                ["ui.open_application"],
                {
                    "platform": app_catalog["platform"],
                    "application_count": len(app_catalog["applications"]),
                    "app_ids": [item["app_id"] for item in app_catalog["applications"]],
                },
                abstract_capability="app.launch",
            )
        )
        registry.add_capability(
            self._capability(
                "screen.capture",
                "observe",
                "MEDIUM",
                "display.main",
                "available" if display_status == "available" else "unavailable",
                ["embodiment.screen_capture", "ui.screenshot"],
            )
        )
        registry.add_capability(
            self._capability(
                "screen.inspect",
                "observe",
                "MEDIUM",
                "display.main",
                "available" if (display_status == "available" and ui_grounding_available) else "unavailable",
                ["ui.predict_coords", "image.analyse"],
                {"ui_grounding_model": self._settings.models.get("ui_grounding_model")},
            )
        )
        registry.add_capability(
            self._capability(
                "pointer.move",
                "interact",
                "HIGH",
                "display.main",
                "available" if display_status == "available" else "unavailable",
                ["ui.move"],
            )
        )
        registry.add_capability(
            self._capability(
                "pointer.click",
                "interact",
                "HIGH",
                "display.main",
                "available" if display_status == "available" else "unavailable",
                ["embodiment.pointer_click", "ui.click"],
            )
        )
        registry.add_capability(
            self._capability(
                "pointer.drag",
                "interact",
                "HIGH",
                "display.main",
                "available" if display_status == "available" else "unavailable",
                ["ui.drag"],
            )
        )
        registry.add_capability(
            self._capability(
                "pointer.scroll",
                "interact",
                "HIGH",
                "display.main",
                "available" if display_status == "available" else "unavailable",
                ["ui.scroll"],
            )
        )
        registry.add_capability(
            self._capability(
                "keyboard.type",
                "interact",
                "HIGH",
                "display.main",
                "available" if display_status == "available" else "unavailable",
                ["embodiment.keyboard_type", "ui.type", "ui.hotkey"],
            )
        )
        registry.add_capability(
            self._capability(
                "window.focus",
                "interact",
                "HIGH",
                "display.main",
                "available" if (display_status == "available" and window_adapter_available) else "unavailable",
                ["embodiment.window_focus", "ui.focus_window"],
            )
        )

        input_count = audio_probe.get("input_count")
        output_count = audio_probe.get("output_count")
        microphone_status = self._availability_status(input_count)
        speaker_status = "available" if tts_available else self._availability_status(output_count)
        registry.add_device(
            EmbodimentDevice(
                device_id="microphone.main",
                kind="microphone",
                name="Primary microphone input",
                status=microphone_status,
                capabilities=["audio.listen"],
                details={
                    "configured_input_device": self._settings.voice.get("input_device"),
                    "input_device_count": input_count,
                    "error": audio_probe.get("error"),
                },
            )
        )
        registry.add_capability(
            self._capability(
                "audio.listen",
                "observe",
                "MEDIUM",
                "microphone.main",
                microphone_status,
                [],
            )
        )
        registry.add_device(
            EmbodimentDevice(
                device_id="speaker.main",
                kind="speaker",
                name="Primary audio output",
                status=speaker_status,
                capabilities=["audio.speak"],
                details={
                    "configured_output_device": self._settings.voice.get("sound_output_device"),
                    "output_device_count": output_count,
                    "tts_enabled": bool(self._settings.voice.get("tts_enabled", False)),
                    "tts_available": tts_available,
                    "error": audio_probe.get("error"),
                },
            )
        )
        registry.add_capability(
            self._capability(
                "audio.speak",
                "interact",
                "MEDIUM",
                "speaker.main",
                speaker_status,
                [],
            )
        )

        camera_status = "available" if camera_adapter_available else "unavailable"
        vision_status = "available" if (camera_status == "available" and vision_adapter_available and vision_model_exists) else "configured"
        if not camera_adapter_available:
            vision_status = "unavailable"
        registry.add_device(
            EmbodimentDevice(
                device_id="camera.primary",
                kind="camera",
                name="Primary camera",
                status=camera_status,
                capabilities=["camera.capture", "vision.detect"],
                details={
                    "camera_index": self._settings.vision.get("camera_index", 0),
                    "preview_enabled": bool(self._settings.vision.get("preview_enabled", False)),
                    "adapter_available": camera_adapter_available,
                    "vision_adapter_available": vision_adapter_available,
                    "vision_model_path": vision_model_path,
                    "vision_model_exists": vision_model_exists,
                },
            )
        )
        registry.add_capability(
            self._capability(
                "camera.capture",
                "observe",
                "MEDIUM",
                "camera.primary",
                camera_status,
                ["embodiment.camera_capture"],
            )
        )
        registry.add_capability(
            self._capability(
                "vision.detect",
                "observe",
                "MEDIUM",
                "camera.primary",
                vision_status,
                [],
                {"model_path": vision_model_path, "model_exists": vision_model_exists},
            )
        )

        if phone_state.get("detected"):
            phone_status = "available" if phone_state.get("adapter_available") else "configured"
            registry.add_device(
                EmbodimentDevice(
                    device_id="phone.primary",
                    kind="phone",
                    name=str(phone_state.get("model") or "Phone device"),
                    status=phone_status,
                    capabilities=[
                        "phone.screen.capture",
                        "phone.pointer.tap",
                        "phone.pointer.swipe",
                        "phone.keyboard.type",
                        "phone.app.focus",
                    ],
                    details=dict(phone_state),
                )
            )
            registry.add_capability(
                self._capability(
                    "phone.screen.capture",
                    "observe",
                    "MEDIUM",
                    "phone.primary",
                    phone_status,
                    ["embodiment.phone_screen_capture"],
                    {"transport": phone_state.get("transport")},
                    abstract_capability="screen.capture",
                )
            )
            registry.add_capability(
                self._capability(
                    "phone.pointer.tap",
                    "interact",
                    "HIGH",
                    "phone.primary",
                    phone_status,
                    ["embodiment.phone_tap"],
                    {"transport": phone_state.get("transport")},
                    abstract_capability="pointer.click",
                )
            )
            registry.add_capability(
                self._capability(
                    "phone.pointer.swipe",
                    "interact",
                    "HIGH",
                    "phone.primary",
                    phone_status,
                    ["embodiment.phone_swipe"],
                    {"transport": phone_state.get("transport")},
                    abstract_capability="pointer.drag",
                )
            )
            registry.add_capability(
                self._capability(
                    "phone.keyboard.type",
                    "interact",
                    "HIGH",
                    "phone.primary",
                    phone_status,
                    ["embodiment.phone_type"],
                    {"transport": phone_state.get("transport")},
                    abstract_capability="keyboard.type",
                )
            )
            registry.add_capability(
                self._capability(
                    "phone.app.focus",
                    "interact",
                    "HIGH",
                    "phone.primary",
                    phone_status,
                    ["embodiment.phone_launch_app"],
                    {"transport": phone_state.get("transport")},
                    abstract_capability="app.focus",
                )
            )

        registry.add_device(
            EmbodimentDevice(
                device_id="workspace.primary",
                kind="filesystem",
                name="Workspace filesystem",
                status="available",
                capabilities=["filesystem.workspace.read", "filesystem.workspace.write"],
                details={
                    "workspace_root": self._settings.workspace_root,
                    "data_dir": self._settings.data_dir,
                },
            )
        )
        registry.add_capability(
            self._capability(
                "filesystem.workspace.read",
                "observe",
                "LOW",
                "workspace.primary",
                "available",
                ["fs.read_file", "fs.list_dir"],
            )
        )
        registry.add_capability(
            self._capability(
                "filesystem.workspace.write",
                "interact",
                "MEDIUM",
                "workspace.primary",
                "available",
                ["fs.write_file", "fs.move", "fs.copy", "fs.mkdir", "fs.delete"],
            )
        )

        network_status = "available" if search_available else "configured"
        registry.add_device(
            EmbodimentDevice(
                device_id="network.primary",
                kind="network",
                name="Primary network interface",
                status=network_status,
                capabilities=["network.search"],
                details={
                    "hostname": host_name,
                    "search_provider": self._settings.search.get("provider"),
                    "search_endpoint": self._settings.search.get("searxng_url"),
                },
            )
        )
        registry.add_capability(
            self._capability(
                "network.search",
                "observe",
                "MEDIUM",
                "network.primary",
                network_status,
                ["net.search"],
            )
        )

        if ros_state.get("detected"):
            robot_status = (
                "available"
                if ros_state.get("command_bridge_available", True)
                else "configured"
            )
            registry.add_device(
                EmbodimentDevice(
                    device_id="robot.controller.1",
                    kind="robot_controller",
                    name="Detected robot controller",
                    status=robot_status,
                    capabilities=["robot.get_state", "robot.control", "robot.move_joint", "robot.set_gripper"],
                    details=dict(ros_state),
                )
            )
            registry.add_capability(
                self._capability(
                    "robot.get_state",
                    "observe",
                    "MEDIUM",
                    "robot.controller.1",
                    robot_status,
                    ["embodiment.robot_get_state"],
                    {"transport": ros_state.get("transport")},
                    abstract_capability="robot.observe",
                )
            )
            registry.add_capability(
                self._capability(
                    "robot.control",
                    "actuate",
                    "VERY_HIGH",
                    "robot.controller.1",
                    robot_status,
                    ["embodiment.robot_move_joint", "embodiment.robot_set_gripper"],
                    {"transport": ros_state.get("transport")},
                    abstract_capability="robot.control",
                )
            )
            registry.add_capability(
                self._capability(
                    "robot.move_joint",
                    "actuate",
                    "VERY_HIGH",
                    "robot.controller.1",
                    robot_status,
                    ["embodiment.robot_move_joint"],
                    {"transport": ros_state.get("transport")},
                    abstract_capability="robot.control",
                )
            )
            registry.add_capability(
                self._capability(
                    "robot.set_gripper",
                    "actuate",
                    "VERY_HIGH",
                    "robot.controller.1",
                    robot_status,
                    ["embodiment.robot_set_gripper"],
                    {"transport": ros_state.get("transport")},
                    abstract_capability="robot.control",
                )
            )

        return registry

    def _capability(
        self,
        capability_id: str,
        category: str,
        safety_class: str,
        device_id: str,
        status: str,
        tool_ids,
        details: Optional[Dict[str, Any]] = None,
        abstract_capability: Optional[str] = None,
    ) -> EmbodimentCapability:
        return EmbodimentCapability(
            capability_id=capability_id,
            category=category,
            safety_class=safety_class,
            status=status,
            device_id=device_id,
            abstract_capability=str(abstract_capability or capability_id),
            tool_ids=list(tool_ids),
            details=dict(details or {}),
        )

    def _build_fallbacks(self, registry: EmbodimentRegistry):
        planner = EmbodimentFallbackPlanner(settings=self._settings)
        return planner.build(registry)

    @staticmethod
    def _unique_capabilities(values) -> list[str]:
        seen = set()
        ordered = []
        for item in values:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _module_available(self, module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _is_headless(self) -> bool:
        if sys.platform in {"win32", "darwin"}:
            return False
        return not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def _host_class(self) -> str:
        if os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA") or sys.platform == "android":
            return "phone"
        if os.environ.get("TERMUX_VERSION"):
            return "phone"
        if sys.platform in {"win32", "darwin"} or sys.platform.startswith("linux"):
            return "computer"
        return "unknown"

    def _host_name(self) -> str:
        value = socket.gethostname().strip()
        return value or "local"

    def _detect_phone_device(self) -> Dict[str, Any]:
        host_class = self._host_class()
        if host_class == "phone":
            return {
                "detected": True,
                "transport": "local",
                "adapter_available": False,
                "host_managed": True,
                "model": None,
                "android_version": None,
                "current_app": None,
            }
        detected = AdbPhoneAdapter.detect()
        if detected.get("detected"):
            detected["host_managed"] = False
            return detected
        detected["host_managed"] = False
        return detected

    def _tts_available(self) -> bool:
        if self._module_available("pyttsx3"):
            return True
        if sys.platform == "darwin" and shutil.which("say"):
            return True
        if sys.platform.startswith("linux") and shutil.which("espeak"):
            return True
        return bool(sys.platform == "win32")

    def _probe_audio_devices(self) -> Dict[str, Any]:
        if not self._module_available("sounddevice"):
            return {"input_count": None, "output_count": None, "error": "sounddevice unavailable"}
        try:
            import sounddevice as sd
        except Exception as e:
            return {"input_count": None, "output_count": None, "error": str(e)}
        try:
            devices = sd.query_devices()
        except Exception as e:
            return {"input_count": None, "output_count": None, "error": str(e)}
        input_count = 0
        output_count = 0
        for item in devices:
            try:
                max_input = int(item.get("max_input_channels", 0))
            except Exception:
                max_input = 0
            try:
                max_output = int(item.get("max_output_channels", 0))
            except Exception:
                max_output = 0
            if max_input > 0:
                input_count += 1
            if max_output > 0:
                output_count += 1
        return {"input_count": input_count, "output_count": output_count, "error": None}

    def _probe_screen_size(self) -> Dict[str, Any]:
        if self._is_headless() or not DesktopAdapter.is_available():
            return {"width": None, "height": None}
        try:
            adapter = DesktopAdapter()
            return adapter.screen_size()
        except DesktopAdapterError:
            return {"width": None, "height": None}

    def _probe_active_window(self) -> Optional[str]:
        if self._is_headless() or not self._module_available("pygetwindow"):
            return None
        try:
            adapter = DesktopAdapter()
            return adapter.active_window_title()
        except DesktopAdapterError:
            return None

    def _probe_battery(self) -> Dict[str, Any]:
        if not self._module_available("psutil"):
            return {"available": False}
        try:
            import psutil
        except Exception:
            return {"available": False}
        try:
            battery = psutil.sensors_battery()
        except Exception:
            battery = None
        if battery is None:
            return {"available": False}
        percent = getattr(battery, "percent", None)
        power_plugged = getattr(battery, "power_plugged", None)
        secsleft = getattr(battery, "secsleft", None)
        return {
            "available": True,
            "percent": percent,
            "power_plugged": power_plugged,
            "secsleft": secsleft,
        }

    def _detect_ros_controller(self) -> Dict[str, Any]:
        return RosCliRobotAdapter().describe()

    def _availability_status(self, count: Optional[int]) -> str:
        if count is None:
            return "unknown"
        if count > 0:
            return "available"
        return "unavailable"

    def _resolve_local_path(self, value: str) -> Optional[str]:
        value = value.strip()
        if not value:
            return None
        if os.path.isabs(value):
            return value
        return os.path.abspath(os.path.join(self._settings.workspace_root, value))

    def _desktop_application_catalog_path(self) -> str:
        raw_path = str((self._settings.tool or {}).get("application_catalog_path", "")).strip()
        if raw_path:
            if os.path.isabs(raw_path):
                return raw_path
            return os.path.abspath(os.path.join(self._settings.workspace_root, raw_path))
        return os.path.join(self._settings.workspace_root, "config", "app_launchers.json")

    def _load_desktop_application_catalog(self) -> Dict[str, Any]:
        catalog_path = self._desktop_application_catalog_path()
        runtime_platform = DesktopAdapter.runtime_platform()
        payload: Dict[str, Any] = {
            "platform": runtime_platform,
            "catalog_path": catalog_path,
            "catalog_exists": os.path.exists(catalog_path),
            "applications": [],
        }
        if not payload["catalog_exists"]:
            return payload
        try:
            with open(catalog_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as exc:
            payload["error"] = str(exc)
            return payload
        if not isinstance(raw, dict):
            payload["error"] = "Application catalog must be a JSON object"
            return payload

        for item in raw.get("applications", []):
            if not isinstance(item, dict):
                continue
            app_id = str(item.get("app_id") or "").strip()
            if not app_id:
                continue
            command = self._command_for_runtime(item.get("commands"), runtime_platform)
            if not command:
                continue
            raw_aliases = item.get("aliases", [])
            if not isinstance(raw_aliases, list):
                raw_aliases = []
            aliases = [
                str(alias).strip()
                for alias in raw_aliases
                if isinstance(alias, str) and str(alias).strip()
            ]
            payload["applications"].append(
                {
                    "app_id": app_id,
                    "name": str(item.get("name") or app_id).strip() or app_id,
                    "description": str(item.get("description") or "").strip(),
                    "aliases": aliases,
                    "command": command,
                    "platform": runtime_platform,
                }
            )
        return payload

    @staticmethod
    def _command_for_runtime(raw_commands: Any, runtime_platform: str) -> list[str]:
        if isinstance(raw_commands, list):
            command = raw_commands
        elif isinstance(raw_commands, dict):
            candidate = raw_commands.get(runtime_platform)
            if candidate is None and runtime_platform == "wsl":
                candidate = raw_commands.get("windows")
            if candidate is None and runtime_platform == "linux":
                candidate = raw_commands.get("posix")
            if candidate is None and runtime_platform == "darwin":
                candidate = raw_commands.get("macos")
            command = candidate
        else:
            command = None
        if not isinstance(command, list):
            return []
        normalized = [str(part).strip() for part in command if str(part).strip()]
        return normalized

    def _resolve_launchable_application(self, app_id: str) -> Dict[str, Any]:
        text = str(app_id or "").strip().lower()
        if not text:
            raise DesktopAdapterError("app_id is required")
        catalog = self._load_desktop_application_catalog()
        for item in catalog["applications"]:
            candidates = {item["app_id"].lower(), item["name"].lower()}
            candidates.update(alias.lower() for alias in item["aliases"])
            if text in candidates:
                return item
        available = ", ".join(item["app_id"] for item in catalog["applications"][:10])
        if not available:
            raise DesktopAdapterError("No launchable desktop applications are configured for this host")
        raise DesktopAdapterError(f"Unknown app_id '{app_id}'. Available app_ids: {available}")
