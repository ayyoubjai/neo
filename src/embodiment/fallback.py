from typing import Any, Dict, List, Optional

from common.config import Settings, load_settings
from embodiment.registry import EmbodimentRegistry


class EmbodimentFallbackPlanner:
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or load_settings()

    def build(self, registry: EmbodimentRegistry) -> List[Dict[str, Any]]:
        devices = registry.devices()
        capabilities = registry.capabilities()
        capability_map = {
            item["capability_id"]: item
            for item in capabilities
            if isinstance(item, dict) and isinstance(item.get("capability_id"), str)
        }
        fallbacks: List[Dict[str, Any]] = []

        if registry.host.host_class == "phone" or self._device_present(devices, "phone"):
            fallbacks.extend(self._phone_fallbacks(capability_map))
        if self._device_present(devices, "robot_controller"):
            fallbacks.extend(self._robot_fallbacks(capability_map))
        return fallbacks

    def _phone_fallbacks(self, capability_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        fallbacks: List[Dict[str, Any]] = []
        specs = [
            (
                "phone.screen.capture",
                "embodiment.generated.phone_screen_capture",
                "Generate a low-level phone screen capture tool for the current host. Prefer locally available Android, Termux, ADB, or vendor SDK APIs. Save the resulting image to the workspace path.",
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "image_ref": {"type": "string"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["image_ref"],
                },
                ["write", "embodiment", "phone", "ui"],
                2,
                ["tier2"],
                {"network": False, "fs": "write"},
            ),
            (
                "phone.pointer.tap",
                "embodiment.generated.phone_tap",
                "Generate a low-level phone tap tool for the current host. Prefer locally available Android, Termux, ADB, or vendor SDK APIs. Perform a single tap on screen coordinates.",
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "tapped": {"type": "boolean"},
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["tapped"],
                },
                ["write", "embodiment", "phone", "ui"],
                2,
                ["tier2"],
                {"network": False, "fs": "none"},
            ),
            (
                "phone.pointer.swipe",
                "embodiment.generated.phone_swipe",
                "Generate a low-level phone swipe tool for the current host. Prefer locally available Android, Termux, ADB, or vendor SDK APIs. Swipe between concrete coordinates.",
                {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                            "required": ["x", "y"],
                        },
                        "end": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                            "required": ["x", "y"],
                        },
                        "duration_ms": {"type": "integer"},
                    },
                    "required": ["start", "end"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"swiped": {"type": "boolean"}},
                    "required": ["swiped"],
                },
                ["write", "embodiment", "phone", "ui"],
                2,
                ["tier2"],
                {"network": False, "fs": "none"},
            ),
            (
                "phone.app.focus",
                "embodiment.generated.phone_launch_app",
                "Generate a low-level phone app launch/focus tool for the current host. Prefer locally available Android, Termux, ADB, or vendor SDK APIs. Launch or focus the requested package/activity.",
                {
                    "type": "object",
                    "properties": {
                        "package": {"type": "string"},
                        "activity": {"type": "string"},
                    },
                    "required": ["package"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"launched": {"type": "boolean"}},
                    "required": ["launched"],
                },
                ["write", "embodiment", "phone", "ui"],
                2,
                ["tier2"],
                {"network": False, "fs": "none"},
            ),
        ]
        for capability_id, tool_id, description, input_schema, output_schema, capabilities, tier, perms, sandbox in specs:
            if self._needs_fallback(capability_map.get(capability_id)):
                fallbacks.append(
                    self._candidate(
                        capability_id=capability_id,
                        abstract_capability=capability_map.get(capability_id, {}).get("abstract_capability", capability_id),
                        reason="Phone capability detected without an available explicit low-level implementation.",
                        tool_spec={
                            "tool_id": tool_id,
                            "name": self._default_name(tool_id),
                            "description": description,
                            "implements_capability_id": capability_id,
                            "implements_abstract_capability": capability_map.get(capability_id, {}).get(
                                "abstract_capability",
                                capability_id,
                            ),
                            "input_schema": input_schema,
                            "output_schema": output_schema,
                            "capabilities": capabilities,
                            "tier": tier,
                            "required_permissions": perms,
                            "risk_level_default": "HIGH",
                            "sandbox_profile": sandbox,
                        },
                    )
                )
        return fallbacks

    def _robot_fallbacks(self, capability_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        fallbacks: List[Dict[str, Any]] = []
        specs = [
            (
                "robot.move_joint",
                "embodiment.generated.robot_move_joint",
                "Generate a low-level robot joint motion tool for the current host. Prefer locally available ROS, robot SDK, serial, or vendor APIs. Execute a single joint target command with strict validation.",
                {
                    "type": "object",
                    "properties": {
                        "joint": {"type": "string"},
                        "position": {"type": "number"},
                        "velocity": {"type": "number"},
                    },
                    "required": ["joint", "position"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"queued": {"type": "boolean"}},
                    "required": ["queued"],
                },
                ["write", "embodiment", "robot"],
                2,
                ["tier2"],
                {"network": False, "fs": "none"},
            ),
            (
                "robot.set_gripper",
                "embodiment.generated.robot_set_gripper",
                "Generate a low-level robot gripper control tool for the current host. Prefer locally available ROS, robot SDK, serial, or vendor APIs. Support open/close and optional width/force arguments.",
                {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string"},
                        "width": {"type": "number"},
                        "force": {"type": "number"},
                    },
                    "required": ["state"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"queued": {"type": "boolean"}},
                    "required": ["queued"],
                },
                ["write", "embodiment", "robot"],
                2,
                ["tier2"],
                {"network": False, "fs": "none"},
            ),
        ]
        for capability_id, tool_id, description, input_schema, output_schema, capabilities, tier, perms, sandbox in specs:
            if self._needs_fallback(capability_map.get(capability_id)):
                fallbacks.append(
                    self._candidate(
                        capability_id=capability_id,
                        abstract_capability=capability_map.get(capability_id, {}).get("abstract_capability", "robot.control"),
                        reason="Robot controller detected without an available explicit low-level implementation.",
                        tool_spec={
                            "tool_id": tool_id,
                            "name": self._default_name(tool_id),
                            "description": description,
                            "implements_capability_id": capability_id,
                            "implements_abstract_capability": capability_map.get(capability_id, {}).get(
                                "abstract_capability",
                                "robot.control",
                            ),
                            "input_schema": input_schema,
                            "output_schema": output_schema,
                            "capabilities": capabilities,
                            "tier": tier,
                            "required_permissions": perms,
                            "risk_level_default": "VERY_HIGH",
                            "sandbox_profile": sandbox,
                        },
                    )
                )
        return fallbacks

    @staticmethod
    def _device_present(devices: List[Dict[str, Any]], kind: str) -> bool:
        for item in devices:
            if isinstance(item, dict) and item.get("kind") == kind:
                return True
        return False

    @staticmethod
    def _needs_fallback(capability: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(capability, dict):
            return True
        return capability.get("status") != "available"

    @staticmethod
    def _candidate(
        capability_id: str,
        abstract_capability: str,
        reason: str,
        tool_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "capability_id": capability_id,
            "abstract_capability": abstract_capability,
            "reason": reason,
            "tool_spec": tool_spec,
        }

    @staticmethod
    def _default_name(tool_id: str) -> str:
        cleaned = tool_id.replace(".", " ").replace("_", " ").strip()
        return " ".join(part.capitalize() for part in cleaned.split()) or "Generated Embodiment Tool"
