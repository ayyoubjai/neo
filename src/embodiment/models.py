from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EmbodimentHostProfile:
    host_id: str
    name: str
    host_class: str
    platform: str
    platform_release: str
    architecture: str
    python_version: str
    interface_mode: str
    headless: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id,
            "name": self.name,
            "host_class": self.host_class,
            "platform": self.platform,
            "platform_release": self.platform_release,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "interface_mode": self.interface_mode,
            "headless": self.headless,
        }


@dataclass
class EmbodimentDevice:
    device_id: str
    kind: str
    name: str
    status: str
    capabilities: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "capabilities": list(self.capabilities),
            "details": dict(self.details),
        }


@dataclass
class EmbodimentCapability:
    capability_id: str
    category: str
    safety_class: str
    status: str
    device_id: str
    abstract_capability: str = ""
    tool_ids: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "abstract_capability": self.abstract_capability or self.capability_id,
            "category": self.category,
            "safety_class": self.safety_class,
            "status": self.status,
            "device_id": self.device_id,
            "tool_ids": list(self.tool_ids),
            "details": dict(self.details),
        }
