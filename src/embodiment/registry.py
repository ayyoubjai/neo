from typing import Any, Dict, List

from embodiment.models import EmbodimentCapability, EmbodimentDevice, EmbodimentHostProfile


class EmbodimentRegistry:
    def __init__(self, host: EmbodimentHostProfile):
        self._host = host
        self._devices: List[EmbodimentDevice] = []
        self._capabilities: List[EmbodimentCapability] = []

    @property
    def host(self) -> EmbodimentHostProfile:
        return self._host

    def add_device(self, device: EmbodimentDevice) -> None:
        self._devices.append(device)

    def add_capability(self, capability: EmbodimentCapability) -> None:
        self._capabilities.append(capability)

    def devices(self) -> List[Dict[str, Any]]:
        return [device.to_dict() for device in self._devices]

    def capabilities(self) -> List[Dict[str, Any]]:
        return [capability.to_dict() for capability in self._capabilities]

    def summary(self) -> Dict[str, int]:
        capability_counts = {"available": 0, "configured": 0, "unknown": 0, "unavailable": 0}
        for item in self._capabilities:
            capability_counts[item.status] = capability_counts.get(item.status, 0) + 1
        device_counts = {"available": 0, "configured": 0, "unknown": 0, "unavailable": 0}
        for item in self._devices:
            device_counts[item.status] = device_counts.get(item.status, 0) + 1
        return {
            "device_count": len(self._devices),
            "capability_count": len(self._capabilities),
            "available_device_count": device_counts.get("available", 0),
            "available_capability_count": capability_counts.get("available", 0),
            "configured_capability_count": capability_counts.get("configured", 0),
            "unknown_capability_count": capability_counts.get("unknown", 0),
            "unavailable_capability_count": capability_counts.get("unavailable", 0),
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "host": self._host.to_dict(),
            "devices": self.devices(),
            "capabilities": self.capabilities(),
            "summary": self.summary(),
        }

    def capabilities_payload(self) -> Dict[str, Any]:
        return {
            "host_id": self._host.host_id,
            "capabilities": self.capabilities(),
            "summary": self.summary(),
        }
