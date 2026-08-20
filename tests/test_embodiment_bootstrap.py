from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import Settings
from embodiment.bootstrap import EmbodimentBootstrap


class _FakeEmbodimentManager:
    def __init__(self, *, devices=None, capabilities=None, fallbacks=None) -> None:
        self._devices = list(devices or [])
        self._capabilities = list(capabilities or [])
        self._fallbacks = list(fallbacks or [])

    def describe_host(self):
        return {
            "host": {"host_id": "host:test", "name": "test-host", "host_class": "computer"},
            "devices": list(self._devices),
            "capabilities": list(self._capabilities),
            "fallbacks": list(self._fallbacks),
            "summary": {
                "device_count": len(self._devices),
                "capability_count": len(self._capabilities),
                "fallback_count": len(self._fallbacks),
            },
        }


class _RegistryState:
    def __init__(self, tools) -> None:
        self._tools = {
            str(tool["tool_id"]): dict(tool)
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("tool_id"), str)
        }

    def registry_cls(self):
        state = self

        class _FakeToolRegistry:
            def __init__(self, embodiment_manager=None):
                self._embodiment_manager = embodiment_manager

            def get_tool(self, tool_id: str):
                tool = state._tools.get(tool_id)
                if tool is None:
                    return None
                return dict(tool)

            def list_active(self):
                return [dict(tool) for tool in state._tools.values()]

        return _FakeToolRegistry

    def add_tool(self, tool: dict) -> None:
        self._tools[str(tool["tool_id"])] = dict(tool)


def _settings(workspace_root: str) -> Settings:
    root = Path(workspace_root)
    return Settings(
        workspace_root=workspace_root,
        data_dir=str(root / "data"),
        rpc={"orch_host": "127.0.0.1", "model_port": 0, "tool_port": 0},
        interface={"mode": "local"},
        voice={},
        vision={},
        video={},
        telegram={},
        google={},
        tool={"active_tool_limit": 100},
        search={},
        orchestrator={},
        evolve={"candidate_dir": str(root / "evolve")},
        models={},
    )


class EmbodimentBootstrapTests(unittest.IsolatedAsyncioTestCase):
    def test_plan_only_queues_missing_fallback_tools(self) -> None:
        devices = [{"device_id": "phone.main", "kind": "phone"}]
        capabilities = [{"capability_id": "phone.pointer.tap", "status": "configured"}]
        fallbacks = [
            {
                "capability_id": "phone.pointer.tap",
                "abstract_capability": "pointer.click",
                "reason": "Phone capability detected without an available explicit low-level implementation.",
                "tool_spec": {
                    "tool_id": "embodiment.generated.phone_tap",
                    "name": "Generated Phone Tap",
                    "description": "Generated phone tap tool.",
                },
            },
            {
                "capability_id": "robot.move_joint",
                "abstract_capability": "robot.control",
                "reason": "Robot controller detected without an available explicit low-level implementation.",
                "tool_spec": {
                    "tool_id": "embodiment.generated.robot_move_joint",
                    "name": "Generated Robot Move Joint",
                    "description": "Generated robot joint tool.",
                },
            },
        ]
        registry_state = _RegistryState(
            [
                {"tool_id": "sys.time", "name": "Time", "description": "Time"},
                {"tool_id": "embodiment.describe_host", "name": "Host", "description": "Host"},
                {
                    "tool_id": "embodiment.generated.phone_tap",
                    "name": "Generated Phone Tap",
                    "description": "Generated phone tap tool.",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bootstrap = EmbodimentBootstrap(
                settings=_settings(tmpdir),
                embodiment_manager=_FakeEmbodimentManager(
                    devices=devices,
                    capabilities=capabilities,
                    fallbacks=fallbacks,
                ),
                tool_registry_cls=registry_state.registry_cls(),
            )

            plan = bootstrap.plan()

        self.assertEqual(plan["summary"]["planner_tool_count"], 3)
        self.assertEqual(plan["summary"]["tool_generation_queue_count"], 1)
        self.assertEqual(
            [item["tool_id"] for item in plan["planner_tools"]],
            ["sys.time", "embodiment.describe_host", "embodiment.generated.phone_tap"],
        )
        self.assertEqual(
            [item["tool_spec"]["tool_id"] for item in plan["tool_generation_queue"]],
            ["embodiment.generated.robot_move_joint"],
        )

    async def test_apply_creates_missing_tools_and_writes_manifest(self) -> None:
        devices = [{"device_id": "phone.main", "kind": "phone"}]
        capabilities = [{"capability_id": "phone.pointer.tap", "status": "configured"}]
        fallbacks = [
            {
                "capability_id": "phone.pointer.tap",
                "abstract_capability": "pointer.click",
                "reason": "Phone capability detected without an available explicit low-level implementation.",
                "tool_spec": {
                    "tool_id": "embodiment.generated.phone_tap",
                    "name": "Generated Phone Tap",
                    "description": "Generated phone tap tool.",
                    "implements_capability_id": "phone.pointer.tap",
                    "implements_abstract_capability": "pointer.click",
                },
            }
        ]
        registry_state = _RegistryState(
            [
                {"tool_id": "sys.time", "name": "Time", "description": "Time"},
                {"tool_id": "embodiment.describe_host", "name": "Host", "description": "Host"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            bootstrap = EmbodimentBootstrap(
                settings=_settings(tmpdir),
                embodiment_manager=_FakeEmbodimentManager(
                    devices=devices,
                    capabilities=capabilities,
                    fallbacks=fallbacks,
                ),
                tool_registry_cls=registry_state.registry_cls(),
            )
            manifest_path = str(Path(tmpdir) / "manifest" / "embodiment.json")
            create_calls = []

            async def _create_tool(spec):
                create_calls.append(dict(spec))
                registry_state.add_tool(spec)
                return {"status": "APPROVED", "result": {"tool_id": spec["tool_id"]}}

            manifest = await bootstrap.apply(_create_tool, manifest_path=manifest_path)

            self.assertEqual(len(create_calls), 1)
            self.assertEqual(create_calls[0]["tool_id"], "embodiment.generated.phone_tap")
            self.assertEqual(manifest["summary"]["created_tool_count"], 1)
            self.assertEqual(manifest["summary"]["failed_tool_count"], 0)
            self.assertEqual(manifest["summary"]["tool_generation_queue_count"], 0)
            self.assertEqual(
                [item["tool_id"] for item in manifest["tool_generation_results"]["created"]],
                ["embodiment.generated.phone_tap"],
            )

            manifest_file = Path(manifest_path)
            self.assertTrue(manifest_file.exists())
            stored = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(stored["summary"]["created_tool_count"], 1)
            self.assertEqual(stored["tool_generation_queue"], [])
            self.assertEqual(
                [item["tool_id"] for item in stored["planner_tools"]],
                ["sys.time", "embodiment.describe_host", "embodiment.generated.phone_tap"],
            )


if __name__ == "__main__":
    unittest.main()
