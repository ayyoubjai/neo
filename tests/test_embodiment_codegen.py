from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import Settings
from orchestrator.main import Orchestrator


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
        tool={
            "active_tool_limit": 100,
            "planner_tool_profile": "default",
            "planner_dynamic_activation_enabled": False,
            "planner_activation_ttl_s": 0.0,
            "planner_hide_duplicate_ui_tools": True,
            "planner_show_embodiment_fallback_tools": False,
        },
        search={},
        orchestrator={"auto_approve_all": True},
        evolve={"candidate_dir": str(root / "evolve")},
        models={},
    )


class EmbodimentCodegenTests(unittest.TestCase):
    def test_embodiment_generated_code_may_use_subprocess_without_shell(self) -> None:
        orchestrator = Orchestrator()
        code = (
            "import subprocess\n"
            "result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)\n"
            "return {'status': 'OK', 'stdout': result.stdout}, {}\n"
        )

        error = orchestrator._validate_generated_tool_code(code, allow_local_subprocess=True)

        self.assertIsNone(error)

    def test_embodiment_generated_code_still_blocks_shell_true(self) -> None:
        orchestrator = Orchestrator()
        code = (
            "import subprocess\n"
            "subprocess.run('adb devices', capture_output=True, text=True, shell=True)\n"
            "return {'status': 'OK'}, {}\n"
        )

        error = orchestrator._validate_generated_tool_code(code, allow_local_subprocess=True)

        self.assertIn("shell=True", error or "")

    def test_non_embodiment_generated_code_still_blocks_subprocess(self) -> None:
        orchestrator = Orchestrator()
        code = (
            "import subprocess\n"
            "subprocess.run(['echo', 'x'], capture_output=True, text=True)\n"
            "return {'status': 'OK'}, {}\n"
        )

        error = orchestrator._validate_generated_tool_code(code, allow_local_subprocess=False)

        self.assertIn("Blocked import", error or "")

    def test_update_tool_registry_persists_embodiment_capability_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            config_dir = workspace_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (workspace_root / "src" / "tool_runtime").mkdir(parents=True, exist_ok=True)
            (workspace_root / "data").mkdir(parents=True, exist_ok=True)
            registry_path = config_dir / "tool_registry.json"
            registry_path.write_text(json.dumps({"tools": []}, ensure_ascii=True), encoding="utf-8")
            settings = _settings(tmpdir)

            with patch("orchestrator.main.load_settings", return_value=settings), patch(
                "orchestrator.tool_selector.load_settings",
                return_value=settings,
            ):
                orchestrator = Orchestrator()

            error = orchestrator._update_tool_registry(
                {
                    "tool_id": "embodiment.generated.phone_tap",
                    "name": "Generated Phone Tap",
                    "description": "Generated phone tap tool.",
                    "implements_capability_id": "phone.pointer.tap",
                    "implements_abstract_capability": "pointer.click",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "capabilities": ["write", "embodiment", "phone", "ui"],
                }
            )

            self.assertIsNone(error)
            stored = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(len(stored["tools"]), 1)
            self.assertEqual(stored["tools"][0]["tool_id"], "embodiment.generated.phone_tap")
            self.assertEqual(stored["tools"][0]["implements_capability_id"], "phone.pointer.tap")
            self.assertEqual(
                stored["tools"][0]["implements_abstract_capability"],
                "pointer.click",
            )

    def test_handle_create_tool_preserves_embodiment_capability_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            config_dir = workspace_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (workspace_root / "src" / "tool_runtime").mkdir(parents=True, exist_ok=True)
            (workspace_root / "data").mkdir(parents=True, exist_ok=True)
            registry_path = config_dir / "tool_registry.json"
            registry_path.write_text(json.dumps({"tools": []}, ensure_ascii=True), encoding="utf-8")
            settings = _settings(tmpdir)

            with patch("orchestrator.main.load_settings", return_value=settings), patch(
                "orchestrator.tool_selector.load_settings",
                return_value=settings,
            ):
                orchestrator = Orchestrator()

                async def _fake_generate_response(prompt, mode, context, trace_id, turn_id):
                    return json.dumps(
                        {
                            "type": "tool_codegen",
                            "name": "Generated Phone Tap",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                },
                                "required": ["x", "y"],
                                "additionalProperties": False,
                            },
                            "output_schema": {
                                "type": "object",
                                "properties": {"tapped": {"type": "boolean"}},
                                "required": ["tapped"],
                            },
                            "code": "return {'tapped': True}, {}",
                            "dependencies": [],
                        },
                        ensure_ascii=True,
                    )

                async def _fake_critic(**kwargs):
                    return {
                        "fulfilled": True,
                        "confidence": 1.0,
                        "issues": [],
                        "fix_instructions": [],
                    }

                orchestrator._generate_response = _fake_generate_response  # type: ignore[method-assign]
                orchestrator._evaluate_tool_codegen_critic = _fake_critic  # type: ignore[method-assign]
                orchestrator._log_mode_event = lambda *args, **kwargs: None  # type: ignore[method-assign]

                result = asyncio.run(
                    orchestrator._handle_create_tool(
                        {
                            "tool_id": "embodiment.generated.phone_tap",
                            "name": "Generated Phone Tap",
                            "description": "Generated phone tap tool.",
                            "implements_capability_id": "phone.pointer.tap",
                            "implements_abstract_capability": "pointer.click",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "integer"},
                                    "y": {"type": "integer"},
                                },
                                "required": ["x", "y"],
                                "additionalProperties": False,
                            },
                            "output_schema": {
                                "type": "object",
                                "properties": {"tapped": {"type": "boolean"}},
                                "required": ["tapped"],
                            },
                            "capabilities": ["write", "embodiment", "phone", "ui"],
                        },
                        "trace-1",
                    )
                )

            self.assertEqual(result["status"], "APPROVED")
            stored_registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_registry["tools"][0]["implements_capability_id"], "phone.pointer.tap")
            self.assertEqual(
                stored_registry["tools"][0]["implements_abstract_capability"],
                "pointer.click",
            )
            generated_specs = json.loads((workspace_root / "data" / "generated_tools.json").read_text(encoding="utf-8"))
            self.assertEqual(generated_specs[0]["implements_capability_id"], "phone.pointer.tap")
            self.assertEqual(
                generated_specs[0]["implements_abstract_capability"],
                "pointer.click",
            )

    def test_call_tool_fetches_tool_metadata_before_logging_and_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            config_dir = workspace_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (workspace_root / "src" / "tool_runtime").mkdir(parents=True, exist_ok=True)
            (workspace_root / "data").mkdir(parents=True, exist_ok=True)
            registry_path = config_dir / "tool_registry.json"
            registry_path.write_text(json.dumps({"tools": []}, ensure_ascii=True), encoding="utf-8")
            settings = _settings(tmpdir)

            class _FakeTools:
                def get_tool(self, tool_id: str):
                    return {
                        "tool_id": tool_id,
                        "name": "Embodiment Screen Capture",
                        "description": "Capture the screen.",
                        "capabilities": ["embodiment", "ui"],
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    }

            async def _fake_send_request(host, port, method, req, timeout):
                return {
                    "status": "APPROVED",
                    "result": {"image_ref": "workspace:/screenshot.png"},
                    "error": "",
                    "logs_ref": None,
                }

            with patch("orchestrator.main.load_settings", return_value=settings), patch(
                "orchestrator.tool_selector.load_settings",
                return_value=settings,
            ), patch("orchestrator.main.send_request", new=_fake_send_request):
                orchestrator = Orchestrator()
                orchestrator._tools = _FakeTools()  # type: ignore[assignment]

                result = asyncio.run(
                    orchestrator._call_tool(
                        "embodiment.screen_capture",
                        {"path": "screenshot.png"},
                        approval_token=None,
                        trace_id="trace-1",
                        required_artifacts=[],
                    )
                )

            self.assertEqual(result["status"], "APPROVED")
            self.assertEqual(result["result"]["image_ref"], "workspace:/screenshot.png")


if __name__ == "__main__":
    unittest.main()
