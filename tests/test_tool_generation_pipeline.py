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
        orchestrator={
            "auto_approve_all": True,
            "create_tool_max_attempts": 2,
            "tool_blueprint_max_attempts": 2,
        },
        evolve={"candidate_dir": str(root / "evolve")},
        models={},
    )


class ToolGenerationPipelineTests(unittest.TestCase):
    def _workspace(self):
        tmpdir = tempfile.TemporaryDirectory()
        workspace_root = Path(tmpdir.name)
        (workspace_root / "config").mkdir(parents=True, exist_ok=True)
        (workspace_root / "src" / "tool_runtime").mkdir(parents=True, exist_ok=True)
        (workspace_root / "data").mkdir(parents=True, exist_ok=True)
        (workspace_root / "config" / "tool_registry.json").write_text(
            json.dumps({"tools": []}, ensure_ascii=True),
            encoding="utf-8",
        )
        return tmpdir, workspace_root

    def test_generate_tools_from_spec_creates_namespaced_tools_and_persists_run(self) -> None:
        tmpdir, workspace_root = self._workspace()
        settings = _settings(tmpdir.name)
        try:
            with patch("orchestrator.main.load_settings", return_value=settings), patch(
                "orchestrator.tool_selector.load_settings",
                return_value=settings,
            ):
                orchestrator = Orchestrator()

                async def _fake_generate_response(prompt, mode, context, trace_id, turn_id):
                    if "TOOL BLUEPRINT PROMPT" in prompt:
                        return json.dumps(
                            {
                                "type": "tool_blueprint_plan",
                                "summary": "Create demo echo and counter tools.",
                                "tools": [
                                    {
                                        "tool_id": "echo",
                                        "name": "Echo",
                                        "description": "Return the provided text.",
                                        "capabilities": ["text", "read"],
                                        "input_schema": {
                                            "type": "object",
                                            "properties": {"text": {"type": "string"}},
                                            "required": ["text"],
                                            "additionalProperties": False,
                                        },
                                        "output_schema": {
                                            "type": "object",
                                            "properties": {"text": {"type": "string"}},
                                            "required": ["text"],
                                        },
                                    },
                                    {
                                        "tool_id": "counter",
                                        "name": "Counter",
                                        "description": "Count the number of items in an array.",
                                        "capabilities": ["data"],
                                        "input_schema": {
                                            "type": "object",
                                            "properties": {"items": {"type": "array"}},
                                            "required": ["items"],
                                            "additionalProperties": False,
                                        },
                                        "output_schema": {
                                            "type": "object",
                                            "properties": {"count": {"type": "integer"}},
                                            "required": ["count"],
                                        },
                                    },
                                ],
                            },
                            ensure_ascii=True,
                        )
                    if "tool_id: demo.echo" in prompt:
                        return json.dumps(
                            {
                                "type": "tool_codegen",
                                "name": "Demo Echo",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"],
                                    "additionalProperties": False,
                                },
                                "output_schema": {
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"],
                                },
                                "code": "text = str(args.get('text', ''))\nreturn {'text': text}, {}",
                                "dependencies": [],
                            },
                            ensure_ascii=True,
                        )
                    if "tool_id: demo.counter" in prompt:
                        return json.dumps(
                            {
                                "type": "tool_codegen",
                                "name": "Demo Counter",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {"items": {"type": "array"}},
                                    "required": ["items"],
                                    "additionalProperties": False,
                                },
                                "output_schema": {
                                    "type": "object",
                                    "properties": {"count": {"type": "integer"}},
                                    "required": ["count"],
                                },
                                "code": (
                                    "items = args.get('items', [])\n"
                                    "if not isinstance(items, list):\n"
                                    "    return {'status': 'ERROR', 'error': 'items must be a list'}, {}\n"
                                    "return {'count': len(items)}, {}"
                                ),
                                "dependencies": [],
                            },
                            ensure_ascii=True,
                        )
                    raise AssertionError(f"Unexpected prompt: {prompt}")

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
                    orchestrator.generate_tools_from_spec(
                        {
                            "request": "Create a demo namespace with echo and counter tools.",
                            "namespace_prefix": "demo",
                            "max_tools": 2,
                            "default_permissions": ["tier1"],
                        },
                        trace_id="trace-pipeline",
                    )
                )

            self.assertEqual(result["status"], "APPROVED")
            self.assertEqual(result["result"]["created_count"], 2)
            self.assertEqual(result["result"]["failed_count"], 0)
            created_ids = [item["tool_id"] for item in result["result"]["tools"] if item["status"] == "APPROVED"]
            self.assertEqual(created_ids, ["demo.echo", "demo.counter"])

            generated_specs = json.loads((workspace_root / "data" / "generated_tools.json").read_text(encoding="utf-8"))
            self.assertEqual([item["tool_id"] for item in generated_specs], ["demo.echo", "demo.counter"])

            stored_registry = json.loads((workspace_root / "config" / "tool_registry.json").read_text(encoding="utf-8"))
            self.assertEqual([item["tool_id"] for item in stored_registry["tools"]], ["demo.echo", "demo.counter"])

            run_path = Path(result["result"]["run_path"])
            self.assertTrue(run_path.exists())
            run_payload = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(run_payload["normalized_spec"]["namespace_prefix"], "demo")
            self.assertEqual(run_payload["summary"]["created_count"], 2)
        finally:
            tmpdir.cleanup()

    def test_handle_create_tool_duplicate_is_idempotent(self) -> None:
        tmpdir, workspace_root = self._workspace()
        settings = _settings(tmpdir.name)
        try:
            existing_tool = {
                "tool_id": "demo.echo",
                "name": "Demo Echo",
                "description": "Existing demo echo tool.",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "capabilities": ["read"],
                "required_permissions": ["tier1"],
            }
            (workspace_root / "config" / "tool_registry.json").write_text(
                json.dumps({"tools": [existing_tool]}, ensure_ascii=True),
                encoding="utf-8",
            )

            with patch("orchestrator.main.load_settings", return_value=settings), patch(
                "orchestrator.tool_selector.load_settings",
                return_value=settings,
            ):
                orchestrator = Orchestrator()
                result = asyncio.run(
                    orchestrator._handle_create_tool(
                        {
                            "tool_id": "demo.echo",
                            "name": "Demo Echo",
                            "description": "Existing demo echo tool.",
                        },
                        "trace-existing",
                    )
                )

            self.assertEqual(result["status"], "APPROVED")
            self.assertTrue(result["result"]["already_exists"])
            self.assertEqual(result["result"]["tool_id"], "demo.echo")
            self.assertFalse((workspace_root / "data" / "generated_tools.json").exists())
        finally:
            tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
