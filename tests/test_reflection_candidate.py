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

from common.soul_mirror import SoulConfig, sync_soul
from reflection_engine.candidate_preparer import (
    ReflectionImproveConfig,
    load_reflection_run,
    prepare_candidate,
)
from reflection_engine.dossier_builder import ReflectionConfig, build_reflection_run, write_reflection_run
from reflection_engine.observer import collect_runtime_observations, load_soul_snapshot


class ReflectionCandidateTests(unittest.TestCase):
    def test_prepare_candidate_builds_isolated_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "orchestrator").mkdir(parents=True)
            (repo_root / "src" / "tool_runtime").mkdir(parents=True)
            (repo_root / "config").mkdir()
            (repo_root / "data").mkdir()

            (repo_root / "src" / "orchestrator" / "main.py").write_text(
                "class Orchestrator:\n"
                "    def _handle_create_tool(self):\n"
                "        return None\n\n"
                "    def _evaluate_tool_codegen_critic(self):\n"
                "        return None\n\n"
                "    def _validate_generated_tool_code(self):\n"
                "        return None\n\n"
                "    def _apply_final_response_critic(self):\n"
                "        return None\n\n"
                "    def _run_cognition_loop(self):\n"
                "        return None\n",
                encoding="utf-8",
            )
            (repo_root / "src" / "tool_runtime" / "main.py").write_text(
                "def execute_tool() -> None:\n"
                "    return None\n",
                encoding="utf-8",
            )
            (repo_root / "src" / "tool_runtime" / "generated_tools.py").write_text(
                "GENERATED_TOOLS = {}\n",
                encoding="utf-8",
            )
            (repo_root / "config" / "tool_registry.json").write_text("{\"tools\": []}\n", encoding="utf-8")

            soul_config = SoulConfig(include=("src", "config"), exclude=(), output_dir="soul")
            sync_soul(repo_root, soul_config)

            (repo_root / "data" / "record.log").write_text(
                json.dumps(
                    {
                        "ts": "2026-03-10T10:00:00Z",
                        "event_type": "user_query",
                        "payload": {"trace_id": "trace-1", "text": "Create a video tool."},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "data" / "complex_events.log").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "ts": "2026-03-10T10:00:02Z",
                                "mode": "COMPLEX",
                                "event_type": "react_parsed",
                                "payload": {
                                    "trace_id": "trace-1",
                                    "parsed": {
                                        "type": "create_tool",
                                        "spec": {"tool_id": "video.analyse", "description": "Analyse video."},
                                    },
                                },
                            },
                            ensure_ascii=True,
                        ),
                        json.dumps(
                            {
                                "ts": "2026-03-10T10:00:03Z",
                                "mode": "COMPLEX",
                                "event_type": "react_step",
                                "payload": {
                                    "trace_id": "trace-1",
                                    "action": {"type": "create_tool"},
                                    "observation": {"status": "ERROR", "error": "Tool already exists"},
                                },
                            },
                            ensure_ascii=True,
                        ),
                        json.dumps(
                            {
                                "ts": "2026-03-10T10:00:04Z",
                                "mode": "COMPLEX",
                                "event_type": "final_critic_result",
                                "payload": {
                                    "trace_id": "trace-1",
                                    "fulfilled": False,
                                    "confidence": 0.0,
                                    "issues": ["Tool already exists", "No evidence of real execution"],
                                },
                            },
                            ensure_ascii=True,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "data" / "audit.log").write_text(
                json.dumps({"action": "memory_write"}, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )

            soul = load_soul_snapshot(repo_root / "soul")
            runtime = collect_runtime_observations(repo_root, repo_root / "data", max_record_events=50, max_complex_events=50)
            reflection_config = ReflectionConfig(
                output_dir="reflection/runs",
                max_dossiers=4,
                max_trace_failures=2,
                max_generated_capabilities=2,
                max_structural_hotspots=0,
            )
            run = build_reflection_run(
                repo_root,
                reflection_config,
                soul,
                runtime,
            )
            run_dir = write_reflection_run(repo_root, reflection_config, run)
            loaded_run = load_reflection_run(run_dir)
            dossier = loaded_run.dossiers[0]

            prepared = prepare_candidate(
                repo_root,
                ReflectionImproveConfig(
                    runs_dir="reflection/runs",
                    candidates_dir="reflection/candidates",
                    workspace_repo_dir="reflection/workspace/repo",
                    backend="copy",
                ),
                loaded_run,
                dossier,
            )
            self.assertEqual(prepared.manifest["dossier_kind"], "trace_failure")
            self.assertTrue((prepared.candidate_dir / "manifest.json").exists())
            self.assertTrue((prepared.candidate_dir / "proposal.json").exists())
            self.assertTrue((prepared.candidate_dir / "instructions.md").exists())
            self.assertTrue((prepared.candidate_dir / "context" / "files.json").exists())
            self.assertTrue((prepared.candidate_dir / "context" / "symbols.json").exists())
            self.assertTrue((prepared.candidate_dir / "context" / "evidence.json").exists())
            self.assertTrue((prepared.worktree_dir / "src" / "orchestrator" / "main.py").exists())
            self.assertTrue((prepared.worktree_dir / "data" / "complex_events.log").exists())
            self.assertIn("src/orchestrator/main.py", prepared.proposal["scope"]["target_files"])
            commands = [item["command"] for item in prepared.proposal["verification"]["commands"]]
            self.assertIn("python3 scripts/reflection_dossier.py", commands)
            self.assertTrue(any(command.startswith("python3 -m py_compile") for command in commands))

    def test_prepare_candidate_adapts_trace_failure_hypotheses_for_model_unavailability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "orchestrator").mkdir(parents=True)
            (repo_root / "src" / "model_server").mkdir(parents=True)
            (repo_root / "config").mkdir()
            (repo_root / "data").mkdir()

            (repo_root / "src" / "orchestrator" / "main.py").write_text(
                "class Orchestrator:\n"
                "    def _apply_final_response_critic(self):\n"
                "        return None\n\n"
                "    def _evaluate_final_response_critic(self):\n"
                "        return None\n",
                encoding="utf-8",
            )
            (repo_root / "src" / "model_server" / "__init__.py").write_text("", encoding="utf-8")
            (repo_root / "src" / "model_server" / "main.py").write_text(
                "class ModelService:\n"
                "    async def Generate(self, params):\n"
                "        return {\"status\": \"OK\"}\n",
                encoding="utf-8",
            )
            (repo_root / "src" / "model_server" / "text_model.py").write_text(
                "def generate(text, context):\n"
                "    return \"ok\"\n",
                encoding="utf-8",
            )

            soul_config = SoulConfig(include=("src", "config"), exclude=(), output_dir="soul")
            sync_soul(repo_root, soul_config)

            (repo_root / "data" / "record.log").write_text(
                json.dumps(
                    {
                        "ts": "2026-03-10T10:00:00Z",
                        "event_type": "user_query",
                        "payload": {"trace_id": "trace-offline", "text": "Answer my question."},
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "data" / "complex_events.log").write_text("", encoding="utf-8")
            (repo_root / "data" / "error.log").write_text(
                json.dumps(
                    {
                        "ts": "2026-03-10T10:00:01Z",
                        "event_type": "model_error",
                        "payload": {
                            "trace_id": "trace-offline",
                            "error_type": "OllamaError",
                            "error": "offline",
                            "traceback": (
                                "Traceback (most recent call last):\n"
                                f"  File \"{repo_root / 'src' / 'model_server' / 'text_model.py'}\", line 1, in generate\n"
                                "    raise RuntimeError('offline')\n"
                            ),
                        },
                    },
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "data" / "audit.log").write_text("", encoding="utf-8")

            soul = load_soul_snapshot(repo_root / "soul")
            runtime = collect_runtime_observations(repo_root, repo_root / "data", max_record_events=50, max_complex_events=50)
            reflection_config = ReflectionConfig(
                output_dir="reflection/runs",
                max_dossiers=2,
                max_trace_failures=2,
                max_generated_capabilities=0,
                max_structural_hotspots=0,
            )
            run = build_reflection_run(repo_root, reflection_config, soul, runtime)
            run_dir = write_reflection_run(repo_root, reflection_config, run)
            loaded_run = load_reflection_run(run_dir)

            prepared = prepare_candidate(
                repo_root,
                ReflectionImproveConfig(
                    runs_dir="reflection/runs",
                    candidates_dir="reflection/candidates",
                    workspace_repo_dir="reflection/workspace/repo",
                    backend="copy",
                ),
                loaded_run,
                loaded_run.dossiers[0],
            )

            hypotheses = [item["text"] for item in prepared.proposal["hypotheses"]]
            constraints = prepared.proposal["constraints"]
            self.assertTrue(any("offline model failures" in item for item in hypotheses))
            self.assertTrue(any("model unavailability" in item.lower() for item in constraints))
