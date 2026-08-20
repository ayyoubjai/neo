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
from reflection_engine.dossier_builder import ReflectionConfig, build_reflection_run, write_reflection_run
from reflection_engine.observer import collect_runtime_observations, load_soul_snapshot


class ReflectionEngineTests(unittest.TestCase):
    def test_builds_ranked_dossiers_from_soul_and_logs(self) -> None:
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
                "\n".join(
                    [
                        json.dumps(
                            {
                                "ts": "2026-03-10T10:00:00Z",
                                "event_type": "user_query",
                                "payload": {"trace_id": "trace-1", "text": "Create a video tool."},
                            },
                            ensure_ascii=True,
                        ),
                        json.dumps(
                            {
                                "ts": "2026-03-10T10:00:05Z",
                                "event_type": "assistant_final",
                                "payload": {"trace_id": "trace-1", "text": "Tool is ready."},
                            },
                            ensure_ascii=True,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (repo_root / "data" / "complex_events.log").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "ts": "2026-03-10T10:00:00Z",
                                "mode": "COMPLEX",
                                "event_type": "turn_start",
                                "payload": {"trace_id": "trace-1", "text": "Create a video tool."},
                            },
                            ensure_ascii=True,
                        ),
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
                                        "reason": "Missing capability.",
                                        "thought": "Create a video capability.",
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
                                    "fix_instructions": ["Handle duplicate create_tool requests"],
                                    "retry_count": 1,
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
            config = ReflectionConfig(
                output_dir="reflection/runs",
                max_dossiers=5,
                max_trace_failures=2,
                max_generated_capabilities=2,
                max_structural_hotspots=1,
                hotspot_score_threshold=0.0,
                hotspot_size_bytes_high=10,
                hotspot_symbol_count_high=2,
                hotspot_dependency_count_high=1,
                hotspot_inbound_count_high=1,
            )
            run = build_reflection_run(repo_root, config, soul, runtime)

            self.assertEqual(run.manifest["dossier_count"], 3)
            kinds = [item["kind"] for item in run.dossiers]
            self.assertEqual(kinds, ["trace_failure", "generated_capability_review", "structural_hotspot"])
            self.assertIn("video.analyse", run.dossiers[0]["title"])
            self.assertIn("src/orchestrator/main.py", run.dossiers[0]["target_files"])
            self.assertIn("config/tool_registry.json", run.dossiers[1]["target_files"])
            self.assertTrue(any(symbol["qualname"] == "Orchestrator._handle_create_tool" for symbol in run.dossiers[1]["target_symbols"]))

            run_dir = write_reflection_run(repo_root, config, run)
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "observations.json").exists())
            self.assertTrue((run_dir / "dossiers.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())

    def test_trace_failure_classifies_model_unavailable_errors(self) -> None:
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
            config = ReflectionConfig(
                output_dir="reflection/runs",
                max_dossiers=3,
                max_trace_failures=3,
                max_generated_capabilities=0,
                max_structural_hotspots=0,
            )
            run = build_reflection_run(repo_root, config, soul, runtime)

            dossier = run.dossiers[0]
            self.assertEqual(dossier["problem_statement"], "model backend was unavailable during a user trace")
            self.assertIn("src/model_server/text_model.py", dossier["target_files"])
            self.assertTrue(
                any(symbol["qualname"] == "generate" for symbol in dossier["target_symbols"]),
                dossier["target_symbols"],
            )
            self.assertTrue(any("model-unavailable response" in item for item in dossier["suggested_actions"]))


if __name__ == "__main__":
    unittest.main()
