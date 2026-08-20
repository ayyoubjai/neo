from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reflection_engine import run_autonomous_reflection
from scripts.evolve_improve import _summarize_candidate_effect


def _write_settings(path: Path, repo_root: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "workspace_root": str(repo_root),
                "data_dir": "./data",
                "rpc": {},
                "interface": {},
                "voice": {},
                "vision": {},
                "video": {},
                "telegram": {},
                "google": {},
                "tool": {"active_tool_limit": 100},
                "search": {},
                "orchestrator": {},
                "evolve": {"candidate_dir": "./evolve/candidates"},
                "models": {"text_model": "test-model", "text_enforce_json": False},
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


class AutonomousImprovementPipelineTests(unittest.TestCase):
    def test_run_autonomous_reflection_builds_focused_targets_from_model_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "orchestrator").mkdir(parents=True)
            (repo_root / "src" / "model_server").mkdir(parents=True)
            (repo_root / "config").mkdir()
            (repo_root / "data").mkdir()

            _write_settings(repo_root / "config" / "settings.json", repo_root)
            (repo_root / "config" / "reflection.json").write_text(
                json.dumps(
                    {
                        "output_dir": "reflection/runs",
                        "max_record_events": 50,
                        "max_complex_events": 50,
                        "max_error_events": 50,
                        "max_audit_events": 50,
                        "max_dossiers": 4,
                        "max_trace_failures": 4,
                        "max_generated_capabilities": 0,
                        "max_structural_hotspots": 0,
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            (repo_root / "config" / "soul.json").write_text(
                json.dumps(
                    {
                        "output_dir": "soul",
                        "include": ["config", "src"],
                        "exclude": [],
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

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
                "def helper():\n"
                "    return 'x'\n\n"
                "def generate(text, context):\n"
                "    return helper()\n",
                encoding="utf-8",
            )

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
                                f"  File \"{repo_root / 'src' / 'model_server' / 'text_model.py'}\", line 4, in generate\n"
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

            plan = run_autonomous_reflection(
                repo_root,
                repo_root / "config" / "reflection.json",
                repo_root / "config" / "soul.json",
                max_dossiers=2,
                max_targets_per_dossier=1,
                target_kinds=["trace_failure"],
            )

            self.assertEqual(len(plan.targets), 1)
            target = plan.targets[0]
            self.assertEqual(target.path, "src/model_server/text_model.py")
            self.assertGreater(target.start_line, 0)
            self.assertIn("model backend was unavailable", target.notes)
            self.assertEqual(target.metadata["reflection"]["selected_symbol"]["qualname"], "generate")

    def test_summarize_candidate_effect_rejects_noop_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            candidate_root = root / "candidate"
            (baseline_root / "src").mkdir(parents=True)
            (candidate_root / "src").mkdir(parents=True)
            (baseline_root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            (candidate_root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

            noop = _summarize_candidate_effect(
                str(candidate_root),
                str(baseline_root),
                {"edits": [{"path": "src/app.py", "before": "VALUE = 1", "after": "VALUE = 1"}]},
            )
            changed = _summarize_candidate_effect(
                str(candidate_root),
                str(baseline_root),
                {"edits": [{"path": "src/app.py", "before": "VALUE = 1", "after": "VALUE = 2"}]},
            )

            self.assertFalse(noop["effective_change"])
            self.assertEqual(noop["changed_file_count"], 0)

            (candidate_root / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
            changed = _summarize_candidate_effect(
                str(candidate_root),
                str(baseline_root),
                {"edits": [{"path": "src/app.py", "before": "VALUE = 1", "after": "VALUE = 2"}]},
            )
            self.assertTrue(changed["effective_change"])
            self.assertEqual(changed["changed_files"], ["src/app.py"])


if __name__ == "__main__":
    unittest.main()
