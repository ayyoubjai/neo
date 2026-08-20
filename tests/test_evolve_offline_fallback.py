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

from scripts.evolve_generate_llm import _build_local_fallback_result
from scripts.evolve_improve import (
    _apply_promoted_candidates_to_repo,
    _candidate_signature,
    _collect_generation_candidates,
    _copy_repo_snapshot,
)
from scripts.evolve_probe_llm import _build_local_probe_output


class EvolveOfflineFallbackTests(unittest.TestCase):
    def test_local_generator_builds_model_unavailable_candidate(self) -> None:
        payload = {
            "target": {
                "path": "src/model_server/text_model.py",
                "content": (
                    "def generate(text, context):\n"
                    "    fallback = (\n"
                    '        "Here is a generic answer even though generation failed. "\n'
                    '        "Please keep going as normal."\n'
                    "    )\n"
                    "    return fallback\n"
                ),
                "notes": "Problem: model backend was unavailable during a user trace",
            },
            "reflection": {
                "problem_statement": "model backend was unavailable during a user trace",
                "suggested_actions": [
                    "Return a clear model-unavailable response instead of optimistic generic fallback text when the backend is offline."
                ],
            },
        }

        result = _build_local_fallback_result(
            payload,
            max_candidates=2,
            provider="ollama",
            model_id="test-model",
            reason="offline",
        )

        self.assertEqual(result["generator"]["mode"], "local_fallback")
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["edits"][0]["path"], "src/model_server/text_model.py")
        self.assertIn("cannot give a reliable answer right now", candidate["edits"][0]["after"])

    def test_local_generator_builds_final_critic_candidate(self) -> None:
        payload = {
            "target": {
                "path": "src/orchestrator/main.py",
                "content": (
                    "async def _evaluate_final_response_critic(self):\n"
                    "    parsed_critic = None\n"
                    "    if parsed_critic:\n"
                    "        return parsed_critic\n"
                    "    return {\n"
                    '        "fulfilled": True,\n'
                    '        "confidence": 1.0,\n'
                    '        "issues": [],\n'
                    '        "fix_instructions": [],\n'
                    "    }\n"
                ),
                "notes": "Issue: final response critic unavailable or parse failed",
            },
            "reflection": {
                "problem_statement": "final response critic unavailable or parse failed",
            },
        }

        result = _build_local_fallback_result(
            payload,
            max_candidates=2,
            provider="ollama",
            model_id="test-model",
            reason="offline",
        )

        self.assertEqual(len(result["candidates"]), 1)
        edit = result["candidates"][0]["edits"][0]
        self.assertIn('"fulfilled": False', edit["after"])
        self.assertIn("degraded execution state", edit["after"])

    def test_local_generator_builds_model_service_status_candidate(self) -> None:
        payload = {
            "target": {
                "path": "src/model_server/main.py",
                "content": (
                    "    async def Generate(self, params):\n"
                    "        output = generate(text, ctx)\n"
                    '        return {"status": "OK", "text": output}\n'
                ),
                "notes": "Problem: model backend was unavailable during a user trace",
            },
            "reflection": {
                "problem_statement": "model backend was unavailable during a user trace",
            },
        }

        result = _build_local_fallback_result(
            payload,
            max_candidates=2,
            provider="ollama",
            model_id="test-model",
            reason="offline",
        )

        self.assertEqual(len(result["candidates"]), 1)
        edit = result["candidates"][0]["edits"][0]
        self.assertIn('status = "DEGRADED"', edit["after"])
        self.assertIn('return {"status": status, "text": output}', edit["after"])

    def test_collect_generation_candidates_falls_back_to_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inbox_dir = Path(tmp) / "inbox"
            inbox_dir.mkdir()
            (inbox_dir / "candidate.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "id": "inbox_1",
                                "summary": "Inbox candidate",
                                "edits": [{"path": "src/app.py", "before": "a", "after": "b"}],
                            }
                        ]
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            candidates, meta = _collect_generation_candidates(
                generator_cmd='python3 -c "import sys; sys.stderr.write(\'boom\'); sys.exit(1)"',
                payload={"target": {"path": "src/app.py", "content": "a"}},
                timeout_s=10,
                inbox_dir=str(inbox_dir),
                fallback_cfg={"fallback_to_inbox_on_error": True, "fallback_to_inbox_on_empty": True},
            )

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["id"], "inbox_1")
            self.assertEqual(meta["mode"], "inbox_fallback")
            self.assertIn("generator failed", meta["reason"])

    def test_local_probe_output_is_empty_and_explicit(self) -> None:
        result = _build_local_probe_output("offline")
        self.assertEqual(result["symbols"], [])
        self.assertEqual(result["files"], [])
        self.assertIn("backend unavailable", result["notes"])

    def test_copy_repo_snapshot_respects_runtime_artifact_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_root = root / "repo"
            dst_root = root / "snapshot"
            (src_root / "src").mkdir(parents=True)
            (src_root / "node_modules" / "@types" / "node").mkdir(parents=True)
            (src_root / "soul" / "meta").mkdir(parents=True)
            (src_root / "reflection" / "runs").mkdir(parents=True)
            (src_root / "screenshots").mkdir(parents=True)

            (src_root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            (src_root / "node_modules" / "@types" / "node" / "test.d.ts").write_text("type T = string;\n", encoding="utf-8")
            (src_root / "soul" / "meta" / "manifest.json").write_text("{}", encoding="utf-8")
            (src_root / "reflection" / "runs" / "summary.md").write_text("# summary\n", encoding="utf-8")
            (src_root / "screenshots" / "a.png").write_text("x", encoding="utf-8")

            _copy_repo_snapshot(
                str(src_root),
                str(dst_root),
                ["node_modules", "soul", "reflection", "screenshots"],
            )

            self.assertTrue((dst_root / "src" / "app.py").exists())
            self.assertFalse((dst_root / "node_modules").exists())
            self.assertFalse((dst_root / "soul").exists())
            self.assertFalse((dst_root / "reflection").exists())
            self.assertFalse((dst_root / "screenshots").exists())

    def test_candidate_signature_ignores_generation_specific_ids(self) -> None:
        first = {
            "id": "cand_a",
            "summary": "same patch",
            "edits": [{"path": "src/app.py", "before": "VALUE = 1", "after": "VALUE = 2"}],
        }
        second = {
            "id": "cand_b",
            "summary": "same patch but different id",
            "edits": [{"path": "src/app.py", "before": "VALUE = 1", "after": "VALUE = 2"}],
        }

        self.assertEqual(_candidate_signature(first), _candidate_signature(second))

    def test_apply_promoted_candidates_to_repo_applies_and_skips_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

            first = {
                "run_candidate_id": "g1_first",
                "score": 0.9,
                "candidate_payload": {
                    "edits": [{"path": "src/app.py", "before": "VALUE = 1", "after": "VALUE = 2"}]
                },
            }
            second = {
                "run_candidate_id": "g1_second",
                "score": 0.8,
                "candidate_payload": {
                    "edits": [{"path": "src/app.py", "before": "VALUE = 1", "after": "VALUE = 3"}]
                },
            }

            results = _apply_promoted_candidates_to_repo(str(root), [second, first])

            self.assertEqual(results[0]["run_candidate_id"], "g1_first")
            self.assertEqual(results[0]["status"], "applied")
            self.assertEqual(results[1]["run_candidate_id"], "g1_second")
            self.assertEqual(results[1]["status"], "skipped_conflict")
            self.assertEqual((root / "src" / "app.py").read_text(encoding="utf-8"), "VALUE = 2\n")


if __name__ == "__main__":
    unittest.main()
