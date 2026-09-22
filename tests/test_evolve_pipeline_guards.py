from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model_server.local_generation import generate_local_text
from scripts.evolve_improve import (
    Target,
    _assert_evolution_baseline_immutable,
    _commit_prepared_evolution_baseline,
    _run_required_preflight,
    _target_evidence_key,
)
from scripts.evolve_test_runner import _discover_tests, _normalize_target, _run_test_file


class LocalGenerationProviderTests(unittest.TestCase):
    @mock.patch("model_server.local_generation.llamacpp_generate", return_value='{"ok": true}')
    def test_llamacpp_dispatch_preserves_generation_controls(self, generate: mock.Mock) -> None:
        output = generate_local_text(
            "prompt",
            "model.gguf",
            provider="llamacpp",
            temperature=0.25,
            max_new_tokens=321,
            system="json only",
            response_format="json",
        )

        self.assertEqual(output, '{"ok": true}')
        generate.assert_called_once_with(
            "prompt",
            "model.gguf",
            system="json only",
            options={"temperature": 0.25, "num_predict": 321},
            response_format="json",
        )

    def test_unknown_provider_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported local model provider"):
            generate_local_text("prompt", "model", provider="unknown")


class FocusedTestEvidenceTests(unittest.TestCase):
    def test_normalization_preserves_traversal_and_hidden_paths(self) -> None:
        self.assertEqual(_normalize_target("../src/app.py"), "../src/app.py")
        self.assertEqual(_normalize_target(".hidden/app.py"), ".hidden/app.py")
        self.assertEqual(_normalize_target("/src/app.py"), "/src/app.py")
        self.assertEqual(_normalize_target(".\\src\\app.py"), "src/app.py")

    def test_cli_rejects_traversal_instead_of_testing_another_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts/evolve_test_runner.py"),
                 "--repo-root", str(root), "--target", "../src/app.py"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(json.loads(proc.stdout)["details"], "target escapes repository root")

    def test_missing_execution_cannot_pass_evidence_gate(self) -> None:
        cases = {
            "empty": "import unittest\n",
            "pytest_only": "def test_failure():\n    assert False\n",
            "skipped": (
                "import unittest\nclass Tests(unittest.TestCase):\n"
                "    @unittest.skip('unavailable')\n"
                "    def test_failure(self):\n        self.fail()\n"
            ),
            "expected_failure": (
                "import unittest\nclass Tests(unittest.TestCase):\n"
                "    @unittest.expectedFailure\n"
                "    def test_failure(self):\n        self.fail()\n"
            ),
            "early_exit": "import os\nos._exit(0)\n",
        }
        for name, content in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                test = root / "test_evidence.py"
                test.write_text(content, encoding="utf-8")
                result = _run_test_file(root, test, timeout_s=10)
                self.assertFalse(result["passed"], result)
                self.assertIn("error", result)

    def test_executed_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test = root / "test_failure.py"
            test.write_text(
                "import unittest\nclass Tests(unittest.TestCase):\n"
                "    def test_failure(self):\n        self.fail('regression')\n",
                encoding="utf-8",
            )
            result = _run_test_file(root, test, timeout_s=10)
            self.assertFalse(result["passed"], result)
            self.assertEqual(result["executed_tests"], 1)
            self.assertIn("regression", result["stderr"])

    def test_passing_test_with_skipped_and_expected_failures_is_valid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test = root / "test_mixed.py"
            test.write_text(
                "import unittest\nclass Tests(unittest.TestCase):\n"
                "    def test_pass(self):\n        self.assertEqual(2 + 2, 4)\n"
                "    @unittest.skip('optional dependency')\n"
                "    def test_skip(self):\n        self.fail()\n"
                "    @unittest.expectedFailure\n"
                "    def test_expected(self):\n        self.fail()\n",
                encoding="utf-8",
            )
            result = _run_test_file(root, test, timeout_s=10)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["tests_run"], 3)
            self.assertEqual(result["executed_tests"], 1)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(result["expected_failures"], 1)

    def test_timeout_preserves_subprocess_byte_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("scripts.evolve_test_runner.subprocess.run", side_effect=
                            subprocess.TimeoutExpired("tests", 1, output=b"progress", stderr=b"diagnostic")):
                result = _run_test_file(root, root / "test_slow.py", timeout_s=1)
            self.assertFalse(result["passed"])
            self.assertEqual(result["stdout"], "progress")
            self.assertEqual(result["stderr"], "diagnostic")

    def test_discovers_and_runs_only_target_related_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "widget").mkdir(parents=True)
            (root / "tests").mkdir()
            (root / "src" / "widget" / "service.py").write_text(
                "def add(left, right):\n    return left + right\n", encoding="utf-8"
            )
            (root / "tests" / "test_service.py").write_text(
                "import unittest\n"
                "from widget.service import add\n\n"
                "class ServiceTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_unrelated.py").write_text(
                "import unittest\n\nclass OtherTests(unittest.TestCase):\n    pass\n",
                encoding="utf-8",
            )

            selected = _discover_tests(root, "src/widget/service.py", max_files=8)

            self.assertEqual([path.name for path in selected], ["test_service.py"])
            result = _run_test_file(root, selected[0], timeout_s=10)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["executed_tests"], 1)

    def test_missing_focused_tests_is_explicitly_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "orphan.py").write_text("VALUE = 1\n", encoding="utf-8")

            self.assertEqual(_discover_tests(root, "src/orphan.py", max_files=8), [])


class EvolutionBaselineGuardTests(unittest.TestCase):
    def _init_repo(self, root: Path) -> None:
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "baseline"], cwd=root, check=True, capture_output=True)

    def test_preparation_is_committed_into_candidate_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            target = root / "app.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            self._init_repo(root)
            target.write_text("# prepared instrumentation\nVALUE = 1\n", encoding="utf-8")

            baseline_ref = _commit_prepared_evolution_baseline(str(root), "test-run")
            _assert_evolution_baseline_immutable(str(root), baseline_ref)

            committed = subprocess.run(
                ["git", "show", f"{baseline_ref}:app.py"], cwd=root, text=True, capture_output=True, check=True
            ).stdout
            self.assertIn("prepared instrumentation", committed)

            target.write_text(committed + "DIRTY = True\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "baseline changed"):
                _assert_evolution_baseline_immutable(str(root), baseline_ref)

    def test_required_neutral_evidence_fails_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            snippet = "import json; print(json.dumps(dict(score=0.5, status='neutral_fallback')))"
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(snippet)}"
            error = _run_required_preflight(
                metrics_cfg=[
                    {
                        "name": "behavioral",
                        "command": command,
                        "required": True,
                        "parse_json": True,
                        "timeout_s": 10,
                    }
                ],
                target=Target(path="src/app.py"),
                default_timeout_s=10,
                repo_root=str(root),
            )

            self.assertIn("required evidence unavailable", error or "")

    def test_target_evidence_keys_include_line_identity(self) -> None:
        first = _target_evidence_key(Target(path="src/app.py", start_line=10, end_line=20))
        second = _target_evidence_key(Target(path="src/app.py", start_line=30, end_line=40))
        self.assertNotEqual(first, second)
        self.assertNotIn("/", first)

    def test_micro_prepare_validates_evidence_before_baseline_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            scenario_dir = root / "evolve" / "scenarios" / "llm"
            scenario_dir.mkdir(parents=True)
            (root / "src" / "sample.py").write_text("def echo(value):\n    return value\n", encoding="utf-8")
            scenarios = {
                "scenarios": [
                    {"object_id": "sample.echo", "call": {"args": [value], "kwargs": {}}}
                    for value in (1, 2, 3)
                ]
            }
            (scenario_dir / "sample.json").write_text(
                json.dumps(scenarios), encoding="utf-8"
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "evolve_score.py"),
                    "--prepare-only",
                    "--mode",
                    "micro",
                    "--macro-policy",
                    "always",
                    "--target",
                    "src/sample.py",
                    "--target-start-line",
                    "1",
                    "--target-end-line",
                    "2",
                    "--repo-root",
                    str(root),
                    "--run-dir",
                    str(root),
                    "--micro-scenarios-file",
                    "evolve/scenarios/llm/sample.json",
                    "--min-cases",
                    "3",
                    "--missing-source-policy",
                    "fail",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["status"], "prepare_ok")
            self.assertEqual(payload["micro_cases"], 3)


if __name__ == "__main__":
    unittest.main()
