import json
from pathlib import Path
import subprocess
import sys

import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "src"))
import neo_code


def test_resume_path_preserves_callers_directory(tmp_path):
    argv = neo_code.build_agent_argv(["--resume=session.json"], tmp_path)
    assert "--resume=" + str(tmp_path / "session.json") in argv


def test_cli_saved_candidate_and_exit_status_from_other_project(tmp_path):
    wrapper = "neo_code.py"
    candidate = tmp_path / "review.json"
    candidate.write_text(json.dumps({"files": [{"path": "result.txt", "content": "reviewed"}]}))
    check = tmp_path / "check.py"
    check.write_text("from pathlib import Path\nassert Path('result.txt').read_text() == 'reviewed'\n")
    result = subprocess.run([sys.executable, str(SCRIPTS / wrapper), "--candidate", "review.json", "--apply",
                             "--test-command", f'"{sys.executable}" check.py'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "result.txt").read_text() == "reviewed"
    # Applying the same new-file candidate twice must fail without overwriting it.
    again = subprocess.run([sys.executable, str(SCRIPTS / wrapper), "--candidate", "review.json", "--apply"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 2
    assert "already exists" in again.stderr


def test_bash_launcher_identifies_neo_code(tmp_path):
    launcher = "neo-code"
    result = subprocess.run(["bash", str(SCRIPTS / launcher), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage: neo-code" in result.stdout
    assert "--resume" in result.stdout
    assert "--allow-exec" in result.stdout


class NeoCodeWrapperTests(unittest.TestCase):
    def test_build_agent_argv_injects_current_directory_as_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            argv = neo_code.build_agent_argv(["implement", "x"], cwd)

            self.assertEqual(argv[2], "--repo-root")
            self.assertEqual(Path(argv[3]), cwd.resolve())
            self.assertEqual(argv[4:], ["implement", "x"])

    def test_build_agent_argv_preserves_explicit_repo_root(self) -> None:
        argv = neo_code.build_agent_argv(["--repo-root", "/tmp/project", "implement", "x"], Path("/ignored"))

        self.assertEqual(argv[2:], ["--repo-root", "/tmp/project", "implement", "x"])

    def test_build_agent_argv_preserves_equals_style_repo_root(self) -> None:
        argv = neo_code.build_agent_argv(["--repo-root=/tmp/project", "implement", "x"], Path("/ignored"))

        self.assertEqual(argv[2:], ["--repo-root=/tmp/project", "implement", "x"])

    def test_relative_paths_are_resolved_before_changing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            argv = neo_code.build_agent_argv(
                ["--repo-root", "project", "--candidate", "review.json", "--apply"], cwd
            )
            self.assertEqual(argv[2:6], ["--repo-root", str(cwd / "project"),
                                         "--candidate", str(cwd / "review.json")])

    def test_equals_candidate_path_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            argv = neo_code.build_agent_argv(["--candidate=review.json"], cwd)
            self.assertIn("--candidate=" + str(cwd / "review.json"), argv)



def test_materialized_coding_bundle_can_launch_neo_code(tmp_path):
    from adaptive_agent.materializer import materialize_profile
    repo = SCRIPTS.parent
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"tools": [{"payload": {"bundles": ["coding_tools"]}}]}))
    target = tmp_path / "cell"
    materialize_profile(profile_path=profile, target_path=target,
                        bundle_path=repo / "config/adaptive_agent/bundles.json", source_path=repo)
    result = subprocess.run([sys.executable, str(target / "scripts/neo_code.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage: neo-code" in result.stdout
    assert (target / "src/coding_agent/session.py").is_file()
    assert (target / "config/tool_registry.json").is_file()
