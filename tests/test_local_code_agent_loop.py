"""Coding sessions use fake generation; commands and files exercise the real loop."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_code_agent as agent


def options(**kwargs):
    values = dict(max_steps=5, file=[], max_files=8, apply=True,
                  test_command="", allow_model_tests=False, test_timeout_s=5)
    values.update(kwargs)
    return argparse.Namespace(**values)


def edit(before, after, path="main.py"):
    return {"edits": [{"path": path, "before": before, "after": after}]}


def test_inspect_patch_fail_repair_uses_actual_observations(tmp_path):
    (tmp_path / "main.py").write_text("value = 0\n")
    (tmp_path / "check.py").write_text("from main import value\nassert value == 200, value\n")
    candidates = iter([
        {"searches": ["value"]},
        {"reads": [{"path": "main.py", "start_line": 1, "end_line": 10}]},
        edit("value = 0", "value = 1"),
        edit("value = 1", "value = 200"),
        {"status": "complete", "plan": [{"id": "task", "description": "fix value", "status": "completed"}]},
    ])
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return next(candidates)

    command = f'{shlex.quote(sys.executable)} -B check.py'
    result = agent._run_agent_loop(tmp_path, tmp_path / "soul", "fix value", options(test_command=command), generate)
    assert result == 0
    assert "main.py:1: value = 0" in prompts[1]
    assert "1: value = 0" in prompts[2]
    assert "AssertionError: 1" in prompts[3]
    assert "already on disk" in prompts[3]
    assert (tmp_path / "main.py").read_text() == "value = 200\n"
    assert len(list((tmp_path / "data/neo_code").glob("*.json"))) == 5


def test_rejected_patch_is_fed_back_without_partial_changes(tmp_path):
    (tmp_path / "main.py").write_text("original")
    calls = []

    def generate(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return {"edits": [edit("original", "changed")["edits"][0],
                              edit("not here", "replacement")["edits"][0]]}
        assert (tmp_path / "main.py").read_text() == "original"
        assert "Patch rejected" in prompt
        return edit("original", "fixed")

    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "fix", options(apply=False), generate) == 0
    assert (tmp_path / "main.py").read_text() == "original"


def test_new_file_never_overwrites_existing(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("user work")
    with pytest.raises(RuntimeError, match="already exists"):
        agent._apply_candidate(tmp_path, {"files": [{"path": "main.py", "content": "lost"}]})
    assert target.read_text() == "user work"


def test_write_failure_rolls_back_existing_files_and_new_directories(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("original")
    original_write = Path.write_bytes

    def fail_one(path, content):
        if path.name == "fail.py":
            raise OSError("simulated write failure")
        return original_write(path, content)

    with patch.object(Path, "write_bytes", fail_one), pytest.raises(OSError):
        agent._apply_candidate(tmp_path, {
            **edit("original", "changed"),
            "files": [{"path": "new/ok.py", "content": "ok"},
                      {"path": "new/fail.py", "content": "fail"}],
        })
    assert target.read_text() == "original"
    assert not (tmp_path / "new").exists()


def test_same_file_edits_are_staged_in_sequence_and_preserve_crlf(tmp_path):
    target = tmp_path / "main.py"
    target.write_bytes(b"a\r\nb\r\n")
    agent._apply_candidate(tmp_path, {"edits": [edit("a", "first")["edits"][0],
                                              edit("first", "last")["edits"][0]]})
    assert target.read_bytes() == b"last\r\nb\r\n"


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", "C:\\outside", ".git/config"])
def test_patch_and_read_paths_are_contained(tmp_path, path):
    with pytest.raises(RuntimeError):
        agent._resolve_repo_path(tmp_path, path)


def test_symlink_escape_is_blocked(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(RuntimeError):
        agent._resolve_repo_path(tmp_path, "link/outside")


@pytest.mark.parametrize("candidate", [[], {"edits": ["bad"]}, {"tests": "pytest"},
                                       {"reads": [{}], **edit("a", "b")}])
def test_malformed_candidates_are_rejected(candidate):
    with pytest.raises(ValueError):
        agent._normalize_candidate(candidate)


def test_discovery_includes_root_sources_and_respects_gitignore(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "main.go").write_text("package main")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / ".env").write_text("example secret")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored/skip.py").write_text("skip")
    include = agent._load_soul_config(tmp_path, "soul").include
    assert "main.go" in include
    assert "README.md" in include
    assert ".env" not in include
    assert "ignored/skip.py" not in include


def test_model_commands_require_opt_in(tmp_path):
    candidate = agent._normalize_candidate({"tests": ["model command"]})
    with patch.object(agent, "_run_command", return_value=(0, "passed")) as run:
        assert agent._verify_candidate(tmp_path, candidate, options())[0] is None
        run.assert_not_called()
        assert agent._verify_candidate(tmp_path, candidate, options(test_command="user command"))[0] == 0
        assert run.call_args.args[1] == "user command"
        assert agent._verify_candidate(tmp_path, candidate, options(allow_model_tests=True))[0] == 0
        assert run.call_args.args[1] == "model command"


def test_step_budget_and_noop_do_not_claim_success(tmp_path):
    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "fix", options(max_steps=2),
                                 lambda _: {"searches": ["missing"]}) == 2
    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "fix", options(), lambda _: {}) == 2


def test_unverified_changes_are_reported_as_incomplete(tmp_path):
    result = agent._run_agent_loop(tmp_path, tmp_path / "soul", "fix", options(),
                                  lambda _: {"files": [{"path": "main.py", "content": "created"}]})
    assert result == 2
    assert (tmp_path / "main.py").read_text() == "created"


def test_saved_candidate_applies_without_generation_or_mirror_sync(tmp_path):
    artifact = tmp_path / "candidate.json"
    artifact.write_text(json.dumps({"candidate": {"files": [{"path": "main.py", "content": "reviewed"}]}}))
    with patch.object(sys, "argv", ["neo-code", "--repo-root", str(tmp_path), "--candidate", str(artifact),
                                    "--apply", "--test-command", "approved check"]), \
         patch.object(agent, "generate_local_text") as generate, \
         patch.object(agent, "sync_soul") as sync, \
         patch.object(agent, "_run_command", return_value=(0, "passed")):
        assert agent.main() == 0
        generate.assert_not_called()
        sync.assert_not_called()
    assert (tmp_path / "main.py").read_text() == "reviewed"


def test_cognition_delegates_execution_to_coding_loop():
    with patch.object(agent, "CognitionClient") as client:
        client.return_value.generate_json = AsyncMock(return_value={"summary": "ready"})
        asyncio.run(agent._generate_candidate_cognition_async("prompt", timeout_s=20, permission_policy="deny"))
        client.return_value.generate_json.assert_awaited_once_with("prompt", timeout=20, max_actions=0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group handling")
def test_command_timeout_returns_feedback(tmp_path):
    command = f'{shlex.quote(sys.executable)} -c "import time; time.sleep(20)"'
    code, output = agent._run_command(tmp_path, command, 0.1)
    assert code == 124
    assert "timed out" in output


def test_failed_existing_file_write_still_restores_earlier_edits(tmp_path):
    (tmp_path / "main.py").write_text("original")
    (tmp_path / "denied.py").write_text("original")
    original_write = Path.write_bytes

    def deny_one(path, content):
        if path.name == "denied.py":
            raise PermissionError("read-only file")
        return original_write(path, content)

    with patch.object(Path, "write_bytes", deny_one), pytest.raises(PermissionError):
        agent._apply_candidate(tmp_path, {"edits": [edit("original", "changed")["edits"][0],
                                                   edit("original", "changed", "denied.py")["edits"][0]]})
    assert (tmp_path / "main.py").read_text() == "original"
    assert (tmp_path / "denied.py").read_text() == "original"


def test_invalid_json_shape_is_repaired_with_feedback(tmp_path):
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return ["invalid"]
        assert "candidate must be a JSON object" in prompt
        return {"files": [{"path": "new.py", "content": "valid"}]}

    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "create file", options(apply=False), generate) == 0
    assert not (tmp_path / "new.py").exists()


def test_incomplete_rollback_stops_the_loop(tmp_path):
    (tmp_path / "main.py").write_text("original")
    with patch.object(agent, "_apply_candidate", side_effect=agent.PatchRollbackError("rollback incomplete")), \
         pytest.raises(agent.PatchRollbackError):
        agent._run_agent_loop(tmp_path, tmp_path / "soul", "fix", options(), lambda _: edit("original", "changed"))


def test_git_metadata_alias_is_blocked(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / ".git", target_is_directory=True)
    with pytest.raises(RuntimeError, match="Git metadata"):
        agent._resolve_repo_path(tmp_path, "alias/config")
