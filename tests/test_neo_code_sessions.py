import argparse
import json
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_code_agent as agent
from coding_agent.session import SessionStore


def options(**overrides):
    values = dict(agent.SESSION_DEFAULTS, max_steps=8, resume=None)
    values.update(apply=True, model="test-model", provider="ollama")
    values.update(overrides)
    return argparse.Namespace(**values)


def milestone(name, status="completed"):
    return {"id": name, "description": name, "status": status}


def create(name, content, **extra):
    return {"files": [{"path": name, "content": content}], **extra}


def completion(*names):
    return {"status": "complete", "plan": [milestone(name) for name in names or ("task",)]}


def session_path(root):
    return next((root / "data/neo_code/sessions").glob("*.json"))


def command():
    return f'{shlex.quote(sys.executable)} -B check.py'


def test_two_successful_stages_continue_until_explicit_completion(tmp_path):
    (tmp_path / "check.py").write_text("from pathlib import Path\nassert Path('one.txt').read_text() == 'one'\n")
    candidates = iter([
        create("one.txt", "one", plan=[milestone("one"), milestone("two", "pending")]),
        create("two.txt", "two", plan=[milestone("one"), milestone("two")]),
        completion("one", "two"),
    ])
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return next(candidates)

    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "create two files", options(test_command=command()), generate) == 0
    assert "Verification passed" in prompts[1]
    assert (tmp_path / "two.txt").read_text() == "two"
    state = SessionStore.read(session_path(tmp_path))
    assert state["status"] == "complete"
    assert state["steps"] == 3
    assert state["changed"] == ["one.txt", "two.txt"]
    assert state["verification"]["revision"] == 2


def test_resume_retains_plan_history_and_rechecks_before_completion(tmp_path):
    (tmp_path / "check.py").write_text("assert True\n")
    args = options(max_steps=1, test_command=command())
    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "build", args,
                                lambda _: create("one.txt", "one", plan=[milestone("one")], notes="Use plain text")) == 2
    saved = session_path(tmp_path)
    candidates = iter([completion("one"), {"verify": True}, completion("one")])
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return next(candidates)

    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "build",
                                options(resume=str(saved), test_command=command()), generate) == 0
    assert "Use plain text" in prompts[0]
    assert '"id": "one"' in prompts[0]
    assert "Applied patch to: one.txt" in prompts[0]
    assert "Completion rejected" in prompts[1]
    state = SessionStore.read(saved)
    assert state["steps"] == 4
    assert state["revision"] == 1  # No replay of the first patch.


def test_main_resume_preserves_permissions_and_cognition_mode(tmp_path):
    with SessionStore(tmp_path, "build") as session:
        session.state["options"] = dict(agent.SESSION_DEFAULTS, mode="cognition", apply=True,
                                        allow_exec=True, allow_network=True, test_command="original checks")
        saved = session.path
        session.save()
    seen = []

    def run(root, soul, task, args, generate):
        assert args.mode == "cognition"
        assert args.apply and args.allow_exec
        assert not args.allow_network  # Explicit override wins.
        assert args.test_command == "original checks"
        assert task == "build"
        seen.append(generate("context"))
        return 2

    with patch.object(sys, "argv", ["neo-code", "--repo-root", str(tmp_path), "--resume", str(saved),
                                    "--no-allow-network", "--no-sync-soul"]), \
         patch.object(agent, "_run_agent_loop", side_effect=run), \
         patch.object(agent, "_generate_candidate_cognition", return_value={"status": "blocked"}) as cognition:
        assert agent.main() == 2
        cognition.assert_called_once()
    assert seen == [{"status": "blocked"}]


def test_external_changes_invalidate_successful_checks(tmp_path):
    (tmp_path / "check.py").write_text("assert True\n")
    calls = 0
    def generate(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            return create("one.txt", "one")
        if calls == 2:
            (tmp_path / "one.txt").write_text("external change")
            return completion()
        if calls == 3:
            assert "repository changed since checks passed" in prompt
            return {"verify": True}
        return completion()
    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "build", options(test_command=command()), generate) == 0
    assert calls == 4


def test_completion_without_checks_never_succeeds(tmp_path):
    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "build", options(max_steps=2), lambda _: completion()) == 2
    state = SessionStore.read(session_path(tmp_path))
    assert not state["verification"]
    assert state["status"] == "budget_exhausted"


def test_unfinished_milestone_blocks_completion(tmp_path):
    candidates = iter([{"plan": [milestone("pending", "pending")]}, {"status": "complete"}])
    assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "build", options(max_steps=2), lambda _: next(candidates)) == 2
    state = SessionStore.read(session_path(tmp_path))
    assert any("unfinished milestones" in entry["text"] for entry in state["history"])


def test_interrupted_actions_are_not_replayed_and_repo_mismatch_rejected(tmp_path):
    with SessionStore(tmp_path, "build") as session:
        session.begin_action({"tools": [{"tool_id": "proc.exec", "args": {"command": ["do-not-replay"]}}]}, mutating=True)
        saved = session.path
    with SessionStore(tmp_path, "", str(saved)) as resumed:
        assert resumed.state["pending_action"] is None
        assert any("not replayed" in entry["text"] for entry in resumed.state["history"])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="different repository"):
        with SessionStore(other, "", str(saved)):
            pass


def test_checkpoint_is_exclusive_and_cannot_drop_milestones(tmp_path):
    with SessionStore(tmp_path, "build") as session:
        session.update_plan([milestone("a", "pending"), milestone("b", "pending")])
        with pytest.raises(ValueError, match="Keep existing"):
            session.update_plan([milestone("a")])
        with pytest.raises(RuntimeError, match="already in use"):
            with SessionStore(tmp_path, "", str(session.path)):
                pass
        session.save()
        assert SessionStore.read(session.path)["plan"][1]["id"] == "b"


def test_keyboard_interrupt_keeps_resumable_checkpoint(tmp_path):
    def interrupt(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        agent._run_agent_loop(tmp_path, tmp_path / "soul", "build", options(), interrupt)
    assert SessionStore.read(session_path(tmp_path))["status"] == "interrupted"


def test_session_storage_cannot_be_edited_by_candidate(tmp_path):
    with pytest.raises(RuntimeError, match="Session storage"):
        agent._apply_candidate(tmp_path, create("data/neo_code/sessions/fake.json", "{}"))


def test_default_session_storage_cannot_follow_data_symlink_outside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "data").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the repository"):
        SessionStore(repo, "task")
