import json
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_code_agent as agent
from coding_agent.tools import CodingTools
from tool_runtime.processes import ProcessManager, run_process
from test_neo_code_sessions import options, completion, session_path


def tool(name, **args):
    return {"tool_id": name, "args": args}


def test_file_operations_reuse_runtime_and_protect_existing_files(tmp_path):
    adapter = CodingTools(tmp_path, apply=True)
    (tmp_path / "one").write_text("one")
    (tmp_path / "existing").write_text("keep")
    assert not adapter.execute(tool("fs.move", src="one", dst="existing"))["ok"]
    assert (tmp_path / "existing").read_text() == "keep"
    assert adapter.execute(tool("fs.copy", src="one", dst="sub/copy"))["ok"]
    assert adapter.execute(tool("fs.move", src="one", dst="sub/moved"))["ok"]
    assert not (tmp_path / "one").exists()
    assert adapter.execute(tool("fs.delete", path="sub/copy"))["ok"]
    assert not adapter.execute(tool("fs.delete", path="sub", recursive=True))["ok"]
    assert (tmp_path / "sub/moved").read_text() == "one"


@pytest.mark.parametrize("name,args", [
    ("fs.delete", {"path": "one"}), ("proc.exec", {"command": ["fake"]}),
    ("proc.start", {"command": ["fake"]}), ("net.search", {"query": "neo"}),
    ("http.request", {"url": "https://example.com"}),
])
def test_permissions_are_enforced_before_runtime_calls(tmp_path, name, args):
    adapter = CodingTools(tmp_path)
    with patch("tool_runtime.tools.proc_exec") as run, patch("tool_runtime.tools.http_request") as http:
        result = adapter.execute(tool(name, **args))
        assert not result["ok"]
        assert "disabled" in result["error"]
        run.assert_not_called()
        http.assert_not_called()


def test_alias_and_path_escapes_are_rejected(tmp_path):
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    (tmp_path / ".git").mkdir()
    (tmp_path / "metadata").symlink_to(tmp_path / ".git", target_is_directory=True)
    adapter = CodingTools(tmp_path, apply=True, allow_exec=True)
    for name in ("../outside", "outside/secret", "metadata/config", "data/neo_code/sessions/a.json", "/tmp/file"):
        assert not adapter.execute(tool("fs.stat", path=name))["ok"]
    assert not adapter.execute(tool("proc.exec", command=[sys.executable], cwd="outside"))["ok"]
    assert not adapter.execute(tool("git.diff", ref="--ext-diff"))["ok"]


def test_git_tools_use_target_repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "untracked.txt").write_text("hello")
    adapter = CodingTools(tmp_path)
    result = adapter.execute(tool("git.status"))
    assert result["ok"]
    assert any(item["path"] == "untracked.txt" for item in result["result"]["changes"])
    assert adapter.execute(tool("git.diff"))["ok"]


def test_network_delegates_with_bounded_read_only_request(tmp_path):
    adapter = CodingTools(tmp_path, allow_network=True)
    with patch("tool_runtime.tools.http_request", return_value=({"status": 200}, {})) as http:
        assert adapter.execute(tool("http.request", url="http://localhost:8000", max_bytes=100000))["ok"]
        assert http.call_args.args[0]["max_bytes"] == 24000
        assert http.call_args.args[1] == str(tmp_path)
        assert not adapter.execute(tool("http.request", url="http://localhost:8000", method="POST"))["ok"]
        assert not adapter.execute(tool("http.request", url="http://localhost:8000", save_to_path="file"))["ok"]
        assert http.call_count == 1


def test_process_execution_returns_failures_as_observations(tmp_path):
    adapter = CodingTools(tmp_path, apply=True, allow_exec=True)
    result = adapter.execute(tool("proc.exec", command=[sys.executable, "-c", "import sys; print('diagnostic'); sys.exit(7)"]))
    assert not result["ok"]
    assert result["result"]["returncode"] == 7
    assert "diagnostic" in result["result"]["stdout"]


def test_process_output_is_bounded_and_stdin_works(tmp_path):
    result = run_process([sys.executable, "-c", "import sys; print(sys.stdin.read()); print('x'*10000)"],
                         cwd=tmp_path, timeout_s=5, stdin="input", max_output_chars=256)
    assert result["returncode"] == 0
    assert len(result["stdout"]) <= 256
    assert result["stdout_truncated"]


def test_background_process_poll_and_stop(tmp_path):
    manager = ProcessManager()
    try:
        started = manager.start({"command": [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(30)"], "timeout_s": 10}, tmp_path)
        output = manager.poll({"handle": started["handle"], "wait_s": 0.2})
        assert output["running"]
        assert "ready" in output["stdout"]
        assert manager.poll({"handle": started["handle"]})["stdout"] == ""
        stopped = manager.stop({"handle": started["handle"]})
        assert not stopped["running"]
        assert not manager.handles
    finally:
        manager.close()


def test_background_lifetime_is_enforced(tmp_path):
    manager = ProcessManager()
    try:
        started = manager.start({"command": [sys.executable, "-c", "import time; time.sleep(30)"], "timeout_s": 1}, tmp_path)
        output = manager.poll({"handle": started["handle"], "wait_s": 3})
        assert not output["running"]
        assert output["timed_out"]
    finally:
        manager.close()


def test_tool_setup_is_followed_by_explicit_verification_and_completion(tmp_path):
    candidates = iter([
        {"tools": [tool("proc.exec", command=[sys.executable, "-c", "from pathlib import Path; Path('generated.txt').write_text('done')"])]},
        {"verify": True, "tests": [f'{shlex.quote(sys.executable)} -c "from pathlib import Path; assert Path(\'generated.txt\').read_text() == \'done\'"']},
        completion(),
    ])
    result = agent._run_agent_loop(tmp_path, tmp_path / "soul", "generate", options(allow_exec=True), lambda _: next(candidates))
    assert result == 0
    state = json.loads(session_path(tmp_path).read_text())
    assert state["verification"]["passed"]
    assert any("Tool result" in item["text"] for item in state["history"])


def test_review_mode_does_not_execute_model_tools(tmp_path):
    candidates = iter([
        {"tools": [tool("proc.exec", command=[sys.executable, "-c", "raise RuntimeError('should not execute')"])]},
        {"status": "blocked", "summary": "execution disabled"},
    ])
    with patch("tool_runtime.tools.proc_exec") as execute:
        assert agent._run_agent_loop(tmp_path, tmp_path / "soul", "build", options(apply=False, allow_exec=True), lambda _: next(candidates)) == 2
        execute.assert_not_called()


def test_session_stops_background_process_when_budget_exhausted(tmp_path):
    adapter = CodingTools(tmp_path, apply=True, allow_exec=True)
    with patch.object(agent, "CodingTools", return_value=adapter):
        result = agent._run_agent_loop(
            tmp_path, tmp_path / "soul", "run server", options(allow_exec=True, max_steps=1),
            lambda _: {"tools": [tool("proc.start", command=[sys.executable, "-c", "import time; time.sleep(30)"])]},
        )
    assert result == 2
    assert not adapter.processes.handles
    state = json.loads(session_path(tmp_path).read_text())
    assert any("Stopped owned background" in item["text"] for item in state["history"])


def test_foreground_timeout_kills_descendants(tmp_path):
    if sys.platform == "win32":
        pytest.skip("POSIX process-group assertion")
    marker = tmp_path / "child-finished"
    child_script = f"import time; from pathlib import Path; time.sleep(0.6); Path({str(marker)!r}).write_text('orphan')"
    parent_script = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child_script!r}]); time.sleep(30)"
    result = run_process([sys.executable, "-c", parent_script], cwd=tmp_path, timeout_s=0.15)
    assert result["timed_out"]
    # Waiting on a separate short process gives a leaked child time to expose itself.
    subprocess.run([sys.executable, "-c", "import time; time.sleep(0.8)"], check=True)
    assert not marker.exists()
