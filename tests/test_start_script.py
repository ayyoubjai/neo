from __future__ import annotations

import os
import json
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
START_SCRIPT = REPO_ROOT / "start.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_launcher(
    tmp_path: Path, docker_script: str, legacy_script: str | None = None
) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "project"
    fake_bin = tmp_path / "bin"
    (root / "config").mkdir(parents=True)
    (root / "scripts").mkdir()
    for package, filename in (("common", "config.py"), ("model_server", "managed_router.py")):
        target = root / "src" / package
        target.mkdir(parents=True, exist_ok=True)
        (target / filename).write_text((REPO_ROOT / "src" / package / filename).read_text(), encoding="utf-8")
    fake_bin.mkdir()

    (root / "start.sh").write_text(
        START_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "config" / "settings.example.json").write_text(
        '{"models": {"provider": "test"}}\n', encoding="utf-8"
    )
    _write_executable(
        root / "scripts" / "run_all.sh",
        "#!/usr/bin/env bash\necho RUN_ALL_REACHED\n",
    )
    _write_executable(fake_bin / "docker", docker_script)
    _write_executable(
        fake_bin / "docker-compose",
        legacy_script or "#!/usr/bin/env bash\nexit 1\n",
    )

    environment = os.environ.copy()
    environment.pop("LLAMA_SERVER_BIN", None)
    environment.pop("AGI_SETTINGS_PATH", None)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["DOCKER_CALL_LOG"] = str(tmp_path / "docker-calls.log")
    return root, environment


def test_docker_permission_error_has_specific_guidance(tmp_path: Path) -> None:
    root, environment = _make_launcher(tmp_path, '''#!/usr/bin/env bash
if [[ "$1 $2" == "compose version" ]]; then exit 0; fi
echo 'permission denied while trying to connect to the docker API' >&2
exit 1
''')
    result = _run_launcher(root, environment)
    assert result.returncode == 0
    assert "Docker socket access denied" in result.stdout
    assert "RUN_ALL_REACHED" in result.stdout


def test_missing_llama_binary_fails_before_preset_generation(tmp_path: Path) -> None:
    root, environment = _make_launcher(tmp_path, "#!/usr/bin/env bash\nexit 1\n")
    environment["LLAMA_SERVER_BIN"] = str(tmp_path / "missing-llama-server")
    (root / "config/settings.local.json").write_text(json.dumps({"models": {
        "provider": "llamacpp", "llamacpp_host": "http://127.0.0.1:0"
    }}), encoding="utf-8")
    result = _run_launcher(root, environment)
    assert result.returncode == 1
    assert "llama-server executable not found" in result.stderr
    assert "models.llama_server" in result.stderr
    assert "Generating models preset" not in result.stdout
    assert "RUN_ALL_REACHED" not in result.stdout


def test_configured_llama_binary_and_environment_override(tmp_path: Path) -> None:
    root, environment = _make_launcher(tmp_path, "#!/usr/bin/env bash\nexit 1\n")
    binary = tmp_path / "custom llama-server"
    _write_executable(binary, "#!/usr/bin/env bash\necho CONFIGURED_BINARY_REACHED\necho SERVER_DIAGNOSTIC >&2\nexit 1\n")
    (root / "config/settings.local.json").write_text(json.dumps({"models": {
        "provider": "llamacpp", "llamacpp_host": "http://127.0.0.1:0",
        "llama_server": str(binary)
    }}), encoding="utf-8")
    result = _run_launcher(root, environment)
    log_path = root / "data/llama-server.log"
    assert "CONFIGURED_BINARY_REACHED" in log_path.read_text(encoding="utf-8")
    assert "SERVER_DIAGNOSTIC" in log_path.read_text(encoding="utf-8")
    assert "CONFIGURED_BINARY_REACHED" not in result.stdout + result.stderr
    assert "SERVER_DIAGNOSTIC" not in result.stdout + result.stderr
    assert str(log_path) in result.stderr
    override = tmp_path / "override llama-server"
    _write_executable(override, "#!/usr/bin/env bash\necho OVERRIDE_BINARY_REACHED\nexit 1\n")
    environment["LLAMA_SERVER_BIN"] = str(override)
    result = _run_launcher(root, environment)
    assert "OVERRIDE_BINARY_REACHED" in log_path.read_text(encoding="utf-8")
    assert "CONFIGURED_BINARY_REACHED" not in log_path.read_text(encoding="utf-8")
    assert "OVERRIDE_BINARY_REACHED" not in result.stdout + result.stderr


def _run_launcher(
    root: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "start.sh"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_missing_compose_is_skipped_and_assistant_starts(tmp_path: Path) -> None:
    root, environment = _make_launcher(tmp_path, "#!/usr/bin/env bash\nexit 1\n")

    result = _run_launcher(root, environment)

    assert result.returncode == 0
    assert "Docker Compose not found — skipping SearXNG." in result.stdout
    assert "RUN_ALL_REACHED" in result.stdout


def test_unavailable_daemon_is_skipped_and_assistant_starts(tmp_path: Path) -> None:
    root, environment = _make_launcher(
        tmp_path,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "compose version" ]]; then exit 0; fi
if [[ "$1" == "info" ]]; then exit 1; fi
echo "$*" >> "$DOCKER_CALL_LOG"
exit 0
""",
    )

    result = _run_launcher(root, environment)

    assert result.returncode == 0
    assert "Docker daemon is unavailable — skipping SearXNG." in result.stdout
    assert "RUN_ALL_REACHED" in result.stdout
    assert not Path(environment["DOCKER_CALL_LOG"]).exists()


def test_compose_v2_starts_searxng(tmp_path: Path) -> None:
    root, environment = _make_launcher(
        tmp_path,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "compose version" || "$1" == "info" ]]; then exit 0; fi
echo "$*" >> "$DOCKER_CALL_LOG"
exit 0
""",
    )

    result = _run_launcher(root, environment)

    assert result.returncode == 0
    assert "SearXNG is running" in result.stdout
    assert "RUN_ALL_REACHED" in result.stdout
    assert Path(environment["DOCKER_CALL_LOG"]).read_text(encoding="utf-8").strip() == (
        "compose -f docker-compose.searxng.yml up -d"
    )


def test_legacy_compose_is_used_as_fallback(tmp_path: Path) -> None:
    root, environment = _make_launcher(
        tmp_path,
        """#!/usr/bin/env bash
if [[ "$1 $2" == "compose version" ]]; then exit 1; fi
if [[ "$1" == "info" ]]; then exit 0; fi
exit 1
""",
        """#!/usr/bin/env bash
if [[ "$1" == "version" ]]; then exit 0; fi
echo "$*" >> "$DOCKER_CALL_LOG"
exit 0
""",
    )

    result = _run_launcher(root, environment)

    assert result.returncode == 0
    assert "SearXNG is running" in result.stdout
    assert "RUN_ALL_REACHED" in result.stdout
    assert Path(environment["DOCKER_CALL_LOG"]).read_text(encoding="utf-8").strip() == (
        "-f docker-compose.searxng.yml up -d"
    )
