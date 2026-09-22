"""Bounded process execution and explicitly owned background process groups."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import uuid


def terminate_process(proc: subprocess.Popen):
    if os.name == "posix":
        # A shell may have exited while its children still occupy the group.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=10, check=False)
        finally:
            if proc.poll() is None:
                proc.kill()
    proc.wait(timeout=10)


class ProcessManager:
    def __init__(self):
        self.handles = {}

    def start(self, args: dict, cwd: Path) -> dict:
        if len(self.handles) >= 4:
            raise ValueError("At most four owned processes; stop an existing handle first")
        command = args.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError("command must be a nonempty argv list")
        if args.get("stdin") is not None:
            raise ValueError("Background processes do not accept stdin")
        env = os.environ.copy()
        raw_env = args.get("env", {})
        if not isinstance(raw_env, dict):
            raise ValueError("env must be an object")
        env.update({str(k): str(v) for k, v in raw_env.items()})
        lifetime = max(1, min(float(args.get("timeout_s", 600)), 3600))
        logs = tempfile.TemporaryDirectory(prefix="neo-process-")
        log_paths = [Path(logs.name) / "stdout", Path(logs.name) / "stderr"]
        stdout, stderr = [path.open("w+b") for path in log_paths]
        try:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr, start_new_session=os.name == "posix")
        except BaseException:
            stdout.close()
            stderr.close()
            logs.cleanup()
            raise
        handle = uuid.uuid4().hex[:12]
        entry = {"proc": proc, "stdout": stdout, "stderr": stderr, "timed_out": False,
                 "command": command, "offsets": [0, 0], "logs": logs, "log_paths": log_paths}

        def expire():
            entry["timed_out"] = proc.poll() is None
            try:
                terminate_process(proc)
            except (OSError, subprocess.TimeoutExpired):
                pass

        timer = threading.Timer(lifetime, expire)
        timer.daemon = True
        entry["timer"] = timer
        self.handles[handle] = entry
        timer.start()
        return {"handle": handle, "pid": proc.pid, "running": proc.poll() is None,
                "lifetime_s": lifetime}

    def poll(self, args: dict) -> dict:
        handle = args.get("handle")
        if handle not in self.handles:
            raise ValueError("Unknown process handle in this invocation")
        entry = self.handles[handle]
        proc = entry["proc"]
        wait = max(0, min(float(args.get("wait_s", 0)), 10))
        if wait and proc.poll() is None:
            try:
                proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                pass
        limit = max(256, min(int(args.get("max_output_chars", 12000)), 24000))
        result = {"handle": handle, "running": proc.poll() is None, "returncode": proc.poll(),
                  "timed_out": entry["timed_out"]}
        for i, key in enumerate(("stdout", "stderr")):
            with entry["log_paths"][i].open("rb") as log:
                log.seek(0, os.SEEK_END)
                size = log.tell()
                offset = max(entry["offsets"][i], size - limit)
                log.seek(offset)
                raw = log.read(limit)
            result[key] = raw.decode("utf-8", errors="replace")
            result[key + "_truncated"] = offset > entry["offsets"][i]
            entry["offsets"][i] = offset + len(raw)
        return result

    def stop(self, args: dict) -> dict:
        handle = args.get("handle")
        if handle not in self.handles:
            raise ValueError("Unknown process handle in this invocation")
        entry = self.handles[handle]
        entry["timer"].cancel()
        entry["timer"].join()
        terminate_process(entry["proc"])
        result = self.poll({"handle": handle})
        for key in ("stdout", "stderr"):
            entry[key].close()
        entry["logs"].cleanup()
        del self.handles[handle]
        return result

    def close(self):
        for handle in list(self.handles):
            self.stop({"handle": handle})


def run_process(command, *, cwd, timeout_s, stdin=None, env=None, max_output_chars=12000, shell=False):
    """Run argv with bounded output and clean up descendants on timeout."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(command, cwd=cwd, env=env, text=True, shell=shell,
                                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                stdout=stdout, stderr=stderr, start_new_session=os.name == "posix")
        timed_out = False
        try:
            proc.communicate(input=stdin, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process(proc)
        except BaseException:
            terminate_process(proc)
            raise
        finally:
            if proc.stdin is not None and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
        result = {"returncode": -1 if timed_out else proc.returncode, "timed_out": timed_out}
        for name, log in (("stdout", stdout), ("stderr", stderr)):
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - max_output_chars))
            result[name] = log.read().decode("utf-8", errors="replace")
            result[name + "_truncated"] = size > max_output_chars
        return result
