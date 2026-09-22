from __future__ import annotations

import json
from pathlib import Path
import re

from tool_runtime.processes import ProcessManager


# Reuse the built-in implementations; do not load generated tools or account integrations.
RUNTIME_TOOLS = {
    "fs.list_dir": ("fs_list_dir", {"path"}),
    "fs.stat": ("fs_stat", {"path"}),
    "fs.mkdir": ("fs_mkdir", {"path"}),
    "fs.move": ("fs_move", {"src", "dst"}),
    "fs.copy": ("fs_copy", {"src", "dst"}),
    "fs.delete": ("fs_delete", {"path"}),
    "git.status": ("git_status", {"repo_path", "timeout_s"}),
    "git.diff": ("git_diff", {"repo_path", "timeout_s", "staged", "ref", "paths"}),
    "git.log": ("git_log", {"repo_path", "timeout_s", "max_commits"}),
    "proc.exec": ("proc_exec", {"command", "cwd", "timeout_s", "env", "stdin", "max_output_chars"}),
    "net.search": ("net_search", {"query", "max_results"}),
    "http.request": ("http_request", {"url", "method", "headers", "params", "timeout_s", "max_bytes"}),
}
MUTATIONS = {"fs.mkdir", "fs.move", "fs.copy", "fs.delete", "proc.exec", "proc.start"}


class CodingTools:
    def __init__(self, root: Path, *, apply=False, allow_exec=False, allow_network=False):
        self.root = root.resolve()
        self.apply = apply
        self.allow_exec = allow_exec
        self.allow_network = allow_network
        self.processes = ProcessManager()

    def catalog(self) -> list[dict]:
        registry_path = Path(__file__).resolve().parents[2] / "config" / "tool_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))["tools"]
        entries = []
        for spec in registry:
            name = spec["tool_id"]
            if name not in RUNTIME_TOOLS:
                continue
            schema = spec.get("input_schema", {})
            allowed = RUNTIME_TOOLS[name][1]
            entries.append({"tool_id": name, "description": spec.get("description", ""),
                            "args": {k: v.get("type", "any") for k, v in schema.get("properties", {}).items() if k in allowed},
                            "required": [k for k in schema.get("required", []) if k in allowed],
                            "enabled": self._enabled(name)})
        entries.extend([
            {"tool_id": "proc.start", "args": {"command": "argv array", "cwd": "relative path", "env": "object", "timeout_s": "lifetime, max 3600"}, "enabled": self._enabled("proc.start")},
            {"tool_id": "proc.poll", "args": {"handle": "string", "wait_s": "0..10", "max_output_chars": "integer"}, "enabled": self._enabled("proc.poll")},
            {"tool_id": "proc.stop", "args": {"handle": "string"}, "enabled": self._enabled("proc.stop")},
        ])
        return entries

    def _enabled(self, name: str) -> bool:
        if name.startswith("proc."):
            return bool(self.apply and self.allow_exec)
        if name in MUTATIONS:
            return bool(self.apply)
        if name in {"net.search", "http.request"}:
            return bool(self.allow_network)
        return True

    def _path(self, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Paths must be strings")
        value = value.removeprefix("workspace:/").replace("\\", "/")
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
            raise ValueError("Use repository-relative paths")
        resolved = (self.root / value).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Path escapes the repository")
        relative = resolved.relative_to(self.root)
        if ".git" in relative.parts or relative.parts[:2] == ("data", "neo_code"):
            raise ValueError("Git metadata and session storage cannot be accessed through file tools")
        return "workspace:/" + relative.as_posix()

    def execute(self, request: dict) -> dict:
        name = request.get("tool_id")
        args = request.get("args", {})
        mutating = name in MUTATIONS
        try:
            if name not in RUNTIME_TOOLS and name not in {"proc.start", "proc.poll", "proc.stop"}:
                raise ValueError(f"Unknown coding tool: {name}")
            if not isinstance(args, dict):
                raise ValueError("Tool args must be an object")
            if not self._enabled(name):
                raise ValueError("Tool disabled by this session's permissions; see enabled tools. "
                                 "Mutations require --apply, processes --allow-exec, network --allow-network.")
            args = dict(args)
            if name in RUNTIME_TOOLS:
                unknown = set(args) - RUNTIME_TOOLS[name][1]
                if unknown:
                    raise ValueError(f"Unsupported arguments: {sorted(unknown)}")
            for key in ("path", "src", "dst", "repo_path", "cwd"):
                if key in args:
                    args[key] = self._path(args[key])
            if "paths" in args:
                if not isinstance(args["paths"], list):
                    raise ValueError("paths must be a list")
                args["paths"] = [self._path(p).removeprefix("workspace:/") for p in args["paths"]]
            if "ref" in args and (not isinstance(args["ref"], str) or args["ref"].startswith("-")):
                raise ValueError("Git ref must be a revision, not an option")
            if name in {"fs.move", "fs.copy", "fs.delete"}:
                key = "path" if name == "fs.delete" else "src"
                if key not in args:
                    raise ValueError(f"{key} is required")
                src = self.root / args[key].removeprefix("workspace:/")
                if not src.is_file():
                    raise ValueError("Move/copy/delete currently operate on individual files only")
                if name != "fs.delete":
                    if "dst" not in args:
                        raise ValueError("dst is required")
                    dst = self.root / args["dst"].removeprefix("workspace:/")
                    if dst.exists():
                        raise ValueError("Destination already exists; use a validated edit")
            if name == "http.request":
                if str(args.get("method", "GET")).upper() not in {"GET", "HEAD"}:
                    raise ValueError("Coding HTTP requests support GET/HEAD only")
                args["max_bytes"] = min(int(args.get("max_bytes", 24000)), 24000)
                args["timeout_s"] = min(float(args.get("timeout_s", 20)), 60)
            if name == "proc.start":
                cwd = self.root / args.get("cwd", "workspace:/").removeprefix("workspace:/")
                result = self.processes.start(args, cwd)
            elif name == "proc.poll":
                result = self.processes.poll(args)
            elif name == "proc.stop":
                result = self.processes.stop(args)
            else:
                from tool_runtime import tools
                if name == "proc.exec":
                    args["max_output_chars"] = min(int(args.get("max_output_chars", 12000)), 24000)
                result, _ = getattr(tools, RUNTIME_TOOLS[name][0])(args, str(self.root))
            ok = not (name == "proc.exec" and (result.get("returncode") != 0 or result.get("timed_out")))
            return {"ok": ok, "tool_id": name, "result": result, "mutating": mutating}
        except Exception as exc:
            return {"ok": False, "tool_id": name, "error": str(exc), "mutating": mutating}

    def close(self):
        self.processes.close()
