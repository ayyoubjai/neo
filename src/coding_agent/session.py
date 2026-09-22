from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from typing import Any


class SessionStore:
    """One atomic checkpoint, including the full transcript, per coding session."""

    def __init__(self, root: Path, task: str, resume: str | None = None):
        self.root = root.resolve()
        self.path = (Path(resume).expanduser().resolve() if resume else
                     self.root / "data" / "neo_code" / "sessions" / f"{uuid.uuid4().hex}.json")
        if not resume and not self.path.resolve().is_relative_to(self.root):
            raise ValueError("Session storage resolves outside the repository")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = None
        self.task = task
        self.resuming = bool(resume)
        self.state: dict[str, Any] = {}

    def __enter__(self):
        self._lock = self.path.with_suffix(".lock").open("a+b")
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                self._lock.seek(0)
                self._lock.write(b"0")
                self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self._lock.close()
            self._lock = None
            raise RuntimeError(f"Session is already in use: {self.path}") from exc
        try:
            if self.resuming:
                self.state = self.read(self.path)
                if Path(self.state["repo_root"]).resolve() != self.root:
                    raise ValueError("Session belongs to a different repository")
                if self.task and self.task != self.state["task"]:
                    raise ValueError("Resume without a new task; start a new session to change the objective")
                previous = self.state.get("pending_action")
                self.observe("Resumed session. Previous process handles are no longer usable. "
                             "Verification must be rerun before completion." +
                             (f" Interrupted action (not replayed): {previous}" if previous else ""))
                self.state["verification"] = None
                self.state["pending_action"] = None
            else:
                self.state = {
                    "version": 1, "repo_root": str(self.root), "task": self.task,
                    "plan": [{"id": "task", "description": self.task, "status": "pending"}],
                    "notes": "", "history": [], "changed": [], "steps": 0,
                    "revision": 0, "verification": None, "pending_action": None,
                    "options": {}, "status": "running",
                }
            self.state["status"] = "running"
            self.save()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported Neo Code session")
        if not isinstance(state.get("task"), str) or not isinstance(state.get("repo_root"), str):
            raise ValueError("Invalid session task or repository")
        for key, kind in (("plan", list), ("history", list), ("changed", list), ("options", dict)):
            if not isinstance(state.get(key), kind):
                raise ValueError(f"Invalid session {key}")
        if any(not isinstance(item, dict) or not isinstance(item.get("text"), str) for item in state["history"]):
            raise ValueError("Invalid session transcript")
        if any(not isinstance(name, str) for name in state["changed"]):
            raise ValueError("Invalid changed-file list")
        for key in ("steps", "revision"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f"Invalid session {key}")
        if not isinstance(state.get("notes"), str) or not state["plan"]:
            raise ValueError("Invalid session notes or plan")
        for item in state["plan"]:
            if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                    or not isinstance(item.get("description"), str)
                    or item.get("status") not in {"pending", "in_progress", "completed"}):
                raise ValueError("Invalid session milestone")
        return state

    def save(self):
        self.state["updated_at"] = time.time()
        descriptor, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def observe(self, text: str):
        self.state["history"].append({"role": "tool", "text": text})

    def update_plan(self, plan: Any):
        if plan is None:
            return
        if not isinstance(plan, list) or not plan or len(plan) > 40:
            raise ValueError("plan must contain between 1 and 40 milestones")
        if len(json.dumps(plan)) > 10000:
            raise ValueError("Keep the serialized plan under 10000 characters")
        seen = set()
        for item in plan:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError("plan milestones need a nonempty string id")
            if item["id"] in seen or item.get("status") not in {"pending", "in_progress", "completed"}:
                raise ValueError("plan ids must be unique and statuses pending/in_progress/completed")
            if not isinstance(item.get("description"), str) or not item["description"].strip():
                raise ValueError("plan milestones need a description")
            seen.add(item["id"])
        previous = {item["id"] for item in self.state["plan"]}
        # The initial placeholder may be decomposed on the first explicit plan.
        if self.state.get("has_plan") and not previous.issubset(seen):
            raise ValueError("Keep existing milestone ids when updating the plan; do not drop unfinished work")
        self.state["plan"] = plan
        self.state["has_plan"] = True

    def begin_action(self, action: Any, mutating: bool = False):
        self.state["pending_action"] = action
        if mutating:
            self.state["revision"] += 1
            self.state["verification"] = None
        self.save()

    def finish_action(self):
        self.state["pending_action"] = None
        self.save()

    def __exit__(self, *_):
        if self._lock is not None:
            if os.name != "posix":
                import msvcrt
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
            self._lock.close()
            self._lock = None
