"""Durable execution checkpoints and conservative daily budget reservations."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


class Journal:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "investigations.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY, plan_hash TEXT NOT NULL, stage TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reservations (
                    day TEXT, scope TEXT, id TEXT, actions INTEGER NOT NULL,
                    PRIMARY KEY(day,scope,id));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def lock(self, investigation_id):
        name = hashlib.sha256(investigation_id.encode()).hexdigest() + ".lock"
        with (self.directory / name).open("a+b") as handle:
            if __import__('os').name == 'nt':
                import msvcrt
                handle.seek(0)
                handle.write(b'0')
                handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RuntimeError("Investigation is already running") from exc
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("Investigation is already running") from exc
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def load(self, investigation_id, plan_hash):
        with self.connect() as db:
            row = db.execute("SELECT plan_hash,stage,data FROM checkpoints WHERE id=?", (investigation_id,)).fetchone()
        if row is None:
            return None
        if row[0] != plan_hash:
            raise ValueError("Saved plan differs from the execution journal")
        return {"stage": row[1], "data": json.loads(row[2])}

    def save(self, investigation_id, plan_hash, stage, data):
        with self.connect() as db:
            db.execute("INSERT INTO checkpoints VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET stage=excluded.stage,data=excluded.data",
                       (investigation_id, plan_hash, stage, json.dumps(data)))

    def reserve(self, scope, investigation_id, *, max_investigations, max_actions, actions, day=None):
        day = day or datetime.now(timezone.utc).date().isoformat()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM reservations WHERE day=? AND scope=? AND id=?", (day,scope,investigation_id)).fetchone():
                return True
            count, used = db.execute("SELECT count(*),coalesce(sum(actions),0) FROM reservations WHERE day=? AND scope=?", (day,scope)).fetchone()
            if count >= max_investigations or used + actions > max_actions:
                return False
            db.execute("INSERT INTO reservations VALUES(?,?,?,?)", (day,scope,investigation_id,actions))
            return True
