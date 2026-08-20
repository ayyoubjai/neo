import os
import sqlite3
from typing import Optional


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            mem_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT,
            data_json TEXT NOT NULL,
            embedding_json TEXT,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            use_count INTEGER NOT NULL,
            ttl_days INTEGER,
            confidence REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            embedding_id TEXT PRIMARY KEY,
            mem_id TEXT NOT NULL,
            modality TEXT NOT NULL,
            purpose TEXT NOT NULL,
            model TEXT,
            embedding_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_embeddings_mem_id ON memory_embeddings(mem_id)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_lookup ON memory_embeddings(modality, purpose, mem_id)"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id TEXT PRIMARY KEY,
            action TEXT NOT NULL,
            tool_id TEXT,
            trace_id TEXT,
            actor TEXT,
            before_hash TEXT,
            after_hash TEXT,
            diff TEXT,
            timestamp TEXT NOT NULL,
            payload_json TEXT
        )
        """
    )
    conn.commit()
