import json
import math
from typing import Any, Dict, List, Optional

from common.ids import new_id
from common.time_utils import utc_now_iso


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class StorageBackend:
    def __init__(self, conn, audit_log_path: str):
        self._conn = conn
        self._audit_log_path = audit_log_path

    def append_audit(self, record: Dict[str, Any]) -> str:
        audit_id = record.get("audit_id") or new_id()
        payload_json = json.dumps(record.get("payload", {}), ensure_ascii=True)
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO audit_log
            (audit_id, action, tool_id, trace_id, actor, before_hash, after_hash, diff, timestamp, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                record.get("action", ""),
                record.get("tool_id"),
                record.get("trace_id"),
                record.get("actor"),
                record.get("before_hash"),
                record.get("after_hash"),
                record.get("diff"),
                record.get("timestamp", utc_now_iso()),
                payload_json,
            ),
        )
        self._conn.commit()
        with open(self._audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        return audit_id

    def write_memory(self, mem: Dict[str, Any]) -> str:
        mem_id = mem.get("mem_id") or new_id()
        embeddings = self._normalize_embeddings(mem.get("embeddings"))
        primary_embedding = self._primary_embedding(mem, embeddings)
        created_at = mem.get("created_at", utc_now_iso())
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO memory
            (mem_id, type, name, data_json, embedding_json, created_at, last_used_at, use_count, ttl_days, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem_id,
                mem.get("type", ""),
                mem.get("name"),
                json.dumps(mem.get("data", {}), ensure_ascii=True),
                json.dumps(primary_embedding) if primary_embedding is not None else None,
                created_at,
                mem.get("last_used_at"),
                mem.get("use_count", 0),
                mem.get("ttl_days"),
                mem.get("confidence"),
            ),
        )
        if embeddings:
            self._write_memory_embeddings(mem_id, embeddings, created_at)
        self._conn.commit()
        return mem_id

    def ensure_memory(self, mem: Dict[str, Any]) -> str:
        mem_id = mem.get("mem_id") or new_id()
        cur = self._conn.cursor()
        cur.execute("SELECT mem_id FROM memory WHERE mem_id = ?", (mem_id,))
        row = cur.fetchone()
        if row:
            return mem_id
        return self.write_memory(mem)

    def search_memory(
        self,
        embedding: List[float],
        type_filter: Optional[str],
        top_k: int,
        modality: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        if type_filter:
            cur.execute("SELECT * FROM memory WHERE type = ?", (type_filter,))
        else:
            cur.execute("SELECT * FROM memory")
        rows = cur.fetchall()
        if not rows:
            return []

        mem_ids = [str(row["mem_id"]) for row in rows if row["mem_id"]]
        embedding_rows = self._list_embedding_rows(mem_ids, modality=modality, purpose=purpose)
        embedding_map: Dict[str, List[Dict[str, Any]]] = {}
        for item in embedding_rows:
            embedding_map.setdefault(str(item["mem_id"]), []).append(item)

        allow_primary_fallback = (modality in (None, "text")) and (purpose in (None, "semantic"))
        scored = []
        for row in rows:
            best_score = None
            best_match = None
            for emb_row in embedding_map.get(str(row["mem_id"]), []):
                stored = json.loads(emb_row["embedding_json"])
                score = _cosine(embedding, stored)
                if best_score is None or score > best_score:
                    best_score = score
                    best_match = {
                        "embedding_id": emb_row["embedding_id"],
                        "modality": emb_row["modality"],
                        "purpose": emb_row["purpose"],
                        "model": emb_row["model"],
                    }
            if best_score is None and allow_primary_fallback and row["embedding_json"]:
                stored = json.loads(row["embedding_json"])
                best_score = _cosine(embedding, stored)
                best_match = {
                    "embedding_id": None,
                    "modality": "text",
                    "purpose": "semantic",
                    "model": "memory.primary",
                }
            if best_score is None:
                continue
            scored.append((best_score, row, best_match))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, row, best_match in scored[:top_k]:
            results.append(
                {
                    "mem_id": row["mem_id"],
                    "type": row["type"],
                    "name": row["name"],
                    "data": json.loads(row["data_json"]),
                    "embedding": json.loads(row["embedding_json"]) if row["embedding_json"] else None,
                    "created_at": row["created_at"],
                    "last_used_at": row["last_used_at"],
                    "use_count": row["use_count"],
                    "ttl_days": row["ttl_days"],
                    "confidence": row["confidence"],
                    "score": score,
                    "matched_embedding": best_match,
                }
            )
        return results

    def get_facts(self, entity_prefix: str) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM memory WHERE type = 'fact' AND name LIKE ?", (f"{entity_prefix}%",))
        rows = cur.fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "mem_id": row["mem_id"],
                    "type": row["type"],
                    "name": row["name"],
                    "data": json.loads(row["data_json"]),
                    "created_at": row["created_at"],
                    "last_used_at": row["last_used_at"],
                    "use_count": row["use_count"],
                    "ttl_days": row["ttl_days"],
                    "confidence": row["confidence"],
                }
            )
        return results

    def list_memory(
        self,
        type_filter: Optional[str],
        limit: int,
        order: str = "asc",
        name_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        order_sql = "ASC" if order.lower() == "asc" else "DESC"
        where = []
        params: List[Any] = []
        if type_filter:
            where.append("type = ?")
            params.append(type_filter)
        if name_filter:
            where.append("name = ?")
            params.append(name_filter)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(limit)
        cur.execute(f"SELECT * FROM memory {where_sql} ORDER BY created_at {order_sql} LIMIT ?", params)
        rows = cur.fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "mem_id": row["mem_id"],
                    "type": row["type"],
                    "name": row["name"],
                    "data": json.loads(row["data_json"]),
                    "embedding": json.loads(row["embedding_json"]) if row["embedding_json"] else None,
                    "created_at": row["created_at"],
                    "last_used_at": row["last_used_at"],
                    "use_count": row["use_count"],
                    "ttl_days": row["ttl_days"],
                    "confidence": row["confidence"],
                }
            )
        return results

    def delete_memory(self, mem_ids: List[str]) -> int:
        if not mem_ids:
            return 0
        placeholders = ",".join("?" for _ in mem_ids)
        cur = self._conn.cursor()
        cur.execute(f"DELETE FROM memory_embeddings WHERE mem_id IN ({placeholders})", mem_ids)
        cur.execute(f"DELETE FROM memory WHERE mem_id IN ({placeholders})", mem_ids)
        self._conn.commit()
        return cur.rowcount

    def add_memory_embeddings(self, mem_id: str, embeddings: List[Dict[str, Any]], created_at: Optional[str] = None) -> int:
        if not mem_id or not embeddings:
            return 0
        normalized = self._normalize_embeddings(embeddings)
        if not normalized:
            return 0
        timestamp = created_at or utc_now_iso()
        cur = self._conn.cursor()
        cur.execute("SELECT mem_id FROM memory WHERE mem_id = ?", (mem_id,))
        row = cur.fetchone()
        if not row:
            return 0
        self._write_memory_embeddings(mem_id, normalized, timestamp)
        self._conn.commit()
        return len(normalized)

    def _normalize_embeddings(self, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            modality = str(item.get("modality") or "").strip() or "text"
            purpose = str(item.get("purpose") or "").strip() or "semantic"
            model = item.get("model")
            normalized.append(
                {
                    "modality": modality,
                    "purpose": purpose,
                    "model": str(model) if model is not None else None,
                    "embedding": embedding,
                }
            )
        return normalized

    def _primary_embedding(self, mem: Dict[str, Any], embeddings: List[Dict[str, Any]]) -> Optional[List[float]]:
        explicit = mem.get("embedding")
        if isinstance(explicit, list) and explicit:
            return explicit
        for item in embeddings:
            if item["modality"] == "text" and item["purpose"] == "semantic":
                return item["embedding"]
        if embeddings:
            return embeddings[0]["embedding"]
        return None

    def _write_memory_embeddings(self, mem_id: str, embeddings: List[Dict[str, Any]], created_at: str) -> None:
        cur = self._conn.cursor()
        for item in embeddings:
            cur.execute(
                """
                INSERT INTO memory_embeddings
                (embedding_id, mem_id, modality, purpose, model, embedding_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id(),
                    mem_id,
                    item["modality"],
                    item["purpose"],
                    item["model"],
                    json.dumps(item["embedding"]),
                    created_at,
                ),
            )

    def _list_embedding_rows(
        self,
        mem_ids: List[str],
        modality: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not mem_ids:
            return []
        placeholders = ",".join("?" for _ in mem_ids)
        query = (
            "SELECT embedding_id, mem_id, modality, purpose, model, embedding_json, created_at "
            f"FROM memory_embeddings WHERE mem_id IN ({placeholders})"
        )
        params: List[Any] = list(mem_ids)
        if modality:
            query += " AND modality = ?"
            params.append(modality)
        if purpose:
            query += " AND purpose = ?"
            params.append(purpose)
        cur = self._conn.cursor()
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]
