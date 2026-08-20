import asyncio
import os
from typing import Any, Dict

from common.config import load_settings
from common.jsonl_rpc import start_server
from common.time_utils import utc_now_iso
from storage.backend import StorageBackend
from storage.db import connect, init_db
from storage.system_seed import build_system_memory


class StorageService:
    def __init__(self, backend: StorageBackend):
        self._backend = backend
        self._lock = asyncio.Lock()

    async def AppendAudit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        record = params.get("record", {})
        if "timestamp" not in record:
            record["timestamp"] = utc_now_iso()
        async with self._lock:
            audit_id = await asyncio.to_thread(self._backend.append_audit, record)
        return {"status": "OK", "audit_id": audit_id}

    async def WriteMemory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mem = params.get("mem", {})
        async with self._lock:
            mem_id = await asyncio.to_thread(self._backend.write_memory, mem)
            await asyncio.to_thread(
                self._backend.append_audit,
                {
                    "audit_id": None,
                    "action": "memory_write",
                    "tool_id": None,
                    "trace_id": mem.get("trace_id"),
                    "actor": "storage",
                    "before_hash": None,
                    "after_hash": None,
                    "diff": None,
                    "timestamp": utc_now_iso(),
                    "payload": {"mem_id": mem_id, "type": mem.get("type"), "name": mem.get("name")},
                },
            )
        return {"status": "OK", "mem_id": mem_id}

    async def SearchMemory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        embedding = params.get("embedding", [])
        type_filter = params.get("type")
        top_k = int(params.get("top_k", 5))
        modality = params.get("modality")
        purpose = params.get("purpose")
        async with self._lock:
            results = await asyncio.to_thread(
                self._backend.search_memory,
                embedding,
                type_filter,
                top_k,
                modality,
                purpose,
            )
        return {"status": "OK", "results": results}

    async def GetFacts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prefix = params.get("entity_prefix", "")
        async with self._lock:
            results = await asyncio.to_thread(self._backend.get_facts, prefix)
        return {"status": "OK", "results": results}

    async def GetRecentMemory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        type_filter = params.get("type")
        name_filter = params.get("name")
        limit = int(params.get("limit", 10))
        order = str(params.get("order", "asc")).lower()
        async with self._lock:
            results = await asyncio.to_thread(self._backend.list_memory, type_filter, limit, order, name_filter)
        return {"status": "OK", "results": results}

    async def DeleteMemory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mem_ids = params.get("mem_ids", [])
        if not isinstance(mem_ids, list):
            mem_ids = []
        async with self._lock:
            deleted = await asyncio.to_thread(self._backend.delete_memory, mem_ids)
            await asyncio.to_thread(
                self._backend.append_audit,
                {
                    "audit_id": None,
                    "action": "memory_delete",
                    "tool_id": None,
                    "trace_id": params.get("trace_id"),
                    "actor": "storage",
                    "before_hash": None,
                    "after_hash": None,
                    "diff": None,
                    "timestamp": utc_now_iso(),
                    "payload": {"mem_ids": mem_ids, "deleted": deleted},
                },
            )
        return {"status": "OK", "deleted": deleted}

    async def AddMemoryEmbeddings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mem_id = str(params.get("mem_id") or "")
        embeddings = params.get("embeddings", [])
        created_at = params.get("created_at")
        async with self._lock:
            added = await asyncio.to_thread(self._backend.add_memory_embeddings, mem_id, embeddings, created_at)
            await asyncio.to_thread(
                self._backend.append_audit,
                {
                    "audit_id": None,
                    "action": "memory_add_embeddings",
                    "tool_id": None,
                    "trace_id": params.get("trace_id"),
                    "actor": "storage",
                    "before_hash": None,
                    "after_hash": None,
                    "diff": None,
                    "timestamp": utc_now_iso(),
                    "payload": {"mem_id": mem_id, "added": added},
                },
            )
        return {"status": "OK", "added": added}

    async def Ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "OK"}


async def serve() -> None:
    settings = load_settings()
    data_dir = settings.data_dir
    db_path = os.path.join(data_dir, "storage.db")
    audit_path = os.path.join(data_dir, "audit.log")

    conn = connect(db_path)
    init_db(conn)

    backend = StorageBackend(conn, audit_path)
    await asyncio.to_thread(backend.ensure_memory, build_system_memory())
    service = StorageService(backend)

    handlers = {
        "storage.AppendAudit": service.AppendAudit,
        "storage.WriteMemory": service.WriteMemory,
        "storage.SearchMemory": service.SearchMemory,
        "storage.GetFacts": service.GetFacts,
        "storage.GetRecentMemory": service.GetRecentMemory,
        "storage.DeleteMemory": service.DeleteMemory,
        "storage.AddMemoryEmbeddings": service.AddMemoryEmbeddings,
        "storage.Ping": service.Ping,
    }

    listen_host = settings.rpc["orch_host"]
    listen_port = settings.rpc["storage_port"]
    server = await start_server(listen_host, listen_port, handlers)
    print(f"[storage] listening on {listen_host}:{listen_port}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve())
