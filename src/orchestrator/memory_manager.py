import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.config import load_settings
from common.ids import new_id
from common.jsonl_rpc import send_request
from common.time_utils import utc_now_iso


class MemoryManager:
    def __init__(self):
        settings = load_settings()
        self._model_host = settings.rpc["orch_host"]
        self._model_port = settings.rpc["model_port"]
        self._storage_host = settings.rpc["orch_host"]
        self._storage_port = settings.rpc["storage_port"]

    async def retrieve(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        embedding, _model = await self._embed_text(text)
        results: List[Dict[str, Any]] = []
        for mem_type in ("episodic", "fact", "entity", "object"):
            resp = await send_request(
                self._storage_host,
                self._storage_port,
                "storage.SearchMemory",
                {
                    "embedding": embedding,
                    "type": mem_type,
                    "top_k": top_k,
                    "modality": "text",
                    "purpose": "semantic",
                },
            )
            results.extend(resp.get("results", []))
        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results[:top_k]

    async def fetch_episodic_batch(
        self,
        limit: int,
        order: str = "oldest",
        name_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        order_value = "asc" if order == "oldest" else "desc"
        resp = await send_request(
            self._storage_host,
            self._storage_port,
            "storage.GetRecentMemory",
            {"type": "episodic", "limit": limit, "order": order_value, "name": name_filter},
        )
        return resp.get("results", [])

    async def get_facts(self, prefix: str = "") -> List[Dict[str, Any]]:
        resp = await send_request(
            self._storage_host,
            self._storage_port,
            "storage.GetFacts",
            {"entity_prefix": prefix},
        )
        return resp.get("results", [])

    async def delete_memory(self, mem_ids: List[str], trace_id: Optional[str] = None) -> int:
        if not mem_ids:
            return 0
        resp = await send_request(
            self._storage_host,
            self._storage_port,
            "storage.DeleteMemory",
            {"mem_ids": mem_ids, "trace_id": trace_id},
        )
        return int(resp.get("deleted", 0))

    async def maybe_store_facts(self, text: str, trace_id: str) -> List[str]:
        facts = []
        match = re.search(r"my name is ([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        if match:
            value = match.group(1)
            mem_id = await self.store_fact("user.preference.name", value, trace_id, confidence=0.6)
            facts.append(mem_id)
        return facts

    async def store_fact(self, name: str, value: str, trace_id: str, confidence: float = 0.6) -> str:
        embedding, model = await self._embed_text(f"{name}: {value}")
        mem = {
            "mem_id": new_id(),
            "type": "fact",
            "name": name,
            "data": {"value": value},
            "embeddings": [self._embedding_record(embedding, modality="text", purpose="semantic", model=model)],
            "confidence": confidence,
            "trace_id": trace_id,
            "created_at": utc_now_iso(),
        }
        await send_request(
            self._storage_host,
            self._storage_port,
            "storage.WriteMemory",
            {"mem": mem},
        )
        return mem["mem_id"]

    async def store_entity(
        self,
        name: str,
        data: Dict[str, Any],
        trace_id: str,
        confidence: float = 0.6,
    ) -> str:
        embedding, model = await self._embed_text(f"{name}: {json.dumps(data, ensure_ascii=True)}")
        mem = {
            "mem_id": new_id(),
            "type": "entity",
            "name": name,
            "data": data,
            "embeddings": [self._embedding_record(embedding, modality="text", purpose="semantic", model=model)],
            "confidence": confidence,
            "trace_id": trace_id,
            "created_at": utc_now_iso(),
        }
        await send_request(
            self._storage_host,
            self._storage_port,
            "storage.WriteMemory",
            {"mem": mem},
        )
        return mem["mem_id"]

    async def store_object(
        self,
        name: str,
        data: Dict[str, Any],
        trace_id: str,
        *,
        confidence: float = 0.6,
        semantic_text: Optional[str] = None,
        appearance_embeddings: Optional[Sequence[Any]] = None,
    ) -> str:
        semantic_source = semantic_text or f"{name}: {json.dumps(data, ensure_ascii=True)}"
        text_embedding, text_model = await self._embed_text(semantic_source)
        embeddings = [self._embedding_record(text_embedding, modality="text", purpose="semantic", model=text_model)]
        embeddings.extend(self._coerce_appearance_embeddings(appearance_embeddings))
        mem = {
            "mem_id": new_id(),
            "type": "object",
            "name": name,
            "data": data,
            "embeddings": embeddings,
            "confidence": confidence,
            "trace_id": trace_id,
            "created_at": utc_now_iso(),
        }
        await send_request(
            self._storage_host,
            self._storage_port,
            "storage.WriteMemory",
            {"mem": mem},
        )
        return mem["mem_id"]

    async def store_episodic(self, summary: str, trace_id: str, name: str = "conversation.summary") -> str:
        embedding, model = await self._embed_text(summary)
        mem = {
            "mem_id": new_id(),
            "type": "episodic",
            "name": name,
            "data": {"summary": summary, "trace_id": trace_id},
            "embeddings": [self._embedding_record(embedding, modality="text", purpose="semantic", model=model)],
            "trace_id": trace_id,
            "created_at": utc_now_iso(),
        }
        await send_request(
            self._storage_host,
            self._storage_port,
            "storage.WriteMemory",
            {"mem": mem},
        )
        return mem["mem_id"]

    async def search_objects_by_appearance(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not embedding:
            return []
        resp = await send_request(
            self._storage_host,
            self._storage_port,
            "storage.SearchMemory",
            {
                "embedding": embedding,
                "type": "object",
                "top_k": top_k,
                "modality": "vision",
                "purpose": "appearance",
            },
        )
        return resp.get("results", [])

    async def add_object_appearance(
        self,
        mem_id: str,
        appearance_embeddings: Sequence[Any],
        trace_id: Optional[str] = None,
    ) -> int:
        embeddings = self._coerce_appearance_embeddings(appearance_embeddings)
        if not mem_id or not embeddings:
            return 0
        resp = await send_request(
            self._storage_host,
            self._storage_port,
            "storage.AddMemoryEmbeddings",
            {
                "mem_id": mem_id,
                "embeddings": embeddings,
                "trace_id": trace_id,
                "created_at": utc_now_iso(),
            },
        )
        return int(resp.get("added", 0))

    async def embed_image(self, *, image_ref: str = "", image_b64: str = "") -> Tuple[List[float], str]:
        resp = await send_request(
            self._model_host,
            self._model_port,
            "model.EmbedImage",
            {"image_ref": image_ref, "image_b64": image_b64},
        )
        embedding = resp.get("embedding", [])
        model = str(resp.get("model") or "")
        return embedding, model

    async def _embed_text(self, text: str) -> Tuple[List[float], str]:
        resp = await send_request(
            self._model_host,
            self._model_port,
            "model.EmbedText",
            {"text": text},
        )
        embedding = resp.get("embedding", [])
        model = str(resp.get("model") or "")
        return embedding, model

    def _embedding_record(
        self,
        embedding: List[float],
        *,
        modality: str,
        purpose: str,
        model: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "modality": modality,
            "purpose": purpose,
            "model": model or None,
            "embedding": embedding,
        }

    def _coerce_appearance_embeddings(self, value: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
        if not value:
            return []
        normalized: List[Dict[str, Any]] = []
        for item in value:
            embedding = None
            model = None
            if isinstance(item, dict):
                candidate = item.get("embedding")
                if isinstance(candidate, list) and candidate:
                    embedding = candidate
                    raw_model = item.get("model")
                    model = str(raw_model) if raw_model is not None else None
            elif isinstance(item, list) and item:
                embedding = item
            if embedding:
                normalized.append(
                    self._embedding_record(
                        embedding,
                        modality="vision",
                        purpose="appearance",
                        model=model,
                    )
                )
        return normalized
