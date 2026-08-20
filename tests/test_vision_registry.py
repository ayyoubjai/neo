from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_daemon.vision_context import VISION_TRACKED_ENTITIES_KEY
from voice_daemon.vision_registry import VisionEntityRegistry


class _FakeContext:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.deleted: List[str] = []

    async def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    async def delete_state(self, key: str) -> None:
        self.deleted.append(key)
        self.state.pop(key, None)


class _FakeVisionService:
    def __init__(self, *, auto_store: bool = False, export_created: bool = False) -> None:
        self._auto_store = auto_store
        self._export_created = export_created
        self.cleaned_payloads: List[Dict[str, Any]] = []
        self.exported_records: List[Dict[str, Any]] = []

    def detection_features(self, cv2, frame: Any, detection: Dict[str, Any], frame_size=None) -> Dict[str, Any]:
        return {
            "color_hist": [0.0, 1.0],
            "aspect_ratio": 1.0,
            "area_ratio": 0.1,
            "zone": "center",
        }

    def track_stable_frames(self) -> int:
        return 2

    def track_max_missing_frames(self) -> int:
        return 3

    def object_memory_enabled(self) -> bool:
        return True

    def object_memory_auto_store(self) -> bool:
        return self._auto_store

    def object_memory_append_embedding_on_match(self) -> bool:
        return True

    def object_memory_match_top_k(self) -> int:
        return 5

    def object_memory_match_threshold(self) -> float:
        return 0.72

    def object_memory_match_margin(self) -> float:
        return 0.08

    def object_memory_label_gate(self) -> bool:
        return True

    def object_memory_object_name(self, label: str) -> str:
        return f"vision.object.{label}"

    def build_embedding_input(self, cv2, frame: Any, detection: Dict[str, Any]) -> Dict[str, Any]:
        return {"image_b64": "ZmFrZQ=="}

    def cleanup_embedding_input(self, payload: Dict[str, Any]) -> None:
        self.cleaned_payloads.append(dict(payload))

    def export_created_entity(self, payload: Dict[str, Any], memory_record: Dict[str, Any]) -> str:
        if not self._export_created:
            return ""
        self.exported_records.append({"payload": dict(payload), "memory_record": dict(memory_record)})
        return "exported"


class _FakeMemoryManager:
    def __init__(self, candidates: Sequence[Dict[str, Any]] | None = None) -> None:
        self._candidates = list(candidates or [])
        self.embed_calls: List[Dict[str, Any]] = []
        self.search_calls: List[Dict[str, Any]] = []
        self.add_calls: List[Dict[str, Any]] = []
        self.store_calls: List[Dict[str, Any]] = []

    async def embed_image(self, *, image_ref: str = "", image_b64: str = ""):
        self.embed_calls.append({"image_ref": image_ref, "image_b64": image_b64})
        return [0.0, 1.0], "vision-test"

    async def search_objects_by_appearance(self, embedding: List[float], top_k: int = 5):
        self.search_calls.append({"embedding": embedding, "top_k": top_k})
        return list(self._candidates)

    async def add_object_appearance(self, mem_id: str, appearance_embeddings, trace_id=None) -> int:
        self.add_calls.append({"mem_id": mem_id, "appearance_embeddings": list(appearance_embeddings)})
        return len(list(appearance_embeddings))

    async def store_object(
        self,
        name: str,
        data: Dict[str, Any],
        trace_id: str,
        *,
        confidence: float = 0.6,
        semantic_text: str | None = None,
        appearance_embeddings=None,
    ) -> str:
        self.store_calls.append(
            {
                "name": name,
                "data": data,
                "confidence": confidence,
                "semantic_text": semantic_text,
                "appearance_embeddings": list(appearance_embeddings or []),
            }
        )
        return "mem-new"


def _scene(track_id: int = 7) -> Dict[str, Any]:
    return {
        "ts": 123.0,
        "frame_size": {"width": 1280, "height": 720},
        "detections": [
            {
                "label": "cup",
                "confidence": 0.9,
                "bbox": [10, 20, 60, 70],
                "track_id": track_id,
            }
        ],
    }


class VisionRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_stable_track_matches_existing_object_memory(self) -> None:
        service = _FakeVisionService()
        memory = _FakeMemoryManager(
            candidates=[
                {
                    "mem_id": "mem-cup",
                    "name": "blue_cup",
                    "score": 0.93,
                    "data": {
                        "label": "cup",
                        "color_hist": [0.0, 1.0],
                        "aspect_ratio": 1.0,
                        "area_ratio": 0.1,
                        "zone": "center",
                    },
                }
            ]
        )
        registry = VisionEntityRegistry(service, memory)
        context = _FakeContext()

        await registry.update(context, object(), object(), _scene())
        scene = await registry.update(context, object(), object(), _scene())

        self.assertEqual(len(memory.embed_calls), 1)
        self.assertEqual(len(memory.add_calls), 1)
        self.assertEqual(memory.add_calls[0]["mem_id"], "mem-cup")
        self.assertEqual(scene["entities"][0]["object_mem_id"], "mem-cup")
        self.assertEqual(scene["entities"][0]["object_name"], "blue_cup")
        self.assertEqual(context.state[VISION_TRACKED_ENTITIES_KEY][0]["object_mem_id"], "mem-cup")
        self.assertEqual(service.cleaned_payloads, [{"image_b64": "ZmFrZQ=="}])

        await registry.clear(context)
        self.assertIn(VISION_TRACKED_ENTITIES_KEY, context.deleted)

    async def test_unmatched_stable_track_can_be_promoted_to_new_object_memory(self) -> None:
        service = _FakeVisionService(auto_store=True)
        memory = _FakeMemoryManager(candidates=[])
        registry = VisionEntityRegistry(service, memory)
        context = _FakeContext()

        await registry.update(context, object(), object(), _scene())
        scene = await registry.update(context, object(), object(), _scene())

        self.assertEqual(len(memory.embed_calls), 1)
        self.assertEqual(len(memory.search_calls), 1)
        self.assertEqual(len(memory.store_calls), 1)
        self.assertEqual(memory.store_calls[0]["name"], "vision.object.cup")
        self.assertEqual(scene["entities"][0]["object_mem_id"], "mem-new")
        self.assertEqual(scene["entities"][0]["match_reason"], "stored")
        self.assertEqual(service.cleaned_payloads, [{"image_b64": "ZmFrZQ=="}])
        self.assertEqual(service.exported_records, [])


class VisionServiceExportTests(unittest.TestCase):
    def test_export_created_entity_writes_crop_and_memory_json(self) -> None:
        from voice_daemon.vision_service import YoloVisionService

        with tempfile.TemporaryDirectory() as tmpdir:
            service = YoloVisionService(
                {
                    "object_memory_export_created_entities": True,
                    "object_memory_export_dir": tmpdir,
                }
            )

            export_dir = service.export_created_entity(
                {"_encoded_bytes": b"fake-jpeg"},
                {
                    "mem_id": "mem-created",
                    "type": "object",
                    "name": "vision.object.cup",
                    "data": {"label": "cup"},
                },
            )

            entity_dir = Path(export_dir)
            self.assertEqual(entity_dir.name, "mem-created")
            self.assertTrue((entity_dir / "crop.jpg").exists())
            self.assertTrue((entity_dir / "memory.json").exists())
            self.assertEqual((entity_dir / "crop.jpg").read_bytes(), b"fake-jpeg")
            self.assertIn("vision.object.cup", (entity_dir / "memory.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
