from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from storage.backend import StorageBackend
from storage.db import connect, init_db


class StorageEmbeddingTests(unittest.TestCase):
    def test_search_memory_uses_modality_and_purpose_specific_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "storage.db"
            audit_path = Path(tmp_dir) / "audit.log"
            conn = connect(str(db_path))
            init_db(conn)
            backend = StorageBackend(conn, str(audit_path))

            backend.write_memory(
                {
                    "mem_id": "obj-1",
                    "type": "object",
                    "name": "red mug",
                    "data": {"label": "cup"},
                    "created_at": "2026-03-12T00:00:00Z",
                    "embeddings": [
                        {
                            "modality": "text",
                            "purpose": "semantic",
                            "model": "text-test",
                            "embedding": [1.0, 0.0],
                        },
                        {
                            "modality": "vision",
                            "purpose": "appearance",
                            "model": "vision-test",
                            "embedding": [0.0, 1.0],
                        },
                        {
                            "modality": "vision",
                            "purpose": "appearance",
                            "model": "vision-test",
                            "embedding": [0.2, 0.98],
                        },
                    ],
                }
            )

            text_results = backend.search_memory([1.0, 0.0], "object", 5, modality="text", purpose="semantic")
            self.assertEqual(len(text_results), 1)
            self.assertEqual(text_results[0]["mem_id"], "obj-1")
            self.assertEqual(text_results[0]["matched_embedding"]["modality"], "text")
            self.assertEqual(text_results[0]["matched_embedding"]["purpose"], "semantic")
            self.assertAlmostEqual(float(text_results[0]["score"]), 1.0, places=6)

            vision_results = backend.search_memory([0.0, 1.0], "object", 5, modality="vision", purpose="appearance")
            self.assertEqual(len(vision_results), 1)
            self.assertEqual(vision_results[0]["mem_id"], "obj-1")
            self.assertEqual(vision_results[0]["matched_embedding"]["modality"], "vision")
            self.assertEqual(vision_results[0]["matched_embedding"]["purpose"], "appearance")
            self.assertGreater(float(vision_results[0]["score"]), 0.99)

    def test_search_memory_falls_back_to_primary_embedding_for_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "storage.db"
            audit_path = Path(tmp_dir) / "audit.log"
            conn = connect(str(db_path))
            init_db(conn)
            backend = StorageBackend(conn, str(audit_path))

            backend.write_memory(
                {
                    "mem_id": "legacy-1",
                    "type": "fact",
                    "name": "legacy.name",
                    "data": {"value": "atlas"},
                    "embedding": [1.0, 0.0],
                    "created_at": "2026-03-12T00:00:00Z",
                }
            )

            results = backend.search_memory([1.0, 0.0], "fact", 5, modality="text", purpose="semantic")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["mem_id"], "legacy-1")
            self.assertEqual(results[0]["matched_embedding"]["model"], "memory.primary")


if __name__ == "__main__":
    unittest.main()
