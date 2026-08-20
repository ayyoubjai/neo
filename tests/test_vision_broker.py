from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.vision_broker import VisionBroker, load_vision_broker_manifest


class _FakeCv2:
    IMWRITE_JPEG_QUALITY = 1

    def imwrite(self, path, frame, params=None):
        Path(path).write_bytes(str(frame.get("id", "frame")).encode("ascii"))
        return True


class VisionBrokerTests(unittest.TestCase):
    def test_broker_persists_latest_stable_and_prunes_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = VisionBroker(
                str(root),
                str(root / "data"),
                {
                    "broker_enabled": True,
                    "broker_dir": str(root / "data" / "vision_broker"),
                    "broker_latest_interval_ms": 0,
                    "broker_buffer_frame_interval_ms": 0,
                    "broker_buffer_retention_s": 1.0,
                    "broker_buffer_max_frames": 2,
                },
            )
            cv2 = _FakeCv2()
            scene = {"summary": "person", "objects": [{"label": "person", "count": 1}]}

            with patch("common.vision_broker.time.time", side_effect=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]):
                broker.publish_latest(cv2, {"id": "frame-1"}, scene)
                broker.publish_latest(cv2, {"id": "frame-2"}, scene)
                broker.publish_latest(cv2, {"id": "frame-3"}, scene)
                broker.publish_stable(cv2, {"id": "stable-1"}, scene)
                broker.close()

            manifest = load_vision_broker_manifest(str(root / "data" / "vision_broker" / "manifest.json"))
            self.assertEqual(manifest["status"], "stopped")
            self.assertEqual(manifest["latest"]["scene"]["summary"], "person")
            self.assertEqual(manifest["stable"]["scene"]["summary"], "person")
            self.assertEqual(len(manifest["buffer"]["frames"]), 2)
            frame_ids = [item["frame_id"] for item in manifest["buffer"]["frames"]]
            self.assertTrue(all(frame_id.startswith("buffer_") for frame_id in frame_ids))
            latest_path = root / manifest["latest"]["image_ref"].replace("workspace:/", "")
            stable_path = root / manifest["stable"]["image_ref"].replace("workspace:/", "")
            self.assertTrue(latest_path.exists())
            self.assertTrue(stable_path.exists())
