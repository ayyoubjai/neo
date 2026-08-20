from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_daemon.vision_context import (
    format_scene_context,
    maybe_augment_turn_with_vision_context,
    wants_vision_context,
)
from voice_daemon.vision_service import YoloVisionService
from runtime_core.runtime import RuntimeContext, RuntimeState


class _FakeBoxes:
    def __init__(self, cls, conf, xyxy, ids=None):
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy
        self.id = ids if ids is not None else []


class _FakeResult:
    def __init__(self):
        self.names = {0: "person", 1: "laptop"}
        self.boxes = _FakeBoxes(
            cls=[0, 1, 1],
            conf=[0.9, 0.8, 0.75],
            xyxy=[[0, 0, 10, 10], [5, 5, 20, 20], [15, 15, 30, 30]],
            ids=[7, 8, 9],
        )


class _FakeSession:
    last_turn_id = None
    pending_permissions = {}

    def resolve_permission_id(self, token=None):
        raise AssertionError("not used")


class _FakePreviewFrame:
    def __init__(self) -> None:
        self.copied = False

    def copy(self):
        duplicate = _FakePreviewFrame()
        duplicate.copied = True
        return duplicate


class _FakeCrop:
    def __init__(self, width: int, height: int) -> None:
        self.shape = (height, width, 3)


class _FakeEncodedBytes:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class _FakeImageFrame:
    def __init__(self, width: int, height: int) -> None:
        self.shape = (height, width, 3)

    def __getitem__(self, key):
        yslice, xslice = key[:2]
        width = max(1, int((xslice.stop or 0) - (xslice.start or 0)))
        height = max(1, int((yslice.stop or 0) - (yslice.start or 0)))
        return _FakeCrop(width, height)


class _FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    IMWRITE_JPEG_QUALITY = 1
    INTER_AREA = 3

    def __init__(self) -> None:
        self.rectangles = []
        self.texts = []
        self.resize_calls = []
        self.imencode_calls = []

    def rectangle(self, frame, pt1, pt2, color, thickness) -> None:
        self.rectangles.append((pt1, pt2, color, thickness))

    def putText(self, frame, text, origin, font, scale, color, thickness, line_type) -> None:
        self.texts.append((text, origin, color, thickness, line_type))

    def resize(self, crop, size, interpolation=None):
        self.resize_calls.append((crop.shape, size, interpolation))
        return _FakeCrop(size[0], size[1])

    def imencode(self, ext, crop, params=None):
        self.imencode_calls.append((ext, crop.shape, params))
        payload = f"{crop.shape[1]}x{crop.shape[0]}".encode("ascii")
        return True, _FakeEncodedBytes(payload)


class VisionServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_scene_from_results_builds_summary_and_signature(self) -> None:
        service = YoloVisionService({"summary_max_objects": 8})

        scene = service.scene_from_results([_FakeResult()], frame_shape=(720, 1280, 3))

        self.assertEqual(scene["summary"], "2 laptops, person")
        self.assertEqual(scene["signature"], (("laptop", 2), ("person", 1)))
        self.assertEqual(scene["frame_size"], {"height": 720, "width": 1280})
        self.assertEqual(scene["detections"][0]["track_id"], 7)
        self.assertEqual(scene["detections"][1]["track_id"], 8)

    async def test_query_turns_can_be_augmented_with_latest_stable_scene(self) -> None:
        context = RuntimeContext(_FakeSession(), RuntimeState())
        await context.set_state(
            "vision.stable_scene",
            {
                "summary": "laptop, person",
                "objects": [{"label": "laptop", "count": 1}, {"label": "person", "count": 1}],
            },
        )

        augmented = await maybe_augment_turn_with_vision_context(context, "what do you see right now")

        self.assertIn("[Local camera scene]", augmented)
        self.assertIn("Summary: laptop, person", augmented)
        self.assertIn("Objects: 1x laptop, 1x person", augmented)

    def test_query_helpers_detect_and_format_scene_context(self) -> None:
        self.assertTrue(wants_vision_context("Please look around and tell me what do you see"))
        self.assertEqual(
            format_scene_context({"summary": "person", "objects": [{"label": "person", "count": 1}]}),
            "Summary: person\nObjects: 1x person",
        )

    def test_render_preview_frame_draws_boxes_labels_confidence_and_summary(self) -> None:
        service = YoloVisionService(
            {
                "preview_enabled": True,
                "preview_show_boxes": True,
                "preview_show_labels": True,
                "preview_show_confidence": True,
                "preview_show_summary": True,
            }
        )
        cv2 = _FakeCv2()
        frame = _FakePreviewFrame()
        scene = {
            "summary": "person",
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.9123,
                    "bbox": [10, 20, 30, 40],
                    "track_id": 7,
                }
            ],
        }

        rendered = service.render_preview_frame(cv2, frame, scene)

        self.assertTrue(getattr(rendered, "copied", False))
        self.assertEqual(len(cv2.rectangles), 1)
        self.assertTrue(any(text.startswith("#7 person 0.91") for text, *_ in cv2.texts))
        self.assertTrue(any(text.startswith("Objects: person") for text, *_ in cv2.texts))

    def test_render_preview_frame_shows_memory_match_name_for_tracked_entity(self) -> None:
        service = YoloVisionService(
            {
                "preview_enabled": True,
                "preview_show_boxes": True,
                "preview_show_labels": True,
                "preview_show_confidence": True,
                "preview_show_summary": False,
            }
        )
        cv2 = _FakeCv2()
        frame = _FakePreviewFrame()
        scene = {
            "detections": [
                {
                    "label": "cup",
                    "confidence": 0.876,
                    "bbox": [10, 20, 30, 40],
                    "track_id": 7,
                }
            ],
            "entities": [
                {
                    "track_id": "7",
                    "object_mem_id": "mem-cup-1",
                    "object_name": "blue_cup",
                    "match_reason": "matched",
                }
            ],
        }

        rendered = service.render_preview_frame(cv2, frame, scene)

        self.assertTrue(getattr(rendered, "copied", False))
        self.assertTrue(any(text.startswith("#7 cup 0.88 -> blue_cup") for text, *_ in cv2.texts))
        self.assertEqual(cv2.rectangles[0][2], (96, 220, 255))

    def test_build_embedding_input_base64_resizes_and_compresses_crop(self) -> None:
        service = YoloVisionService(
            {
                "object_memory_embed_transport": "base64",
                "object_memory_crop_max_width": 64,
                "object_memory_crop_max_height": 64,
                "object_memory_crop_jpeg_quality": 70,
            }
        )
        cv2 = _FakeCv2()
        frame = _FakeImageFrame(320, 240)

        payload = service.build_embedding_input(
            cv2,
            frame,
            {"bbox": [0, 0, 160, 80]},
        )

        self.assertIn("image_b64", payload)
        self.assertNotIn("image_ref", payload)
        self.assertEqual(cv2.resize_calls[0][1], (64, 32))
        self.assertEqual(cv2.imencode_calls[0][2], [cv2.IMWRITE_JPEG_QUALITY, 70])

    def test_build_embedding_input_temp_file_writes_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = YoloVisionService(
                {
                    "object_memory_embed_transport": "temp_file",
                    "object_memory_crop_temp_dir": tmpdir,
                    "object_memory_crop_keep_temp_files": False,
                }
            )
            cv2 = _FakeCv2()
            frame = _FakeImageFrame(320, 240)

            payload = service.build_embedding_input(
                cv2,
                frame,
                {"bbox": [0, 0, 80, 80]},
            )

            image_ref = payload.get("image_ref")
            self.assertTrue(image_ref)
            self.assertTrue(Path(str(image_ref)).exists())
            self.assertEqual(payload.get("_cleanup_path"), image_ref)

            service.cleanup_embedding_input(payload)

            self.assertFalse(Path(str(image_ref)).exists())


if __name__ == "__main__":
    unittest.main()
