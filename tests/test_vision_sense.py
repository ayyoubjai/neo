from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_daemon.vision_context import VISION_LATEST_SCENE_KEY, VISION_STABLE_SCENE_KEY
from voice_daemon.vision_sense import VisionSense


class _FakeCapture:
    def __init__(self) -> None:
        self.released = False
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return True, {"frame": self.read_count}

    def release(self) -> None:
        self.released = True


class _FakeVisionService:
    def __init__(self, *, auto_submit: bool = False) -> None:
        self._capture = _FakeCapture()
        self._auto_submit = auto_submit
        self.logged_scenes: List[Dict[str, Any]] = []
        self.preview_calls = 0
        self.preview_closed = False
        self._scenes = [
            {
                "summary": "person, laptop",
                "objects": [{"label": "person", "count": 1}, {"label": "laptop", "count": 1}],
                "signature": (("laptop", 1), ("person", 1)),
            },
            {
                "summary": "person, laptop",
                "objects": [{"label": "person", "count": 1}, {"label": "laptop", "count": 1}],
                "signature": (("laptop", 1), ("person", 1)),
            },
        ]

    def load_runtime(self):
        return object(), object()

    def load_model(self, yolo_cls):
        return "model"

    def open_camera(self, cv2):
        return self._capture

    def detect_scene(self, model, frame) -> Dict[str, Any]:
        index = min(self._capture.read_count - 1, len(self._scenes) - 1)
        return dict(self._scenes[index])

    def show_preview(self, cv2, frame, scene) -> bool:
        self.preview_calls += 1
        return True

    def close_preview(self, cv2) -> None:
        self.preview_closed = True

    def scene_signature(self, scene) -> Any:
        return scene["signature"]

    def capture_interval_s(self) -> float:
        return 0.0

    def min_stable_frames(self) -> int:
        return 2

    def scene_cooldown_s(self) -> float:
        return 0.0

    def log_scene(self, scene) -> None:
        self.logged_scenes.append(dict(scene))

    def print_scene_changes(self) -> bool:
        return False

    def auto_submit_scene_changes(self) -> bool:
        return self._auto_submit

    def submit_requested_mode(self) -> str:
        return "COGNITION"

    def turn_text(self, scene) -> str:
        return f"Local camera observation: {scene['summary']}"


class _FakeContext:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.state_history: List[tuple[str, Any]] = []
        self.deleted: List[str] = []
        self.turns: List[tuple[str, str | None]] = []
        self.stable_scene_ready = asyncio.Event()

    async def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value
        self.state_history.append((key, value))
        if key == VISION_STABLE_SCENE_KEY:
            self.stable_scene_ready.set()

    async def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    async def delete_state(self, key: str) -> None:
        self.deleted.append(key)
        self.state.pop(key, None)

    async def submit_turn(self, text: str, *, requested_mode: str | None = None, turn_id: str | None = None) -> str:
        self.turns.append((text, requested_mode))
        return "turn-1"


class _FakeBroker:
    def __init__(self) -> None:
        self.enabled = True
        self.latest_calls: List[Dict[str, Any]] = []
        self.stable_calls: List[Dict[str, Any]] = []
        self.closed = False

    def publish_latest(self, cv2, frame, scene) -> None:
        self.latest_calls.append({"frame": frame, "scene": dict(scene)})

    def publish_stable(self, cv2, frame, scene) -> None:
        self.stable_calls.append({"frame": frame, "scene": dict(scene)})

    def close(self) -> None:
        self.closed = True


class VisionSenseTests(unittest.IsolatedAsyncioTestCase):
    async def test_promotes_stable_scene_and_cleans_up_state(self) -> None:
        service = _FakeVisionService()
        context = _FakeContext()
        broker = _FakeBroker()
        task = asyncio.create_task(VisionSense(service, broker=broker).run(context))

        await asyncio.wait_for(context.stable_scene_ready.wait(), timeout=1.0)
        for _ in range(20):
            if service.logged_scenes:
                break
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        stable_entries = [value for key, value in context.state_history if key == VISION_STABLE_SCENE_KEY]
        self.assertEqual(len(stable_entries), 1)
        self.assertEqual(stable_entries[0]["summary"], "person, laptop")
        self.assertEqual(service.logged_scenes[0]["summary"], "person, laptop")
        self.assertIn(VISION_LATEST_SCENE_KEY, context.deleted)
        self.assertIn(VISION_STABLE_SCENE_KEY, context.deleted)
        self.assertTrue(service._capture.released)
        self.assertGreaterEqual(service.preview_calls, 1)
        self.assertTrue(service.preview_closed)
        self.assertGreaterEqual(len(broker.latest_calls), 1)
        self.assertEqual(len(broker.stable_calls), 1)
        self.assertTrue(broker.closed)

    async def test_can_auto_submit_scene_changes(self) -> None:
        service = _FakeVisionService(auto_submit=True)
        context = _FakeContext()
        task = asyncio.create_task(VisionSense(service).run(context))

        await asyncio.wait_for(context.stable_scene_ready.wait(), timeout=1.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(context.turns, [("Local camera observation: person, laptop", "COGNITION")])


if __name__ == "__main__":
    unittest.main()
