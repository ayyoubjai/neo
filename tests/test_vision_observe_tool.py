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

from common.config import Settings
from tool_runtime.tools import vision_observe


def _settings(workspace_root: str, broker_dir: str) -> Settings:
    return Settings(
        workspace_root=workspace_root,
        data_dir=str(Path(workspace_root) / "data"),
        rpc={},
        interface={"mode": "local"},
        voice={},
        vision={"broker_dir": broker_dir},
        video={},
        telegram={},
        google={},
        tool={"active_tool_limit": 100, "planner_tool_profile": "initial"},
        search={},
        orchestrator={},
        evolve={"candidate_dir": str(Path(workspace_root) / "evolve")},
        models={},
    )


class VisionObserveToolTests(unittest.TestCase):
    def test_now_scope_uses_single_frame_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker_dir = root / "data" / "vision_broker"
            settings = _settings(tmpdir, str(broker_dir))
            manifest = {
                "updated_ts": 10.0,
                "latest": {
                    "frame_id": "latest_1",
                    "ts": 10.0,
                    "image_ref": "workspace:/data/vision_broker/frames/latest.jpg",
                    "scene": {"summary": "screen"},
                },
                "stable": None,
                "buffer": {"frames": []},
            }

            with patch("tool_runtime.tools.load_settings", return_value=settings), patch(
                "tool_runtime.tools.load_vision_broker_manifest_for_settings",
                return_value=manifest,
            ), patch(
                "tool_runtime.tools.analyze",
                return_value={
                    "ok": True,
                    "summary": "login screen",
                    "answer": "The screen shows a login form.",
                    "text": "Username Password",
                    "objects": [{"name": "form", "count": 1}],
                },
            ):
                result, io = vision_observe({"question": "What is on the screen?", "time_scope": "now"}, tmpdir)

            self.assertEqual(io, {})
            self.assertTrue(result["ok"])
            self.assertEqual(result["summary"], "login screen")
            self.assertEqual(result["text"], "Username Password")
            self.assertEqual(len(result["frames"]), 1)
            self.assertEqual(result["scene"]["summary"], "screen")

    def test_recent_scope_uses_video_analysis_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker_dir = root / "data" / "vision_broker"
            settings = _settings(tmpdir, str(broker_dir))
            manifest = {
                "updated_ts": 20.0,
                "latest": {
                    "frame_id": "latest_3",
                    "ts": 20.0,
                    "image_ref": "workspace:/data/vision_broker/frames/frame3.jpg",
                    "scene": {"summary": "person turns"},
                },
                "stable": None,
                "buffer": {
                    "frames": [
                        {"frame_id": "buffer_1", "ts": 16.0, "image_ref": "workspace:/data/vision_broker/frames/frame1.jpg", "scene": {"summary": "person seated"}},
                        {"frame_id": "buffer_2", "ts": 18.0, "image_ref": "workspace:/data/vision_broker/frames/frame2.jpg", "scene": {"summary": "person stands"}},
                        {"frame_id": "buffer_3", "ts": 20.0, "image_ref": "workspace:/data/vision_broker/frames/frame3.jpg", "scene": {"summary": "person turns"}},
                    ]
                },
            }

            with patch("tool_runtime.tools.load_settings", return_value=settings), patch(
                "tool_runtime.tools.load_vision_broker_manifest_for_settings",
                return_value=manifest,
            ), patch(
                "tool_runtime.tools._vision_export_clip_from_frames",
                return_value=("workspace:/data/vision_broker/clips/recent.avi", str(root / "data" / "vision_broker" / "clips" / "recent.avi"), False),
            ), patch(
                "tool_runtime.tools.analyze_video",
                return_value={
                    "ok": True,
                    "summary": "person stands up and turns",
                    "answer": "A person stands up and turns.",
                    "text": "EXIT",
                    "objects": [{"name": "person", "count": 1}],
                    "keyframes": [{"timestamp_s": 0.0, "summary": "person seated"}, {"timestamp_s": 2.0, "summary": "person stands"}],
                    "scenes": [],
                    "events": [],
                    "duration_s": 4.0,
                },
            ):
                result, _ = vision_observe(
                    {"question": "What just happened?", "time_scope": "recent", "lookback_s": 6, "sample_count": 3},
                    tmpdir,
                )

            self.assertEqual(result["analysis_mode"], "video")
            self.assertEqual(result["summary"], "person stands up and turns")
            self.assertEqual(result["answer"], "A person stands up and turns.")
            self.assertIn("EXIT", result["text"])
            self.assertEqual(len(result["frames"]), 2)
            self.assertEqual(result["video_ref"], "workspace:/data/vision_broker/clips/recent.avi")

    def test_recent_scope_can_force_image_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker_dir = root / "data" / "vision_broker"
            settings = _settings(tmpdir, str(broker_dir))
            manifest = {
                "updated_ts": 20.0,
                "latest": {
                    "frame_id": "latest_3",
                    "ts": 20.0,
                    "image_ref": "workspace:/data/vision_broker/frames/frame3.jpg",
                    "scene": {"summary": "person turns"},
                },
                "stable": None,
                "buffer": {
                    "frames": [
                        {"frame_id": "buffer_1", "ts": 16.0, "image_ref": "workspace:/data/vision_broker/frames/frame1.jpg", "scene": {"summary": "person seated"}},
                        {"frame_id": "buffer_2", "ts": 18.0, "image_ref": "workspace:/data/vision_broker/frames/frame2.jpg", "scene": {"summary": "person stands"}},
                        {"frame_id": "buffer_3", "ts": 20.0, "image_ref": "workspace:/data/vision_broker/frames/frame3.jpg", "scene": {"summary": "person turns"}},
                    ]
                },
            }

            def _analyze(image_ref, question=None):
                if image_ref.endswith("frame1.jpg"):
                    return {"ok": True, "summary": "person seated", "answer": "", "text": "", "objects": [{"name": "person", "count": 1}]}
                if image_ref.endswith("frame2.jpg"):
                    return {"ok": True, "summary": "person stands", "answer": "", "text": "EXIT", "objects": [{"name": "person", "count": 1}]}
                return {"ok": True, "summary": "person turns", "answer": "", "text": "", "objects": [{"name": "person", "count": 1}]}

            with patch("tool_runtime.tools.load_settings", return_value=settings), patch(
                "tool_runtime.tools.load_vision_broker_manifest_for_settings",
                return_value=manifest,
            ), patch("tool_runtime.tools.analyze", side_effect=_analyze), patch(
                "tool_runtime.tools.answer_from_frame_sequence",
                return_value="A person stands up and turns.",
            ):
                result, _ = vision_observe(
                    {
                        "question": "What just happened?",
                        "time_scope": "recent",
                        "analysis_mode": "image",
                        "lookback_s": 6,
                        "sample_count": 3,
                    },
                    tmpdir,
                )

            self.assertEqual(result["analysis_mode"], "image")
            self.assertEqual(result["summary"], "person seated; person stands; person turns")
            self.assertEqual(result["answer"], "A person stands up and turns.")
            self.assertIn("EXIT", result["text"])
            self.assertEqual(len(result["frames"]), 3)
