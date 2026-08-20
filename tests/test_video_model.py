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

from model_server.frame_sequence import answer_from_keyframes
from model_server.video_model import _extract_audio_to_wav, build_sampling_policy
from tool_runtime.tools import ToolError, video_analyse


class VideoSamplingPolicyTests(unittest.TestCase):
    def test_policy_increases_spacing_for_longer_videos(self) -> None:
        short_policy = build_sampling_policy(12.0, question="")
        long_policy = build_sampling_policy(900.0, question="")

        self.assertLessEqual(short_policy["baseline_frames"], long_policy["baseline_frames"])
        self.assertLess(short_policy["baseline_spacing_s"], long_policy["baseline_spacing_s"])
        self.assertLessEqual(long_policy["baseline_frames"], 24)
        self.assertLessEqual(long_policy["max_total_frames"], 32)

    def test_policy_boosts_budget_for_text_focused_questions(self) -> None:
        plain_policy = build_sampling_policy(60.0, question="what happens in the clip")
        text_policy = build_sampling_policy(60.0, question="read the text shown on screen")

        self.assertGreaterEqual(text_policy["baseline_frames"], plain_policy["baseline_frames"])
        self.assertLessEqual(text_policy["min_timestamp_gap_s"], plain_policy["min_timestamp_gap_s"])


class VideoToolTests(unittest.TestCase):
    def test_video_analyse_normalizes_workspace_path_and_forwards_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "sample.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"video")

            with patch(
                "tool_runtime.tools.analyze_video",
                return_value={
                    "ok": True,
                    "video_ref": "workspace:/videos/sample.mp4",
                    "duration_s": 12.0,
                    "summary": "summary",
                    "answer": "answer",
                    "text": "",
                    "transcript": "",
                    "objects": [],
                    "keyframes": [],
                    "scenes": [],
                    "events": [],
                },
            ) as mock_analyze:
                result, io = video_analyse(
                    {"path": str(video_path), "question": "what happens?", "start_s": 1, "end_s": "3.5"},
                    str(root),
                )

            self.assertEqual(io, {})
            self.assertEqual(result["summary"], "summary")
            self.assertEqual(mock_analyze.call_args.args[0], "workspace:/videos/sample.mp4")
            self.assertEqual(mock_analyze.call_args.kwargs["question"], "what happens?")
            self.assertEqual(mock_analyze.call_args.kwargs["start_s"], 1.0)
            self.assertEqual(mock_analyze.call_args.kwargs["end_s"], 3.5)

    def test_video_analyse_raises_tool_error_for_failed_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "broken.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"video")

            with patch(
                "tool_runtime.tools.analyze_video",
                return_value={"ok": False, "error": "Video analysis failed."},
            ):
                with self.assertRaisesRegex(ToolError, "Video analysis failed"):
                    video_analyse({"path": str(video_path)}, str(root))

    def test_extract_audio_to_wav_passes_clip_range_to_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample.mp4"
            video_path.write_bytes(b"video")
            audio_dir = Path(tmpdir) / "audio"
            audio_dir.mkdir()
            output_path = audio_dir / "audio.wav"

            def _run(command, capture_output, text, check):
                self.assertEqual(command[0], "ffmpeg")
                self.assertIn("-ss", command)
                self.assertIn("1.500", command)
                self.assertIn("-t", command)
                self.assertIn("3.000", command)
                self.assertIn("-vn", command)
                self.assertIn("-ac", command)
                self.assertIn("-ar", command)
                output_path.write_bytes(b"wav")

                class Result:
                    returncode = 0
                    stderr = ""
                    stdout = ""

                return Result()

            with patch("model_server.video_model.subprocess.run", side_effect=_run) as mock_run:
                result = _extract_audio_to_wav(str(video_path), str(audio_dir), "ffmpeg", 16000, 1.5, 4.5)

            self.assertEqual(result, str(output_path))
            self.assertEqual(mock_run.call_count, 1)

    def test_answer_prompt_includes_transcript_context(self) -> None:
        with patch("model_server.frame_sequence.generate_text", return_value="answer") as mock_generate:
            result = answer_from_keyframes(
                "what is said?",
                "person presenting",
                [{"timestamp_s": 1.0, "summary": "person at desk"}],
                evidence_label="video",
                duration_s=12.0,
                transcript="hello world from the speaker",
            )

        self.assertEqual(result, "answer")
        prompt = mock_generate.call_args.args[0]
        self.assertIn("Transcript excerpt", prompt)
        self.assertIn("hello world from the speaker", prompt)


if __name__ == "__main__":
    unittest.main()
