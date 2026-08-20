from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.system_entity import SystemEntity, detect_wake_word
from voice_daemon.audio_controller import AudioControllerConfig, VoiceAudioController


class _FakeContext:
    def __init__(self) -> None:
        self.turns: List[str] = []
        self.wakes: List[Dict[str, Any]] = []

    async def submit_turn(self, text: str) -> str:
        self.turns.append(text)
        return f"turn-{len(self.turns)}"

    async def send_wake(self, payload=None) -> None:
        self.wakes.append(dict(payload or {}))


class _FakeDelegate:
    def __init__(self) -> None:
        self._config = AudioControllerConfig(
            voice={"frame_ms": 30, "wake_mode": "stt_prefix"},
            debug_audio=False,
            debug_audio_interval_ms=1000,
            no_speech_prob_threshold=None,
        )
        self._wake_mode = "stt_prefix"
        self._system_entity = SystemEntity(
            mem_id="system.object",
            name="Atlas",
            aliases=["atlas"],
            properties={},
            pins=[],
        )
        self._listening_mode = "wake"
        self._segment_counter = 0
        self._last_user_text = ""
        self._last_user_time = 0.0
        self.activation_count = 0
        self.sleep_reasons: List[str] = []
        self.user_transcripts: List[Tuple[str, str, bool]] = []
        self.listening_mode_changes: List[str] = []
        self.transcript_logs: List[Dict[str, Any]] = []
        self.echo_logs: List[Tuple[str, Dict[str, Any]]] = []

    @property
    def config(self) -> AudioControllerConfig:
        return self._config

    def maybe_sleep_for_timeout(self, now: float) -> None:
        return None

    def should_suppress_capture(self, now: float) -> bool:
        return False

    def note_conversation_activity(self, now: float) -> None:
        return None

    def listening_mode(self) -> str:
        return self._listening_mode

    def set_wake_mode(self, mode: str) -> None:
        self._wake_mode = mode

    def wake_mode(self) -> Optional[str]:
        return self._wake_mode

    def init_wake_model(self, openwakeword, wake_model_cls, wake_models: List[Any]):
        return None

    def open_input_stream(self, sd, sample_rate: int, frame_samples: int, callback):
        raise AssertionError("open_input_stream should not be called in this test")

    def print_stream_info(self, sd, stream) -> None:
        return None

    def predict_wake(self, wake_model, audio_int16, audio_float, wake_use_float: Optional[bool], threshold: float):
        return None

    def on_activation(self) -> None:
        self.activation_count += 1

    def on_sleep(self, reason: str) -> None:
        self.sleep_reasons.append(reason)

    def log_audio_debug(self, avg_level: float, is_speech: bool, capture_active: bool) -> None:
        return None

    def next_segment_id(self) -> int:
        self._segment_counter += 1
        return self._segment_counter

    def normalize_spoken_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def log_echo_debug(self, event: str, payload: Dict[str, Any]) -> None:
        self.echo_logs.append((event, payload))

    def filter_echo_text(self, text: str, segment_id: Optional[int] = None) -> Optional[str]:
        return text

    def is_sleep_command(self, text: str) -> bool:
        return False

    def print_user_transcript(self, original_text: str, final_text: str, accepted: bool) -> None:
        self.user_transcripts.append((original_text, final_text, accepted))

    def log_transcript(
        self,
        original_text: str,
        final_text: str,
        wake_mode: str,
        wake_hint: bool,
        is_wake: bool,
        accepted: bool,
    ) -> None:
        self.transcript_logs.append(
            {
                "original_text": original_text,
                "final_text": final_text,
                "wake_mode": wake_mode,
                "wake_hint": wake_hint,
                "is_wake": is_wake,
                "accepted": accepted,
            }
        )

    async def handle_voice_approval(self, text: str, context) -> bool:
        return False

    def detect_wake_phrase(self, text: str) -> Tuple[bool, str]:
        return detect_wake_word(text, self._system_entity)

    def wake_parts(self, text: str) -> Tuple[bool, str, str]:
        is_wake, remainder = detect_wake_word(text, self._system_entity)
        return is_wake, remainder, remainder

    def set_listening_mode(self, mode: str, note: Optional[str] = None) -> None:
        self._listening_mode = mode
        self.listening_mode_changes.append(mode)

    def record_user_turn(self, text: str, at: float) -> None:
        self._last_user_text = text
        self._last_user_time = at


class VoiceAudioControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_finalize_audio_segment_enqueues_segment_metadata(self) -> None:
        delegate = _FakeDelegate()
        controller = VoiceAudioController(delegate, sd=None, np=None, webrtcvad=None, whisper_model_cls=None)
        stt_queue: asyncio.Queue = asyncio.Queue()

        await controller._finalize_audio_segment(stt_queue, [b"a", b"b"], 16000, True)

        item = await asyncio.wait_for(stt_queue.get(), timeout=1.0)
        self.assertEqual(delegate._segment_counter, 1)
        self.assertEqual(item["sample_rate"], 16000)
        self.assertEqual(item["frames"], [b"a", b"b"])
        self.assertTrue(item["wake_hint"])
        self.assertEqual(item["duration_ms"], 60)
        self.assertEqual(item["segment_id"], 1)

    async def test_stt_worker_routes_wake_prefixed_text_to_runtime_turn(self) -> None:
        delegate = _FakeDelegate()
        controller = VoiceAudioController(delegate, sd=None, np=None, webrtcvad=None, whisper_model_cls=None)
        controller._transcribe = lambda stt_model, frames, sample_rate: "Atlas check status"
        context = _FakeContext()
        stt_queue: asyncio.Queue = asyncio.Queue()
        await stt_queue.put(
            {
                "frames": [b"pcm"],
                "sample_rate": 16000,
                "wake_hint": False,
                "segment_id": 7,
                "duration_ms": 90,
            }
        )
        await stt_queue.put(None)

        async def _to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("voice_daemon.audio_controller.asyncio.to_thread", _to_thread):
            await controller._stt_worker(context, stt_queue, stt_model=object())

        self.assertEqual(context.turns, ["check status"])
        self.assertEqual(len(context.wakes), 1)
        self.assertEqual(context.wakes[0]["wake_word"], "Atlas check status")
        self.assertEqual(delegate.activation_count, 1)
        self.assertEqual(delegate._listening_mode, "conversation")
        self.assertIn("conversation", delegate.listening_mode_changes)
        self.assertEqual(delegate._last_user_text, "check status")
        self.assertEqual(delegate.user_transcripts[-1], ("Atlas check status", "check status", True))

    def test_trim_trailing_silence_keeps_tail_after_last_speech_frame(self) -> None:
        delegate = _FakeDelegate()
        controller = VoiceAudioController(delegate, sd=None, np=None, webrtcvad=None, whisper_model_cls=None)

        trimmed = controller._trim_trailing_silence([b"a", b"b", b"c", b"d"], last_speech_idx=1, tail_frames=1)

        self.assertEqual(trimmed, [b"a", b"b", b"c"])


if __name__ == "__main__":
    unittest.main()
