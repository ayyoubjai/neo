from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from common.system_entity import detect_wake_word
from runtime_core.runtime import RuntimeContext
from voice_daemon.audio_controller import AudioControllerConfig, AudioControllerDelegate

if TYPE_CHECKING:
    from voice_daemon.main import VoiceDaemon


class VoiceDaemonAudioDelegate(AudioControllerDelegate):
    def __init__(self, daemon: "VoiceDaemon"):
        self._daemon = daemon

    @property
    def config(self) -> AudioControllerConfig:
        return AudioControllerConfig(
            voice=self._daemon._voice_cfg,
            debug_audio=self._daemon._debug_audio,
            debug_audio_interval_ms=self._daemon._debug_audio_interval_ms,
            no_speech_prob_threshold=self._daemon._no_speech_prob_threshold,
        )

    def maybe_sleep_for_timeout(self, now: float) -> None:
        if self._daemon._listening_mode != "conversation":
            return
        timeout_s = self._daemon._conversation_timeout_s
        if timeout_s <= 0:
            return
        if now - self._daemon._conversation_last_activity >= timeout_s:
            self._daemon._on_sleep("timeout")

    def should_suppress_capture(self, now: float) -> bool:
        return self._daemon._output.should_suppress_capture(now)

    def note_conversation_activity(self, now: float) -> None:
        if self._daemon._listening_mode == "conversation":
            self._daemon._conversation_last_activity = now

    def listening_mode(self) -> str:
        return self._daemon._listening_mode

    def set_wake_mode(self, mode: str) -> None:
        self._daemon._wake_mode_current = mode

    def wake_mode(self) -> Optional[str]:
        return self._daemon._wake_mode_current

    def init_wake_model(self, openwakeword, wake_model_cls, wake_models: List[Any]):
        return self._daemon._input.init_wake_model(openwakeword, wake_model_cls, wake_models)

    def open_input_stream(self, sd, sample_rate: int, frame_samples: int, callback):
        return self._daemon._input.open_input_stream(sd, sample_rate, frame_samples, callback)

    def print_stream_info(self, sd, stream) -> None:
        self._daemon._input.print_stream_info(sd, stream)

    def predict_wake(self, wake_model, audio_int16, audio_float, wake_use_float: Optional[bool], threshold: float):
        return self._daemon._input.predict_wake(wake_model, audio_int16, audio_float, wake_use_float, threshold)

    def on_activation(self) -> None:
        self._daemon._on_activation()

    def on_sleep(self, reason: str) -> None:
        self._daemon._on_sleep(reason)

    def log_audio_debug(self, avg_level: float, is_speech: bool, capture_active: bool) -> None:
        self._daemon._input.log_audio_debug(avg_level, is_speech, capture_active)

    def next_segment_id(self) -> int:
        self._daemon._segment_counter += 1
        return self._daemon._segment_counter

    def log_echo_debug(self, event: str, payload: Dict[str, Any]) -> None:
        self._daemon._output.log_echo_debug(event, payload)

    def normalize_spoken_text(self, text: str) -> str:
        return self._daemon._output.normalize_spoken_text(text)

    def filter_echo_text(self, text: str, segment_id: Optional[int] = None) -> Optional[str]:
        return self._daemon._output.filter_echo_text(text, segment_id)

    def is_sleep_command(self, text: str) -> bool:
        return self._daemon._is_sleep_command(text)

    async def handle_voice_approval(self, text: str, context: RuntimeContext) -> bool:
        return await self._daemon._handle_voice_approval(text, context)

    def detect_wake_phrase(self, text: str) -> Tuple[bool, str]:
        return detect_wake_word(text, self._daemon._system_entity)

    def wake_parts(self, text: str) -> Tuple[bool, str, str]:
        return self._daemon.wake_parts(text)

    def print_user_transcript(self, original_text: str, final_text: str, accepted: bool) -> None:
        self._daemon._print_user_transcript(original_text, final_text, accepted)

    def log_transcript(
        self,
        original_text: str,
        final_text: str,
        wake_mode: str,
        wake_hint: bool,
        is_wake: bool,
        accepted: bool,
    ) -> None:
        self._daemon._log_transcript(original_text, final_text, wake_mode, wake_hint, is_wake, accepted)

    def set_listening_mode(self, mode: str, note: Optional[str] = None) -> None:
        self._daemon._set_listening_mode(mode, note)

    def record_user_turn(self, text: str, at: float) -> None:
        self._daemon._output.record_user_turn(text, at)

    def vision_query_triggers(self) -> List[str]:
        return self._daemon._vision_service.query_triggers
