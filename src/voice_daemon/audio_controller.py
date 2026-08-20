import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Protocol, Tuple

from runtime_core.runtime import RuntimeContext
from voice_daemon.vision_context import maybe_augment_turn_with_vision_context


@dataclass(frozen=True)
class AudioControllerConfig:
    voice: Mapping[str, Any]
    debug_audio: bool
    debug_audio_interval_ms: int
    no_speech_prob_threshold: Optional[float]


class AudioControllerDelegate(Protocol):
    @property
    def config(self) -> AudioControllerConfig:
        raise NotImplementedError

    def maybe_sleep_for_timeout(self, now: float) -> None:
        raise NotImplementedError

    def should_suppress_capture(self, now: float) -> bool:
        raise NotImplementedError

    def note_conversation_activity(self, now: float) -> None:
        raise NotImplementedError

    def listening_mode(self) -> str:
        raise NotImplementedError

    def set_wake_mode(self, mode: str) -> None:
        raise NotImplementedError

    def wake_mode(self) -> Optional[str]:
        raise NotImplementedError

    def init_wake_model(self, openwakeword, wake_model_cls, wake_models: List[Any]):
        raise NotImplementedError

    def open_input_stream(self, sd, sample_rate: int, frame_samples: int, callback):
        raise NotImplementedError

    def print_stream_info(self, sd, stream) -> None:
        raise NotImplementedError

    def predict_wake(self, wake_model, audio_int16, audio_float, wake_use_float: Optional[bool], threshold: float):
        raise NotImplementedError

    def on_activation(self) -> None:
        raise NotImplementedError

    def on_sleep(self, reason: str) -> None:
        raise NotImplementedError

    def log_audio_debug(self, avg_level: float, is_speech: bool, capture_active: bool) -> None:
        raise NotImplementedError

    def next_segment_id(self) -> int:
        raise NotImplementedError

    def log_echo_debug(self, event: str, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    def normalize_spoken_text(self, text: str) -> str:
        raise NotImplementedError

    def filter_echo_text(self, text: str, segment_id: Optional[int] = None) -> Optional[str]:
        raise NotImplementedError

    def is_sleep_command(self, text: str) -> bool:
        raise NotImplementedError

    async def handle_voice_approval(self, text: str, context: RuntimeContext) -> bool:
        raise NotImplementedError

    def detect_wake_phrase(self, text: str) -> Tuple[bool, str]:
        raise NotImplementedError

    def wake_parts(self, text: str) -> Tuple[bool, str, str]:
        raise NotImplementedError

    def print_user_transcript(self, original_text: str, final_text: str, accepted: bool) -> None:
        raise NotImplementedError

    def log_transcript(
        self,
        original_text: str,
        final_text: str,
        wake_mode: str,
        wake_hint: bool,
        is_wake: bool,
        accepted: bool,
    ) -> None:
        raise NotImplementedError

    def set_listening_mode(self, mode: str, note: Optional[str] = None) -> None:
        raise NotImplementedError

    def record_user_turn(self, text: str, at: float) -> None:
        raise NotImplementedError


class VoiceAudioController:
    def __init__(self, delegate: AudioControllerDelegate, sd, np, webrtcvad, whisper_model_cls):
        self._delegate = delegate
        self._sd = sd
        self._np = np
        self._webrtcvad = webrtcvad
        self._whisper_model_cls = whisper_model_cls

    async def run(self, context: RuntimeContext) -> None:
        delegate = self._delegate
        controller_cfg = delegate.config
        cfg = controller_cfg.voice
        sample_rate = int(cfg.get("sample_rate", 16000))
        frame_ms = int(cfg.get("frame_ms", 30))
        frame_samples = int(sample_rate * frame_ms / 1000)
        frame_bytes = frame_samples * 2

        silence_ms = int(cfg.get("silence_ms", 700))
        silence_frames = max(1, silence_ms // frame_ms)
        silence_tail_ms = int(cfg.get("silence_tail_ms", 250))
        silence_tail_frames = max(0, silence_tail_ms // frame_ms)
        wake_timeout_ms = int(cfg.get("wake_timeout_ms", 4000))
        wake_timeout_frames = max(1, wake_timeout_ms // frame_ms)
        pre_roll_ms = int(cfg.get("pre_roll_ms", 500))
        pre_roll_frames = max(1, pre_roll_ms // frame_ms)

        vad = self._webrtcvad.Vad(int(cfg.get("vad_mode", 2)))
        wake_threshold = float(cfg.get("wake_threshold", 0.6))
        wake_mode = str(cfg.get("wake_mode", "openwakeword")).lower()
        wake_fallback = bool(cfg.get("wake_fallback_to_stt", True))
        wake_required = bool(cfg.get("wake_required", True))
        wake_models = cfg.get("wakeword_models", [])
        wake_model = None
        if wake_required and wake_mode in ("openwakeword", "hybrid"):
            try:
                import openwakeword
                from openwakeword.model import Model as WakeModel
            except ImportError as e:
                print(f"[voice] Missing openwakeword dependency: {e}")
                openwakeword = None
                WakeModel = None
            if openwakeword and WakeModel:
                wake_model = delegate.init_wake_model(openwakeword, WakeModel, wake_models)

        if not wake_required:
            wake_mode = "always"
        elif wake_mode in ("openwakeword", "hybrid") and wake_model is None and wake_fallback:
            wake_mode = "stt_prefix"
            print("[voice] Falling back to STT prefix wake mode.")
        elif wake_mode in ("openwakeword", "hybrid") and wake_model is None:
            print("[voice] No wakeword models available; cannot start audio mode.")
            return

        stt_model_name = str(cfg.get("stt_model", "small.en"))
        stt_device = str(cfg.get("stt_device", "cpu"))
        stt_compute = str(cfg.get("stt_compute_type", "int8"))
        try:
            stt_model = self._whisper_model_cls(stt_model_name, device=stt_device, compute_type=stt_compute)
        except Exception as e:
            print(f"[voice] STT model init failed: {e}")
            return

        audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        stt_queue: asyncio.Queue = asyncio.Queue()
        pre_roll: Deque[bytes] = deque(maxlen=pre_roll_frames)
        loop = asyncio.get_running_loop()
        last_status_log = 0.0

        def _enqueue_audio(data: bytes) -> None:
            try:
                audio_queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

        def callback(indata, frames, time_info, status):
            nonlocal last_status_log
            if status and controller_cfg.debug_audio:
                now = time.monotonic()
                if now - last_status_log >= 1.0:
                    print(f"[voice] audio status={status}", flush=True)
                    last_status_log = now
            data = bytes(indata)
            if len(data) < frame_bytes:
                return
            try:
                loop.call_soon_threadsafe(_enqueue_audio, data)
            except RuntimeError:
                pass

        stream = delegate.open_input_stream(self._sd, sample_rate, frame_samples, callback)
        if stream is None:
            return
        stream.start()
        delegate.print_stream_info(self._sd, stream)
        print("[voice] Listening for wake word...")

        delegate.set_wake_mode(wake_mode)
        stt_task = asyncio.create_task(self._stt_worker(context, stt_queue, stt_model))

        capture_frames: List[bytes] = []
        capture_active = False
        speaking = False
        silence_count = 0
        last_speech_idx = -1
        wake_wait = 0
        wake_use_float: Optional[bool] = None
        wake_hint = False
        last_debug = time.monotonic()
        level_sum = 0.0
        level_count = 0

        try:
            while True:
                now = time.monotonic()
                delegate.maybe_sleep_for_timeout(now)
                frame = await audio_queue.get()
                now = time.monotonic()
                if delegate.should_suppress_capture(now):
                    capture_frames = []
                    capture_active = False
                    speaking = False
                    silence_count = 0
                    last_speech_idx = -1
                    wake_wait = 0
                    wake_hint = False
                    continue
                pre_roll.append(frame)
                audio_int16 = self._np.frombuffer(frame, dtype=self._np.int16)
                frame_included = False
                use_wakeword = wake_mode in ("openwakeword", "hybrid") and wake_model is not None
                if use_wakeword:
                    audio_float = audio_int16.astype("float32") / 32768.0
                    pred = delegate.predict_wake(wake_model, audio_int16, audio_float, wake_use_float, wake_threshold)
                    if pred is not None:
                        wake_use_float = pred["use_float"]
                        if pred["triggered"]:
                            delegate.on_activation()
                            await context.send_wake({"ts": time.time(), "wake_word": pred["name"]})
                            wake_hint = True
                            if not capture_active:
                                capture_frames = list(pre_roll)
                                capture_active = True
                                speaking = False
                                silence_count = 0
                                wake_wait = 0
                                frame_included = True

                is_speech = vad.is_speech(frame, sample_rate)
                if controller_cfg.debug_audio:
                    level_sum += float(self._np.abs(audio_int16).mean()) / 32768.0
                    level_count += 1
                    if (now - last_debug) * 1000 >= controller_cfg.debug_audio_interval_ms:
                        avg_level = level_sum / max(level_count, 1)
                        delegate.log_audio_debug(avg_level, is_speech, capture_active)
                        level_sum = 0.0
                        level_count = 0
                        last_debug = now
                if is_speech:
                    delegate.note_conversation_activity(now)

                if capture_active:
                    if not frame_included:
                        capture_frames.append(frame)
                    if is_speech:
                        speaking = True
                        silence_count = 0
                        wake_wait = 0
                        last_speech_idx = len(capture_frames) - 1
                    else:
                        if speaking:
                            silence_count += 1
                            if silence_count >= silence_frames:
                                trimmed = self._trim_trailing_silence(
                                    capture_frames, last_speech_idx, silence_tail_frames
                                )
                                await self._finalize_audio_segment(stt_queue, trimmed, sample_rate, wake_hint)
                                capture_frames = []
                                capture_active = False
                                speaking = False
                                silence_count = 0
                                last_speech_idx = -1
                                wake_wait = 0
                                wake_hint = False
                        else:
                            wake_wait += 1
                            if wake_wait >= wake_timeout_frames:
                                capture_frames = []
                                capture_active = False
                                speaking = False
                                silence_count = 0
                                last_speech_idx = -1
                                wake_wait = 0
                                wake_hint = False
                    continue

                if wake_mode == "openwakeword" and delegate.listening_mode() == "wake":
                    continue

                if is_speech:
                    capture_frames = list(pre_roll)
                    capture_active = True
                    speaking = True
                    silence_count = 0
                    last_speech_idx = len(capture_frames) - 1
                    wake_wait = 0
        finally:
            stream.stop()
            stream.close()
            await stt_queue.put(None)
            await stt_task

    def _trim_trailing_silence(self, frames: List[bytes], last_speech_idx: int, tail_frames: int) -> List[bytes]:
        if not frames:
            return frames
        if last_speech_idx < 0:
            return frames
        tail = max(0, int(tail_frames))
        trim_idx = min(len(frames), last_speech_idx + 1 + tail)
        if trim_idx >= len(frames):
            return frames
        if trim_idx <= 0:
            return frames
        return frames[:trim_idx]

    async def _finalize_audio_segment(
        self, stt_queue: asyncio.Queue, frames: List[bytes], sample_rate: int, wake_hint: bool
    ) -> None:
        delegate = self._delegate
        if not frames:
            return
        segment_id = delegate.next_segment_id()
        frame_ms = int(delegate.config.voice.get("frame_ms", 30))
        duration_ms = len(frames) * frame_ms
        delegate.log_echo_debug(
            "segment_finalized",
            {
                "segment_id": segment_id,
                "duration_ms": duration_ms,
                "frames": len(frames),
                "wake_hint": wake_hint,
            },
        )
        await stt_queue.put(
            {
                "frames": frames,
                "sample_rate": sample_rate,
                "wake_hint": wake_hint,
                "segment_id": segment_id,
                "duration_ms": duration_ms,
            }
        )

    async def _stt_worker(self, context: RuntimeContext, stt_queue: asyncio.Queue, stt_model) -> None:
        delegate = self._delegate
        while True:
            item = await stt_queue.get()
            if item is None:
                break
            frames = item["frames"]
            sample_rate = item["sample_rate"]
            wake_hint = item.get("wake_hint", False)
            segment_id = item.get("segment_id")
            duration_ms = item.get("duration_ms")
            text = await asyncio.to_thread(self._transcribe, stt_model, frames, sample_rate)
            if not text:
                continue

            wake_mode = delegate.wake_mode() or str(delegate.config.voice.get("wake_mode", "openwakeword")).lower()
            original_text = text
            normalized_text = delegate.normalize_spoken_text(original_text)
            delegate.log_echo_debug(
                "stt_raw",
                {
                    "segment_id": segment_id,
                    "duration_ms": duration_ms,
                    "text": original_text,
                    "normalized_text": normalized_text,
                    "normalized_tokens": normalized_text.split() if normalized_text else [],
                },
            )
            filtered_text = delegate.filter_echo_text(original_text, segment_id)
            if filtered_text is None:
                continue
            text = filtered_text
            if delegate.is_sleep_command(text):
                delegate.print_user_transcript(original_text, text, False)
                delegate.on_sleep("command")
                delegate.log_transcript(original_text, text, wake_mode, wake_hint, False, False)
                continue

            approval_text = text
            approval_is_wake = False
            wake_probe, remainder = delegate.detect_wake_phrase(text)
            if wake_probe and remainder:
                approval_text = remainder
                approval_is_wake = True
            if approval_text:
                handled = await delegate.handle_voice_approval(approval_text, context)
                if handled:
                    if approval_is_wake:
                        delegate.on_activation()
                    delegate.print_user_transcript(original_text, approval_text, False)
                    delegate.log_transcript(
                        original_text, approval_text, wake_mode, wake_hint, approval_is_wake, False
                    )
                    continue

            accepted = True
            is_wake = False
            if delegate.listening_mode() == "conversation":
                is_wake, remainder, forward_text = delegate.wake_parts(text)
                if is_wake and remainder:
                    text = forward_text
                elif is_wake and not remainder:
                    accepted = False
            elif wake_mode in ("stt_prefix", "hybrid"):
                is_wake, remainder, forward_text = delegate.wake_parts(text)
                if not is_wake and not wake_hint:
                    accepted = False
                if is_wake:
                    await context.send_wake({"ts": time.time(), "wake_word": text})
                    if remainder:
                        text = forward_text
                    elif not wake_hint:
                        delegate.on_activation()
                        delegate.set_listening_mode("conversation")
                        delegate.print_user_transcript(original_text, text, False)
                        delegate.log_transcript(original_text, text, wake_mode, wake_hint, True, False)
                        continue
            else:
                is_wake, remainder, forward_text = delegate.wake_parts(text)
                if is_wake and remainder:
                    text = forward_text
                elif is_wake and not remainder:
                    accepted = False

            if wake_hint or is_wake:
                delegate.on_activation()
            delegate.print_user_transcript(original_text, text, accepted)
            delegate.log_transcript(original_text, text, wake_mode, wake_hint, is_wake, accepted)
            if not accepted:
                continue

            vision_query_triggers = None
            get_vision_query_triggers = getattr(delegate, "vision_query_triggers", None)
            if callable(get_vision_query_triggers):
                vision_query_triggers = get_vision_query_triggers()
            text = await maybe_augment_turn_with_vision_context(
                context,
                text,
                triggers=vision_query_triggers,
            )
            delegate.set_listening_mode("conversation")
            delegate.record_user_turn(text, time.monotonic())
            await context.submit_turn(text)

    def _transcribe(self, stt_model, frames: List[bytes], sample_rate: int) -> str:
        audio_bytes = b"".join(frames)
        audio_int16 = self._np.frombuffer(audio_bytes, dtype=self._np.int16)
        audio = audio_int16.astype("float32") / 32768.0
        segments, _ = stt_model.transcribe(audio, language=None, vad_filter=False)
        parts: List[str] = []
        no_speech_probs: List[float] = []
        for seg in segments:
            parts.append(seg.text)
            prob = getattr(seg, "no_speech_prob", None)
            if prob is not None:
                try:
                    no_speech_probs.append(float(prob))
                except (TypeError, ValueError):
                    pass
        text = "".join(parts).strip()
        if not text:
            return ""
        no_speech_prob_threshold = self._delegate.config.no_speech_prob_threshold
        if no_speech_prob_threshold is not None and no_speech_probs:
            avg_prob = sum(no_speech_probs) / len(no_speech_probs)
            if avg_prob >= no_speech_prob_threshold:
                return ""
        return text
