import argparse
import asyncio
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common.config import load_settings
from common.jsonlog import append_jsonl
from common.system_entity import detect_wake_word, load_system_entity
from common.vision_broker import VisionBroker
from orchestrator.memory_manager import MemoryManager
from runtime_core import LocalRuntime, RuntimeContext
from runtime_core.session import OrchestratorSession
from voice_daemon.audio_controller import VoiceAudioController
from voice_daemon.audio_delegate import VoiceDaemonAudioDelegate
from voice_daemon.input_service import VoiceInputService
from voice_daemon.output_manager import VoiceOutputManager
from voice_daemon.senses import AudioSense, ConsoleApprovalSense, TextSense
from voice_daemon.vision_registry import VisionEntityRegistry
from voice_daemon.vision_sense import VisionSense
from voice_daemon.vision_service import YoloVisionService


class VoiceDaemon:
    def __init__(self):
        settings = load_settings()
        self._interface_cfg = dict(settings.interface or {})
        source_mode = self._normalize_source_mode(self._interface_cfg.get("mode", "voice"))
        self._host = settings.rpc["orch_host"]
        self._port = settings.rpc["orch_port"]
        self._voice_cfg = self._build_effective_voice_cfg(settings.voice or {}, source_mode)
        self._vision_cfg = dict(settings.vision or {})
        self._transcript_log_path = settings.voice.get("transcript_log_path")
        self._log_transcripts = bool(settings.voice.get("log_transcripts", False))
        self._debug_audio = bool(settings.voice.get("debug_audio", False))
        self._debug_audio_interval_ms = int(settings.voice.get("debug_audio_interval_ms", 1000))
        self._audio_debug_log_path = settings.voice.get("audio_debug_log_path")
        self._print_user_transcripts = bool(settings.voice.get("print_user_transcripts", False))
        self._strip_wake_word = bool(settings.voice.get("strip_wake_word", False))
        self._voice_approvals_enabled = bool(settings.voice.get("voice_approvals_enabled", True))
        self._no_speech_prob_threshold = settings.voice.get("no_speech_prob_threshold")
        if self._no_speech_prob_threshold is not None:
            try:
                self._no_speech_prob_threshold = float(self._no_speech_prob_threshold)
            except (TypeError, ValueError):
                self._no_speech_prob_threshold = None
        self._conversation_timeout_s = float(settings.voice.get("conversation_timeout_s", 15.0))
        self._sleep_phrases = self._normalize_phrases(
            settings.voice.get("sleep_phrases", settings.voice.get("sleep_phrase"))
        )
        if not self._sleep_phrases:
            self._sleep_phrases = ["sleep"]
        self._listening_mode = "conversation" if not self._voice_cfg.get("wake_required", True) else "wake"
        self._conversation_last_activity = time.monotonic() if self._listening_mode == "conversation" else 0.0
        self._system_entity = load_system_entity()
        self._wake_mode_current: Optional[str] = None
        self._input = VoiceInputService(
            settings.voice,
            debug_audio=self._debug_audio,
            audio_debug_log_path=self._audio_debug_log_path,
        )
        self._vision_service = YoloVisionService(settings.vision)
        self._vision_broker = VisionBroker(settings.workspace_root, settings.data_dir, settings.vision)
        self._vision_memory = None
        if self._vision_service.track_enabled() and self._vision_service.object_memory_enabled():
            self._vision_memory = MemoryManager()
        self._output = VoiceOutputManager(self._voice_cfg)
        self._session = OrchestratorSession(
            self._host,
            self._port,
            source_id=source_mode,
            on_assistant_final=self._handle_assistant_final,
            on_assistant_draft=self._handle_assistant_draft,
            on_permission_request=self._handle_permission_request,
        )
        self._segment_counter = 0
        if self._log_transcripts and self._transcript_log_path:
            os.makedirs(os.path.dirname(self._transcript_log_path), exist_ok=True)
        if self._debug_audio and self._audio_debug_log_path:
            os.makedirs(os.path.dirname(self._audio_debug_log_path), exist_ok=True)

    def _normalize_source_mode(self, mode: Any) -> str:
        source_mode = str(mode or "voice").strip().lower()
        if source_mode not in {"text", "audio", "vision", "local"}:
            return "voice"
        return source_mode

    def _build_effective_voice_cfg(self, voice_cfg: Dict[str, Any], source_mode: str) -> Dict[str, Any]:
        effective = dict(voice_cfg or {})
        if source_mode == "text":
            effective["tts_enabled"] = False
            effective["permission_prompt_tts_enabled"] = False
        return effective

    async def run(self, mode: str = "text", *, senses: Optional[Sequence[str]] = None) -> None:
        requested_senses = self._resolve_requested_senses(mode, senses)
        await self._run_local(requested_senses)

    async def _run_local(self, senses: Sequence[str]) -> None:
        normalized = list(senses)
        sd = None
        np = None
        webrtcvad = None
        whisper_model = None
        if "audio" in normalized:
            try:
                import numpy as np  # type: ignore[no-redef]
                import sounddevice as sd  # type: ignore[no-redef]
                import webrtcvad  # type: ignore[no-redef]
                from faster_whisper import WhisperModel  # type: ignore[no-redef]
            except ImportError as e:
                print(f"[voice] Missing audio dependency: {e}")
                print("[voice] Install audio deps from requirements.txt or remove the audio sense.")
                return
            whisper_model = WhisperModel

        self._start_helpers(sd, np)
        try:
            runtime = LocalRuntime(self._session, self._build_senses(normalized, sd, np, webrtcvad, whisper_model))
            print("[voice] Local senses active: {senses}".format(senses=", ".join(normalized)), flush=True)
            await runtime.run()
        finally:
            await self._stop_helpers()

    def _build_senses(
        self,
        senses: Sequence[str],
        sd,
        np,
        webrtcvad,
        whisper_model,
    ) -> List[Any]:
        built: List[Any] = []
        if "text" in senses:
            built.append(
                TextSense(
                    self._system_entity.name,
                    self._system_entity.aliases,
                    self,
                    vision_query_triggers=self._vision_service.query_triggers,
                )
            )
        if "audio" in senses:
            audio_controller = VoiceAudioController(VoiceDaemonAudioDelegate(self), sd, np, webrtcvad, whisper_model)
            built.append(ConsoleApprovalSense(self))
            built.append(AudioSense(audio_controller.run))
        if "vision" in senses:
            registry = None
            if self._vision_service.track_enabled():
                registry = VisionEntityRegistry(self._vision_service, self._vision_memory)
            built.append(VisionSense(self._vision_service, registry=registry, broker=self._vision_broker))
        return built

    def _resolve_requested_senses(self, mode: str, senses: Optional[Sequence[str]]) -> List[str]:
        requested: List[str] = []
        if senses:
            for item in senses:
                requested.extend(str(item).split(","))
        else:
            normalized_mode = str(mode or "text").strip().lower()
            if normalized_mode in {"", "text"}:
                requested = ["text"]
            elif normalized_mode in {"audio", "vision"}:
                requested = [normalized_mode]
            elif normalized_mode == "local":
                requested = list(self._interface_cfg.get("senses") or ["text"])
            else:
                raise ValueError(f"Unsupported local mode '{mode}'")

        normalized: List[str] = []
        for item in requested:
            name = str(item or "").strip().lower()
            if not name:
                continue
            if name not in {"text", "audio", "vision"}:
                raise ValueError(f"Unsupported local sense '{name}'")
            if name not in normalized:
                normalized.append(name)
        if not normalized:
            normalized = ["text"]
        if "text" in normalized and "audio" in normalized:
            raise ValueError("Text and audio senses cannot be active together because both consume stdin.")
        return normalized

    def _start_helpers(self, sd=None, np=None) -> None:
        self._output.start(sd=sd, np=np)

    async def _stop_helpers(self) -> None:
        await self._output.stop()

    async def _handle_assistant_final(self, payload: Dict[str, Any]) -> None:
        text = payload.get("text", "")
        print(f"\n[assistant] {text}\n")
        await self._output.enqueue_tts(text)

    async def _handle_assistant_draft(self, payload: Dict[str, Any]) -> None:
        text = payload.get("text", "")
        print(f"\n[assistant draft] {text}\n")

    async def _handle_permission_request(self, payload: Dict[str, Any]) -> None:
        request_id = payload.get("request_id")
        tool_id = payload.get("tool_id")
        justification = payload.get("justification")
        if not request_id:
            return
        print(
            "\n[permission] request_id={rid} tool={tool} reason={reason}".format(
                rid=request_id, tool=tool_id, reason=justification
            )
        )
        print("Type /approve {rid} or /deny {rid}".format(rid=request_id))
        if self._output.permission_prompt_tts_enabled:
            speak_tool = str(tool_id or "this action")
            await self._output.enqueue_tts(f"Permission needed for {speak_tool}. Say approve or deny.")

    def _print_user_transcript(self, original_text: str, final_text: Optional[str], accepted: bool) -> None:
        if not self._print_user_transcripts:
            return
        if accepted and final_text:
            display = final_text
        else:
            display = original_text
        if not display:
            return
        print(f"\n[user] {display}\n", flush=True)

    def _on_activation(self) -> None:
        previous = self._listening_mode
        note = "[voice] Wake detected; conversation active." if previous != "conversation" else None
        self._set_listening_mode("conversation", note)
        if previous != "conversation":
            self._output.notify_activation(time.monotonic())

    def _on_sleep(self, reason: str) -> None:
        previous = self._listening_mode
        note = None
        if reason == "command":
            note = "[voice] Sleep command received; listening for wake word."
        elif reason == "timeout":
            note = "[voice] Inactivity timeout; listening for wake word."
        self._set_listening_mode("wake", note)
        if previous != "wake":
            self._output.notify_sleep(time.monotonic())

    def wake_parts(self, text: str) -> Tuple[bool, str, str]:
        return self._wake_parts(text)

    async def handle_console_approval(self, line: str, context: RuntimeContext) -> bool:
        if line.startswith("/approve") or line.startswith("/deny"):
            parts = line.split()
            if len(parts) >= 2:
                action = parts[0]
                request_id = parts[1]
                approved = action == "/approve"
                await self._send_permission_decision(request_id, approved, context)
            return True
        return False

    def _resolve_permission_id(self, token: Optional[str], context: RuntimeContext) -> Optional[str]:
        resolution = context.resolve_permission_id(token)
        if resolution.status == "matched":
            return resolution.request_id
        if resolution.status == "ambiguous":
            print("[voice] Multiple matching permission requests; please say the full request_id.")
            return None
        if resolution.status == "multiple_pending":
            print("[voice] Multiple pending permission requests; please specify the request_id.")
        return None

    async def _send_permission_decision(
        self,
        request_id: str,
        approved: bool,
        context: RuntimeContext,
    ) -> bool:
        sent = await context.send_permission_decision(request_id, approved)
        if not sent:
            print("[voice] Unknown permission request_id")
            return False
        return True

    async def _handle_voice_approval(self, text: str, context: RuntimeContext) -> bool:
        if not self._voice_approvals_enabled:
            return False
        normalized = text.strip().lower()
        if not normalized:
            return False
        normalized = re.sub(r"[.!?,]+$", "", normalized)
        match = re.match(r"^(approve|deny)(?:\s+([0-9a-fA-F-]{4,}))?$", normalized)
        if not match:
            return False
        action = match.group(1)
        token = match.group(2)
        request_id = self._resolve_permission_id(token, context)
        if not request_id:
            return True
        approved = action == "approve"
        await self._send_permission_decision(request_id, approved, context)
        return True

    def _wake_parts(self, text: str) -> Tuple[bool, str, str]:
        is_wake, remainder = detect_wake_word(text, self._system_entity)
        forward_text = remainder if self._strip_wake_word else text
        return is_wake, remainder, forward_text

    def _normalize_phrases(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            return []
        normalized: List[str] = []
        for item in values:
            cleaned = self._normalize_command_text(str(item))
            if cleaned:
                normalized.append(cleaned)
        return normalized

    def _normalize_command_text(self, text: str) -> str:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(cleaned.split())

    def _set_listening_mode(self, mode: str, note: Optional[str] = None) -> None:
        if mode == self._listening_mode:
            if mode == "conversation":
                self._conversation_last_activity = time.monotonic()
            return
        self._listening_mode = mode
        if mode == "conversation":
            self._conversation_last_activity = time.monotonic()
        else:
            self._conversation_last_activity = 0.0
        if note:
            print(note, flush=True)

    def _is_sleep_command(self, text: str) -> bool:
        cleaned = self._normalize_command_text(text)
        if not cleaned:
            return False
        if cleaned in self._sleep_phrases:
            return True
        is_wake, remainder = detect_wake_word(text, self._system_entity)
        if is_wake and self._normalize_command_text(remainder) in self._sleep_phrases:
            return True
        return False

    def _log_transcript(
        self,
        original_text: str,
        final_text: str,
        wake_mode: str,
        wake_hint: bool,
        is_wake: bool,
        accepted: bool,
    ) -> None:
        if not self._log_transcripts or not self._transcript_log_path:
            return
        append_jsonl(
            self._transcript_log_path,
            {
                "ts": time.time(),
                "text": original_text,
                "final_text": final_text,
                "wake_mode": wake_mode,
                "wake_hint": wake_hint,
                "wake_match": is_wake,
                "accepted": accepted,
            },
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice daemon")
    parser.add_argument(
        "--mode",
        choices=["text", "audio", "vision", "local"],
        default="text",
        help="Primary local mode.",
    )
    parser.add_argument(
        "--senses",
        default="",
        help="Comma-separated local senses to activate, for example 'audio,vision' or 'text,vision'.",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    daemon = VoiceDaemon()
    requested_senses = [part for part in str(args.senses or "").split(",") if part.strip()]
    await daemon.run(mode=args.mode, senses=requested_senses or None)


if __name__ == "__main__":
    asyncio.run(main())
