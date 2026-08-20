import asyncio
import io
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from common.config import load_settings
from common.ids import new_id
from common.jsonl_rpc import RpcError, send_request
from runtime_core.session import OrchestratorSession


class TelegramDaemon:
    _TEXT_FILE_EXTENSIONS = {
        ".txt",
        ".md",
        ".rst",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".conf",
        ".log",
        ".csv",
        ".tsv",
        ".xml",
        ".html",
        ".css",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".py",
        ".c",
        ".cpp",
        ".cc",
        ".cxx",
        ".h",
        ".hpp",
        ".java",
        ".kt",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".sh",
        ".ps1",
        ".sql",
    }
    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    _VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}

    def __init__(self):
        self._load_dotenv_defaults()
        settings = load_settings()
        self._host = settings.rpc["orch_host"]
        self._port = settings.rpc["orch_port"]
        self._model_host = settings.rpc["orch_host"]
        self._model_port = settings.rpc["model_port"]
        self._workspace_root = settings.workspace_root
        self._cfg = dict(settings.telegram or {})
        self._voice_cfg = dict(settings.voice or {})

        token_cfg = str(self._cfg.get("bot_token", "")).strip()
        token_env = self._read_env("TELEGRAM_BOT_TOKEN")
        self._token = token_env or token_cfg
        api_base = str(
            self._read_env("TELEGRAM_API_BASE_URL")
            or self._cfg.get("api_base_url", "https://api.telegram.org")
        ).strip()
        if not api_base:
            api_base = "https://api.telegram.org"
        self._api_base = api_base.rstrip("/")

        self._poll_timeout_s = self._coerce_int(
            self._read_env("TELEGRAM_POLL_TIMEOUT_S") or self._cfg.get("poll_timeout_s"),
            25,
            minimum=1,
            maximum=50,
        )
        default_request_timeout = max(35, self._poll_timeout_s + 5)
        self._request_timeout_s = self._coerce_int(
            self._read_env("TELEGRAM_REQUEST_TIMEOUT_S") or self._cfg.get("request_timeout_s"),
            default_request_timeout,
            minimum=5,
            maximum=120,
        )
        if self._request_timeout_s <= self._poll_timeout_s:
            self._request_timeout_s = self._poll_timeout_s + 5
        self._model_rpc_timeout_s = self._coerce_int(
            self._read_env("TELEGRAM_MODEL_RPC_TIMEOUT_S")
            or self._cfg.get("model_rpc_timeout_s")
            or settings.orchestrator.get("model_rpc_timeout_s"),
            60,
            minimum=0,
            maximum=7 * 24 * 60 * 60,
        )
        self._max_message_chars = self._coerce_int(
            self._read_env("TELEGRAM_MAX_MESSAGE_CHARS") or self._cfg.get("max_message_chars"),
            3500,
            minimum=200,
            maximum=3900,
        )
        response_mode = str(
            self._read_env("TELEGRAM_RESPONSE_MODE") or self._cfg.get("response_mode", "text")
        ).strip().lower()
        if response_mode not in {"text", "audio", "same"}:
            response_mode = "text"
        self._response_mode = response_mode
        self._audio_response_max_chars = self._coerce_int(
            self._read_env("TELEGRAM_AUDIO_RESPONSE_MAX_CHARS")
            or self._cfg.get("audio_response_max_chars"),
            1800,
            minimum=128,
            maximum=20000,
        )
        self._response_tts_rate = self._coerce_int(
            self._read_env("TELEGRAM_TTS_RATE")
            or self._cfg.get("tts_rate")
            or self._voice_cfg.get("tts_rate"),
            175,
            minimum=50,
            maximum=400,
        )
        self._response_tts_volume = self._coerce_float(
            self._read_env("TELEGRAM_TTS_VOLUME")
            or self._cfg.get("tts_volume")
            or self._voice_cfg.get("tts_volume"),
            1.0,
            minimum=0.0,
            maximum=1.0,
        )
        self._max_download_bytes = self._coerce_int(
            self._read_env("TELEGRAM_MAX_DOWNLOAD_BYTES") or self._cfg.get("max_download_bytes"),
            10 * 1024 * 1024,
            minimum=32 * 1024,
            maximum=100 * 1024 * 1024,
        )
        self._text_attachment_max_chars = self._coerce_int(
            self._read_env("TELEGRAM_TEXT_ATTACHMENT_MAX_CHARS")
            or self._cfg.get("text_attachment_max_chars"),
            12000,
            minimum=512,
            maximum=200000,
        )
        self._vision_question_fallback = str(
            self._cfg.get("vision_question_fallback", "Describe this image.")
        ).strip() or "Describe this image."
        self._media_submit_requested_mode = self._normalize_media_requested_mode(
            self._read_env("TELEGRAM_MEDIA_SUBMIT_REQUESTED_MODE")
            or self._cfg.get("media_submit_requested_mode"),
            default="SYSTEM0",
        )
        self._media_skip_distillation = self._coerce_bool(
            self._read_env("TELEGRAM_MEDIA_SKIP_DISTILLATION")
            if self._read_env("TELEGRAM_MEDIA_SKIP_DISTILLATION") is not None
            else self._cfg.get("media_skip_distillation"),
            default=False,
        )
        self._pdf_max_pages = self._coerce_int(
            self._read_env("TELEGRAM_PDF_MAX_PAGES") or self._cfg.get("pdf_max_pages"),
            40,
            minimum=1,
            maximum=1000,
        )
        self._pdf_max_chars = self._coerce_int(
            self._read_env("TELEGRAM_PDF_MAX_CHARS") or self._cfg.get("pdf_max_chars"),
            20000,
            minimum=512,
            maximum=500000,
        )
        self._stt_model_name = str(
            self._read_env("TELEGRAM_STT_MODEL")
            or self._cfg.get("stt_model")
            or self._voice_cfg.get("stt_model", "small.en")
        ).strip() or "small.en"
        self._stt_device = str(
            self._read_env("TELEGRAM_STT_DEVICE")
            or self._cfg.get("stt_device")
            or self._voice_cfg.get("stt_device", "cpu")
        ).strip() or "cpu"
        self._stt_compute_type = str(
            self._read_env("TELEGRAM_STT_COMPUTE_TYPE")
            or self._cfg.get("stt_compute_type")
            or self._voice_cfg.get("stt_compute_type", "int8")
        ).strip() or "int8"
        typing_env = self._read_env("TELEGRAM_SEND_TYPING_ACTION")
        self._send_typing_action = self._coerce_bool(
            typing_env if typing_env is not None else self._cfg.get("send_typing_action"),
            default=True,
        )
        updates_env = self._read_env("TELEGRAM_SEND_USER_UPDATES")
        self._send_user_updates = self._coerce_bool(
            updates_env if updates_env is not None else self._cfg.get("send_user_updates"),
            default=True,
        )
        self._unsupported_files_to_agent = self._coerce_bool(
            self._cfg.get("unsupported_files_to_agent"),
            default=False,
        )

        allowed_cfg = (
            self._cfg.get("allowed_chat_ids")
            if "allowed_chat_ids" in self._cfg
            else self._cfg.get("allowed_chat_id")
        )
        allowed_env = self._read_env("TELEGRAM_ALLOWED_CHAT_IDS")
        if allowed_env is None:
            allowed_env = self._read_env("TELEGRAM_ALLOWED_CHAT_ID")
        self._allowed_chat_ids = self._parse_chat_ids(
            allowed_env if allowed_env is not None else allowed_cfg
        )
        self._active_chat_id: Any = (
            next(iter(self._allowed_chat_ids))
            if self._allowed_chat_ids and len(self._allowed_chat_ids) == 1
            else None
        )
        self._offset = 0
        self._turn_chat: Dict[str, Any] = {}
        self._turn_input_mode: Dict[str, str] = {}
        self._permission_chat: Dict[str, Any] = {}
        self._last_error_log_ts = 0.0
        self._whisper_model: Any = None
        self._whisper_model_error: Optional[str] = None
        self._tts_error: Optional[str] = None
        self._session = OrchestratorSession(
            self._host,
            self._port,
            source_id="telegram",
            on_assistant_final=self._handle_assistant_final,
            on_permission_request=self._handle_permission_request,
        )

        self._incoming_dir = os.path.join(settings.data_dir, "telegram_inbox")
        os.makedirs(self._incoming_dir, exist_ok=True)

    @staticmethod
    def _load_dotenv_defaults() -> None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env_path = os.path.join(root_dir, ".env")
        if not os.path.isfile(env_path):
            return
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if not key:
                        continue
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]
                    os.environ.setdefault(key, value)
        except OSError:
            return

    @staticmethod
    def _read_env(name: str) -> Optional[str]:
        value = os.environ.get(name)
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        return stripped

    @staticmethod
    def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        if parsed < minimum:
            return minimum
        if parsed > maximum:
            return maximum
        return parsed

    @staticmethod
    def _coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        if parsed < minimum:
            return minimum
        if parsed > maximum:
            return maximum
        return parsed

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return default

    @staticmethod
    def _normalize_media_requested_mode(value: Any, default: str = "SYSTEM0") -> str:
        text = str(value or "").strip().upper()
        if text in {"COGNITION", "SYSTEM0"}:
            return text
        return default

    @staticmethod
    def _parse_chat_ids(value: Any) -> Optional[set[int]]:
        if value is None:
            return None
        if isinstance(value, int):
            return {value}
        if isinstance(value, list):
            valid = set()
            for x in value:
                try:
                    valid.add(int(x))
                except (ValueError, TypeError):
                    pass
            return valid if valid else None
        
        try:
            text = str(value).strip()
            if not text:
                return None
            if text.startswith('[') and text.endswith(']'):
                text = text[1:-1]
            valid_ids = set()
            for p in text.split(','):
                p = p.strip()
                if p:
                    try:
                        valid_ids.add(int(p))
                    except ValueError:
                        pass
            return valid_ids if valid_ids else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_chat_id(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        try:
            text = str(value).strip()
            if not text:
                return None
            return int(text)
        except (TypeError, ValueError):
            return None

    def _chat_allowed(self, chat_id: int) -> bool:
        if self._allowed_chat_ids is None:
            return True
        return chat_id in self._allowed_chat_ids

    async def run(self) -> None:
        await self._session.connect()
        print(f"[telegram] connected to orchestrator at {self._host}:{self._port}")
        if not self._token:
            print("[telegram] Missing bot token. Set TELEGRAM_BOT_TOKEN (for example in .env).")
            await self._session.close()
            return
        if self._allowed_chat_ids is not None:
            print(f"[telegram] bot mode restricted to chat_ids={self._allowed_chat_ids}")
        else:
            print("[telegram] bot mode unrestricted; first incoming chat becomes active.")
        print("[telegram] polling Bot API updates...")
        transport_task = asyncio.create_task(self._poll_updates())

        reader_task = asyncio.create_task(self._read_events())
        tasks = [reader_task, transport_task]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc:
                    raise exc
        finally:
            await self._session.close()

    async def _read_events(self) -> None:
        try:
            await self._session.read_events()
        finally:
            print("[telegram] orchestrator connection closed.")

    async def _handle_assistant_final(self, payload: Dict[str, Any]) -> None:
        text = str(payload.get("text", "")).strip()
        turn_id = payload.get("turn_id")
        chat_id: Any = None
        input_mode = "text"
        if isinstance(turn_id, str) and turn_id:
            chat_id = self._turn_chat.pop(turn_id, None)
            input_mode = self._turn_input_mode.pop(turn_id, "text")
        if chat_id is None:
            chat_id = self._active_chat_id
        if chat_id is None:
            print("[telegram] Dropping assistant response because no active chat is available.")
            return
        if not text:
            return
        response_mode = self._resolve_response_mode(input_mode)
        if response_mode == "audio":
            if await self._send_audio_message(chat_id, text):
                return
        await self._send_message(chat_id, text)

    def _resolve_response_mode(self, input_mode: str) -> str:
        if self._response_mode == "same":
            return "audio" if input_mode == "audio" else "text"
        return self._response_mode

    async def _handle_permission_request(self, payload: Dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "")).strip()
        if not request_id:
            return

        chat_id: Any = None
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str) and turn_id:
            chat_id = self._turn_chat.get(turn_id)
        if chat_id is None:
            chat_id = self._active_chat_id
        if chat_id is not None:
            self._permission_chat[request_id] = chat_id
            tool_id = str(payload.get("tool_id", "tool"))
            reason = str(payload.get("justification", "No reason provided"))
            await self._send_message(
                chat_id,
                (
                    f"Permission required for {tool_id}.\n"
                    f"Reason: {reason}\n"
                    f"Reply with /approve {request_id} or /deny {request_id}."
                ),
            )

    async def _poll_updates(self) -> None:
        while True:
            params = {
                "offset": self._offset,
                "timeout": self._poll_timeout_s,
                "allowed_updates": ["message"],
            }
            try:
                resp = await asyncio.to_thread(self._telegram_api_call, "getUpdates", params)
            except Exception as e:
                self._log_api_error("getUpdates", e)
                await asyncio.sleep(1.0)
                continue

            updates = resp.get("result", [])
            if not isinstance(updates, list):
                continue
            for update in updates:
                if not isinstance(update, dict):
                    continue
                update_id = update.get("update_id")
                if isinstance(update_id, int) and update_id >= self._offset:
                    self._offset = update_id + 1
                message = update.get("message")
                if not isinstance(message, dict):
                    continue
                await self._handle_message(message)

    async def _handle_text_line(self, line: str, chat_id: Any) -> bool:
        cleaned = line.strip()
        if not cleaned:
            return False
        if cleaned in {"/start", "/help"}:
            await self._send_help(chat_id)
            return True
        if cleaned.startswith("/approve") or cleaned.startswith("/deny"):
            await self._handle_permission_command(cleaned, chat_id)
            return True
        if cleaned == "/status":
            await self._send_status(chat_id)
            return True
        if cleaned.startswith("/wake"):
            await self._session.send_wake()
            await self._send_message(chat_id, "Wake event sent.")
            return True
        if cleaned.startswith("+"):
            await self._handle_patch(cleaned, chat_id)
            return True
        if cleaned.startswith("/"):
            await self._send_help(chat_id)
            return True
        await self._submit_turn(chat_id, cleaned, input_mode="text")
        return True

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        chat = message.get("chat", {})
        chat_id = self._parse_chat_id(chat.get("id"))
        if chat_id is None:
            return
        if not self._chat_allowed(chat_id):
            await self._send_message(chat_id, "This bot is restricted to a different chat.")
            return

        self._active_chat_id = chat_id

        text = message.get("text")
        if isinstance(text, str) and text.strip():
            await self._handle_text_line(text, chat_id)
            return

        if self._has_audio_attachment(message):
            await self._handle_audio_attachment(message, chat_id)
            return
        if self._has_image_attachment(message):
            await self._handle_image_attachment(message, chat_id)
            return
        if self._has_video_attachment(message):
            await self._handle_video_attachment(message, chat_id)
            return

        document = message.get("document")
        if isinstance(document, dict):
            await self._handle_document_attachment(message, chat_id)
            return

        if self._unsupported_files_to_agent:
            if await self._handle_other_attachment_with_complex(message, chat_id):
                return

        await self._send_message(
            chat_id,
            "I can currently handle text, audio messages, image attachments, video attachments, PDFs, and text/code files.",
        )

    async def _submit_turn(
        self,
        chat_id: Any,
        text: str,
        input_mode: str = "text",
        requested_mode: Optional[str] = None,
        skip_distillation: bool = False,
    ) -> None:
        line = text.strip()
        if not line:
            await self._send_message(chat_id, "Message is empty.")
            return
        if self._send_typing_action:
            await self._send_chat_action(chat_id, "typing")
        turn_id = await self._session.submit_turn(
            line,
            requested_mode=requested_mode,
            skip_distillation=skip_distillation,
        )
        self._turn_chat[turn_id] = chat_id
        self._turn_input_mode[turn_id] = input_mode

    def _has_audio_attachment(self, message: Dict[str, Any]) -> bool:
        return isinstance(message.get("voice"), dict) or isinstance(message.get("audio"), dict)

    def _has_image_attachment(self, message: Dict[str, Any]) -> bool:
        photo = message.get("photo")
        if isinstance(photo, list) and photo:
            return True
        document = message.get("document")
        return self._is_image_document(document)

    def _has_video_attachment(self, message: Dict[str, Any]) -> bool:
        if isinstance(message.get("video"), dict):
            return True
        if isinstance(message.get("animation"), dict):
            return True
        if isinstance(message.get("video_note"), dict):
            return True
        document = message.get("document")
        return self._is_video_document(document)

    @staticmethod
    def _extract_caption(message: Dict[str, Any]) -> str:
        caption = message.get("caption")
        if isinstance(caption, str):
            return caption.strip()
        return ""

    def _file_extension(self, file_name: str) -> str:
        return os.path.splitext(file_name or "")[1].lower()

    def _is_image_document(self, document: Any) -> bool:
        if not isinstance(document, dict):
            return False
        mime = str(document.get("mime_type", "")).lower()
        if mime.startswith("image/"):
            return True
        file_name = str(document.get("file_name", ""))
        return self._file_extension(file_name) in self._IMAGE_EXTENSIONS

    def _is_text_document(self, document: Any) -> bool:
        if not isinstance(document, dict):
            return False
        mime = str(document.get("mime_type", "")).lower()
        if mime.startswith("text/"):
            return True
        file_name = str(document.get("file_name", ""))
        return self._file_extension(file_name) in self._TEXT_FILE_EXTENSIONS

    def _is_pdf_document(self, document: Any) -> bool:
        if not isinstance(document, dict):
            return False
        mime = str(document.get("mime_type", "")).lower()
        if mime == "application/pdf":
            return True
        file_name = str(document.get("file_name", ""))
        return self._file_extension(file_name) == ".pdf"

    def _is_video_document(self, document: Any) -> bool:
        if not isinstance(document, dict):
            return False
        mime = str(document.get("mime_type", "")).lower()
        if mime.startswith("video/"):
            return True
        file_name = str(document.get("file_name", ""))
        return self._file_extension(file_name) in self._VIDEO_EXTENSIONS

    async def _handle_audio_attachment(
        self,
        message: Dict[str, Any],
        chat_id: Any,
    ) -> None:
        voice = message.get("voice")
        audio = message.get("audio")
        media = voice if isinstance(voice, dict) else audio if isinstance(audio, dict) else None
        if not isinstance(media, dict):
            await self._send_message(chat_id, "Audio payload is invalid.")
            return
        file_id = str(media.get("file_id", "")).strip()
        if not file_id:
            await self._send_message(chat_id, "Audio payload is missing file_id.")
            return

        file_name = str(media.get("file_name", "")).strip()
        if not file_name:
            file_name = "voice.ogg" if isinstance(voice, dict) else "audio.bin"

        try:
            data, resolved_name = await self._download_telegram_file(file_id, file_name)
        except Exception as e:
            self._log_api_error("audio_transcribe", e)
            await self._send_message(chat_id, f"Failed to read audio: {e}")
            return
        await self._process_audio_bytes(chat_id, data, resolved_name, self._extract_caption(message))

    async def _process_audio_bytes(
        self,
        chat_id: Any,
        data: bytes,
        file_name: str,
        caption: str,
    ) -> None:
        await self._send_user_update(chat_id, "Transcribing audio...")
        try:
            transcript = await asyncio.to_thread(self._transcribe_audio_bytes, data, file_name)
        except Exception as e:
            self._log_api_error("audio_transcribe", e)
            await self._send_message(chat_id, f"Failed to transcribe audio: {e}")
            return

        if not transcript:
            await self._send_message(chat_id, "I couldn't detect speech in that audio.")
            return

        if caption:
            turn_text = (
                "User sent an audio message with an instruction.\n"
                f"Instruction: {caption}\n"
                f"Transcribed audio: {transcript}"
            )
        else:
            turn_text = transcript
        await self._submit_turn(chat_id, turn_text, input_mode="audio")

    async def _handle_image_attachment(
        self,
        message: Dict[str, Any],
        chat_id: Any,
    ) -> None:
        file_id, suggested_name = self._extract_image_file_ref(message)
        if not file_id:
            await self._send_message(chat_id, "Image payload is invalid.")
            return
        question = self._extract_caption(message) or self._vision_question_fallback
        try:
            data, resolved_name = await self._download_telegram_file(file_id, suggested_name)
        except Exception as e:
            self._log_api_error("image_analyze", e)
            await self._send_message(chat_id, f"Failed to read image: {e}")
            return
        await self._process_image_bytes(chat_id, data, resolved_name, question)

    async def _process_image_bytes(
        self,
        chat_id: Any,
        data: bytes,
        file_name: str,
        question: str,
    ) -> None:
        await self._send_user_update(chat_id, "Analyzing image...")
        try:
            local_path = self._save_attachment_bytes(data, "image", file_name)
            image_ref = self._to_file_ref(local_path)
            vision_result = await self._analyze_image(image_ref, question)
            turn_text = self._build_image_turn_text(
                question=question,
                file_name=file_name,
                result=vision_result,
            )
        except Exception as e:
            self._log_api_error("image_analyze", e)
            await self._send_message(chat_id, f"Failed to process image: {e}")
            return
        await self._submit_turn(
            chat_id,
            turn_text,
            input_mode="text",
            requested_mode=self._media_submit_requested_mode,
            skip_distillation=self._media_skip_distillation,
        )

    def _extract_image_file_ref(self, message: Dict[str, Any]) -> Tuple[str, str]:
        photo = message.get("photo")
        if isinstance(photo, list) and photo:
            selected = None
            best_size = -1
            for entry in photo:
                if not isinstance(entry, dict):
                    continue
                size = entry.get("file_size")
                size_int = int(size) if isinstance(size, int) else 0
                if selected is None or size_int >= best_size:
                    selected = entry
                    best_size = size_int
            if isinstance(selected, dict):
                file_id = str(selected.get("file_id", "")).strip()
                unique = str(selected.get("file_unique_id", "")).strip()
                name = f"photo_{unique}.jpg" if unique else "photo.jpg"
                return file_id, name

        document = message.get("document")
        if self._is_image_document(document):
            assert isinstance(document, dict)
            file_id = str(document.get("file_id", "")).strip()
            name = str(document.get("file_name", "")).strip() or "image.bin"
            return file_id, name
        return "", ""

    def _extract_video_file_ref(self, message: Dict[str, Any]) -> Tuple[str, str]:
        candidates: List[Tuple[str, Any, str]] = [
            ("video", message.get("video"), "video.mp4"),
            ("animation", message.get("animation"), "animation.mp4"),
            ("video_note", message.get("video_note"), "video_note.mp4"),
        ]
        for kind, payload, default_name in candidates:
            if not isinstance(payload, dict):
                continue
            file_id = str(payload.get("file_id", "")).strip()
            if not file_id:
                continue
            file_name = str(payload.get("file_name", "")).strip()
            if not file_name:
                unique = str(payload.get("file_unique_id", "")).strip()
                file_name = f"{kind}_{unique}.mp4" if unique else default_name
            return file_id, file_name

        document = message.get("document")
        if self._is_video_document(document):
            assert isinstance(document, dict)
            file_id = str(document.get("file_id", "")).strip()
            name = str(document.get("file_name", "")).strip() or "video.bin"
            return file_id, name
        return "", ""

    async def _handle_video_attachment(
        self,
        message: Dict[str, Any],
        chat_id: Any,
    ) -> None:
        file_id, suggested_name = self._extract_video_file_ref(message)
        if not file_id:
            await self._send_message(chat_id, "Video payload is invalid.")
            return
        question = self._extract_caption(message) or f"Please analyze this video ({suggested_name})."
        try:
            data, resolved_name = await self._download_telegram_file(file_id, suggested_name)
        except Exception as e:
            self._log_api_error("video_analyze", e)
            await self._send_message(chat_id, f"Failed to read video: {e}")
            return
        await self._process_video_bytes(chat_id, data, resolved_name, question)

    async def _process_video_bytes(
        self,
        chat_id: Any,
        data: bytes,
        file_name: str,
        question: str,
    ) -> None:
        await self._send_user_update(chat_id, "Analyzing video...")
        try:
            local_path = self._save_attachment_bytes(data, "video", file_name)
            video_ref = self._to_file_ref(local_path)
            video_result = await self._analyze_video(video_ref, question)
            turn_text = self._build_video_turn_text(
                question=question,
                file_name=file_name,
                result=video_result,
            )
        except Exception as e:
            self._log_api_error("video_analyze", e)
            await self._send_message(chat_id, f"Failed to process video: {e}")
            return
        await self._submit_turn(
            chat_id,
            turn_text,
            input_mode="text",
            requested_mode=self._media_submit_requested_mode,
            skip_distillation=self._media_skip_distillation,
        )

    async def _handle_document_attachment(
        self,
        message: Dict[str, Any],
        chat_id: Any,
    ) -> None:
        document = message.get("document")
        if not isinstance(document, dict):
            await self._send_message(chat_id, "Document payload is invalid.")
            return
        if self._is_image_document(document):
            await self._handle_image_attachment(message, chat_id)
            return
        if self._is_video_document(document):
            await self._handle_video_attachment(message, chat_id)
            return

        file_name = str(document.get("file_name", "")).strip() or "document.txt"
        if self._is_pdf_document(document):
            await self._handle_pdf_attachment(message, chat_id)
            return
        if not self._is_text_document(document):
            if self._unsupported_files_to_agent:
                await self._handle_unsupported_document_with_complex(document, message, chat_id)
            else:
                await self._send_message(chat_id, "Unsupported file type. Send a text/code file for now.")
            return

        file_id = str(document.get("file_id", "")).strip()
        if not file_id:
            await self._send_message(chat_id, "Document payload is missing file_id.")
            return

        try:
            data, resolved_name = await self._download_telegram_file(file_id, file_name)
        except Exception as e:
            self._log_api_error("document_read", e)
            await self._send_message(chat_id, f"Failed to read file: {e}")
            return

        await self._process_text_document_bytes(chat_id, data, resolved_name, self._extract_caption(message))

    async def _process_text_document_bytes(
        self,
        chat_id: Any,
        data: bytes,
        file_name: str,
        caption: str,
    ) -> None:
        await self._send_user_update(chat_id, "Reading file...")
        try:
            text_content, truncated = self._decode_text_attachment(data)
        except Exception as e:
            self._log_api_error("document_read", e)
            await self._send_message(chat_id, f"Failed to read file: {e}")
            return

        if not text_content:
            await self._send_message(chat_id, "The file appears empty.")
            return

        turn_text = self._build_text_document_turn_text(
            file_name=file_name,
            user_query=caption,
            content=text_content,
            truncated=truncated,
        )
        await self._submit_turn(chat_id, turn_text, input_mode="text")

    async def _handle_pdf_attachment(
        self,
        message: Dict[str, Any],
        chat_id: Any,
    ) -> None:
        document = message.get("document")
        if not isinstance(document, dict):
            await self._send_message(chat_id, "PDF payload is invalid.")
            return
        file_name = str(document.get("file_name", "")).strip() or "document.pdf"
        file_id = str(document.get("file_id", "")).strip()
        if not file_id:
            await self._send_message(chat_id, "PDF payload is missing file_id.")
            return

        try:
            data, resolved_name = await self._download_telegram_file(file_id, file_name)
        except Exception as e:
            self._log_api_error("pdf_read", e)
            await self._send_message(chat_id, f"Failed to read PDF: {e}")
            return

        await self._process_pdf_bytes(chat_id, data, resolved_name, self._extract_caption(message))

    async def _process_pdf_bytes(
        self,
        chat_id: Any,
        data: bytes,
        file_name: str,
        caption: str,
    ) -> None:
        await self._send_user_update(chat_id, "Reading PDF...")
        try:
            pdf_text, meta = self._extract_pdf_text(data)
        except Exception as e:
            self._log_api_error("pdf_read", e)
            await self._send_message(chat_id, f"Failed to read PDF: {e}")
            return

        if not pdf_text:
            await self._send_message(
                chat_id,
                (
                    "I couldn't extract text from this PDF. "
                    "It may be scanned/image-only; OCR for PDFs is not enabled yet."
                ),
            )
            return

        turn_text = self._build_pdf_turn_text(
            file_name=file_name,
            user_query=caption,
            content=pdf_text,
            meta=meta,
        )
        await self._submit_turn(chat_id, turn_text, input_mode="text")

    async def _handle_unsupported_document_with_complex(
        self,
        document: Dict[str, Any],
        message: Dict[str, Any],
        chat_id: Any,
    ) -> None:
        file_id = str(document.get("file_id", "")).strip()
        if not file_id:
            await self._send_message(chat_id, "Document payload is missing file_id.")
            return
        file_name = str(document.get("file_name", "")).strip() or "document.bin"
        mime_type = str(document.get("mime_type", "")).strip() or "unknown"

        try:
            data, resolved_name = await self._download_telegram_file(file_id, file_name)
        except Exception as e:
            self._log_api_error("unsupported_document_read", e)
            await self._send_message(chat_id, f"Failed to read file: {e}")
            return

        await self._process_unsupported_bytes(
            chat_id=chat_id,
            data=data,
            file_name=resolved_name,
            mime_type=mime_type,
            caption=self._extract_caption(message),
            intro_line="User sent an unsupported file attachment on Telegram.",
        )

    def _extract_other_attachment_ref(self, message: Dict[str, Any]) -> Tuple[str, str, str, str]:
        candidates: List[Tuple[str, Any, str]] = [
            ("video", message.get("video"), "video.mp4"),
            ("animation", message.get("animation"), "animation.mp4"),
            ("video_note", message.get("video_note"), "video_note.mp4"),
            ("sticker", message.get("sticker"), "sticker.webp"),
        ]
        for kind, payload, default_name in candidates:
            if not isinstance(payload, dict):
                continue
            file_id = str(payload.get("file_id", "")).strip()
            if not file_id:
                continue
            file_name = str(payload.get("file_name", "")).strip()
            if not file_name:
                unique = str(payload.get("file_unique_id", "")).strip()
                file_name = f"{kind}_{unique}" if unique else default_name
            mime_type = str(payload.get("mime_type", "")).strip() or "unknown"
            return kind, file_id, file_name, mime_type
        return "", "", "", ""

    async def _handle_other_attachment_with_complex(
        self,
        message: Dict[str, Any],
        chat_id: Any,
    ) -> bool:
        kind, file_id, file_name, mime_type = self._extract_other_attachment_ref(message)
        if not file_id:
            return False

        try:
            data, resolved_name = await self._download_telegram_file(file_id, file_name)
        except Exception as e:
            self._log_api_error("other_attachment_read", e)
            await self._send_message(chat_id, f"Failed to read file: {e}")
            return True

        await self._process_unsupported_bytes(
            chat_id=chat_id,
            data=data,
            file_name=resolved_name,
            mime_type=mime_type,
            caption=self._extract_caption(message),
            intro_line="User sent a non-document attachment on Telegram.",
            attachment_kind=kind,
        )
        return True

    async def _process_unsupported_bytes(
        self,
        chat_id: Any,
        data: bytes,
        file_name: str,
        mime_type: str,
        caption: str,
        intro_line: str,
        attachment_kind: Optional[str] = None,
    ) -> None:
        kind_label = attachment_kind or "file"
        await self._send_user_update(
            chat_id,
            f"Received {kind_label} attachment. Forwarding to COGNITION analysis...",
        )
        local_path = self._save_attachment_bytes(data, kind_label, file_name)
        file_ref = self._to_file_ref(local_path)
        user_request = caption or f"Please analyze this {kind_label} ({file_name})."
        details = [
            intro_line,
            *( [f"Attachment kind: {attachment_kind}"] if attachment_kind else [] ),
            f"Attachment: {file_name}",
            f"MIME type: {mime_type or 'unknown'}",
            f"File size bytes: {len(data)}",
            f"Saved path: {file_ref}",
            f"User request: {user_request}",
            "Inspect and analyze this file.",
        ]
        await self._submit_turn(
            chat_id,
            "\n".join(details),
            input_mode="text",
            requested_mode="COGNITION",
        )

    async def _send_audio_message(self, chat_id: Any, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if len(cleaned) > self._audio_response_max_chars:
            return False
        try:
            audio_path = await asyncio.to_thread(self._synthesize_tts_audio, cleaned)
        except Exception as e:
            self._log_api_error("tts_generate", e)
            return False
        try:
            return await asyncio.to_thread(self._upload_audio_response, chat_id, audio_path)
        except Exception as e:
            self._log_api_error("tts_upload", e)
            return False
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    def _synthesize_tts_audio(self, text: str) -> str:
        stamp = int(time.time() * 1000)
        path = os.path.join(self._incoming_dir, f"tts_{stamp}_{new_id().replace('-', '')[:8]}.wav")
        try:
            import pyttsx3

            engine = pyttsx3.init()
            try:
                engine.setProperty("rate", int(self._response_tts_rate))
            except Exception:
                pass
            try:
                engine.setProperty("volume", float(self._response_tts_volume))
            except Exception:
                pass
            engine.save_to_file(text, path)
            engine.runAndWait()
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
        except Exception:
            pass

        if shutil.which("espeak"):
            cmd = [
                "espeak",
                "-s",
                str(int(self._response_tts_rate)),
                "-a",
                str(int(self._response_tts_volume * 200)),
                "-w",
                path,
                text,
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    return path
            except Exception:
                pass

        self._tts_error = "No available TTS backend (pyttsx3/espeak)."
        raise RuntimeError(self._tts_error)

    def _upload_audio_response(self, chat_id: int, audio_path: str) -> bool:
        file_name = os.path.basename(audio_path) or "response.wav"
        try:
            self._telegram_api_upload_file(
                "sendAudio",
                {"chat_id": chat_id},
                "audio",
                audio_path,
                file_name=file_name,
                content_type="audio/wav",
            )
            return True
        except Exception:
            self._telegram_api_upload_file(
                "sendDocument",
                {"chat_id": chat_id},
                "document",
                audio_path,
                file_name=file_name,
                content_type="audio/wav",
            )
            return True

    async def _download_telegram_file(self, file_id: str, suggested_name: str) -> Tuple[bytes, str]:
        return await asyncio.to_thread(self._download_telegram_file_sync, file_id, suggested_name)

    def _download_telegram_file_sync(self, file_id: str, suggested_name: str) -> Tuple[bytes, str]:
        response = self._telegram_api_call("getFile", {"file_id": file_id})
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise RuntimeError("Telegram getFile returned an invalid payload.")
        file_path = str(result.get("file_path", "")).strip()
        if not file_path:
            raise RuntimeError("Telegram getFile did not return file_path.")
        file_url = f"{self._api_base}/file/bot{self._token}/{file_path}"
        request = urllib.request.Request(url=file_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_s) as resp:
                data = resp.read(self._max_download_bytes + 1)
        except urllib.error.HTTPError as e:
            details = ""
            try:
                details = e.read().decode("utf-8", errors="replace")
            except Exception:
                details = ""
            raise RuntimeError(f"Telegram file download HTTP error {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Telegram file download error: {e.reason}") from e
        if len(data) > self._max_download_bytes:
            raise RuntimeError(
                f"File is too large ({len(data)} bytes). Limit is {self._max_download_bytes} bytes."
            )
        resolved_name = os.path.basename(file_path) or suggested_name
        return data, resolved_name

    def _save_attachment_bytes(self, data: bytes, prefix: str, file_name: str) -> str:
        extension = self._file_extension(file_name)
        if not extension:
            extension = ".bin"
        stamp = int(time.time() * 1000)
        file_token = new_id().replace("-", "")[:8]
        out_name = f"{prefix}_{stamp}_{file_token}{extension}"
        out_path = os.path.join(self._incoming_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path

    def _to_file_ref(self, local_path: str) -> str:
        try:
            rel = os.path.relpath(local_path, self._workspace_root)
            if not rel.startswith(".."):
                return f"workspace:/{rel.replace(os.sep, '/')}"
        except Exception:
            pass
        return f"file:{local_path}"

    async def _analyze_image(self, image_ref: str, question: str) -> Dict[str, Any]:
        try:
            response = await send_request(
                self._model_host,
                self._model_port,
                "model.Vision",
                {"image_ref": image_ref, "question": question},
                timeout=self._model_rpc_timeout_s,
            )
        except RpcError as e:
            raise RuntimeError(f"Vision RPC failed: {e}") from e
        if not isinstance(response, dict):
            raise RuntimeError("Vision RPC returned an invalid payload.")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Vision result is missing.")
        if result.get("ok") is not True:
            reason = result.get("error") or result.get("summary") or "Vision failed."
            raise RuntimeError(str(reason))
        return result

    async def _analyze_video(self, video_ref: str, question: str) -> Dict[str, Any]:
        try:
            response = await send_request(
                self._model_host,
                self._model_port,
                "model.Video",
                {"video_ref": video_ref, "question": question},
                timeout=self._model_rpc_timeout_s,
            )
        except RpcError as e:
            raise RuntimeError(f"Video RPC failed: {e}") from e
        if not isinstance(response, dict):
            raise RuntimeError("Video RPC returned an invalid payload.")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Video result is missing.")
        if result.get("ok") is not True:
            reason = result.get("error") or result.get("summary") or "Video analysis failed."
            raise RuntimeError(str(reason))
        return result

    def _build_image_turn_text(self, question: str, file_name: str, result: Dict[str, Any]) -> str:
        summary = self._truncate(str(result.get("summary", "")).strip(), 1500)
        answer = self._truncate(str(result.get("answer", "")).strip(), 1500)
        ocr_text = self._truncate(str(result.get("text", "")).strip(), 3000)
        objects_text = self._format_objects(result.get("objects"))
        return (
            "User sent an image attachment on Telegram.\n"
            f"Attachment: {file_name}\n"
            f"User request: {question}\n"
            f"Vision summary: {summary or 'n/a'}\n"
            f"Vision direct answer: {answer or 'n/a'}\n"
            f"Vision OCR text: {ocr_text or 'n/a'}\n"
            f"Detected objects: {objects_text}\n"
            "Use this context to answer the user request."
        )

    def _build_video_turn_text(self, question: str, file_name: str, result: Dict[str, Any]) -> str:
        summary = self._truncate(str(result.get("summary", "")).strip(), 1800)
        answer = self._truncate(str(result.get("answer", "")).strip(), 1800)
        visible_text = self._truncate(str(result.get("text", "")).strip(), 3000)
        transcript = self._truncate(str(result.get("transcript", "")).strip(), 4000)
        objects_text = self._format_objects(result.get("objects"))
        duration_s = float(result.get("duration_s", 0.0) or 0.0)
        return (
            "User sent a video attachment on Telegram.\n"
            f"Attachment: {file_name}\n"
            f"User request: {question}\n"
            f"Video duration seconds: {duration_s:.3f}\n"
            f"Video summary: {summary or 'n/a'}\n"
            f"Video direct answer: {answer or 'n/a'}\n"
            f"Visible text: {visible_text or 'n/a'}\n"
            f"Audio transcript: {transcript or 'n/a'}\n"
            f"Detected objects: {objects_text}\n"
            "Use this context to answer the user request."
        )

    @staticmethod
    def _format_objects(objects: Any) -> str:
        if not isinstance(objects, list) or not objects:
            return "none"
        entries: List[str] = []
        for item in objects[:20]:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                count = item.get("count")
                if not name:
                    continue
                if isinstance(count, int) and count > 0:
                    entries.append(f"{name} x{count}")
                else:
                    entries.append(name)
            elif isinstance(item, str):
                value = item.strip()
                if value:
                    entries.append(value)
        return ", ".join(entries) if entries else "none"

    def _transcribe_audio_bytes(self, data: bytes, file_name: str) -> str:
        local_path = self._save_attachment_bytes(data, "audio", file_name)
        model = self._get_whisper_model()
        segments, _ = model.transcribe(local_path)
        parts: List[str] = []
        for seg in segments:
            text = str(getattr(seg, "text", "")).strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    def _get_whisper_model(self):
        if self._whisper_model is not None:
            return self._whisper_model
        if self._whisper_model_error:
            raise RuntimeError(self._whisper_model_error)
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            self._whisper_model_error = f"faster-whisper is unavailable: {e}"
            raise RuntimeError(self._whisper_model_error) from e

        try:
            self._whisper_model = WhisperModel(
                self._stt_model_name,
                device=self._stt_device,
                compute_type=self._stt_compute_type,
            )
        except Exception as e:
            if self._stt_device.lower() != "cpu":
                try:
                    self._whisper_model = WhisperModel(
                        self._stt_model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                except Exception:
                    self._whisper_model_error = (
                        f"Failed to initialize faster-whisper with device={self._stt_device}: {e}"
                    )
                    raise RuntimeError(self._whisper_model_error) from e
            else:
                self._whisper_model_error = (
                    f"Failed to initialize faster-whisper with device={self._stt_device}: {e}"
                )
                raise RuntimeError(self._whisper_model_error) from e
        return self._whisper_model

    def _decode_text_attachment(self, data: bytes) -> Tuple[str, bool]:
        if not data:
            return "", False
        if b"\x00" in data:
            raise RuntimeError("File appears to be binary.")
        text = data.decode("utf-8", errors="replace")
        truncated = False
        if len(text) > self._text_attachment_max_chars:
            text = text[: self._text_attachment_max_chars]
            truncated = True
        return text.strip(), truncated

    def _extract_pdf_text(self, data: bytes) -> Tuple[str, Dict[str, Any]]:
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError(f"pypdf is unavailable: {e}") from e

        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as e:
            raise RuntimeError(f"Invalid PDF file: {e}") from e

        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, self._pdf_max_pages)
        extracted_pages = 0
        chunks: List[str] = []
        for idx in range(pages_to_read):
            page = reader.pages[idx]
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            page_text = page_text.strip()
            if not page_text:
                continue
            extracted_pages += 1
            chunks.append(f"[Page {idx + 1}]\n{page_text}")

        text = "\n\n".join(chunks).strip()
        truncated = False
        if len(text) > self._pdf_max_chars:
            text = text[: self._pdf_max_chars]
            truncated = True
        meta = {
            "total_pages": total_pages,
            "pages_read": pages_to_read,
            "pages_with_text": extracted_pages,
            "truncated": truncated,
        }
        return text, meta

    def _build_text_document_turn_text(
        self,
        file_name: str,
        user_query: str,
        content: str,
        truncated: bool,
    ) -> str:
        if not user_query:
            user_query = f"Please analyze this file ({file_name})."
        truncation_note = "yes" if truncated else "no"
        return (
            "User sent a text/code file attachment on Telegram.\n"
            f"Attachment: {file_name}\n"
            f"User request: {user_query}\n"
            f"Content truncated: {truncation_note}\n"
            "File content:\n"
            f"{content}"
        )

    def _build_pdf_turn_text(
        self,
        file_name: str,
        user_query: str,
        content: str,
        meta: Dict[str, Any],
    ) -> str:
        if not user_query:
            user_query = f"Please analyze this PDF ({file_name})."
        total_pages = int(meta.get("total_pages", 0) or 0)
        pages_read = int(meta.get("pages_read", 0) or 0)
        pages_with_text = int(meta.get("pages_with_text", 0) or 0)
        truncated = "yes" if bool(meta.get("truncated")) else "no"
        return (
            "User sent a PDF attachment on Telegram.\n"
            f"Attachment: {file_name}\n"
            f"User request: {user_query}\n"
            f"PDF total pages: {total_pages}\n"
            f"PDF pages read: {pages_read}\n"
            f"PDF pages with extracted text: {pages_with_text}\n"
            f"Content truncated: {truncated}\n"
            "Extracted PDF text:\n"
            f"{content}"
        )

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    async def _handle_patch(self, line: str, chat_id: Any) -> None:
        if not self._session.last_turn_id:
            await self._send_message(chat_id, "No turn to patch yet.")
            return
        appended = line[1:].strip()
        if not appended:
            await self._send_message(chat_id, "Patch text is empty.")
            return
        await self._session.patch_turn(self._session.last_turn_id, appended)
        await self._send_message(chat_id, "Patch sent.")

    async def _handle_permission_command(
        self,
        line: str,
        chat_id: Any,
    ) -> None:
        parts = line.split(maxsplit=1)
        action = parts[0]
        token = parts[1].strip() if len(parts) > 1 else None
        request_id = self._resolve_permission_id(token)
        if not request_id:
            await self._send_message(chat_id, "No matching pending permission request.")
            return
        owner_chat = self._permission_chat.get(request_id)
        if owner_chat is not None and owner_chat != chat_id:
            await self._send_message(chat_id, "This permission request belongs to another chat.")
            return
        approved = action == "/approve"
        sent = await self._send_permission_decision(request_id, approved)
        if not sent:
            await self._send_message(chat_id, "Permission request is no longer pending.")
            return
        decision = "Approved" if approved else "Denied"
        await self._send_message(chat_id, f"{decision} request {request_id}.")

    def _resolve_permission_id(self, token: Optional[str]) -> Optional[str]:
        resolution = self._session.resolve_permission_id(token)
        if resolution.status == "matched":
            return resolution.request_id
        return None

    async def _send_permission_decision(
        self,
        request_id: str,
        approved: bool,
    ) -> bool:
        sent = await self._session.send_permission_decision(request_id, approved)
        self._permission_chat.pop(request_id, None)
        return sent

    async def _send_help(self, chat_id: Any) -> None:
        await self._send_message(
            chat_id,
            (
                "Send text, audio, images, videos, PDFs, or text/code files to chat with the assistant.\n"
                "Commands:\n"
                "/status - show daemon status\n"
                "/approve <request_id> - approve a permission request\n"
                "/deny <request_id> - deny a permission request\n"
                "+ <extra text> - patch the previous turn\n"
                "/wake - send a wake event"
            ),
        )

    async def _send_status(self, chat_id: Any) -> None:
        await self._send_message(
            chat_id,
            (
                f"Active chat: {self._active_chat_id}\n"
                f"Pending permissions: {len(self._session.pending_permissions)}\n"
                f"Last turn id: {self._session.last_turn_id or 'none'}\n"
                f"Response mode: {self._response_mode}\n"
                f"User updates: {'on' if self._send_user_updates else 'off'}\n"
                f"Unsupported->COGNITION: {'on' if self._unsupported_files_to_agent else 'off'}\n"
                f"Download limit: {self._max_download_bytes} bytes\n"
                f"Media mode: {self._media_submit_requested_mode}\n"
                f"Media skip distill: {'on' if self._media_skip_distillation else 'off'}\n"
                f"Text attachment limit: {self._text_attachment_max_chars} chars\n"
                f"PDF page limit: {self._pdf_max_pages}\n"
                f"PDF text limit: {self._pdf_max_chars} chars"
            ),
        )

    async def _send_chat_action(self, chat_id: Any, action: str) -> None:
        payload = {"chat_id": chat_id, "action": action}
        try:
            await asyncio.to_thread(self._telegram_api_call, "sendChatAction", payload)
        except Exception:
            pass

    async def _send_message(self, chat_id: Any, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        for chunk in self._split_message(cleaned):
            payload = {"chat_id": chat_id, "text": chunk}
            try:
                await asyncio.to_thread(self._telegram_api_call, "sendMessage", payload)
            except Exception as e:
                self._log_api_error("sendMessage", e)
                return

    async def _send_user_update(self, chat_id: Any, text: str) -> None:
        if not self._send_user_updates:
            return
        await self._send_message(chat_id, text)

    def _split_message(self, text: str) -> List[str]:
        if len(text) <= self._max_message_chars:
            return [text]
        chunks: List[str] = []
        remaining = text
        limit = self._max_message_chars
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break
            cut = remaining.rfind("\n", 0, limit)
            if cut < limit // 3:
                cut = remaining.rfind(" ", 0, limit)
            if cut < limit // 3:
                cut = limit
            chunk = remaining[:cut]
            if not chunk.strip():
                chunk = remaining[:limit]
                cut = len(chunk)
            chunks.append(chunk.strip())
            remaining = remaining[cut:].lstrip()
        return chunks

    def _telegram_api_call(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._api_base}/bot{self._token}/{method}"
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(url=url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            details = ""
            try:
                details = e.read().decode("utf-8", errors="replace")
            except Exception:
                details = ""
            raise RuntimeError(f"Telegram HTTP error {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(self._format_telegram_network_error(e.reason)) from e

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError("Telegram returned an invalid JSON payload.") from e
        if not isinstance(parsed, dict):
            raise RuntimeError("Telegram returned an unexpected response type.")
        if parsed.get("ok") is not True:
            description = parsed.get("description", "unknown Telegram API error")
            raise RuntimeError(f"Telegram API error: {description}")
        return parsed

    def _telegram_api_upload_file(
        self,
        method: str,
        fields: Dict[str, Any],
        file_field: str,
        file_path: str,
        file_name: str,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        url = f"{self._api_base}/bot{self._token}/{method}"
        boundary = "----telegram-boundary-" + new_id().replace("-", "")
        body = bytearray()

        for key, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        request = urllib.request.Request(url=url, data=bytes(body), method="POST")
        request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            details = ""
            try:
                details = e.read().decode("utf-8", errors="replace")
            except Exception:
                details = ""
            raise RuntimeError(f"Telegram HTTP error {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(self._format_telegram_network_error(e.reason)) from e

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError("Telegram returned an invalid JSON payload.") from e
        if not isinstance(parsed, dict):
            raise RuntimeError("Telegram returned an unexpected response type.")
        if parsed.get("ok") is not True:
            description = parsed.get("description", "unknown Telegram API error")
            raise RuntimeError(f"Telegram API error: {description}")
        return parsed

    def _format_telegram_network_error(self, reason: Any) -> str:
        detail = str(reason).strip() or "unknown network error"
        host = urllib.parse.urlparse(self._api_base).hostname or self._api_base
        hints: List[str] = []
        if isinstance(reason, socket.gaierror) or "getaddrinfo failed" in detail.lower():
            hints.append(f"DNS lookup failed for Telegram host '{host}'.")
            hints.append(
                "Check TELEGRAM_API_BASE_URL or telegram.api_base_url, and verify this machine can resolve and reach that host."
            )
        if hints:
            return f"Telegram network error: {detail}. {' '.join(hints)}"
        return f"Telegram network error: {detail}"

    def _log_api_error(self, source: str, err: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_log_ts < 5.0:
            return
        self._last_error_log_ts = now
        print(f"[telegram] {source} failed: {err}", flush=True)


async def main() -> None:
    daemon = TelegramDaemon()
    await daemon.run()


if __name__ == "__main__":
    asyncio.run(main())
