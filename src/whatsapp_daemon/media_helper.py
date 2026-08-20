import argparse
import asyncio
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from common.config import load_settings
from common.jsonl_rpc import RpcError, send_request


class MediaHelper:
    def __init__(self) -> None:
        self._settings = load_settings()
        self._workspace_root = self._settings.workspace_root
        self._whisper_model: Any = None
        self._whisper_model_error: Optional[str] = None

    def to_file_ref(self, local_path: str) -> str:
        abs_path = os.path.abspath(local_path)
        try:
            rel = os.path.relpath(abs_path, self._workspace_root)
            if not rel.startswith(".."):
                return f"workspace:/{rel.replace(os.sep, '/')}"
        except Exception:
            pass
        return f"file:{abs_path}"

    def decode_text_attachment(self, local_path: str, max_chars: int) -> Tuple[str, bool]:
        with open(local_path, "rb") as f:
            data = f.read()
        if not data:
            return "", False
        if b"\x00" in data:
            raise RuntimeError("File appears to be binary.")
        text = data.decode("utf-8", errors="replace")
        truncated = False
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        return text.strip(), truncated

    def extract_pdf_text(self, local_path: str, max_pages: int, max_chars: int) -> Tuple[str, Dict[str, Any]]:
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError(f"pypdf is unavailable: {e}") from e

        try:
            with open(local_path, "rb") as f:
                reader = PdfReader(io.BytesIO(f.read()))
        except Exception as e:
            raise RuntimeError(f"Invalid PDF file: {e}") from e

        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages)
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
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        meta = {
            "total_pages": total_pages,
            "pages_read": pages_to_read,
            "pages_with_text": extracted_pages,
            "truncated": truncated,
        }
        return text, meta

    def transcribe_audio_path(self, local_path: str, model_name: str, device: str, compute_type: str) -> str:
        model = self._get_whisper_model(model_name, device, compute_type)
        segments, _ = model.transcribe(local_path)
        parts: List[str] = []
        for seg in segments:
            text = str(getattr(seg, "text", "")).strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    def _get_whisper_model(self, model_name: str, device: str, compute_type: str):
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
            self._whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:
            if device.lower() != "cpu":
                try:
                    self._whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
                except Exception:
                    self._whisper_model_error = (
                        f"Failed to initialize faster-whisper with device={device}: {e}"
                    )
                    raise RuntimeError(self._whisper_model_error) from e
            else:
                self._whisper_model_error = (
                    f"Failed to initialize faster-whisper with device={device}: {e}"
                )
                raise RuntimeError(self._whisper_model_error) from e
        return self._whisper_model

    def synthesize_tts_audio(
        self,
        text: str,
        out_dir: str,
        tts_rate: int,
        tts_volume: float,
        ffmpeg_binary: str,
    ) -> Dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            raise RuntimeError("TTS text is empty.")

        os.makedirs(out_dir, exist_ok=True)
        stamp = int(time.time() * 1000)
        token = secrets.token_hex(4)
        wav_path = os.path.join(out_dir, f"tts_{stamp}_{token}.wav")

        try:
            import pyttsx3

            engine = pyttsx3.init()
            try:
                engine.setProperty("rate", int(tts_rate))
            except Exception:
                pass
            try:
                engine.setProperty("volume", float(tts_volume))
            except Exception:
                pass
            engine.save_to_file(cleaned, wav_path)
            engine.runAndWait()
            if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
                return self._finalize_tts_audio(wav_path, ffmpeg_binary)
        except Exception:
            pass

        if shutil.which("espeak"):
            cmd = [
                "espeak",
                "-s",
                str(int(tts_rate)),
                "-a",
                str(int(max(0.0, min(1.0, float(tts_volume))) * 200)),
                "-w",
                wav_path,
                cleaned,
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
                    return self._finalize_tts_audio(wav_path, ffmpeg_binary)
            except Exception:
                pass

        raise RuntimeError("No available TTS backend (pyttsx3/espeak).")

    def _finalize_tts_audio(self, wav_path: str, ffmpeg_binary: str) -> Dict[str, Any]:
        ffmpeg_name = str(ffmpeg_binary or "ffmpeg").strip() or "ffmpeg"
        if shutil.which(ffmpeg_name):
            ogg_path = os.path.splitext(wav_path)[0] + ".ogg"
            command = [
                ffmpeg_name,
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                wav_path,
                "-avoid_negative_ts",
                "make_zero",
                "-ac",
                "1",
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                ogg_path,
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode == 0 and os.path.isfile(ogg_path) and os.path.getsize(ogg_path) > 0:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                return {
                    "audio_path": ogg_path,
                    "mimetype": "audio/ogg; codecs=opus",
                }
        return {
            "audio_path": wav_path,
            "mimetype": "audio/wav",
        }

    async def analyze_image(
        self,
        local_path: str,
        question: str,
        model_host: str,
        model_port: int,
        timeout_s: int,
    ) -> Dict[str, Any]:
        image_ref = self.to_file_ref(local_path)
        try:
            response = await send_request(
                model_host,
                model_port,
                "model.Vision",
                {"image_ref": image_ref, "question": question},
                timeout=timeout_s,
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

    async def analyze_video(
        self,
        local_path: str,
        question: str,
        model_host: str,
        model_port: int,
        timeout_s: int,
    ) -> Dict[str, Any]:
        video_ref = self.to_file_ref(local_path)
        try:
            response = await send_request(
                model_host,
                model_port,
                "model.Video",
                {"video_ref": video_ref, "question": question},
                timeout=timeout_s,
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

    @staticmethod
    def truncate(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    @staticmethod
    def format_objects(objects: Any) -> str:
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

    def build_audio_turn_text(self, platform: str, transcript: str, caption: str) -> str:
        if caption:
            return (
                f"User sent an audio message on {platform} with an instruction.\n"
                f"Instruction: {caption}\n"
                f"Transcribed audio: {transcript}"
            )
        return transcript

    def build_image_turn_text(self, platform: str, question: str, file_name: str, result: Dict[str, Any]) -> str:
        summary = self.truncate(str(result.get("summary", "")).strip(), 1500)
        answer = self.truncate(str(result.get("answer", "")).strip(), 1500)
        ocr_text = self.truncate(str(result.get("text", "")).strip(), 3000)
        objects_text = self.format_objects(result.get("objects"))
        return (
            f"User sent an image attachment on {platform}.\n"
            f"Attachment: {file_name}\n"
            f"User request: {question}\n"
            f"Vision summary: {summary or 'n/a'}\n"
            f"Vision direct answer: {answer or 'n/a'}\n"
            f"Vision OCR text: {ocr_text or 'n/a'}\n"
            f"Detected objects: {objects_text}\n"
            "Use this context to answer the user request."
        )

    def build_video_turn_text(self, platform: str, question: str, file_name: str, result: Dict[str, Any]) -> str:
        summary = self.truncate(str(result.get("summary", "")).strip(), 1800)
        answer = self.truncate(str(result.get("answer", "")).strip(), 1800)
        visible_text = self.truncate(str(result.get("text", "")).strip(), 3000)
        transcript = self.truncate(str(result.get("transcript", "")).strip(), 4000)
        objects_text = self.format_objects(result.get("objects"))
        duration_s = float(result.get("duration_s", 0.0) or 0.0)
        return (
            f"User sent a video attachment on {platform}.\n"
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

    def build_text_document_turn_text(
        self,
        platform: str,
        file_name: str,
        user_query: str,
        content: str,
        truncated: bool,
    ) -> str:
        if not user_query:
            user_query = f"Please analyze this file ({file_name})."
        truncation_note = "yes" if truncated else "no"
        return (
            f"User sent a text/code file attachment on {platform}.\n"
            f"Attachment: {file_name}\n"
            f"User request: {user_query}\n"
            f"Content truncated: {truncation_note}\n"
            "File content:\n"
            f"{content}"
        )

    def build_pdf_turn_text(
        self,
        platform: str,
        file_name: str,
        user_query: str,
        content: str,
        meta: Mapping[str, Any],
    ) -> str:
        if not user_query:
            user_query = f"Please analyze this PDF ({file_name})."
        total_pages = int(meta.get("total_pages", 0) or 0)
        pages_read = int(meta.get("pages_read", 0) or 0)
        pages_with_text = int(meta.get("pages_with_text", 0) or 0)
        truncated = "yes" if bool(meta.get("truncated")) else "no"
        return (
            f"User sent a PDF attachment on {platform}.\n"
            f"Attachment: {file_name}\n"
            f"User request: {user_query}\n"
            f"PDF total pages: {total_pages}\n"
            f"PDF pages read: {pages_read}\n"
            f"PDF pages with extracted text: {pages_with_text}\n"
            f"Content truncated: {truncated}\n"
            "Extracted PDF text:\n"
            f"{content}"
        )


def ok_result(turn_text: str, input_mode: str = "text") -> Dict[str, Any]:
    return {"ok": True, "turn_text": turn_text, "input_mode": input_mode}


def error_result(message: str) -> Dict[str, Any]:
    return {"ok": False, "user_message": message}


async def handle_audio(args: argparse.Namespace, helper: MediaHelper) -> Dict[str, Any]:
    transcript = helper.transcribe_audio_path(args.file, args.stt_model, args.stt_device, args.stt_compute_type)
    if not transcript:
        return error_result("I couldn't detect speech in that audio.")
    turn_text = helper.build_audio_turn_text(args.platform, transcript, args.caption or "")
    return ok_result(turn_text, input_mode="audio")


async def handle_image(args: argparse.Namespace, helper: MediaHelper) -> Dict[str, Any]:
    result = await helper.analyze_image(
        args.file,
        args.question,
        args.model_host,
        int(args.model_port),
        int(args.timeout_s),
    )
    file_name = args.display_name or os.path.basename(args.file)
    turn_text = helper.build_image_turn_text(args.platform, args.question, file_name, result)
    return ok_result(turn_text)


async def handle_video(args: argparse.Namespace, helper: MediaHelper) -> Dict[str, Any]:
    result = await helper.analyze_video(
        args.file,
        args.question,
        args.model_host,
        int(args.model_port),
        int(args.timeout_s),
    )
    file_name = args.display_name or os.path.basename(args.file)
    turn_text = helper.build_video_turn_text(args.platform, args.question, file_name, result)
    return ok_result(turn_text)


async def handle_text(args: argparse.Namespace, helper: MediaHelper) -> Dict[str, Any]:
    text_content, truncated = helper.decode_text_attachment(args.file, int(args.max_chars))
    if not text_content:
        return error_result("The file appears empty.")
    file_name = args.display_name or os.path.basename(args.file)
    turn_text = helper.build_text_document_turn_text(
        args.platform,
        file_name,
        args.caption or "",
        text_content,
        truncated,
    )
    return ok_result(turn_text)


async def handle_pdf(args: argparse.Namespace, helper: MediaHelper) -> Dict[str, Any]:
    pdf_text, meta = helper.extract_pdf_text(args.file, int(args.max_pages), int(args.max_chars))
    if not pdf_text:
        return error_result(
            "I couldn't extract text from this PDF. It may be scanned/image-only; OCR for PDFs is not enabled yet."
        )
    file_name = args.display_name or os.path.basename(args.file)
    turn_text = helper.build_pdf_turn_text(
        args.platform,
        file_name,
        args.caption or "",
        pdf_text,
        meta,
    )
    return ok_result(turn_text)


async def handle_tts(args: argparse.Namespace, helper: MediaHelper) -> Dict[str, Any]:
    result = helper.synthesize_tts_audio(
        args.text,
        args.out_dir,
        int(args.tts_rate),
        float(args.tts_volume),
        args.ffmpeg_binary,
    )
    return {
        "ok": True,
        "audio_path": result["audio_path"],
        "mimetype": result["mimetype"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WhatsApp media helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audio = subparsers.add_parser("audio")
    audio.add_argument("--file", required=True)
    audio.add_argument("--caption", default="")
    audio.add_argument("--platform", default="WhatsApp")
    audio.add_argument("--stt-model", required=True)
    audio.add_argument("--stt-device", required=True)
    audio.add_argument("--stt-compute-type", required=True)

    image = subparsers.add_parser("image")
    image.add_argument("--file", required=True)
    image.add_argument("--display-name", default="")
    image.add_argument("--question", required=True)
    image.add_argument("--platform", default="WhatsApp")
    image.add_argument("--model-host", required=True)
    image.add_argument("--model-port", required=True)
    image.add_argument("--timeout-s", required=True)

    video = subparsers.add_parser("video")
    video.add_argument("--file", required=True)
    video.add_argument("--display-name", default="")
    video.add_argument("--question", required=True)
    video.add_argument("--platform", default="WhatsApp")
    video.add_argument("--model-host", required=True)
    video.add_argument("--model-port", required=True)
    video.add_argument("--timeout-s", required=True)

    text = subparsers.add_parser("text")
    text.add_argument("--file", required=True)
    text.add_argument("--display-name", default="")
    text.add_argument("--caption", default="")
    text.add_argument("--platform", default="WhatsApp")
    text.add_argument("--max-chars", required=True)

    pdf = subparsers.add_parser("pdf")
    pdf.add_argument("--file", required=True)
    pdf.add_argument("--display-name", default="")
    pdf.add_argument("--caption", default="")
    pdf.add_argument("--platform", default="WhatsApp")
    pdf.add_argument("--max-pages", required=True)
    pdf.add_argument("--max-chars", required=True)

    tts = subparsers.add_parser("tts")
    tts.add_argument("--text", required=True)
    tts.add_argument("--out-dir", required=True)
    tts.add_argument("--tts-rate", required=True)
    tts.add_argument("--tts-volume", required=True)
    tts.add_argument("--ffmpeg-binary", default="ffmpeg")

    return parser


async def main_async() -> int:
    parser = build_parser()
    args = parser.parse_args()
    helper = MediaHelper()

    try:
        if args.command == "audio":
            result = await handle_audio(args, helper)
        elif args.command == "image":
            result = await handle_image(args, helper)
        elif args.command == "video":
            result = await handle_video(args, helper)
        elif args.command == "text":
            result = await handle_text(args, helper)
        elif args.command == "pdf":
            result = await handle_pdf(args, helper)
        elif args.command == "tts":
            result = await handle_tts(args, helper)
        else:
            result = error_result(f"Unsupported helper command: {args.command}")
    except Exception as e:
        result = error_result(str(e))

    json.dump(result, sys.stdout, ensure_ascii=True)
    sys.stdout.write("\n")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
