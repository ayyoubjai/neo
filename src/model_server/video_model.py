from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from common.config import load_settings
from common.error_log import log_exception
from common.record_log import record_event
from model_server.frame_sequence import (
    aggregate_objects,
    answer_from_keyframes,
    build_frame_prompt,
    collect_visible_text,
    summary_from_keyframes,
)
from model_server.vision_model import analyze as analyze_image
from tool_runtime.sandbox import SandboxViolation, resolve_workspace_path


_WHISPER_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
_WHISPER_MODEL_ERROR_CACHE: Dict[Tuple[str, str, str], str] = {}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_sampling_policy(duration_s: float, question: str = "", cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = dict(cfg or {})
    duration_s = max(1.0, float(duration_s))
    sampling_mode = str(config.get("sampling_mode", "adaptive")).strip().lower()
    if sampling_mode != "adaptive":
        sampling_mode = "adaptive"

    min_baseline_frames = int(config.get("min_baseline_frames", 6))
    max_baseline_frames = int(config.get("max_baseline_frames", 24))
    min_spacing_s = float(config.get("min_spacing_s", 1.0))
    max_spacing_s = float(config.get("max_spacing_s", 90.0))
    scene_ratio = float(config.get("scene_ratio", 0.5))
    min_scene_frames = int(config.get("min_scene_frames", 2))
    max_scene_frames = int(config.get("max_scene_frames", 12))
    max_total_frames = int(config.get("max_total_frames", 32))
    growth_divisor = float(config.get("duration_growth_divisor", 20.0))
    growth_weight = float(config.get("duration_growth_weight", 4.5))
    question_frame_bonus = int(config.get("question_frame_bonus", 2))
    question_scene_bonus = int(config.get("question_scene_bonus", 3))

    baseline_frames = round(min_baseline_frames + growth_weight * math.log1p(duration_s / max(1.0, growth_divisor)))
    baseline_frames = int(clamp(baseline_frames, min_baseline_frames, max_baseline_frames))

    if baseline_frames <= 1:
        baseline_spacing_s = duration_s
    else:
        baseline_spacing_s = duration_s / float(baseline_frames - 1)
    baseline_spacing_s = float(clamp(baseline_spacing_s, min_spacing_s, max_spacing_s))

    scene_frame_budget = round(baseline_frames * scene_ratio)
    scene_frame_budget = int(clamp(scene_frame_budget, min_scene_frames, max_scene_frames))
    min_timestamp_gap_s = max(0.75, baseline_spacing_s * 0.35)

    normalized = " ".join((question or "").lower().split())
    if any(token in normalized for token in ("read", "text", "screen", "caption", "subtitle", "sign")):
        baseline_frames = int(clamp(baseline_frames + question_frame_bonus, min_baseline_frames, max_baseline_frames))
        scene_frame_budget = int(clamp(scene_frame_budget + question_frame_bonus, min_scene_frames, max_scene_frames))
        min_timestamp_gap_s = max(0.5, min_timestamp_gap_s * 0.8)

    if any(token in normalized for token in ("when", "moment", "happens", "enter", "leave", "change")):
        scene_frame_budget = int(clamp(scene_frame_budget + question_scene_bonus, min_scene_frames, max_scene_frames))

    max_total_frames = max(1, max_total_frames)
    total_cap = min(max_total_frames, baseline_frames + scene_frame_budget)
    return {
        "sampling_mode": sampling_mode,
        "baseline_frames": baseline_frames,
        "baseline_spacing_s": baseline_spacing_s,
        "scene_frame_budget": scene_frame_budget,
        "min_timestamp_gap_s": min_timestamp_gap_s,
        "max_total_frames": total_cap,
    }


def _resolve_video_path(video_ref: str) -> str:
    if video_ref.startswith("workspace:/"):
        settings = load_settings()
        return resolve_workspace_path(settings.workspace_root, video_ref)
    if video_ref.startswith("file:"):
        return video_ref[len("file:") :]
    return video_ref


def _to_workspace_or_file_ref(path: str, workspace_root: str) -> str:
    abs_root = os.path.abspath(workspace_root)
    abs_path = os.path.abspath(path)
    if abs_path == abs_root or abs_path.startswith(abs_root + os.sep):
        rel = os.path.relpath(abs_path, abs_root).replace("\\", "/")
        return f"workspace:/{rel}"
    return f"file:{abs_path}"


def _load_runtime():
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"Video analysis requires opencv-python and numpy: {e}") from e
    return cv2, np


def _open_capture(cv2, video_path: str):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Unable to open video")
    return capture


def _video_metadata(cv2, capture, video_ref: str, video_path: str) -> Dict[str, Any]:
    fps = float(capture.get(getattr(cv2, "CAP_PROP_FPS", 5)) or 0.0)
    frame_count = int(capture.get(getattr(cv2, "CAP_PROP_FRAME_COUNT", 7)) or 0)
    width = int(capture.get(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)) or 0)
    height = int(capture.get(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)) or 0)
    duration_s = 0.0
    if fps > 0 and frame_count > 0:
        duration_s = frame_count / fps
    return {
        "video_ref": video_ref,
        "video_path": video_path,
        "duration_s": round(float(duration_s), 3),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def _clip_interval(duration_s: float, start_s: Optional[float], end_s: Optional[float]) -> Tuple[float, float]:
    start = 0.0 if start_s is None else max(0.0, float(start_s))
    end = duration_s if end_s is None else min(duration_s, float(end_s))
    if end <= start:
        end = duration_s
    if end <= start:
        end = start + 1.0
    return start, end


def _baseline_timestamps(start_s: float, end_s: float, frame_count: int) -> List[float]:
    if frame_count <= 1 or end_s <= start_s:
        return [round(start_s, 3)]
    span = end_s - start_s
    return [round(start_s + span * (index / float(frame_count - 1)), 3) for index in range(frame_count)]


def _read_frame_at(capture, cv2, timestamp_s: float):
    capture.set(getattr(cv2, "CAP_PROP_POS_MSEC", 0), max(0.0, float(timestamp_s)) * 1000.0)
    ok, frame = capture.read()
    if not ok:
        return None
    return frame


def _frame_signature(np, cv2, frame: Any) -> List[float]:
    resized = cv2.resize(frame, (64, 64))
    features: List[float] = []
    for channel_index in range(3):
        channel = resized[:, :, channel_index]
        hist, _ = np.histogram(channel, bins=16, range=(0, 256))
        hist = hist.astype(np.float32)
        total = float(hist.sum()) or 1.0
        features.extend((hist / total).tolist())
    return features


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = sum(float(a) * float(a) for a in left) ** 0.5
    norm_right = sum(float(b) * float(b) for b in right) ** 0.5
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _scene_change_timestamps(
    capture,
    cv2,
    np,
    start_s: float,
    end_s: float,
    probe_spacing_s: float,
    threshold: float,
) -> List[float]:
    timestamps: List[float] = []
    last_signature: Optional[List[float]] = None
    current = start_s
    while current <= end_s:
        frame = _read_frame_at(capture, cv2, current)
        if frame is None:
            current += probe_spacing_s
            continue
        signature = _frame_signature(np, cv2, frame)
        if last_signature is not None:
            diff = 1.0 - _cosine(last_signature, signature)
            if diff >= threshold:
                timestamps.append(round(current, 3))
        last_signature = signature
        current += probe_spacing_s
    return timestamps


def _merge_timestamps(
    baseline: Sequence[float],
    scene_changes: Sequence[float],
    start_s: float,
    end_s: float,
    min_gap_s: float,
    max_total_frames: int,
) -> List[float]:
    merged = sorted({round(start_s, 3), round(end_s, 3), *baseline, *scene_changes})
    result: List[float] = []
    for timestamp in merged:
        if not result:
            result.append(timestamp)
            continue
        if timestamp - result[-1] < min_gap_s:
            continue
        result.append(timestamp)
        if len(result) >= max_total_frames:
            break
    if result and result[-1] != round(end_s, 3) and len(result) < max_total_frames:
        if end_s - result[-1] >= min_gap_s:
            result.append(round(end_s, 3))
    return result[:max_total_frames]


def _ensure_frame_dir(settings, keep_frames: bool) -> Tuple[str, bool]:
    frame_root = str(settings.video.get("sampled_frame_dir", os.path.join(settings.data_dir, "video_frames")) or "")
    os.makedirs(frame_root, exist_ok=True)
    if keep_frames:
        frame_dir = os.path.join(frame_root, f"video_sample_{int(time.time() * 1000)}")
        os.makedirs(frame_dir, exist_ok=True)
        return frame_dir, False
    return tempfile.mkdtemp(prefix="video_sample_", dir=frame_root), True


def _ensure_audio_dir(settings, keep_audio: bool) -> Tuple[str, bool]:
    audio_root = str(settings.video.get("audio_extract_dir", os.path.join(settings.data_dir, "video_audio")) or "")
    os.makedirs(audio_root, exist_ok=True)
    if keep_audio:
        audio_dir = os.path.join(audio_root, f"video_audio_{int(time.time() * 1000)}")
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir, False
    return tempfile.mkdtemp(prefix="video_audio_", dir=audio_root), True


def _save_frame(cv2, frame_dir: str, frame_index: int, timestamp_s: float, frame: Any, jpeg_quality: int) -> str:
    safe_ts = f"{timestamp_s:08.3f}".replace(".", "_")
    frame_path = os.path.join(frame_dir, f"frame_{frame_index:03d}_{safe_ts}.jpg")
    params = []
    quality_flag = getattr(cv2, "IMWRITE_JPEG_QUALITY", None)
    if quality_flag is not None:
        params = [int(quality_flag), jpeg_quality]
    if params:
        ok = cv2.imwrite(frame_path, frame, params)
    else:
        ok = cv2.imwrite(frame_path, frame)
    if not ok:
        raise RuntimeError(f"Failed to write sampled frame {frame_path}")
    return frame_path


def _video_stt_config(settings) -> Tuple[str, str, str]:
    model_name = str(
        settings.video.get("stt_model")
        or settings.telegram.get("stt_model")
        or settings.voice.get("stt_model", "small.en")
    ).strip() or "small.en"
    device = str(
        settings.video.get("stt_device")
        or settings.telegram.get("stt_device")
        or settings.voice.get("stt_device", "cpu")
    ).strip() or "cpu"
    compute_type = str(
        settings.video.get("stt_compute_type")
        or settings.telegram.get("stt_compute_type")
        or settings.voice.get("stt_compute_type", "int8")
    ).strip() or "int8"
    return model_name, device, compute_type


def _get_whisper_model(settings):
    key = _video_stt_config(settings)
    if key in _WHISPER_MODEL_CACHE:
        return _WHISPER_MODEL_CACHE[key]
    if key in _WHISPER_MODEL_ERROR_CACHE:
        raise RuntimeError(_WHISPER_MODEL_ERROR_CACHE[key])
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        message = f"faster-whisper is unavailable: {e}"
        _WHISPER_MODEL_ERROR_CACHE[key] = message
        raise RuntimeError(message) from e

    model_name, device, compute_type = key
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        if device.lower() == "cpu":
            message = f"Failed to initialize faster-whisper with device={device}: {e}"
            _WHISPER_MODEL_ERROR_CACHE[key] = message
            raise RuntimeError(message) from e
        try:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception:
            message = f"Failed to initialize faster-whisper with device={device}: {e}"
            _WHISPER_MODEL_ERROR_CACHE[key] = message
            raise RuntimeError(message) from e
    _WHISPER_MODEL_CACHE[key] = model
    return model


def _extract_audio_to_wav(
    video_path: str,
    audio_dir: str,
    ffmpeg_binary: str,
    sample_rate_hz: int,
    start_s: float,
    end_s: float,
) -> str:
    output_path = os.path.join(audio_dir, "audio.wav")
    command = [ffmpeg_binary, "-nostdin", "-loglevel", "error", "-y"]
    if start_s > 0.0:
        command.extend(["-ss", f"{start_s:.3f}"])
    command.extend(["-i", video_path])
    clip_duration_s = max(0.1, float(end_s) - float(start_s))
    command.extend(
        [
            "-t",
            f"{clip_duration_s:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(max(8000, int(sample_rate_hz))),
            "-f",
            "wav",
            output_path,
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        if details:
            raise RuntimeError(f"ffmpeg audio extraction failed: {details}")
        raise RuntimeError("ffmpeg audio extraction failed")
    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError("ffmpeg audio extraction did not produce an audio file")
    return output_path


def _transcribe_audio_path(audio_path: str, settings) -> str:
    model = _get_whisper_model(settings)
    segments, _ = model.transcribe(audio_path)
    parts: List[str] = []
    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()
        if text:
            parts.append(text)
    transcript = " ".join(parts).strip()
    max_chars = max(0, int(settings.video.get("transcript_max_chars", 12000)))
    if max_chars and len(transcript) > max_chars:
        transcript = transcript[:max_chars].rstrip()
    return transcript


def _transcribe_video_audio(video_path: str, settings, start_s: float, end_s: float) -> str:
    if not bool(settings.video.get("transcription_enabled", True)):
        return ""
    ffmpeg_binary = str(settings.video.get("ffmpeg_binary", "ffmpeg") or "ffmpeg").strip() or "ffmpeg"
    sample_rate_hz = int(settings.video.get("audio_extract_sample_rate_hz", 16000))
    keep_audio = bool(settings.video.get("keep_audio_extracts", False))
    audio_dir = ""
    cleanup_dir = False
    try:
        audio_dir, cleanup_dir = _ensure_audio_dir(settings, keep_audio)
        audio_path = _extract_audio_to_wav(video_path, audio_dir, ffmpeg_binary, sample_rate_hz, start_s, end_s)
        return _transcribe_audio_path(audio_path, settings)
    finally:
        if cleanup_dir and audio_dir:
            shutil.rmtree(audio_dir, ignore_errors=True)


def analyze(
    video_ref: str,
    question: Optional[str] = None,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    transcription_enabled_override: Optional[bool] = None,
) -> Dict[str, Any]:
    settings = load_settings()
    keep_frames = bool(settings.video.get("keep_sampled_frames", False))
    scene_threshold = float(settings.video.get("scene_change_threshold", 0.18))
    scene_probe_scale = float(settings.video.get("scene_probe_scale", 0.5))
    jpeg_quality = int(settings.video.get("frame_jpeg_quality", 85))
    jpeg_quality = int(clamp(jpeg_quality, 30, 100))

    frame_dir = ""
    cleanup_dir = False
    capture = None
    transcript = ""
    try:
        video_path = _resolve_video_path(video_ref)
        if not os.path.exists(video_path):
            return {"ok": False, "video_ref": video_ref, "summary": "Video file not found.", "answer": "", "error": "Video file not found."}

        cv2, np = _load_runtime()
        capture = _open_capture(cv2, video_path)
        metadata = _video_metadata(cv2, capture, video_ref, video_path)
        duration_s = float(metadata.get("duration_s") or 0.0)
        if duration_s <= 0.0:
            return {"ok": False, "video_ref": video_ref, "summary": "Unable to read video duration.", "answer": "", "error": "Unable to read video duration."}

        clip_start_s, clip_end_s = _clip_interval(duration_s, start_s, end_s)
        clip_duration_s = max(1.0, clip_end_s - clip_start_s)
        transcription_enabled = (
            bool(transcription_enabled_override)
            if transcription_enabled_override is not None
            else bool(settings.video.get("transcription_enabled", True))
        )
        if transcription_enabled:
            try:
                transcript = _transcribe_video_audio(video_path, settings, clip_start_s, clip_end_s)
            except (RuntimeError, FileNotFoundError, OSError, subprocess.SubprocessError) as e:
                log_exception("video_transcription_error", e, {"video_ref": video_ref, "question": question or ""})
                transcript = ""
        policy = build_sampling_policy(clip_duration_s, question=question or "", cfg=settings.video)
        baseline = _baseline_timestamps(clip_start_s, clip_end_s, int(policy["baseline_frames"]))
        probe_spacing_s = max(float(policy["min_timestamp_gap_s"]), float(policy["baseline_spacing_s"]) * max(0.1, scene_probe_scale))
        scene_changes = _scene_change_timestamps(
            capture,
            cv2,
            np,
            clip_start_s,
            clip_end_s,
            probe_spacing_s,
            scene_threshold,
        )
        timestamps = _merge_timestamps(
            baseline,
            scene_changes[: int(policy["scene_frame_budget"])],
            clip_start_s,
            clip_end_s,
            float(policy["min_timestamp_gap_s"]),
            int(policy["max_total_frames"]),
        )
        if not timestamps:
            return {"ok": False, "video_ref": video_ref, "summary": "No frames could be sampled.", "answer": "", "error": "No frames could be sampled."}

        frame_dir, cleanup_dir = _ensure_frame_dir(settings, keep_frames)
        keyframes: List[Dict[str, Any]] = []
        prompt = build_frame_prompt(question, source_label="video frame")
        for index, timestamp_s in enumerate(timestamps):
            frame = _read_frame_at(capture, cv2, timestamp_s)
            if frame is None:
                continue
            frame_path = _save_frame(cv2, frame_dir, index, timestamp_s, frame, jpeg_quality)
            frame_ref = _to_workspace_or_file_ref(frame_path, settings.workspace_root)
            frame_result = analyze_image(frame_ref, question=prompt)
            if isinstance(frame_result, dict) and frame_result.get("ok") is False:
                frame_summary = str(frame_result.get("summary") or "Frame analysis failed.").strip()
                frame_text = ""
                frame_objects: List[Any] = []
            else:
                frame_summary = str(frame_result.get("summary") or "").strip()
                frame_text = str(frame_result.get("text") or "").strip()
                frame_objects = frame_result.get("objects") or []
            record: Dict[str, Any] = {
                "timestamp_s": round(float(timestamp_s), 3),
                "summary": frame_summary,
                "text": frame_text,
                "objects": frame_objects,
            }
            if keep_frames:
                record["frame_ref"] = frame_ref
            keyframes.append(record)

        if not keyframes:
            return {"ok": False, "video_ref": video_ref, "summary": "No sampled frames could be analyzed.", "answer": "", "error": "No sampled frames could be analyzed."}

        summary = summary_from_keyframes(keyframes)
        answer = answer_from_keyframes(
            question or "",
            summary,
            keyframes,
            evidence_label="video",
            duration_s=float(metadata.get("duration_s") or 0.0),
            transcript=transcript,
        )
        visible_text = collect_visible_text(keyframes)
        objects = aggregate_objects(keyframes)
        scenes = [{"timestamp_s": item["timestamp_s"], "summary": item["summary"]} for item in keyframes]
        events = [{"timestamp_s": item["timestamp_s"], "event": item["summary"]} for item in keyframes]

        result = {
            "ok": True,
            "video_ref": video_ref,
            "duration_s": metadata["duration_s"],
            "fps": metadata["fps"],
            "width": metadata["width"],
            "height": metadata["height"],
            "summary": summary,
            "answer": answer,
            "text": visible_text,
            "transcript": transcript,
            "objects": objects,
            "keyframes": keyframes,
            "scenes": scenes,
            "events": events,
            "sampling": {
                **policy,
                "clip_start_s": round(clip_start_s, 3),
                "clip_end_s": round(clip_end_s, 3),
                "timestamps": timestamps,
            },
        }
        record_event(
            "video_analysis",
            {
                "video_ref": video_ref,
                "duration_s": metadata["duration_s"],
                "sampled_frames": len(keyframes),
                "question": question or "",
                "transcript_chars": len(transcript),
            },
        )
        return result
    except (RuntimeError, FileNotFoundError, SandboxViolation, OSError) as e:
        log_exception("video_error", e, {"video_ref": video_ref, "question": question or ""})
        return {
            "ok": False,
            "video_ref": video_ref,
            "summary": f"Video analysis failed: {e}",
            "answer": "",
            "text": "",
            "transcript": transcript,
            "objects": [],
            "keyframes": [],
            "scenes": [],
            "events": [],
            "error": str(e),
        }
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if cleanup_dir and frame_dir:
            shutil.rmtree(frame_dir, ignore_errors=True)
