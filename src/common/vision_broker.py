from __future__ import annotations

import json
import os
import tempfile
import time
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional


VISION_BROKER_VERSION = 1
VISION_BROKER_MANIFEST_FILENAME = "manifest.json"


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_float(value: Any, default: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _coerce_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def broker_enabled(vision_cfg: Mapping[str, Any]) -> bool:
    return _coerce_bool(vision_cfg.get("broker_enabled", True), True)


def broker_dir_from_cfg(vision_cfg: Mapping[str, Any], data_dir: str) -> str:
    value = str(vision_cfg.get("broker_dir", "") or "").strip()
    if value:
        return value
    return os.path.join(data_dir, "vision_broker")


def broker_manifest_path(broker_dir: str) -> str:
    return os.path.join(broker_dir, VISION_BROKER_MANIFEST_FILENAME)


def workspace_ref_for_path(path: str, workspace_root: str) -> str:
    abs_root = os.path.abspath(workspace_root)
    abs_path = os.path.abspath(path)
    if abs_path == abs_root or abs_path.startswith(abs_root + os.sep):
        rel = os.path.relpath(abs_path, abs_root).replace("\\", "/")
        return f"workspace:/{rel}"
    return f"file:{abs_path}"


def scene_digest(scene: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(scene, Mapping):
        return {}
    digest: Dict[str, Any] = {}
    summary = str(scene.get("summary") or "").strip()
    if summary:
        digest["summary"] = summary
    objects = scene.get("objects")
    if isinstance(objects, list):
        compact_objects: List[Dict[str, Any]] = []
        for item in objects[:16]:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label") or item.get("name") or "").strip()
            if not label:
                continue
            try:
                count = max(1, int(item.get("count", 1)))
            except (TypeError, ValueError):
                count = 1
            compact_objects.append({"label": label, "count": count})
        if compact_objects:
            digest["objects"] = compact_objects
    entities = scene.get("entities")
    if isinstance(entities, list):
        compact_entities: List[Dict[str, Any]] = []
        for item in entities[:16]:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("object_name") or item.get("label") or "").strip()
            if not label:
                continue
            compact_entities.append({"track_id": item.get("track_id"), "label": label})
        if compact_entities:
            digest["entities"] = compact_entities
    frame_size = scene.get("frame_size")
    if isinstance(frame_size, Mapping):
        width = frame_size.get("width")
        height = frame_size.get("height")
        if isinstance(width, int) and isinstance(height, int):
            digest["frame_size"] = {"width": width, "height": height}
    signature = scene.get("signature")
    if isinstance(signature, (list, tuple)):
        digest["signature"] = list(signature)
    return digest


def load_vision_broker_manifest(manifest_path: str) -> Dict[str, Any]:
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def load_vision_broker_manifest_for_settings(settings: Any) -> Dict[str, Any]:
    broker_dir = broker_dir_from_cfg(getattr(settings, "vision", {}) or {}, getattr(settings, "data_dir", ""))
    return load_vision_broker_manifest(broker_manifest_path(broker_dir))


class VisionBroker:
    def __init__(self, workspace_root: str, data_dir: str, vision_cfg: Mapping[str, Any]):
        self._workspace_root = workspace_root
        self._vision_cfg = dict(vision_cfg or {})
        self._enabled = broker_enabled(self._vision_cfg)
        self._broker_dir = broker_dir_from_cfg(self._vision_cfg, data_dir)
        self._frames_dir = os.path.join(self._broker_dir, "frames")
        self._manifest_path = broker_manifest_path(self._broker_dir)
        self._latest_interval_s = _coerce_float(self._vision_cfg.get("broker_latest_interval_ms", 500), 500.0) / 1000.0
        self._buffer_interval_s = _coerce_float(self._vision_cfg.get("broker_buffer_frame_interval_ms", 500), 500.0) / 1000.0
        self._buffer_retention_s = _coerce_float(self._vision_cfg.get("broker_buffer_retention_s", 10.0), 1.0)
        self._buffer_max_frames = _coerce_int(self._vision_cfg.get("broker_buffer_max_frames", 32), 1)
        self._jpeg_quality = min(100, _coerce_int(self._vision_cfg.get("broker_jpeg_quality", 85), 30))
        self._latest_entry: Optional[Dict[str, Any]] = None
        self._stable_entry: Optional[Dict[str, Any]] = None
        self._buffer_entries: Deque[Dict[str, Any]] = deque()
        self._last_latest_write_ts = 0.0
        self._last_buffer_write_ts = 0.0
        self._sequence = 0
        if self._enabled:
            os.makedirs(self._frames_dir, exist_ok=True)
            self._write_manifest(status="active")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def publish_latest(self, cv2, frame: Any, scene: Optional[Mapping[str, Any]] = None) -> None:
        if not self._enabled:
            return
        now = time.time()
        digest = scene_digest(scene)
        wrote_any = False
        if self._last_latest_write_ts <= 0.0 or now - self._last_latest_write_ts >= self._latest_interval_s:
            self._latest_entry = self._write_entry(cv2, frame, now, digest, kind="latest")
            self._last_latest_write_ts = now
            wrote_any = True
        if self._last_buffer_write_ts <= 0.0 or now - self._last_buffer_write_ts >= self._buffer_interval_s:
            self._buffer_entries.append(self._write_entry(cv2, frame, now, digest, kind="buffer"))
            self._last_buffer_write_ts = now
            wrote_any = True
        if self._prune_buffer(now):
            wrote_any = True
        if wrote_any:
            self._write_manifest(status="active")

    def publish_stable(self, cv2, frame: Any, scene: Optional[Mapping[str, Any]] = None) -> None:
        if not self._enabled:
            return
        self._stable_entry = self._write_entry(cv2, frame, time.time(), scene_digest(scene), kind="stable")
        self._write_manifest(status="active")

    def close(self) -> None:
        if not self._enabled:
            return
        self._write_manifest(status="stopped")

    def _write_entry(
        self,
        cv2,
        frame: Any,
        ts: float,
        digest: Dict[str, Any],
        *,
        kind: str,
    ) -> Dict[str, Any]:
        previous_entry = self._latest_entry if kind == "latest" else self._stable_entry if kind == "stable" else None
        self._sequence += 1
        frame_id = f"{kind}_{int(ts * 1000)}_{self._sequence:06d}"
        abs_path = os.path.join(self._frames_dir, f"{frame_id}.jpg")
        self._write_frame_atomic(cv2, frame, abs_path)
        entry = {
            "frame_id": frame_id,
            "kind": kind,
            "ts": round(float(ts), 3),
            "image_ref": workspace_ref_for_path(abs_path, self._workspace_root),
            "scene": digest,
        }
        if previous_entry and kind in {"latest", "stable"}:
            previous_path = self._path_from_entry(previous_entry)
            if previous_path:
                try:
                    os.unlink(previous_path)
                except OSError:
                    pass
        return entry

    def _write_frame_atomic(self, cv2, frame: Any, abs_path: str) -> None:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".vision_broker_", suffix=".jpg", dir=os.path.dirname(abs_path))
        os.close(fd)
        params = []
        quality_flag = getattr(cv2, "IMWRITE_JPEG_QUALITY", None)
        if quality_flag is not None:
            params = [int(quality_flag), int(self._jpeg_quality)]
        try:
            if params:
                ok = cv2.imwrite(temp_path, frame, params)
            else:
                ok = cv2.imwrite(temp_path, frame)
            if not ok:
                raise RuntimeError(f"Failed to write broker frame {abs_path}")
            os.replace(temp_path, abs_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _prune_buffer(self, now: float) -> bool:
        changed = False
        cutoff = now - self._buffer_retention_s
        while self._buffer_entries:
            entry = self._buffer_entries[0]
            ts = float(entry.get("ts") or 0.0)
            if ts >= cutoff and len(self._buffer_entries) <= self._buffer_max_frames:
                break
            removed = self._buffer_entries.popleft()
            self._delete_if_unreferenced(self._path_from_entry(removed))
            changed = True
        while len(self._buffer_entries) > self._buffer_max_frames:
            removed = self._buffer_entries.popleft()
            self._delete_if_unreferenced(self._path_from_entry(removed))
            changed = True
        return changed

    def _path_from_entry(self, entry: Optional[Mapping[str, Any]]) -> str:
        if not isinstance(entry, Mapping):
            return ""
        image_ref = str(entry.get("image_ref") or "").strip()
        if image_ref.startswith("workspace:/"):
            rel = image_ref[len("workspace:/") :].lstrip("/").replace("/", os.sep)
            return os.path.abspath(os.path.join(self._workspace_root, rel))
        if image_ref.startswith("file:"):
            return image_ref[len("file:") :]
        return image_ref

    def _delete_if_unreferenced(self, abs_path: str) -> None:
        if not abs_path:
            return
        refs = {self._path_from_entry(self._latest_entry), self._path_from_entry(self._stable_entry)}
        refs.update(self._path_from_entry(entry) for entry in self._buffer_entries)
        if abs_path in refs:
            return
        try:
            os.unlink(abs_path)
        except OSError:
            pass

    def _write_manifest(self, *, status: str) -> None:
        payload = {
            "version": VISION_BROKER_VERSION,
            "status": status,
            "updated_ts": round(time.time(), 3),
            "latest": dict(self._latest_entry) if self._latest_entry else None,
            "stable": dict(self._stable_entry) if self._stable_entry else None,
            "buffer": {
                "retention_s": self._buffer_retention_s,
                "max_frames": self._buffer_max_frames,
                "frames": [dict(entry) for entry in self._buffer_entries],
            },
        }
        os.makedirs(os.path.dirname(self._manifest_path), exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".vision_manifest_", suffix=".json", dir=os.path.dirname(self._manifest_path))
        os.close(fd)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))
            os.replace(temp_path, self._manifest_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
