from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from common.jsonlog import append_jsonl


class YoloVisionService:
    def __init__(self, vision_cfg: Mapping[str, Any]):
        self._vision_cfg = dict(vision_cfg or {})
        self._scene_log_path = self._vision_cfg.get("scene_log_path")
        self._preview_failed = False
        if self._scene_log_path:
            scene_log_dir = os.path.dirname(self._scene_log_path)
            if scene_log_dir:
                os.makedirs(scene_log_dir, exist_ok=True)
        if self.object_memory_export_created_entities():
            os.makedirs(self.object_memory_export_dir(), exist_ok=True)

    @property
    def query_triggers(self) -> List[str]:
        triggers = self._vision_cfg.get("query_triggers") or []
        return [str(item).strip().lower() for item in triggers if str(item).strip()]

    def capture_interval_s(self) -> float:
        return max(0.05, float(self._vision_cfg.get("capture_interval_ms", 750)) / 1000.0)

    def min_stable_frames(self) -> int:
        return max(1, int(self._vision_cfg.get("min_stable_frames", 2)))

    def scene_cooldown_s(self) -> float:
        return max(0.0, float(self._vision_cfg.get("scene_cooldown_s", 2.0)))

    def print_scene_changes(self) -> bool:
        return bool(self._vision_cfg.get("print_scene_changes", True))

    def auto_submit_scene_changes(self) -> bool:
        return bool(self._vision_cfg.get("auto_submit_scene_changes", False))

    def submit_requested_mode(self) -> str:
        return str(self._vision_cfg.get("submit_requested_mode", "COGNITION"))

    def track_enabled(self) -> bool:
        return bool(self._vision_cfg.get("track_enabled", False))

    def tracker_config(self) -> str:
        value = str(self._vision_cfg.get("tracker_config", "botsort.yaml")).strip()
        return value or "botsort.yaml"

    def track_persist(self) -> bool:
        return bool(self._vision_cfg.get("track_persist", True))

    def track_stable_frames(self) -> int:
        return max(1, int(self._vision_cfg.get("track_stable_frames", 3)))

    def track_max_missing_frames(self) -> int:
        return max(1, int(self._vision_cfg.get("track_max_missing_frames", 8)))

    def object_memory_enabled(self) -> bool:
        return bool(self._vision_cfg.get("object_memory_enabled", False))

    def object_memory_auto_store(self) -> bool:
        return bool(self._vision_cfg.get("object_memory_auto_store", False))

    def object_memory_append_embedding_on_match(self) -> bool:
        return bool(self._vision_cfg.get("object_memory_append_embedding_on_match", True))

    def object_memory_match_top_k(self) -> int:
        return max(1, int(self._vision_cfg.get("object_memory_match_top_k", 5)))

    def object_memory_match_threshold(self) -> float:
        return max(0.0, float(self._vision_cfg.get("object_memory_match_threshold", 0.72)))

    def object_memory_match_margin(self) -> float:
        return max(0.0, float(self._vision_cfg.get("object_memory_match_margin", 0.08)))

    def object_memory_label_gate(self) -> bool:
        return bool(self._vision_cfg.get("object_memory_label_gate", True))

    def object_memory_object_name(self, label: str) -> str:
        prefix = str(self._vision_cfg.get("object_memory_name_prefix", "vision.object")).strip() or "vision.object"
        cleaned_label = str(label or "object").strip().lower().replace(" ", "_")
        return f"{prefix}.{cleaned_label}"

    def object_memory_embed_transport(self) -> str:
        value = str(self._vision_cfg.get("object_memory_embed_transport", "temp_file")).strip().lower()
        if value in {"base64", "temp_file"}:
            return value
        return "temp_file"

    def object_memory_crop_max_width(self) -> int:
        return max(0, int(self._vision_cfg.get("object_memory_crop_max_width", 256)))

    def object_memory_crop_max_height(self) -> int:
        return max(0, int(self._vision_cfg.get("object_memory_crop_max_height", 256)))

    def object_memory_crop_jpeg_quality(self) -> int:
        value = int(self._vision_cfg.get("object_memory_crop_jpeg_quality", 80))
        return max(30, min(100, value))

    def object_memory_crop_temp_dir(self) -> str:
        value = str(self._vision_cfg.get("object_memory_crop_temp_dir", "") or "").strip()
        return value or os.path.join(".", "data", "vision_crops")

    def object_memory_crop_keep_temp_files(self) -> bool:
        return bool(self._vision_cfg.get("object_memory_crop_keep_temp_files", False))

    def object_memory_export_created_entities(self) -> bool:
        return bool(self._vision_cfg.get("object_memory_export_created_entities", False))

    def object_memory_export_dir(self) -> str:
        value = str(self._vision_cfg.get("object_memory_export_dir", "") or "").strip()
        return value or os.path.join(".", "data", "vision_entities")

    def preview_enabled(self) -> bool:
        return bool(self._vision_cfg.get("preview_enabled", False))

    def preview_show_boxes(self) -> bool:
        return bool(self._vision_cfg.get("preview_show_boxes", True))

    def preview_show_labels(self) -> bool:
        return bool(self._vision_cfg.get("preview_show_labels", True))

    def preview_show_confidence(self) -> bool:
        return bool(self._vision_cfg.get("preview_show_confidence", True))

    def preview_show_summary(self) -> bool:
        return bool(self._vision_cfg.get("preview_show_summary", True))

    def preview_window_name(self) -> str:
        value = str(self._vision_cfg.get("preview_window_name", "YOLO Vision")).strip()
        return value or "YOLO Vision"

    def load_runtime(self):
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(f"Vision mode requires opencv-python: {e}") from e
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(f"Vision mode requires ultralytics: {e}") from e
        return cv2, YOLO

    def load_model(self, yolo_cls):
        model_path = str(self._vision_cfg.get("model_path", "yolov8n.pt")).strip() or "yolov8n.pt"
        return yolo_cls(model_path)

    def open_camera(self, cv2):
        camera_index = self._vision_cfg.get("camera_index", 0)
        cap = cv2.VideoCapture(camera_index)
        width = int(self._vision_cfg.get("frame_width", 0) or 0)
        height = int(self._vision_cfg.get("frame_height", 0) or 0)
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open camera index {camera_index}")
        return cap

    def detect_scene(self, model, frame) -> Dict[str, Any]:
        conf_threshold = float(self._vision_cfg.get("conf_threshold", 0.35))
        iou_threshold = float(self._vision_cfg.get("iou_threshold", 0.45))
        max_det = int(self._vision_cfg.get("max_det", 20))
        if self.track_enabled():
            track_kwargs: Dict[str, Any] = {
                "source": frame,
                "conf": conf_threshold,
                "iou": iou_threshold,
                "max_det": max_det,
                "verbose": False,
                "persist": self.track_persist(),
            }
            tracker = self.tracker_config()
            if tracker:
                track_kwargs["tracker"] = tracker
            results = model.track(**track_kwargs)
        else:
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=False,
            )
        return self.scene_from_results(results, frame_shape=getattr(frame, "shape", None))

    def scene_from_results(self, results: Sequence[Any], frame_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        detections = self._extract_detections(results)
        objects = self._aggregate_objects(detections)
        summary = self._scene_summary(objects)
        scene: Dict[str, Any] = {
            "ts": time.time(),
            "summary": summary,
            "objects": objects,
            "detections": detections,
            "signature": self.scene_signature_from_objects(objects),
        }
        if frame_shape and len(frame_shape) >= 2:
            scene["frame_size"] = {"height": int(frame_shape[0]), "width": int(frame_shape[1])}
        return scene

    def scene_signature(self, scene: Mapping[str, Any]) -> Tuple[Tuple[str, int], ...]:
        objects = scene.get("objects") or []
        return self.scene_signature_from_objects(objects)

    def scene_signature_from_objects(self, objects: Sequence[Mapping[str, Any]]) -> Tuple[Tuple[str, int], ...]:
        signature = []
        for item in objects:
            label = str(item.get("label") or "").strip()
            count = int(item.get("count") or 0)
            if label and count > 0:
                signature.append((label, count))
        return tuple(signature)

    def turn_text(self, scene: Mapping[str, Any]) -> str:
        summary = str(scene.get("summary") or "no known objects detected").strip()
        return f"Local camera observation: {summary}"

    def log_scene(self, scene: Mapping[str, Any]) -> None:
        if not self._scene_log_path:
            return
        append_jsonl(self._scene_log_path, dict(scene))

    def render_preview_frame(self, cv2, frame: Any, scene: Mapping[str, Any]) -> Any:
        rendered = frame.copy() if hasattr(frame, "copy") else frame
        font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
        line_type = getattr(cv2, "LINE_AA", 16)
        summary_color = (32, 240, 240)
        entity_map = self._preview_entity_map(scene.get("entities") or [])

        detections = scene.get("detections") or []
        if self.preview_show_boxes():
            for item in detections:
                if not isinstance(item, Mapping):
                    continue
                bbox = item.get("bbox") or []
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
                except (TypeError, ValueError):
                    continue
                entity = entity_map.get(self._track_key(item.get("track_id")))
                box_color = self._preview_box_color(entity)
                cv2.rectangle(rendered, (x1, y1), (x2, y2), box_color, 2)

                label_text = self._preview_label_text(item, entity)
                if label_text:
                    cv2.putText(
                        rendered,
                        label_text,
                        (x1, max(24, y1 - 8)),
                        font,
                        0.55,
                        box_color,
                        2,
                        line_type,
                    )

        if self.preview_show_summary():
            summary = str(scene.get("summary") or "").strip()
            if summary:
                cv2.putText(
                    rendered,
                    f"Objects: {summary}",
                    (10, 28),
                    font,
                    0.75,
                    summary_color,
                    2,
                    line_type,
                )
        return rendered

    def show_preview(self, cv2, frame: Any, scene: Mapping[str, Any]) -> bool:
        if not self.preview_enabled() or self._preview_failed:
            return True
        try:
            rendered = self.render_preview_frame(cv2, frame, scene)
            cv2.imshow(self.preview_window_name(), rendered)
            key = int(cv2.waitKey(1)) & 0xFF
            return key not in (27, ord("q"), ord("Q"))
        except Exception as e:
            self._preview_failed = True
            print(f"[vision] Preview disabled: {e}", flush=True)
            return True

    def close_preview(self, cv2) -> None:
        if not self.preview_enabled():
            return
        try:
            destroy_window = getattr(cv2, "destroyWindow", None)
            if callable(destroy_window):
                destroy_window(self.preview_window_name())
                return
            destroy_all_windows = getattr(cv2, "destroyAllWindows", None)
            if callable(destroy_all_windows):
                destroy_all_windows()
        except Exception:
            pass

    def build_embedding_input(self, cv2, frame: Any, detection: Mapping[str, Any]) -> Dict[str, Any]:
        encoded = self._encode_detection_crop_bytes(cv2, frame, detection)
        if not encoded:
            return {}
        transport = self.object_memory_embed_transport()
        if transport == "temp_file":
            path = self._write_embedding_temp_file(encoded)
            if path:
                payload = {"image_ref": path, "_encoded_bytes": encoded}
                if not self.object_memory_crop_keep_temp_files():
                    payload["_cleanup_path"] = path
                return payload
        return {"image_b64": base64.b64encode(encoded).decode("ascii"), "_encoded_bytes": encoded}

    def cleanup_embedding_input(self, payload: Mapping[str, Any]) -> None:
        cleanup_path = str(payload.get("_cleanup_path") or "").strip()
        if not cleanup_path:
            return
        try:
            os.remove(cleanup_path)
        except FileNotFoundError:
            return
        except OSError:
            return

    def export_created_entity(self, payload: Mapping[str, Any], memory_record: Mapping[str, Any]) -> str:
        if not self.object_memory_export_created_entities():
            return ""
        mem_id = str(memory_record.get("mem_id") or "").strip()
        if not mem_id:
            return ""
        export_dir = os.path.join(self.object_memory_export_dir(), mem_id)
        os.makedirs(export_dir, exist_ok=True)

        image_bytes = self._payload_image_bytes(payload)
        if image_bytes:
            crop_path = os.path.join(export_dir, "crop.jpg")
            with open(crop_path, "wb") as f:
                f.write(image_bytes)

        memory_path = os.path.join(export_dir, "memory.json")
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(dict(memory_record), f, ensure_ascii=True, indent=2, sort_keys=True)
        return export_dir

    def encode_detection_crop(self, cv2, frame: Any, detection: Mapping[str, Any]) -> str:
        encoded = self._encode_detection_crop_bytes(cv2, frame, detection)
        if not encoded:
            return ""
        return base64.b64encode(encoded).decode("ascii")

    def _encode_detection_crop_bytes(self, cv2, frame: Any, detection: Mapping[str, Any]) -> bytes:
        crop = self._crop_frame(frame, detection.get("bbox") or [])
        if crop is None:
            return b""
        prepared = self._resize_embedding_crop(cv2, crop)
        encode_params = self._jpeg_encode_params(cv2)
        try:
            if encode_params:
                ok, encoded = cv2.imencode(".jpg", prepared, encode_params)
            else:
                ok, encoded = cv2.imencode(".jpg", prepared)
        except TypeError:
            try:
                ok, encoded = cv2.imencode(".jpg", prepared)
            except Exception:
                return b""
        except Exception:
            return b""
        if not ok:
            return b""
        data = encoded.tobytes() if hasattr(encoded, "tobytes") else bytes(encoded)
        return data

    def detection_features(
        self,
        cv2,
        frame: Any,
        detection: Mapping[str, Any],
        frame_size: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        bbox = detection.get("bbox") or []
        width, height = self._frame_dimensions(frame, frame_size)
        aspect_ratio = None
        area_ratio = None
        zone = ""
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and width > 0 and height > 0:
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
                box_width = max(1.0, x2 - x1)
                box_height = max(1.0, y2 - y1)
                aspect_ratio = box_width / box_height
                area_ratio = (box_width * box_height) / float(width * height)
                center_x = ((x1 + x2) / 2.0) / float(width)
                center_y = ((y1 + y2) / 2.0) / float(height)
                zone = self._zone_name(center_x, center_y)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        return {
            "color_hist": self._color_histogram(self._crop_frame(frame, bbox)),
            "aspect_ratio": aspect_ratio,
            "area_ratio": area_ratio,
            "zone": zone,
        }

    def _extract_detections(self, results: Sequence[Any]) -> List[Dict[str, Any]]:
        if not results:
            return []
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        cls_values = self._to_list(getattr(boxes, "cls", []))
        conf_values = self._to_list(getattr(boxes, "conf", []))
        xyxy_values = self._to_list(getattr(boxes, "xyxy", []))
        id_values = self._to_list(getattr(boxes, "id", []))
        detections: List[Dict[str, Any]] = []
        for index, raw_cls in enumerate(cls_values):
            cls_id = int(raw_cls)
            label = self._class_label(names, cls_id)
            confidence = float(conf_values[index]) if index < len(conf_values) else 0.0
            bbox_value = xyxy_values[index] if index < len(xyxy_values) else []
            bbox = [float(value) for value in bbox_value] if isinstance(bbox_value, (list, tuple)) else []
            track_id = None
            if index < len(id_values):
                raw_track_id = id_values[index]
                if raw_track_id is not None:
                    try:
                        numeric_id = float(raw_track_id)
                        track_id = int(numeric_id) if numeric_id.is_integer() else numeric_id
                    except (TypeError, ValueError):
                        track_id = str(raw_track_id)
            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                    "track_id": track_id,
                }
            )
        return detections

    def _aggregate_objects(self, detections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for item in detections:
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            confidence = float(item.get("confidence") or 0.0)
            bucket = summary.setdefault(label, {"label": label, "count": 0, "confidence": 0.0})
            bucket["count"] += 1
            bucket["confidence"] = max(bucket["confidence"], confidence)
        max_objects = max(1, int(self._vision_cfg.get("summary_max_objects", 8)))
        objects = list(summary.values())
        objects.sort(key=lambda item: (-int(item["count"]), -float(item["confidence"]), str(item["label"])))
        return objects[:max_objects]

    def _scene_summary(self, objects: Sequence[Mapping[str, Any]]) -> str:
        if not objects:
            return "no known objects detected"
        parts = []
        for item in objects:
            label = str(item.get("label") or "").strip()
            count = int(item.get("count") or 0)
            if not label or count <= 0:
                continue
            if count == 1:
                parts.append(label)
            else:
                parts.append(f"{count} {self._pluralize(label)}")
        return ", ".join(parts) if parts else "no known objects detected"

    def _class_label(self, names: Any, cls_id: int) -> str:
        if isinstance(names, dict):
            label = names.get(cls_id)
            if label is None:
                label = names.get(str(cls_id))
            if label is not None:
                return str(label)
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return f"class_{cls_id}"

    def _pluralize(self, label: str) -> str:
        if label.endswith("s"):
            return label
        return label + "s"

    def _to_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
            return [converted]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _crop_frame(self, frame: Any, bbox: Sequence[Any]):
        if frame is None or not hasattr(frame, "shape"):
            return None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            height, width = int(frame.shape[0]), int(frame.shape[1])
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError, AttributeError, IndexError):
            return None
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        try:
            return frame[y1:y2, x1:x2]
        except Exception:
            return None

    def _frame_dimensions(self, frame: Any, frame_size: Optional[Mapping[str, Any]]) -> Tuple[int, int]:
        if isinstance(frame_size, Mapping):
            try:
                height = int(frame_size.get("height") or 0)
                width = int(frame_size.get("width") or 0)
                if width > 0 and height > 0:
                    return width, height
            except (TypeError, ValueError):
                pass
        if hasattr(frame, "shape"):
            try:
                return int(frame.shape[1]), int(frame.shape[0])
            except (TypeError, ValueError, IndexError):
                return 0, 0
        return 0, 0

    def _zone_name(self, center_x: float, center_y: float) -> str:
        if center_x < 1.0 / 3.0:
            x_label = "left"
        elif center_x < 2.0 / 3.0:
            x_label = "center"
        else:
            x_label = "right"
        if center_y < 1.0 / 3.0:
            y_label = "top"
        elif center_y < 2.0 / 3.0:
            y_label = "middle"
        else:
            y_label = "bottom"
        return f"{x_label}-{y_label}"

    def _color_histogram(self, crop: Any) -> List[float]:
        if crop is None:
            return []
        try:
            import numpy as np
        except Exception:
            return []
        try:
            arr = crop.astype(np.float32)
        except Exception:
            return []
        if arr.ndim != 3 or arr.shape[2] < 3:
            return []
        features: List[float] = []
        for channel_index in range(3):
            channel = arr[:, :, channel_index]
            hist, _ = np.histogram(channel, bins=16, range=(0, 256))
            hist = hist.astype(np.float32)
            total = float(hist.sum()) or 1.0
            features.extend((hist / total).tolist())
        return features

    def _resize_embedding_crop(self, cv2, crop: Any) -> Any:
        if crop is None or not hasattr(crop, "shape"):
            return crop
        try:
            height = int(crop.shape[0])
            width = int(crop.shape[1])
        except (TypeError, ValueError, IndexError):
            return crop
        if width <= 0 or height <= 0:
            return crop
        max_width = self.object_memory_crop_max_width()
        max_height = self.object_memory_crop_max_height()
        scale = 1.0
        if max_width > 0 and width > max_width:
            scale = min(scale, float(max_width) / float(width))
        if max_height > 0 and height > max_height:
            scale = min(scale, float(max_height) / float(height))
        if scale >= 0.999:
            return crop
        resize_fn = getattr(cv2, "resize", None)
        if not callable(resize_fn):
            return crop
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        interpolation = getattr(cv2, "INTER_AREA", None)
        try:
            if interpolation is None:
                return resize_fn(crop, (new_width, new_height))
            return resize_fn(crop, (new_width, new_height), interpolation=interpolation)
        except TypeError:
            try:
                return resize_fn(crop, (new_width, new_height))
            except Exception:
                return crop
        except Exception:
            return crop

    def _jpeg_encode_params(self, cv2) -> List[int]:
        quality_flag = getattr(cv2, "IMWRITE_JPEG_QUALITY", None)
        if quality_flag is None:
            return []
        return [int(quality_flag), self.object_memory_crop_jpeg_quality()]

    def _write_embedding_temp_file(self, encoded: bytes) -> str:
        temp_dir = self.object_memory_crop_temp_dir()
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".jpg",
                prefix="vision_crop_",
                dir=temp_dir or None,
                delete=False,
            ) as handle:
                handle.write(encoded)
                return handle.name
        except OSError:
            return ""

    def _payload_image_bytes(self, payload: Mapping[str, Any]) -> bytes:
        encoded = payload.get("_encoded_bytes")
        if isinstance(encoded, bytes) and encoded:
            return encoded
        image_b64 = str(payload.get("image_b64") or "").strip()
        if image_b64:
            try:
                return base64.b64decode(image_b64.encode("ascii"), validate=True)
            except Exception:
                return b""
        image_ref = str(payload.get("image_ref") or "").strip()
        if image_ref:
            try:
                with open(image_ref, "rb") as f:
                    return f.read()
            except OSError:
                return b""
        return b""

    def _preview_entity_map(self, entities: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
        mapped: Dict[str, Mapping[str, Any]] = {}
        for item in entities:
            if not isinstance(item, Mapping):
                continue
            track_key = self._track_key(item.get("track_id"))
            if track_key:
                mapped[track_key] = item
        return mapped

    def _preview_box_color(self, entity: Optional[Mapping[str, Any]]) -> Tuple[int, int, int]:
        match_reason = str((entity or {}).get("match_reason") or "").strip().lower()
        if match_reason == "matched":
            return (96, 220, 255)
        if match_reason == "stored":
            return (0, 215, 255)
        return (64, 220, 160)

    def _preview_label_text(self, detection: Mapping[str, Any], entity: Optional[Mapping[str, Any]]) -> str:
        label_parts: List[str] = []
        track_id = detection.get("track_id")
        if track_id not in (None, ""):
            label_parts.append(f"#{track_id}")
        if self.preview_show_labels():
            label = str(detection.get("label") or "").strip()
            if label:
                label_parts.append(label)
        if self.preview_show_confidence():
            try:
                confidence = float(detection.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            label_parts.append(f"{confidence:.2f}")
        if entity:
            resolved_name = str(entity.get("object_name") or "").strip()
            object_mem_id = str(entity.get("object_mem_id") or "").strip()
            match_reason = str(entity.get("match_reason") or "").strip().lower()
            if resolved_name:
                label_parts.append(f"-> {resolved_name}")
            elif object_mem_id:
                label_parts.append(f"-> mem:{object_mem_id[:8]}")
            elif match_reason == "stored":
                label_parts.append("[new]")
        return " ".join(label_parts).strip()

    def _track_key(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value).strip()
