from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common.ids import new_id
from runtime_core.runtime import RuntimeContext
from voice_daemon.vision_context import VISION_TRACKED_ENTITIES_KEY


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = sum(float(x) * float(x) for x in a) ** 0.5
    norm_b = sum(float(y) * float(y) for y in b) ** 0.5
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VisionEntityRegistry:
    def __init__(self, service, memory_manager=None):
        self._service = service
        self._memory = memory_manager
        self._tracks: Dict[str, Dict[str, Any]] = {}

    async def update(self, context: RuntimeContext, cv2, frame: Any, scene: Dict[str, Any]) -> Dict[str, Any]:
        detections = scene.get("detections") or []
        now = float(scene.get("ts") or time.time())
        seen_track_ids = set()

        for detection in detections:
            track_id = self._track_id(detection)
            if not track_id:
                continue
            seen_track_ids.add(track_id)
            features = self._service.detection_features(cv2, frame, detection, scene.get("frame_size"))
            profile = self._tracks.get(track_id)
            if profile is None:
                profile = self._new_profile(track_id, now)
                self._tracks[track_id] = profile
            self._update_profile(profile, detection, features, now)
            if self._should_resolve(profile):
                await self._resolve_profile(profile, cv2, frame, detection)

        max_missing = self._service.track_max_missing_frames()
        for track_id in list(self._tracks.keys()):
            if track_id in seen_track_ids:
                continue
            profile = self._tracks[track_id]
            profile["missing_frames"] = int(profile.get("missing_frames", 0)) + 1
            if profile["missing_frames"] > max_missing:
                del self._tracks[track_id]

        entities = [self._entity_snapshot(profile) for profile in self._tracks.values() if profile.get("stable")]
        entities.sort(key=lambda item: (str(item.get("label") or ""), self._sort_track_id(item.get("track_id"))))
        scene["entities"] = entities
        await context.set_state(VISION_TRACKED_ENTITIES_KEY, entities)
        return scene

    async def clear(self, context: RuntimeContext) -> None:
        self._tracks.clear()
        await context.delete_state(VISION_TRACKED_ENTITIES_KEY)

    def _new_profile(self, track_id: str, now: float) -> Dict[str, Any]:
        return {
            "track_id": track_id,
            "first_seen_ts": now,
            "last_seen_ts": now,
            "seen_frames": 0,
            "missing_frames": 0,
            "stable": False,
            "label_votes": {},
            "label": "",
            "confidence_ema": 0.0,
            "bbox": [],
            "aspect_ratio": None,
            "area_ratio": None,
            "zone": "",
            "color_hist": [],
            "object_mem_id": None,
            "object_name": None,
            "match_score": None,
            "match_reason": "unresolved",
            "resolve_attempted": False,
            "appearance_synced": False,
        }

    def _update_profile(
        self,
        profile: Dict[str, Any],
        detection: Mapping[str, Any],
        features: Mapping[str, Any],
        now: float,
    ) -> None:
        label = str(detection.get("label") or "").strip()
        confidence = float(detection.get("confidence") or 0.0)
        label_votes = profile["label_votes"]
        if label:
            label_votes[label] = int(label_votes.get(label, 0)) + 1
            profile["label"] = max(label_votes.items(), key=lambda item: item[1])[0]
        previous_conf = float(profile.get("confidence_ema") or 0.0)
        profile["confidence_ema"] = confidence if previous_conf <= 0.0 else (previous_conf * 0.7 + confidence * 0.3)
        profile["bbox"] = list(detection.get("bbox") or [])
        profile["last_seen_ts"] = now
        profile["seen_frames"] = int(profile.get("seen_frames", 0)) + 1
        profile["missing_frames"] = 0
        profile["stable"] = int(profile["seen_frames"]) >= self._service.track_stable_frames()
        for key in ("aspect_ratio", "area_ratio", "zone"):
            value = features.get(key)
            if value not in (None, ""):
                profile[key] = value
        new_hist = features.get("color_hist") or []
        if new_hist:
            previous_hist = profile.get("color_hist") or []
            if previous_hist and len(previous_hist) == len(new_hist):
                profile["color_hist"] = [float(old) * 0.7 + float(new) * 0.3 for old, new in zip(previous_hist, new_hist)]
            else:
                profile["color_hist"] = list(new_hist)

    def _should_resolve(self, profile: Mapping[str, Any]) -> bool:
        if not profile.get("stable"):
            return False
        if profile.get("resolve_attempted"):
            return False
        if not self._service.object_memory_enabled():
            return False
        if self._memory is None:
            return False
        return True

    async def _resolve_profile(self, profile: Dict[str, Any], cv2, frame: Any, detection: Mapping[str, Any]) -> None:
        profile["resolve_attempted"] = True
        embed_input = self._service.build_embedding_input(cv2, frame, detection)
        image_ref = str(embed_input.get("image_ref") or "").strip()
        image_b64 = str(embed_input.get("image_b64") or "").strip()
        if not image_ref and not image_b64:
            profile["match_reason"] = "no_crop"
            return
        try:
            embedding, model = await self._memory.embed_image(image_ref=image_ref, image_b64=image_b64)
            if not embedding:
                profile["match_reason"] = "no_embedding"
                return

            candidates = await self._memory.search_objects_by_appearance(
                embedding,
                top_k=self._service.object_memory_match_top_k(),
            )
            match = self._select_match(profile, candidates)
            appearance_payload = [{"embedding": embedding, "model": model}]
            if match is not None:
                profile["object_mem_id"] = match["mem_id"]
                profile["object_name"] = match.get("name")
                profile["match_score"] = match["combined_score"]
                profile["match_reason"] = "matched"
                if self._service.object_memory_append_embedding_on_match():
                    await self._memory.add_object_appearance(match["mem_id"], appearance_payload, trace_id=new_id())
                    profile["appearance_synced"] = True
                return

            profile["match_reason"] = "unmatched"
            if not self._service.object_memory_auto_store():
                return

            object_name = self._service.object_memory_object_name(str(profile.get("label") or "object"))
            object_data = {
                "label": profile.get("label"),
                "source": "vision.yolo.track",
                "first_seen_ts": profile.get("first_seen_ts"),
                "last_seen_ts": profile.get("last_seen_ts"),
                "zone": profile.get("zone"),
                "aspect_ratio": profile.get("aspect_ratio"),
                "area_ratio": profile.get("area_ratio"),
                "color_hist": profile.get("color_hist") or [],
                "confidence_ema": profile.get("confidence_ema"),
            }
            trace_id = new_id()
            confidence = float(profile.get("confidence_ema") or 0.6)
            semantic_text = f"vision object {object_name} label={profile.get('label')} zone={profile.get('zone')}"
            mem_id = await self._memory.store_object(
                object_name,
                object_data,
                trace_id=trace_id,
                confidence=confidence,
                semantic_text=semantic_text,
                appearance_embeddings=appearance_payload,
            )
            profile["object_mem_id"] = mem_id
            profile["object_name"] = object_name
            profile["match_reason"] = "stored"
            self._service.export_created_entity(
                embed_input,
                {
                    "mem_id": mem_id,
                    "type": "object",
                    "name": object_name,
                    "data": object_data,
                    "confidence": confidence,
                    "trace_id": trace_id,
                    "semantic_text": semantic_text,
                    "appearance_embeddings": appearance_payload,
                },
            )
        finally:
            self._service.cleanup_embedding_input(embed_input)

    def _select_match(self, profile: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for candidate in candidates:
            combined = self._candidate_score(profile, candidate)
            if combined is None:
                continue
            payload = dict(candidate)
            payload["combined_score"] = combined
            scored.append(payload)
        if not scored:
            return None
        scored.sort(key=lambda item: float(item.get("combined_score") or 0.0), reverse=True)
        top = scored[0]
        top_score = float(top.get("combined_score") or 0.0)
        if top_score < self._service.object_memory_match_threshold():
            return None
        second_score = float(scored[1].get("combined_score") or 0.0) if len(scored) > 1 else 0.0
        if len(scored) > 1 and (top_score - second_score) < self._service.object_memory_match_margin():
            return None
        return top

    def _candidate_score(self, profile: Mapping[str, Any], candidate: Mapping[str, Any]) -> Optional[float]:
        appearance_score = float(candidate.get("score") or 0.0)
        data = candidate.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        profile_label = str(profile.get("label") or "").strip()
        candidate_label = str(data.get("label") or "").strip()
        if self._service.object_memory_label_gate() and profile_label and candidate_label and profile_label != candidate_label:
            return None

        hist_score = _cosine(profile.get("color_hist") or [], data.get("color_hist") or [])
        profile_aspect = profile.get("aspect_ratio")
        candidate_aspect = data.get("aspect_ratio")
        aspect_score = self._ratio_similarity(profile_aspect, candidate_aspect)
        profile_area = profile.get("area_ratio")
        candidate_area = data.get("area_ratio")
        area_score = self._ratio_similarity(profile_area, candidate_area)
        geometry_score = (aspect_score + area_score) / 2.0
        zone_score = 1.0 if data.get("zone") and data.get("zone") == profile.get("zone") else 0.0

        return 0.65 * appearance_score + 0.2 * hist_score + 0.1 * geometry_score + 0.05 * zone_score

    def _ratio_similarity(self, left: Any, right: Any) -> float:
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            return 0.0
        if left_value <= 0.0 or right_value <= 0.0:
            return 0.0
        return max(0.0, 1.0 - abs(left_value - right_value) / max(left_value, right_value))

    def _entity_snapshot(self, profile: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "track_id": profile.get("track_id"),
            "label": profile.get("label"),
            "zone": profile.get("zone"),
            "bbox": list(profile.get("bbox") or []),
            "confidence": round(float(profile.get("confidence_ema") or 0.0), 4),
            "object_mem_id": profile.get("object_mem_id"),
            "object_name": profile.get("object_name"),
            "match_reason": profile.get("match_reason"),
            "match_score": profile.get("match_score"),
        }

    def _track_id(self, detection: Mapping[str, Any]) -> Optional[str]:
        value = detection.get("track_id")
        if value is None:
            return None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        text = str(value).strip()
        return text or None

    def _sort_track_id(self, value: Any) -> tuple[int, Any]:
        if value is None:
            return (1, "")
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))
