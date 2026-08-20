from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from common.vision_broker import VisionBroker
from runtime_core.runtime import RuntimeContext, Sense
from voice_daemon.vision_context import VISION_LATEST_SCENE_KEY, VISION_STABLE_SCENE_KEY
from voice_daemon.vision_registry import VisionEntityRegistry
from voice_daemon.vision_service import YoloVisionService


class VisionSense(Sense):
    name = "vision"

    def __init__(
        self,
        service: YoloVisionService,
        registry: Optional[VisionEntityRegistry] = None,
        broker: Optional[VisionBroker] = None,
    ):
        self._service = service
        self._registry = registry
        self._broker = broker

    async def run(self, context: RuntimeContext) -> None:
        cv2, yolo_cls = self._service.load_runtime()
        model = await asyncio.to_thread(self._service.load_model, yolo_cls)
        capture = self._service.open_camera(cv2)
        print("[vision] Camera active.", flush=True)

        active_signature = None
        candidate_signature = None
        candidate_count = 0
        candidate_scene = None
        last_scene_emit = 0.0
        interval_s = self._service.capture_interval_s()
        min_stable_frames = self._service.min_stable_frames()
        cooldown_s = self._service.scene_cooldown_s()

        try:
            while True:
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok:
                    await asyncio.sleep(interval_s)
                    continue
                scene = await asyncio.to_thread(self._service.detect_scene, model, frame)
                if self._registry is not None:
                    scene = await self._registry.update(context, cv2, frame, scene)
                await context.set_state(VISION_LATEST_SCENE_KEY, scene)
                if self._broker is not None and self._broker.enabled:
                    await asyncio.to_thread(self._broker.publish_latest, cv2, frame, scene)
                keep_running = await asyncio.to_thread(self._service.show_preview, cv2, frame, scene)
                if not keep_running:
                    print("[vision] Preview closed.", flush=True)
                    break

                signature = self._service.scene_signature(scene)
                if signature == candidate_signature:
                    candidate_count += 1
                else:
                    candidate_signature = signature
                    candidate_count = 1
                    candidate_scene = scene

                if candidate_count >= min_stable_frames:
                    now = time.time()
                    if signature != active_signature and now - last_scene_emit >= cooldown_s:
                        active_signature = signature
                        last_scene_emit = now
                        stable_scene = dict(candidate_scene or scene)
                        stable_scene["stable_ts"] = now
                        await context.set_state(VISION_STABLE_SCENE_KEY, stable_scene)
                        if self._broker is not None and self._broker.enabled:
                            await asyncio.to_thread(self._broker.publish_stable, cv2, frame, stable_scene)
                        self._service.log_scene(stable_scene)
                        if self._service.print_scene_changes():
                            print(f"[vision] {stable_scene.get('summary', 'scene updated')}", flush=True)
                        if self._service.auto_submit_scene_changes():
                            await context.submit_turn(
                                self._service.turn_text(stable_scene),
                                requested_mode=self._service.submit_requested_mode(),
                            )
                await asyncio.sleep(interval_s)
        finally:
            self._service.close_preview(cv2)
            capture.release()
            if self._broker is not None and self._broker.enabled:
                await asyncio.to_thread(self._broker.close)
            if self._registry is not None:
                await self._registry.clear(context)
            await context.delete_state(VISION_LATEST_SCENE_KEY)
            await context.delete_state(VISION_STABLE_SCENE_KEY)
