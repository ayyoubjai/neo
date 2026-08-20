import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from common.ids import new_id
from common.jsonl import read_jsonl, write_jsonl
from common.types import (
    EVENT_ASSISTANT_DRAFT,
    EVENT_ASSISTANT_FINAL,
    EVENT_PERMISSION_REQUEST,
    EVENT_PERMISSION_RESPONSE,
    EVENT_STT_FINAL,
    EVENT_TURN_PATCH,
    EVENT_WAKE,
    make_event,
)


EventHandler = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class PermissionResolution:
    request_id: Optional[str]
    status: str

    @property
    def matched(self) -> bool:
        return self.request_id is not None and self.status == "matched"


class OrchestratorSession:
    _VALID_REQUESTED_MODES = {"COGNITION", "SYSTEM0", "SYSTEM1", "SYSTEM2", "SYSTEM3"}

    def __init__(
        self,
        host: str,
        port: int,
        *,
        source_id: Optional[str] = None,
        on_assistant_final: Optional[EventHandler] = None,
        on_assistant_draft: Optional[EventHandler] = None,
        on_permission_request: Optional[EventHandler] = None,
    ):
        self._host = host
        self._port = port
        source_text = str(source_id or "").strip()
        self._source_id = source_text or None
        self._on_assistant_final = on_assistant_final
        self._on_assistant_draft = on_assistant_draft
        self._on_permission_request = on_permission_request
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._write_lock = asyncio.Lock()
        self._pending_permissions: Dict[str, Dict[str, Any]] = {}
        self._last_turn_id: Optional[str] = None

    @property
    def last_turn_id(self) -> Optional[str]:
        return self._last_turn_id

    @property
    def pending_permissions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._pending_permissions)

    async def connect(self, retries: int = 15, delay: float = 1.0) -> None:
        if self._reader is not None or self._writer is not None:
            raise RuntimeError("Session is already connected.")
        for attempt in range(retries):
            try:
                self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
                return
            except ConnectionRefusedError:
                if attempt < retries - 1:
                    print(f"[session] Connection refused by orchestrator at {self._host}:{self._port}, retrying in {delay}s ({attempt + 1}/{retries})...")
                    await asyncio.sleep(delay)
                else:
                    raise

    async def close(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is None:
            return
        writer.close()
        await writer.wait_closed()

    async def read_events(self) -> None:
        reader = self._reader
        if reader is None:
            raise RuntimeError("Session is not connected.")
        while True:
            event = await read_jsonl(reader)
            if event is None:
                break
            await self._dispatch_event(event)

    async def _dispatch_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if event_type == EVENT_PERMISSION_REQUEST:
            request_id = payload.get("request_id")
            if isinstance(request_id, str) and request_id:
                self._pending_permissions[request_id] = payload
            if self._on_permission_request is not None:
                await self._on_permission_request(payload)
            return

        if event_type == EVENT_ASSISTANT_FINAL and self._on_assistant_final is not None:
            await self._on_assistant_final(payload)
            return

        if event_type == EVENT_ASSISTANT_DRAFT and self._on_assistant_draft is not None:
            await self._on_assistant_draft(payload)

    async def send_event(self, event: Dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            raise RuntimeError("Session is not connected.")
        outgoing = event
        if self._source_id and isinstance(event, dict):
            payload = event.get("payload")
            if isinstance(payload, dict) and "source_id" not in payload:
                outgoing = dict(event)
                outgoing["payload"] = dict(payload)
                outgoing["payload"]["source_id"] = self._source_id
        async with self._write_lock:
            await write_jsonl(writer, outgoing)

    async def send_wake(self, payload: Optional[Dict[str, Any]] = None) -> None:
        wake_payload = dict(payload or {})
        wake_payload.setdefault("ts", time.time())
        await self.send_event(make_event(EVENT_WAKE, wake_payload))

    async def submit_turn(
        self,
        text: str,
        *,
        requested_mode: Optional[str] = None,
        skip_distillation: bool = False,
        turn_id: Optional[str] = None,
    ) -> str:
        resolved_turn_id = str(turn_id or new_id())
        now = time.time()
        payload: Dict[str, Any] = {
            "turn_id": resolved_turn_id,
            "text": text,
            "t0": now,
            "t1": now,
            "is_final": True,
        }
        normalized_mode = self._normalize_requested_mode(requested_mode)
        if normalized_mode:
            payload["requested_mode"] = normalized_mode
        if skip_distillation:
            payload["skip_distillation"] = True
        self._last_turn_id = resolved_turn_id
        await self.send_event(make_event(EVENT_STT_FINAL, payload))
        return resolved_turn_id

    async def patch_turn(self, turn_id: str, appended_text: str) -> None:
        await self.send_event(
            make_event(EVENT_TURN_PATCH, {"turn_id": turn_id, "appended_text": appended_text})
        )

    def resolve_permission_id(self, token: Optional[str] = None) -> PermissionResolution:
        if token:
            matches = [rid for rid in self._pending_permissions if rid.startswith(token)]
            if len(matches) == 1:
                return PermissionResolution(matches[0], "matched")
            if len(matches) > 1:
                return PermissionResolution(None, "ambiguous")
        if len(self._pending_permissions) == 1:
            return PermissionResolution(next(iter(self._pending_permissions)), "matched")
        if len(self._pending_permissions) > 1:
            return PermissionResolution(None, "multiple_pending")
        return PermissionResolution(None, "missing")

    async def send_permission_decision(self, request_id: str, approved: bool) -> bool:
        pending = self._pending_permissions.pop(request_id, None)
        if pending is None:
            return False
        payload: Dict[str, Any] = {"request_id": request_id, "approved": approved}
        if approved:
            payload["token"] = new_id()
        await self.send_event(make_event(EVENT_PERMISSION_RESPONSE, payload))
        return True

    @classmethod
    def _normalize_requested_mode(cls, value: Optional[str]) -> Optional[str]:
        normalized = str(value or "").strip().upper()
        if normalized in cls._VALID_REQUESTED_MODES:
            return normalized
        return None
