import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from runtime_core.session import OrchestratorSession, PermissionResolution


class RuntimeState:
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._data[key] = value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            return dict(self._data)


class RuntimeContext:
    def __init__(self, session: OrchestratorSession, state: RuntimeState):
        self._session = session
        self._state = state

    @property
    def last_turn_id(self) -> Optional[str]:
        return self._session.last_turn_id

    @property
    def pending_permissions(self):
        return self._session.pending_permissions

    def resolve_permission_id(self, token: Optional[str] = None) -> PermissionResolution:
        return self._session.resolve_permission_id(token)

    async def get_state(self, key: str, default: Any = None) -> Any:
        return await self._state.get(key, default)

    async def set_state(self, key: str, value: Any) -> None:
        await self._state.set(key, value)

    async def delete_state(self, key: str) -> None:
        await self._state.delete(key)

    async def state_snapshot(self) -> Dict[str, Any]:
        return await self._state.snapshot()

    async def submit_turn(
        self,
        text: str,
        *,
        requested_mode: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> str:
        return await self._session.submit_turn(text, requested_mode=requested_mode, turn_id=turn_id)

    async def patch_turn(self, turn_id: str, appended_text: str) -> None:
        await self._session.patch_turn(turn_id, appended_text)

    async def patch_last_turn(self, appended_text: str) -> bool:
        last_turn_id = self.last_turn_id
        if not last_turn_id:
            return False
        await self.patch_turn(last_turn_id, appended_text)
        return True

    async def send_wake(self, payload=None) -> None:
        await self._session.send_wake(payload)

    async def send_permission_decision(self, request_id: str, approved: bool) -> bool:
        return await self._session.send_permission_decision(request_id, approved)


class Sense(ABC):
    name = "sense"

    @abstractmethod
    async def run(self, context: RuntimeContext) -> None:
        raise NotImplementedError


class LocalRuntime:
    def __init__(self, session: OrchestratorSession, senses: Iterable[Sense]):
        self._session = session
        self._senses = list(senses)
        self._state = RuntimeState()

    async def run(self) -> None:
        await self._session.connect()
        context = RuntimeContext(self._session, self._state)
        tasks: List[asyncio.Task] = [asyncio.create_task(self._session.read_events())]
        tasks.extend(asyncio.create_task(sense.run(context)) for sense in self._senses)
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
