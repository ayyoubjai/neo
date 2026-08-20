from __future__ import annotations

import asyncio
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional, Type, TypeVar

from autonomy.json_utils import parse_json_object
from common.config import Settings, load_settings
from runtime_core.session import OrchestratorSession


T = TypeVar("T")
_VALID_PERMISSION_POLICIES = {"deny", "approve"}
_VALID_REQUESTED_MODES = {"COGNITION", "SYSTEM0", "SYSTEM1", "SYSTEM2", "SYSTEM3"}


class CognitionClientError(RuntimeError):
    pass


class CognitionJsonError(CognitionClientError):
    pass


def _validate_model(model_cls: Type[T], payload: Dict[str, Any]) -> T:
    if hasattr(model_cls, "from_dict"):
        return model_cls.from_dict(payload)  # type: ignore[return-value]
    if is_dataclass(model_cls):
        allowed = {field.name for field in fields(model_cls)}
        return model_cls(**{key: value for key, value in payload.items() if key in allowed})
    return model_cls(**payload)


def _normalize_permission_policy(value: object) -> str:
    policy = str(value or "").strip().lower()
    if policy in _VALID_PERMISSION_POLICIES:
        return policy
    return "deny"


def _normalize_requested_mode(value: object) -> str:
    mode = str(value or "").strip().upper()
    if mode in _VALID_REQUESTED_MODES:
        return mode
    return "COGNITION"


class CognitionClient:
    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        source_id: Optional[str] = None,
        permission_policy: Optional[str] = None,
        requested_mode: Optional[str] = None,
        timeout_s: Optional[int] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.host = host or self.settings.rpc.get("orch_host", "127.0.0.1")
        self.port = int(port or self.settings.rpc.get("orch_port", 50051))
        autonomy = dict(self.settings.autonomy or {})
        raw_source_id = source_id if source_id is not None else autonomy.get("source_id")
        resolved_source_id = str(raw_source_id or "").strip()
        self.source_id = resolved_source_id or None
        self.permission_policy = _normalize_permission_policy(
            permission_policy if permission_policy is not None else autonomy.get("permission_policy")
        )
        self.requested_mode = _normalize_requested_mode(
            requested_mode if requested_mode is not None else autonomy.get("requested_mode", "COGNITION")
        )
        configured_timeout = timeout_s if timeout_s is not None else autonomy.get("cognition_timeout_s", 180)
        try:
            self.default_timeout_s = max(1, int(configured_timeout))
        except (TypeError, ValueError):
            self.default_timeout_s = 180

    async def generate_text(self, prompt: str, *, timeout: Optional[int] = None) -> str:
        text, _ = await self._run_turn_with_trace(prompt, timeout=timeout or self.default_timeout_s)
        return text

    async def generate_json(self, prompt: str, *, extraction_schema: str = "", timeout: Optional[int] = None) -> Dict[str, Any]:
        effective_timeout = timeout or self.default_timeout_s
        if self.requested_mode and extraction_schema:
            text, trace = await self._run_turn_with_trace(prompt, timeout=effective_timeout)
            extraction_prompt = self._build_extraction_prompt(text, trace, extraction_schema)
            raw_output, _ = await self._run_turn_with_trace(extraction_prompt, timeout=effective_timeout, requested_mode="SYSTEM0")
        else:
            raw_output, _ = await self._run_turn_with_trace(self._wrap_json_prompt(prompt), timeout=effective_timeout)
            
        payload = parse_json_object(raw_output)
        if payload:
            return payload
        repaired_output, _ = await self._run_turn_with_trace(
            self._wrap_json_repair_prompt(raw_output),
            timeout=effective_timeout,
            requested_mode="SYSTEM0" if self.requested_mode else None
        )
        repaired_payload = parse_json_object(repaired_output)
        if repaired_payload:
            return repaired_payload
        preview = repaired_output.strip().replace("\n", " ")[:300]
        raise CognitionJsonError(f"COGNITION returned non-JSON output: {preview}")

    async def generate_model(
        self,
        prompt: str,
        model_cls: Type[T],
        *,
        timeout: Optional[int] = None,
    ) -> T:
        schema = self._build_schema_hint(model_cls)
        payload = await self.generate_json(prompt, extraction_schema=schema, timeout=timeout)
        try:
            return _validate_model(model_cls, payload)
        except Exception as exc:
            raise CognitionJsonError(f"COGNITION output failed schema validation: {exc}") from exc

    async def wait_until_available(self, *, timeout_s: float = 30.0, interval_s: float = 0.5) -> None:
        """Wait until the orchestrator COGNITION socket accepts connections."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout_s))
        last_error: Optional[BaseException] = None
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError as exc:
                last_error = exc
            if loop.time() >= deadline:
                detail = f": {last_error}" if last_error else ""
                raise CognitionClientError(
                    f"COGNITION orchestrator is not reachable at {self.host}:{self.port}{detail}. "
                    "Start the local stack with ./scripts/run_all.sh, or start orchestrator.main "
                    "before running autonomy directly."
                )
            await asyncio.sleep(max(0.05, float(interval_s)))

    @staticmethod
    def _wrap_json_prompt(prompt: str) -> str:
        return prompt.strip()

    @staticmethod
    def _wrap_json_repair_prompt(raw_output: str) -> str:
        return (
            "Convert the following response into exactly one JSON object.\n\n"
            "[RESPONSE]\n"
            f"{raw_output.strip()}"
        )

    @staticmethod
    def _build_schema_hint(model_cls: Type[T]) -> str:
        if is_dataclass(model_cls):
            schema = {field.name: field.type.__name__ if hasattr(field.type, "__name__") else str(field.type) for field in fields(model_cls)}
            import json
            return json.dumps(schema)
        return "{}"

    @staticmethod
    def _build_extraction_prompt(result_text: str, thinking_trace: list, schema: str) -> str:
        import json
        return (
            "Extract the structured result from the completed execution below.\n\n"
            f"SCHEMA:\n{schema}\n\n"
            f"RESULT:\n{result_text}\n\n"
            f"THINKING TRACE:\n{json.dumps(thinking_trace, indent=2)}\n"
        )

    async def _run_turn_with_trace(self, prompt: str, *, timeout: int, requested_mode: Optional[str] = None) -> tuple[str, list]:
        session: Optional[OrchestratorSession] = None
        read_task: Optional[asyncio.Task] = None
        final_future: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()
        turn_id_box: Dict[str, Optional[str]] = {"value": None}

        async def on_assistant_final(payload: Dict[str, Any]) -> None:
            expected_turn_id = turn_id_box["value"]
            payload_turn_id = payload.get("turn_id")
            if expected_turn_id and payload_turn_id not in {None, expected_turn_id}:
                return
            if not final_future.done():
                final_future.set_result(payload)

        async def on_permission_request(payload: Dict[str, Any]) -> None:
            if session is None:
                return
            request_id = str(payload.get("request_id", "")).strip()
            if not request_id:
                return
            approved = self.permission_policy == "approve"
            await session.send_permission_decision(request_id, approved)

        session = OrchestratorSession(
            self.host,
            self.port,
            source_id=self.source_id,
            on_assistant_final=on_assistant_final,
            on_permission_request=on_permission_request,
        )
        try:
            await session.connect()
            read_task = asyncio.create_task(session.read_events(), name="autonomy-cognition-read-events")
            mode = requested_mode or self.requested_mode
            turn_id = await session.submit_turn(prompt, requested_mode=mode)
            turn_id_box["value"] = turn_id
            wait_task = asyncio.create_task(
                asyncio.wait_for(final_future, timeout=timeout),
                name="autonomy-cognition-wait-final",
            )
            done, pending = await asyncio.wait({wait_task, read_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if wait_task in done:
                payload = wait_task.result()
            elif final_future.done():
                payload = final_future.result()
            else:
                await asyncio.gather(*pending, return_exceptions=True)
                exc = read_task.exception() if read_task and read_task.done() else None
                raise CognitionClientError("COGNITION session ended before returning a final response.") from exc
            text = str(payload.get("text", "") or "").strip()
            if not text:
                raise CognitionClientError("COGNITION returned an empty final response.")
            trace = payload.get("thinking_trace", [])
            if not isinstance(trace, list):
                trace = []
            return text, trace
        except asyncio.TimeoutError as exc:
            raise CognitionClientError(f"COGNITION request timed out after {timeout} seconds.") from exc
        finally:
            if read_task is not None:
                read_task.cancel()
                await asyncio.gather(read_task, return_exceptions=True)
            if session is not None:
                await session.close()
