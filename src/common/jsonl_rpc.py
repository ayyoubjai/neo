import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from common.ids import new_id
from common.jsonl import read_jsonl, write_jsonl


class RpcError(Exception):
    pass


Handler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _format_success(rpc_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if "rpc_id" in payload:
        payload = dict(payload)
        payload.pop("rpc_id", None)
    return {"rpc_id": rpc_id, **payload}


def _format_error(rpc_id: str, message: str) -> Dict[str, Any]:
    return {"rpc_id": rpc_id, "status": "ERROR", "error": message}


async def send_request(
    host: str,
    port: int,
    method: str,
    params: Dict[str, Any],
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    reader, writer = await asyncio.open_connection(host, port)
    rpc_id = new_id()
    await write_jsonl(writer, {"rpc_id": rpc_id, "method": method, "params": params})
    try:
        if timeout is not None and timeout > 0:
            resp = await asyncio.wait_for(read_jsonl(reader), timeout=timeout)
        else:
            resp = await read_jsonl(reader)
    except asyncio.TimeoutError as e:
        writer.close()
        await writer.wait_closed()
        raise RpcError("RPC timeout") from e
    writer.close()
    await writer.wait_closed()
    if resp is None:
        raise RpcError("Empty RPC response")
    return resp


async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, handlers: Dict[str, Handler]) -> None:
    try:
        while True:
            msg = await read_jsonl(reader)
            if msg is None:
                break
            method = msg.get("method", "")
            rpc_id = msg.get("rpc_id") or new_id()
            handler = handlers.get(method)
            if not handler:
                await write_jsonl(writer, _format_error(rpc_id, "Unknown method"))
                continue
            try:
                result = await handler(msg.get("params", {}))
                await write_jsonl(writer, _format_success(rpc_id, result))
            except Exception:
                await write_jsonl(writer, _format_error(rpc_id, "Server error"))
    finally:
        writer.close()
        await writer.wait_closed()


async def start_server(host: str, port: int, handlers: Dict[str, Handler]) -> asyncio.AbstractServer:
    return await asyncio.start_server(
        lambda r, w: _handle_client(r, w, handlers),
        host,
        port,
    )
