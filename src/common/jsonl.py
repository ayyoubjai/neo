import json
from typing import Any, Dict, Optional

import asyncio


def _dumps(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _loads(line: bytes) -> Optional[Dict[str, Any]]:
    if not line:
        return None
    try:
        return json.loads(line.decode("utf-8"))
    except json.JSONDecodeError:
        return None


async def read_jsonl(reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
    line = await reader.readline()
    return _loads(line)


async def write_jsonl(writer: asyncio.StreamWriter, obj: Dict[str, Any]) -> None:
    writer.write(_dumps(obj))
    await writer.drain()
