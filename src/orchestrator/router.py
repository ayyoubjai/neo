from typing import Any, Dict, Optional

from common.config import load_settings
from common.jsonl_rpc import send_request


class RouterClient:
    def __init__(self):
        settings = load_settings()
        self._host = settings.rpc["orch_host"]
        self._port = settings.rpc["model_port"]
        self._timeout_s = int(settings.orchestrator.get("model_rpc_timeout_s", 0))
        if self._timeout_s < 0:
            self._timeout_s = 0

    async def route(self, text: str, trace_id: Optional[str] = None, turn_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"text": text}
        if trace_id:
            payload["trace_id"] = trace_id
        if turn_id:
            payload["turn_id"] = turn_id
        resp = await send_request(self._host, self._port, "model.Route", payload, timeout=self._timeout_s)
        return resp.get("result", {})
