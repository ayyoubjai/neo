from typing import Dict, Optional


PRIMARY_ROUTE = {
    "mode": "COGNITION",
    "needs_memory": False,
}


def _coerce_route(parsed: Dict[str, object]) -> Dict[str, object]:
    return dict(PRIMARY_ROUTE)


def route(text: str, trace_id: Optional[str] = None, turn_id: Optional[str] = None) -> Dict[str, object]:
    return dict(PRIMARY_ROUTE)
