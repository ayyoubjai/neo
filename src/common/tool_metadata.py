from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from common.config import load_settings


_DEFAULT_LOG_POLICY = {
    "redact_args": False,
    "redact_result": False,
    "redact_io": False,
}


def load_tool_metadata(registry_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    if registry_path is None:
        try:
            settings = load_settings()
            registry_path = os.path.join(settings.workspace_root, "config", "tool_registry.json")
        except Exception:
            return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    metadata: Dict[str, Dict[str, Any]] = {}
    for item in raw.get("tools", []):
        if not isinstance(item, dict):
            continue
        tool_id = item.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id:
            continue
        metadata[tool_id] = item
    return metadata


def tool_log_policy(tool_def: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    policy = dict(_DEFAULT_LOG_POLICY)
    if not isinstance(tool_def, dict):
        return policy

    raw_policy = tool_def.get("log_policy", {})
    if tool_def.get("sensitive") is True:
        raw_policy = {
            "redact_args": True,
            "redact_result": True,
            "redact_io": True,
        }
    if not isinstance(raw_policy, dict):
        return policy

    for key in policy:
        if key in raw_policy:
            policy[key] = bool(raw_policy.get(key))
    return policy


def redact_tool_payload(tool_def: Optional[Dict[str, Any]], kind: str, value: Any) -> Any:
    policy = tool_log_policy(tool_def)
    key = f"redact_{kind}"
    if policy.get(key, False):
        return {"_redacted": True, "kind": kind}
    return value


def merge_runtime_tool_metadata(registry: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    metadata = load_tool_metadata()
    if not metadata:
        return registry

    merged: Dict[str, Dict[str, Any]] = {}
    for tool_id, spec in registry.items():
        item = dict(spec)
        meta = metadata.get(tool_id)
        if isinstance(meta, dict):
            merged_item = dict(meta)
            merged_item.update(item)
            item = merged_item
        merged[tool_id] = item
    return merged
