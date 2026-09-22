import asyncio
import difflib
import hashlib
import inspect
import os
import re
from urllib.parse import unquote, urlparse
from typing import Any, Dict, List, Optional

from common.config import load_settings
from common.ids import new_id
from common.jsonlog import append_jsonl
from common.jsonl_rpc import send_request, start_server
from common.record_log import record_event
from common.tool_metadata import redact_tool_payload
from common.tool_recovery import failure_details
from common.time_utils import utc_now_iso
from integrations.google_workspace import GoogleAuthError, GoogleWorkspaceManager
from tool_runtime.sandbox import SandboxViolation, resolve_workspace_path
from tool_runtime.tools import ToolError, load_generated_registry, tool_registry


DEFAULT_GOOGLE_AUTO_BUNDLES = tuple(GoogleWorkspaceManager.BUNDLE_SCOPES.keys())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unified_diff(before: str, after: str, path: str) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(before_lines, after_lines, fromfile=path, tofile=path)
    return "".join(diff)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _is_abs_path(path: str) -> bool:
    if os.path.isabs(path):
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", path))


def _bool_setting(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _google_bundle_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",")]
    elif isinstance(raw_value, (list, tuple)):
        items = [str(item).strip() for item in raw_value]
    else:
        return []
    seen = set()
    bundles: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        bundles.append(item)
    return bundles


def _auto_google_authorize(settings: Optional[Any] = None) -> None:
    resolved_settings = settings or load_settings()
    google_cfg = getattr(resolved_settings, "google", {}) or {}
    if not _bool_setting(google_cfg.get("enabled", True)):
        return
    if not _bool_setting(google_cfg.get("auto_google_authorize", False)):
        return

    bundles = _google_bundle_list(google_cfg.get("auto_google_authorize_bundles"))
    if not bundles:
        bundles = list(DEFAULT_GOOGLE_AUTO_BUNDLES)

    account_name = google_cfg.get("default_account_name")
    manager = GoogleWorkspaceManager(settings=resolved_settings)
    for bundle in bundles:
        try:
            result = manager.ensure_authorized(bundle=bundle, account_name=account_name)
            mode = "interactive" if result.get("interactive") else "cached"
            print(f"[tool-runtime] google bundle '{bundle}' ready ({mode})")
        except GoogleAuthError as exc:
            print(f"[tool-runtime] google auto authorize failed for '{bundle}': {exc}")


class ToolRuntime:
    def __init__(self, workspace_root: str, storage_host: str, storage_port: int, data_dir: str):
        self._workspace_root = workspace_root
        self._tools = tool_registry()
        self._storage_host = storage_host
        self._storage_port = storage_port
        self._generated_path = os.path.join(workspace_root, "src", "tool_runtime", "generated_tools.py")
        self._generated_mtime: Optional[float] = None
        self._tool_io_path = os.path.join(data_dir, "tool_io.log")

    def _append_tool_io(self, payload: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self._tool_io_path), exist_ok=True)
            record = {"ts": utc_now_iso()}
            record.update(_json_safe(payload))
            append_jsonl(self._tool_io_path, record)
        except Exception:
            pass

    async def _append_audit(self, record: Dict[str, Any]) -> Optional[str]:
        try:
            resp = await send_request(
                self._storage_host,
                self._storage_port,
                "storage.AppendAudit",
                {"record": record},
            )
        except Exception:
            return None
        return resp.get("audit_id")

    def _looks_like_path_key(self, key: str) -> bool:
        if not key:
            return False
        name = key.lower()
        explicit = {
            "path",
            "src",
            "dst",
            "file",
            "filepath",
            "file_path",
            "directory",
            "dir",
            "image_ref",
            "video_path",
            "audio_path",
            "input_path",
            "output_path",
            "source_path",
            "target_path",
            "document_path",
            "doc_path",
        }
        if name in explicit:
            return True
        return name.endswith(("_path", "_file", "_dir", "_directory", "_ref", "_src", "_dst"))

    def _resolve_generated_path_arg(self, raw_path: str) -> str:
        candidate = raw_path.strip()
        if candidate.startswith("workspace:/"):
            try:
                return resolve_workspace_path(self._workspace_root, candidate)
            except SandboxViolation as exc:
                raise ToolError(str(exc))

        if candidate.startswith("file://"):
            parsed = urlparse(candidate)
            candidate = unquote(parsed.path or "")
            if re.match(r"^/[A-Za-z]:[\\/]", candidate):
                candidate = candidate[1:]
        elif candidate.startswith("file:"):
            candidate = candidate[len("file:"):]

        candidate = candidate.strip()
        if not candidate:
            raise ToolError("Path is empty")

        win_abs = re.match(r"^([A-Za-z]):[\\/](.*)$", candidate)
        if win_abs:
            drive = win_abs.group(1).lower()
            tail = win_abs.group(2).replace("\\", "/")
            candidate = f"/mnt/{drive}/{tail}"

        if _is_abs_path(candidate):
            abs_path = os.path.abspath(candidate)
        else:
            abs_path = os.path.abspath(os.path.join(self._workspace_root, candidate))

        root = os.path.abspath(self._workspace_root)
        if abs_path != root and not abs_path.startswith(root + os.sep):
            raise ToolError("Path escapes workspace")
        return abs_path

    def _normalize_generated_value(self, key_hint: str, value: Any) -> Any:
        if isinstance(value, dict):
            normalized: Dict[str, Any] = {}
            for child_key, child_value in value.items():
                child_hint = child_key.lower() if isinstance(child_key, str) else key_hint
                normalized[child_key] = self._normalize_generated_value(child_hint, child_value)
            return normalized
        if isinstance(value, list):
            return [self._normalize_generated_value(key_hint, item) for item in value]
        if not isinstance(value, str):
            return value
        if not self._looks_like_path_key(key_hint):
            return value
        stripped = value.strip()
        if not stripped:
            return value
        if "://" in stripped and not stripped.startswith("file://") and not stripped.startswith("workspace:/"):
            return value
        return self._resolve_generated_path_arg(stripped)

    def _normalize_generated_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in args.items():
            key_hint = key.lower() if isinstance(key, str) else ""
            try:
                normalized[key] = self._normalize_generated_value(key_hint, value)
            except ToolError as exc:
                if isinstance(key, str) and key:
                    raise ToolError(f"Invalid path for '{key}': {exc}")
                raise
        return normalized

    def _make_nested_tool_caller(
        self,
        approval_token: Any,
        trace_id: Any,
        parent_tier: int,
        stack: tuple[str, ...],
    ):
        """Build the synchronous composition API exposed to generated tools."""
        def call_tool(nested_tool_id: Any, nested_args: Any):
            if not isinstance(nested_tool_id, str) or not nested_tool_id.strip():
                return {"status": "ERROR", "error": "Nested tool_id must be a string"}, {}
            if not isinstance(nested_args, dict):
                return {"status": "ERROR", "error": "Nested tool args must be an object"}, {}
            nested_tool_id = nested_tool_id.strip()
            if nested_tool_id in stack:
                return {"status": "ERROR", "error": "Nested tool cycle detected"}, {}
            nested_tool = self._tools.get(nested_tool_id)
            if not isinstance(nested_tool, dict):
                return {"status": "ERROR", "error": f"Unknown nested tool: {nested_tool_id}"}, {}
            try:
                nested_tier = int(nested_tool.get("tier", 0))
            except (TypeError, ValueError):
                nested_tier = 0
            if nested_tier > parent_tier:
                return {
                    "status": "ERROR",
                    "error": f"Nested tool '{nested_tool_id}' requires a higher permission tier",
                }, {}
            try:
                run_args = nested_args
                if bool(nested_tool.get("generated")):
                    run_args = self._normalize_generated_args(nested_args)
                nested_caller = self._make_nested_tool_caller(
                    approval_token,
                    trace_id,
                    nested_tier,
                    stack + (nested_tool_id,),
                )
                if bool(nested_tool.get("generated")):
                    parameters = inspect.signature(nested_tool["fn"]).parameters
                    if len(parameters) >= 3:
                        return nested_tool["fn"](run_args, self._workspace_root, nested_caller)
                    # Support generated modules produced before the optional
                    # composition argument was introduced.
                    return nested_tool["fn"](run_args, self._workspace_root)
                return nested_tool["fn"](run_args, self._workspace_root)
            except ToolError as exc:
                return {"status": "ERROR", "error": str(exc), "error_details": exc.details}, getattr(exc, "io", {})
            except Exception as exc:
                return {"status": "ERROR", "error": f"Nested tool exception: {exc}"}, {}

        return call_tool

    async def Execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._refresh_generated_tools()
        tool_id = params.get("tool_id", "")
        args = params.get("args", {})
        request_id = params.get("request_id", new_id())
        trace_id = params.get("trace_id")
        required_artifacts = params.get("required_artifacts", [])
        approval_token = params.get("approval_token")
        tool = self._tools.get(tool_id, {})
        self._append_tool_io(
            {
                "event_type": "tool_input",
                "request_id": request_id,
                "trace_id": trace_id,
                "tool_id": tool_id,
                "args": redact_tool_payload(tool, "args", args),
                "required_artifacts": required_artifacts,
                "has_approval_token": bool(approval_token),
            }
        )

        if tool_id not in self._tools:
            self._append_tool_io(
                {
                    "event_type": "tool_output",
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "DENIED",
                    "error": "Unknown tool",
                }
            )
            record_event(
                "tool_execute_result",
                {
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "DENIED",
                    "error": "Unknown tool",
                },
            )
            return {"request_id": request_id, "status": "DENIED", "error": "Unknown tool",
                    "error_details": failure_details('MISSING_CAPABILITY')}

        tool = self._tools[tool_id]
        tier = tool.get("tier", 0)
        if tier >= 1 and not approval_token:
            self._append_tool_io(
                {
                    "event_type": "tool_output",
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "NEEDS_CONFIRMATION",
                }
            )
            record_event(
                "tool_execute_result",
                {
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "NEEDS_CONFIRMATION",
                },
            )
            return {"request_id": request_id, "status": "NEEDS_CONFIRMATION",
                    "error_details": failure_details('PERMISSION_DENIED')}

        artifacts: List[Dict[str, Any]] = []
        logs_ref = None

        try:
            run_args = args
            if bool(tool.get("generated")) and isinstance(args, dict):
                run_args = self._normalize_generated_args(args)

            start_payload = {
                "request_id": request_id,
                "trace_id": trace_id,
                "tool_id": tool_id,
                "args": redact_tool_payload(tool, "args", run_args),
            }
            if run_args != args:
                start_payload["raw_args"] = redact_tool_payload(tool, "args", args)
            record_event(
                "tool_execute_start",
                start_payload,
            )
            if bool(tool.get("generated")):
                nested_caller = self._make_nested_tool_caller(
                    approval_token,
                    trace_id,
                    int(tier),
                    (str(tool_id),),
                )
                parameters = inspect.signature(tool["fn"]).parameters
                if len(parameters) >= 3:
                    result, io = tool["fn"](run_args, self._workspace_root, nested_caller)
                else:
                    # Support generated modules produced before the optional
                    # composition argument was introduced.
                    result, io = tool["fn"](run_args, self._workspace_root)
            else:
                result, io = tool["fn"](run_args, self._workspace_root)
            if "diff" in required_artifacts and "before" in io and "after" in io:
                path_for_diff = run_args.get("path", "") if isinstance(run_args, dict) else ""
                diff_text = _unified_diff(io["before"], io["after"], path_for_diff)
                artifacts.append({"type": "diff", "data": diff_text})
            if "hash" in required_artifacts and "after" in io:
                artifacts.append({"type": "sha256", "data": _sha256(io["after"])})

            if "before" in io or "after" in io:
                audit = {
                    "audit_id": new_id(),
                    "action": "tool_execute",
                    "tool_id": tool_id,
                    "trace_id": trace_id,
                    "actor": "tool_runtime",
                    "before_hash": _sha256(io.get("before", "")),
                    "after_hash": _sha256(io.get("after", "")),
                    "diff": artifacts[0]["data"] if artifacts and artifacts[0]["type"] == "diff" else None,
                    "timestamp": utc_now_iso(),
                    "payload": {"args": redact_tool_payload(tool, "args", run_args)},
                }
                logs_ref_id = await self._append_audit(audit)
                if logs_ref_id:
                    logs_ref = f"audit://{logs_ref_id}"

            record_event(
                "tool_execute_result",
                {
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "APPROVED",
                    "result": redact_tool_payload(tool, "result", result),
                    "logs_ref": logs_ref,
                },
            )
            self._append_tool_io(
                {
                    "event_type": "tool_output",
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "APPROVED",
                    "result": redact_tool_payload(tool, "result", result),
                    "io": redact_tool_payload(tool, "io", io),
                    "artifacts": artifacts,
                    "logs_ref": logs_ref,
                }
            )
            return {
                "request_id": request_id,
                "status": "APPROVED",
                "result": result,
                "artifacts": artifacts,
                "logs_ref": logs_ref,
            }
        except ToolError as e:
            err_payload = {
                "event_type": "tool_output",
                "request_id": request_id,
                "trace_id": trace_id,
                "tool_id": tool_id,
                "status": "ERROR",
                "error": str(e),
                "error_details": e.details,
            }
            err_io = getattr(e, "io", None)
            if isinstance(err_io, dict) and err_io:
                err_payload["io"] = err_io
            self._append_tool_io(err_payload)
            record_event(
                "tool_execute_result",
                {
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "ERROR",
                    "error": str(e),
                    "error_details": e.details,
                },
            )
            return {"request_id": request_id, "status": "ERROR", "error": str(e), "error_details": e.details}
        except Exception:
            self._append_tool_io(
                {
                    "event_type": "tool_output",
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "ERROR",
                    "error": "Unhandled tool error",
                }
            )
            record_event(
                "tool_execute_result",
                {
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "ERROR",
                    "error": "Unhandled tool error",
                },
            )
            return {"request_id": request_id, "status": "ERROR", "error": "Unhandled tool error"}

    async def Ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "OK"}

    def _refresh_generated_tools(self) -> None:
        try:
            mtime = os.path.getmtime(self._generated_path)
        except OSError:
            mtime = None
        if mtime is None or mtime == self._generated_mtime:
            return
        self._generated_mtime = mtime
        generated = load_generated_registry(reload_module=True)
        if generated:
            self._tools.update(generated)


async def serve() -> None:
    settings = load_settings()
    runtime = ToolRuntime(
        settings.workspace_root,
        settings.rpc["orch_host"],
        settings.rpc["storage_port"],
        settings.data_dir,
    )
    _auto_google_authorize(settings)

    handlers = {
        "tool.Execute": runtime.Execute,
        "tool.Ping": runtime.Ping,
    }

    listen_host = settings.rpc["orch_host"]
    listen_port = settings.rpc["tool_port"]
    server = await start_server(listen_host, listen_port, handlers)
    print(f"[tool-runtime] listening on {listen_host}:{listen_port}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve())
