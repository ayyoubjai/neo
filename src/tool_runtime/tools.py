import ast
import base64
import copy
import csv
import fnmatch
import importlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Dict, List, Tuple

from common.config import load_settings
from common.tool_metadata import merge_runtime_tool_metadata
from common.time_utils import utc_now_iso
from common.vision_broker import load_vision_broker_manifest_for_settings, workspace_ref_for_path
from embodiment.adapters import (
    AdbPhoneAdapter,
    CameraAdapterError,
    DesktopAdapter,
    DesktopAdapterError,
    OpenCvCameraAdapter,
    PhoneAdapterError,
    RobotAdapterError,
    RosCliRobotAdapter,
)
from embodiment.manager import EmbodimentManager
from integrations.google_workspace import GoogleAuthError, GoogleWorkspaceManager
from model_server.frame_sequence import (
    aggregate_objects as aggregate_frame_objects,
    answer_from_keyframes as answer_from_frame_sequence,
    build_frame_prompt,
    collect_visible_text as collect_frame_text,
    summary_from_keyframes as summarize_frame_sequence,
)
from model_server.video_model import analyze as analyze_video
from model_server.vision_model import analyze
from model_server.uground_model import predict as uground_predict
from tool_runtime.sandbox import resolve_workspace_path, SandboxViolation


class ToolError(Exception):
    def __init__(self, message: str, io: Dict[str, Any] = None):
        super().__init__(message)
        self.io = io or {}


def sys_time(_: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return {"iso": utc_now_iso()}, {}


def sys_list_tools(_: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tools = []
    for tool_id, spec in sorted(tool_registry().items()):
        tools.append(
            {
                "tool_id": tool_id,
                "name": spec.get("name", tool_id),
                "description": spec.get("description", ""),
                "capabilities": list(spec.get("capabilities", [])),
                "required_permissions": list(spec.get("required_permissions", [])),
                "risk_level_default": spec.get("risk_level_default", "LOW"),
                "version": spec.get("version", ""),
                "generated": bool(spec.get("generated", False)),
                "tier": int(spec.get("tier", 0)),
            }
        )
    return {"count": len(tools), "tools": tools}, {}


def _is_abs_path(path: str) -> bool:
    if os.path.isabs(path):
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", path))


def _normalize_workspace_path_arg(path: Any, workspace_root: str) -> Any:
    if not isinstance(path, str) or not path:
        return path
    if path.startswith("workspace:/"):
        return path
    if _is_abs_path(path):
        abs_path = os.path.abspath(path)
        root = os.path.abspath(workspace_root)
        if abs_path == root or abs_path.startswith(root + os.sep):
            rel = os.path.relpath(abs_path, root)
            return f"workspace:/{rel}".replace("\\", "/")
        return path
    return f"workspace:/{path.lstrip('/')}"


def _normalize_workspace_args(args: Dict[str, Any], workspace_root: str, keys: List[str]) -> Dict[str, Any]:
    normalized = dict(args)
    for key in keys:
        if key not in args:
            continue
        value = _normalize_workspace_path_arg(args.get(key), workspace_root)
        if value in (None, ""):
            normalized.pop(key, None)
        else:
            normalized[key] = value
    return normalized


def _resolve_output_workspace_path(
    workspace_root: str,
    path_arg: Any,
    default_workspace_path: str,
) -> Tuple[str, str]:
    if isinstance(path_arg, str) and path_arg:
        workspace_path = path_arg if path_arg.startswith("workspace:/") else f"workspace:/{path_arg.lstrip('/')}"
    else:
        workspace_path = default_workspace_path
    try:
        abs_path = resolve_workspace_path(workspace_root, workspace_path)
    except SandboxViolation as e:
        raise ToolError(str(e))
    return workspace_path, abs_path


def _iso_from_epoch(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def _truncate_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else str(value or "")
    if limit < 1:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _safe_filename_fragment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _list_workspace_path_arg(path_values: Any, workspace_root: str) -> List[Tuple[str, str]]:
    if not isinstance(path_values, list) or not path_values:
        raise ToolError("Expected a non-empty list of workspace paths")
    resolved: List[Tuple[str, str]] = []
    for item in path_values:
        normalized = _normalize_workspace_path_arg(item, workspace_root)
        if not isinstance(normalized, str) or not normalized:
            raise ToolError("Expected each path to be a string")
        try:
            abs_path = resolve_workspace_path(workspace_root, normalized)
        except SandboxViolation as e:
            raise ToolError(str(e))
        resolved.append((normalized, abs_path))
    return resolved


def _repo_workspace_path(args: Dict[str, Any], workspace_root: str) -> Tuple[str, str]:
    args = _normalize_workspace_args(args, workspace_root, ["repo_path"])
    repo_path = args.get("repo_path", "workspace:/")
    try:
        abs_repo = resolve_workspace_path(workspace_root, repo_path)
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.isdir(abs_repo):
        raise ToolError("Repository path not found")
    return repo_path, abs_repo


def _run_subprocess(
    command: List[str],
    *,
    cwd: str,
    timeout_s: float,
    stdin: str | None = None,
    env: Dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    except FileNotFoundError as e:
        raise ToolError(f"Command not found: {e.filename or command[0]}")
    except OSError as e:
        raise ToolError(f"Process launch failed: {e}")


def _require_success(result: subprocess.CompletedProcess[str], action: str) -> subprocess.CompletedProcess[str]:
    if result.returncode == 0:
        return result
    detail = (result.stderr or result.stdout or "").strip()
    if detail:
        raise ToolError(f"{action} failed: {_truncate_text(detail, 500)}")
    raise ToolError(f"{action} failed with exit code {result.returncode}")


def _iter_search_files(root_path: str, file_glob: str) -> List[str]:
    if os.path.isfile(root_path):
        return [root_path]
    files: List[str] = []
    for current_root, _, names in os.walk(root_path):
        for name in names:
            if file_glob and not fnmatch.fnmatch(name, file_glob):
                continue
            files.append(os.path.join(current_root, name))
    return files


def _json_pointer_lookup(payload: Any, pointer: str) -> Tuple[bool, Any]:
    if not pointer:
        return True, payload
    if pointer == "/":
        return True, payload
    if not pointer.startswith("/"):
        raise ToolError("pointer must start with '/'")
    current = payload
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return False, None
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
            continue
        if isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
            continue
        return False, None
    return True, current


def _textual_content_type(content_type: str) -> bool:
    normalized = (content_type or "").lower()
    if normalized.startswith("text/"):
        return True
    return any(token in normalized for token in ("json", "xml", "yaml", "javascript"))


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
        ast.Load,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed):
            raise ToolError("Expression contains unsafe nodes")
    return eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})


def math_eval(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    expr = args.get("expr", "")
    value = _safe_eval(expr)
    return {"value": value}, {}


def fs_read_file(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["path"])
    try:
        path = resolve_workspace_path(workspace_root, args.get("path", ""))
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.exists(path):
        raise ToolError("File not found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content}, {}


def fs_write_file(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["path"])
    try:
        path = resolve_workspace_path(workspace_root, args.get("path", ""))
    except SandboxViolation as e:
        raise ToolError(str(e))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    before = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            before = f.read()
    content = args.get("content", "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"bytes_written": len(content)}, {"before": before, "after": content}


def fs_list_dir(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["path"])
    try:
        path = resolve_workspace_path(workspace_root, args.get("path", "workspace:/"))
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.isdir(path):
        raise ToolError("Directory not found")
    entries = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        entries.append({"name": name, "is_dir": os.path.isdir(full)})
    return {"entries": entries}, {}


def fs_delete(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["path"])
    try:
        path = resolve_workspace_path(workspace_root, args.get("path", ""))
    except SandboxViolation as e:
        raise ToolError(str(e))
    recursive = bool(args.get("recursive", False))
    if not os.path.exists(path):
        raise ToolError("Path not found")
    if os.path.isdir(path):
        if not recursive:
            raise ToolError("Directory delete requires recursive=true")
        shutil.rmtree(path)
    else:
        os.remove(path)
    return {"deleted": True, "path": args.get("path")}, {}


def fs_move(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["src", "dst"])
    try:
        src = resolve_workspace_path(workspace_root, args.get("src", ""))
        dst = resolve_workspace_path(workspace_root, args.get("dst", ""))
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.exists(src):
        raise ToolError("Source not found")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return {"moved": True, "src": args.get("src"), "dst": args.get("dst")}, {}


def fs_copy(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["src", "dst"])
    try:
        src = resolve_workspace_path(workspace_root, args.get("src", ""))
        dst = resolve_workspace_path(workspace_root, args.get("dst", ""))
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.exists(src):
        raise ToolError("Source not found")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return {"copied": True, "src": args.get("src"), "dst": args.get("dst")}, {}


def fs_mkdir(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["path"])
    try:
        path = resolve_workspace_path(workspace_root, args.get("path", ""))
    except SandboxViolation as e:
        raise ToolError(str(e))
    os.makedirs(path, exist_ok=True)
    return {"created": True, "path": args.get("path")}, {}


def fs_stat(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["path"])
    path_arg = args.get("path", "")
    if not isinstance(path_arg, str) or not path_arg:
        raise ToolError("path is required")
    try:
        abs_path = resolve_workspace_path(workspace_root, path_arg)
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.exists(abs_path):
        return {"path": path_arg, "exists": False}, {}

    stat_result = os.stat(abs_path)
    kind = "dir" if os.path.isdir(abs_path) else "file" if os.path.isfile(abs_path) else "other"
    return {
        "path": path_arg,
        "exists": True,
        "kind": kind,
        "is_dir": os.path.isdir(abs_path),
        "is_file": os.path.isfile(abs_path),
        "size_bytes": int(stat_result.st_size),
        "modified_iso": _iso_from_epoch(stat_result.st_mtime),
    }, {}


def fs_find(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["root"])
    root_arg = args.get("root", "workspace:/")
    try:
        root_path = resolve_workspace_path(workspace_root, root_arg)
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.isdir(root_path):
        raise ToolError("root must be a directory")

    pattern = str(args.get("pattern", "*") or "*")
    regex_text = str(args.get("regex", "") or "").strip()
    kind = str(args.get("kind", "any") or "any").strip().lower()
    if kind not in {"any", "file", "dir"}:
        raise ToolError("kind must be one of: any, file, dir")
    max_results = _coerce_int(args.get("max_results", 100), 100, 1, 1000)
    flags = 0 if bool(args.get("case_sensitive", False)) else re.IGNORECASE
    regex = None
    if regex_text:
        try:
            regex = re.compile(regex_text, flags)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}")

    entries: List[Dict[str, Any]] = []
    for current_root, dirnames, filenames in os.walk(root_path):
        candidates: List[Tuple[str, bool]] = []
        if kind in {"any", "dir"}:
            candidates.extend((name, True) for name in dirnames)
        if kind in {"any", "file"}:
            candidates.extend((name, False) for name in filenames)
        for name, is_dir in candidates:
            full_path = os.path.join(current_root, name)
            rel_path = os.path.relpath(full_path, workspace_root).replace("\\", "/")
            workspace_path = f"workspace:/{rel_path}"
            if pattern and not fnmatch.fnmatch(name, pattern) and not fnmatch.fnmatch(workspace_path, pattern):
                continue
            if regex and not regex.search(workspace_path):
                continue
            entries.append({"path": workspace_path, "name": name, "is_dir": is_dir})
            if len(entries) >= max_results:
                return {"root": root_arg, "count": len(entries), "entries": entries}, {}
    return {"root": root_arg, "count": len(entries), "entries": entries}, {}


def fs_search_text(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["root"])
    root_arg = args.get("root", "workspace:/")
    query = args.get("query", "")
    if not isinstance(query, str) or not query:
        raise ToolError("query is required")
    try:
        root_path = resolve_workspace_path(workspace_root, root_arg)
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.exists(root_path):
        raise ToolError("root path not found")

    max_matches = _coerce_int(args.get("max_matches", 50), 50, 1, 500)
    max_file_bytes = _coerce_int(args.get("max_file_bytes", 1_000_000), 1_000_000, 1_024, 10_000_000)
    file_glob = str(args.get("file_glob", "*") or "*")
    case_sensitive = bool(args.get("case_sensitive", False))
    needle = query if case_sensitive else query.lower()
    matches: List[Dict[str, Any]] = []

    for file_path in _iter_search_files(root_path, file_glob):
        try:
            if os.path.getsize(file_path) > max_file_bytes:
                continue
        except OSError:
            continue
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line_number, line in enumerate(handle, start=1):
                    haystack = line if case_sensitive else line.lower()
                    if needle not in haystack:
                        continue
                    rel_path = os.path.relpath(file_path, workspace_root).replace("\\", "/")
                    matches.append(
                        {
                            "path": f"workspace:/{rel_path}",
                            "line_number": line_number,
                            "line": line.rstrip("\n"),
                        }
                    )
                    if len(matches) >= max_matches:
                        return {"query": query, "count": len(matches), "matches": matches}, {}
        except (OSError, UnicodeError):
            continue
    return {"query": query, "count": len(matches), "matches": matches}, {}


def data_json_query(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["path"])
    path_arg = args.get("path", "")
    if not isinstance(path_arg, str) or not path_arg:
        raise ToolError("path is required")
    try:
        abs_path = resolve_workspace_path(workspace_root, path_arg)
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.exists(abs_path):
        raise ToolError("File not found")
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as e:
        raise ToolError(f"Invalid JSON: {e}")
    pointer = str(args.get("pointer", "") or "")
    found, value = _json_pointer_lookup(payload, pointer)
    return {"path": path_arg, "pointer": pointer, "found": found, "value": value}, {}


def data_csv_read(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["path"])
    path_arg = args.get("path", "")
    if not isinstance(path_arg, str) or not path_arg:
        raise ToolError("path is required")
    try:
        abs_path = resolve_workspace_path(workspace_root, path_arg)
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.exists(abs_path):
        raise ToolError("File not found")

    delimiter = str(args.get("delimiter", ",") or ",")
    max_rows = _coerce_int(args.get("max_rows", 100), 100, 1, 5000)
    has_header = bool(args.get("has_header", True))

    with open(abs_path, "r", encoding="utf-8", newline="") as handle:
        if has_header:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows: List[Dict[str, Any]] = []
            for index, row in enumerate(reader):
                if index >= max_rows:
                    break
                rows.append({str(key): value for key, value in (row or {}).items()})
            return {
                "path": path_arg,
                "columns": list(reader.fieldnames or []),
                "row_count": len(rows),
                "rows": rows,
            }, {}

        reader_plain = csv.reader(handle, delimiter=delimiter)
        rows_plain: List[Dict[str, Any]] = []
        columns: List[str] = []
        for index, row in enumerate(reader_plain):
            if index >= max_rows:
                break
            if not columns:
                columns = [f"column_{i + 1}" for i in range(len(row))]
            rows_plain.append({columns[i]: value for i, value in enumerate(row)})
        return {"path": path_arg, "columns": columns, "row_count": len(rows_plain), "rows": rows_plain}, {}


def archive_zip_create(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sources = _list_workspace_path_arg(args.get("sources", args.get("paths")), workspace_root)
    ts = int(time.time() * 1000)
    workspace_path, abs_output = _resolve_output_workspace_path(
        workspace_root,
        args.get("output_path", args.get("path")),
        f"workspace:/archives/archive_{ts}.zip",
    )
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    entry_count = 0
    with zipfile.ZipFile(abs_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for _, abs_source in sources:
            if not os.path.exists(abs_source):
                raise ToolError(f"Source not found: {abs_source}")
            if os.path.isdir(abs_source):
                for current_root, _, filenames in os.walk(abs_source):
                    for name in filenames:
                        full_path = os.path.join(current_root, name)
                        if os.path.abspath(full_path) == os.path.abspath(abs_output):
                            continue
                        arcname = os.path.relpath(full_path, workspace_root).replace("\\", "/")
                        archive.write(full_path, arcname=arcname)
                        entry_count += 1
            else:
                arcname = os.path.relpath(abs_source, workspace_root).replace("\\", "/")
                archive.write(abs_source, arcname=arcname)
                entry_count += 1
    return {
        "archive_path": workspace_path,
        "entry_count": entry_count,
        "size_bytes": os.path.getsize(abs_output),
    }, {}


def archive_zip_extract(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args = _normalize_workspace_args(args, workspace_root, ["archive_path", "output_dir"])
    archive_arg = args.get("archive_path", "")
    if not isinstance(archive_arg, str) or not archive_arg:
        raise ToolError("archive_path is required")
    try:
        abs_archive = resolve_workspace_path(workspace_root, archive_arg)
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.exists(abs_archive):
        raise ToolError("Archive not found")
    output_workspace_path, abs_output_dir = _resolve_output_workspace_path(
        workspace_root,
        args.get("output_dir"),
        f"workspace:/extracted/{_safe_filename_fragment(os.path.splitext(os.path.basename(abs_archive))[0], 'archive')}",
    )
    overwrite = bool(args.get("overwrite", False))
    os.makedirs(abs_output_dir, exist_ok=True)

    extracted_count = 0
    with zipfile.ZipFile(abs_archive, "r") as archive:
        for member in archive.infolist():
            target_path = os.path.abspath(os.path.join(abs_output_dir, member.filename))
            if target_path != abs_output_dir and not target_path.startswith(abs_output_dir + os.sep):
                raise ToolError("Archive entry escapes output directory")
            if member.is_dir():
                os.makedirs(target_path, exist_ok=True)
                continue
            if os.path.exists(target_path) and not overwrite:
                raise ToolError(f"Refusing to overwrite existing file: {member.filename}")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with archive.open(member, "r") as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
            extracted_count += 1
    return {"archive_path": archive_arg, "output_dir": output_workspace_path, "extracted_count": extracted_count}, {}


def _sympy_locals() -> Dict[str, Any]:
    import sympy as sp

    allowed = {
        "symbols": sp.symbols,
        "Symbol": sp.Symbol,
        "Integer": sp.Integer,
        "Rational": sp.Rational,
        "simplify": sp.simplify,
        "expand": sp.expand,
        "factor": sp.factor,
        "diff": sp.diff,
        "integrate": sp.integrate,
        "limit": sp.limit,
        "series": sp.series,
        "solve": sp.solve,
        "Eq": sp.Eq,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "E": sp.E,
    }
    return allowed


#from typing import Dict, Any, Tuple

def math_sympy(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    expr = args.get("expr", "")
    if not isinstance(expr, str) or not expr.strip():
        raise ToolError("expr is required")

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )
    except ImportError as e:
        raise ToolError(f"Sympy not available: {e}")

    s = expr.strip()

    # Parser upgrades:
    # - allow "2x" (implicit multiplication)
    # - treat "^" as exponent (convert_xor)
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    def parse(s_: str) -> sp.Expr:
        return parse_expr(
            s_,
            local_dict=_sympy_locals(),
            transformations=transformations,
            evaluate=True,
        )

    def to_text(obj: Any) -> str:
        try:
            return sp.sstr(obj)
        except Exception:
            return str(obj)

    def to_latex(obj: Any) -> str:
        try:
            return sp.latex(obj)
        except Exception:
            return ""

    # Optional user hints (but still works without them)
    # vars can be ["x","y"] or "x,y"
    vars_arg = args.get("vars", None)
    evalf = bool(args.get("evalf", False))  # numeric evaluation for non-solve results

    def parse_vars(v) -> list[sp.Symbol]:
        if v is None:
            return []
        if isinstance(v, str):
            names = [p.strip() for p in v.replace(";", ",").split(",") if p.strip()]
            return [sp.Symbol(n) for n in names]
        if isinstance(v, (list, tuple)):
            out = []
            for item in v:
                if isinstance(item, str) and item.strip():
                    out.append(sp.Symbol(item.strip()))
            return out
        return []

    # --- AUTO-DETECT: equation/system/inequality vs plain expression ---

    # Split potential systems: "x+y=2; x-y=0" or separated by commas
    parts = [p.strip() for p in s.replace("\n", ";").split(";") if p.strip()]
    if len(parts) == 1:
        # also allow comma-separated systems, but only if it seems like equations
        if "," in parts[0] and any(op in parts[0] for op in ["=", "<", ">"]):
            parts = [p.strip() for p in parts[0].split(",") if p.strip()]

    def is_inequality(txt: str) -> bool:
        return any(op in txt for op in [">=", "<=", ">", "<"])

    def is_equation(txt: str) -> bool:
        # "=" but not just ">=, <=" (handled by inequality)
        return ("=" in txt or "==" in txt) and not is_inequality(txt)

    try:
        # Case A: inequality (single or multiple)
        if any(is_inequality(p) for p in parts):
            ineqs = []
            for p in parts:
                # normalize "==" to "=" for parsing splits
                p_norm = p.replace("==", "=").strip()
                if ">=" in p_norm:
                    a, b = [q.strip() for q in p_norm.split(">=", 1)]
                    ineqs.append(sp.Ge(parse(a), parse(b)))
                elif "<=" in p_norm:
                    a, b = [q.strip() for q in p_norm.split("<=", 1)]
                    ineqs.append(sp.Le(parse(a), parse(b)))
                elif ">" in p_norm:
                    a, b = [q.strip() for q in p_norm.split(">", 1)]
                    ineqs.append(sp.Gt(parse(a), parse(b)))
                elif "<" in p_norm:
                    a, b = [q.strip() for q in p_norm.split("<", 1)]
                    ineqs.append(sp.Lt(parse(a), parse(b)))
                else:
                    # fallback: treat as expression
                    ineqs.append(parse(p_norm))

            # choose variable: args vars > free_symbols
            var_syms = parse_vars(vars_arg)
            if not var_syms:
                free = sorted(set().union(*[getattr(i, "free_symbols", set()) for i in ineqs]), key=lambda x: x.name)
                var_syms = free[:1]  # prefer univariate

            if len(var_syms) == 1:
                sol = sp.reduce_inequalities(ineqs, var_syms[0])
            else:
                sol = sp.reduce_inequalities(ineqs)

            return {"result": to_text(sol), "latex": to_latex(sol)}, {}

        # Case B: equation / system of equations
        if any(is_equation(p) for p in parts):
            eqs = []
            for p in parts:
                p_norm = p.replace("==", "=").strip()
                if "=" not in p_norm:
                    # allow raw expressions in a system; treat as "= 0"
                    eqs.append(sp.Eq(parse(p_norm), 0))
                    continue
                left, right = [q.strip() for q in p_norm.split("=", 1)]
                eqs.append(sp.Eq(parse(left), parse(right)))

            # decide variables to solve for
            var_syms = parse_vars(vars_arg)
            if not var_syms:
                free = sorted(set().union(*[eq.free_symbols for eq in eqs]), key=lambda x: x.name)
                var_syms = free

            if not var_syms:
                # nothing to solve for; just simplify the equations
                out = [sp.simplify(eq) for eq in eqs]
                return {"result": to_text(out), "latex": to_latex(out)}, {}

            sol = sp.solve(eqs, var_syms, dict=True)

            # If sympy returns [] (no solutions) or something odd, try linsolve for linear systems
            if sol == [] and len(eqs) > 1:
                try:
                    sol = sp.linsolve(eqs, *var_syms)
                except Exception:
                    pass

            return {"result": to_text(sol), "latex": to_latex(sol)}, {}

        # Case C: plain expression -> simplify (and optionally evalf)
        parsed = parse(s)
        simplified = sp.simplify(parsed)
        if evalf:
            try:
                simplified = simplified.evalf()
            except Exception:
                pass

        return {"result": to_text(simplified), "latex": to_latex(simplified)}, {}

    except Exception as e:
        raise ToolError(f"Sympy error: {e}")



def _net_search_searxng(query: str, max_results: int, base_url: str, timeout_s: int) -> List[Dict[str, Any]]:
    params = {
        "q": query,
        "format": "json",
        "safesearch": 0,
    }
    url = base_url.rstrip("/") + "/search?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        raise ToolError(f"Search request failed: {e}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise ToolError("Invalid search response")
    results: List[Dict[str, Any]] = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content") or item.get("snippet"),
        })
        if len(results) >= max_results:
            break
    return results[:max_results]


def net_search(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    query = args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query is required")
    max_results = args.get("max_results", 5)
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 5
    max_results = max(1, min(max_results, 10))
    settings = load_settings()
    provider = str(settings.search.get("provider", "searxng")).lower()
    if provider != "searxng":
        provider = "searxng"
    base_url = str(settings.search.get("searxng_url", "")).strip()
    if not base_url:
        raise ToolError("Missing search.searxng_url in settings")
    timeout_s = int(settings.search.get("timeout_s", 20))
    results = _net_search_searxng(query, max_results, base_url, timeout_s)
    return {"query": query, "results": results, "provider": provider}, {}


def http_request(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    url = args.get("url", "")
    if not isinstance(url, str) or not url.strip():
        raise ToolError("url is required")
    url = url.strip()
    parsed_input_url = urllib.parse.urlsplit(url)
    if parsed_input_url.scheme.lower() not in {"http", "https"}:
        raise ToolError("url must use http or https")
    method = str(args.get("method", "GET") or "GET").upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        raise ToolError("Unsupported HTTP method")

    params = args.get("params", {})
    if params:
        if not isinstance(params, dict):
            raise ToolError("params must be an object")
        parsed_url = urllib.parse.urlsplit(url)
        existing_params = urllib.parse.parse_qsl(parsed_url.query, keep_blank_values=True)
        for key, value in params.items():
            if isinstance(value, list):
                for item in value:
                    existing_params.append((str(key), str(item)))
            else:
                existing_params.append((str(key), str(value)))
        url = urllib.parse.urlunsplit(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                urllib.parse.urlencode(existing_params, doseq=True),
                parsed_url.fragment,
            )
        )

    headers = {}
    raw_headers = args.get("headers", {})
    if raw_headers:
        if not isinstance(raw_headers, dict):
            raise ToolError("headers must be an object")
        headers = {str(key): str(value) for key, value in raw_headers.items()}

    body_text = args.get("body_text")
    json_body = args.get("json_body")
    if body_text is not None and json_body is not None:
        raise ToolError("Provide only one of body_text or json_body")
    request_data = None
    if json_body is not None:
        request_data = json.dumps(json_body, ensure_ascii=True).encode("utf-8")
        if not any(key.lower() == "content-type" for key in headers):
            headers["Content-Type"] = "application/json"
    elif body_text is not None:
        request_data = str(body_text).encode("utf-8")

    timeout_s = _coerce_float(args.get("timeout_s", 20), 20.0, 1.0, 300.0)
    max_bytes = _coerce_int(args.get("max_bytes", 1_000_000), 1_000_000, 1_024, 10_000_000)

    save_to_path = args.get("save_to_path")
    output_workspace_path = None
    abs_output_path = None
    if save_to_path is not None:
        output_workspace_path, abs_output_path = _resolve_output_workspace_path(
            workspace_root,
            save_to_path,
            "workspace:/downloads/http_response.bin",
        )
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    request = urllib.request.Request(url, data=request_data, headers=headers, method=method)
    status_code = 0
    response_headers: Dict[str, str] = {}
    content_type = ""
    raw_body = b""
    body_truncated = False
    response_obj = None
    try:
        response_obj = urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.HTTPError as response_obj:
        status_code = int(response_obj.code)
        response_headers = {str(key): str(value) for key, value in response_obj.headers.items()}
        content_type = response_obj.headers.get("Content-Type", "")
        raw_body = response_obj.read(max_bytes + 1)
    except urllib.error.URLError as e:
        raise ToolError(f"HTTP request failed: {e}")
    else:
        status_code = int(getattr(response_obj, "status", 200))
        response_headers = {str(key): str(value) for key, value in response_obj.headers.items()}
        content_type = response_obj.headers.get("Content-Type", "")
        raw_body = response_obj.read(max_bytes + 1)
    finally:
        if response_obj is not None:
            try:
                response_obj.close()
            except Exception:
                pass

    if len(raw_body) > max_bytes:
        body_truncated = True
        raw_body = raw_body[:max_bytes]

    if abs_output_path:
        with open(abs_output_path, "wb") as handle:
            handle.write(raw_body)

    result: Dict[str, Any] = {
        "url": url,
        "method": method,
        "status_code": status_code,
        "content_type": content_type,
        "headers": response_headers,
        "response_bytes": len(raw_body),
        "body_truncated": body_truncated,
    }
    if output_workspace_path:
        result["saved_to_path"] = output_workspace_path

    if method != "HEAD":
        if _textual_content_type(content_type):
            charset = "utf-8"
            match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type, re.IGNORECASE)
            if match:
                charset = match.group(1)
            try:
                result["body_text"] = raw_body.decode(charset, errors="replace")
            except LookupError:
                result["body_text"] = raw_body.decode("utf-8", errors="replace")
        else:
            guessed_type = mimetypes.guess_type(url)[0] or ""
            if _textual_content_type(guessed_type):
                result["body_text"] = raw_body.decode("utf-8", errors="replace")
            else:
                result["body_base64"] = base64.b64encode(raw_body).decode("ascii")
    return result, {}


def google_status(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        result = GoogleWorkspaceManager().status(account_name=args.get("account_name"))
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_authorize(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = args.get("bundle", "")
    if not isinstance(bundle, str) or not bundle.strip():
        raise ToolError("bundle is required")
    try:
        result = GoogleWorkspaceManager().authorize(
            bundle=bundle.strip(),
            account_name=args.get("account_name"),
            force_reconsent=bool(args.get("force_reconsent", False)),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_disconnect(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = args.get("bundle")
    try:
        result = GoogleWorkspaceManager().disconnect(
            account_name=args.get("account_name"),
            bundle=bundle if isinstance(bundle, str) and bundle.strip() else None,
            revoke=bool(args.get("revoke", False)),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_gmail_list_messages(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        result = GoogleWorkspaceManager().gmail_list_messages(
            account_name=args.get("account_name"),
            query=str(args.get("query", "") or ""),
            max_results=args.get("max_results", 10),
            label_ids=args.get("label_ids"),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_gmail_get_message(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    message_id = args.get("message_id", "")
    if not isinstance(message_id, str) or not message_id.strip():
        raise ToolError("message_id is required")
    try:
        result = GoogleWorkspaceManager().gmail_get_message(
            message_id=message_id.strip(),
            account_name=args.get("account_name"),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_gmail_create_draft(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    subject = args.get("subject", "")
    to = args.get("to")
    if not isinstance(subject, str):
        raise ToolError("subject must be a string")
    try:
        result = GoogleWorkspaceManager().gmail_create_draft(
            subject=subject,
            to=to,
            body_text=str(args.get("body_text", args.get("body", "")) or ""),
            account_name=args.get("account_name"),
            cc=args.get("cc"),
            bcc=args.get("bcc"),
            body_html=str(args.get("body_html", "") or ""),
            reply_to_message_id=str(args.get("reply_to_message_id", "") or ""),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_gmail_send_message(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    subject = args.get("subject", "")
    to = args.get("to")
    if not isinstance(subject, str):
        raise ToolError("subject must be a string")
    try:
        result = GoogleWorkspaceManager().gmail_send_message(
            subject=subject,
            to=to,
            body_text=str(args.get("body_text", args.get("body", "")) or ""),
            account_name=args.get("account_name"),
            cc=args.get("cc"),
            bcc=args.get("bcc"),
            body_html=str(args.get("body_html", "") or ""),
            reply_to_message_id=str(args.get("reply_to_message_id", "") or ""),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_calendar_list_events(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        result = GoogleWorkspaceManager().calendar_list_events(
            account_name=args.get("account_name"),
            calendar_id=str(args.get("calendar_id", "primary") or "primary"),
            time_min=str(args.get("time_min", "") or ""),
            time_max=str(args.get("time_max", "") or ""),
            max_results=args.get("max_results", 10),
            query=str(args.get("query", "") or ""),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_calendar_create_event(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        result = GoogleWorkspaceManager().calendar_create_event(
            summary=args.get("summary", ""),
            start=args.get("start"),
            end=args.get("end"),
            account_name=args.get("account_name"),
            calendar_id=str(args.get("calendar_id", "primary") or "primary"),
            description=str(args.get("description", "") or ""),
            location=str(args.get("location", "") or ""),
            attendees=args.get("attendees"),
            send_updates=str(args.get("send_updates", "none") or "none"),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_docs_get_document(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    document_id = args.get("document_id", "")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ToolError("document_id is required")
    try:
        result = GoogleWorkspaceManager().docs_get_document(
            document_id=document_id.strip(),
            account_name=args.get("account_name"),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_docs_create_document(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    title = args.get("title", "")
    if not isinstance(title, str) or not title.strip():
        raise ToolError("title is required")
    try:
        result = GoogleWorkspaceManager().docs_create_document(
            title=title.strip(),
            account_name=args.get("account_name"),
            content=str(args.get("content", "") or ""),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def google_docs_append_text(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    document_id = args.get("document_id", "")
    text = args.get("text", "")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ToolError("document_id is required")
    if not isinstance(text, str) or not text:
        raise ToolError("text is required")
    try:
        result = GoogleWorkspaceManager().docs_append_text(
            document_id=document_id.strip(),
            text=text,
            account_name=args.get("account_name"),
        )
    except GoogleAuthError as e:
        raise ToolError(str(e))
    return result, {}


def git_status(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    repo_arg, abs_repo = _repo_workspace_path(args, workspace_root)
    result = _require_success(
        _run_subprocess(
            ["git", "-C", abs_repo, "status", "--short", "--branch"],
            cwd=abs_repo,
            timeout_s=_coerce_float(args.get("timeout_s", 20), 20.0, 1.0, 120.0),
        ),
        "git status",
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    branch = lines[0] if lines else ""
    changes = []
    for line in lines[1:]:
        if len(line) < 3:
            continue
        changes.append({"status": line[:2], "path": line[3:]})
    return {"repo_path": repo_arg, "branch": branch, "change_count": len(changes), "changes": changes}, {}


def git_diff(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    repo_arg, abs_repo = _repo_workspace_path(args, workspace_root)
    command = ["git", "-C", abs_repo, "diff"]
    if bool(args.get("staged", False)):
        command.append("--staged")
    ref = args.get("ref")
    if isinstance(ref, str) and ref.strip():
        command.append(ref.strip())
    pathspecs = args.get("paths")
    if pathspecs is not None:
        if not isinstance(pathspecs, list) or any(not isinstance(item, str) or not item for item in pathspecs):
            raise ToolError("paths must be a list of strings")
        if pathspecs:
            command.append("--")
            command.extend(pathspecs)
    result = _require_success(
        _run_subprocess(
            command,
            cwd=abs_repo,
            timeout_s=_coerce_float(args.get("timeout_s", 20), 20.0, 1.0, 120.0),
        ),
        "git diff",
    )
    return {"repo_path": repo_arg, "diff": result.stdout}, {}


def git_log(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    repo_arg, abs_repo = _repo_workspace_path(args, workspace_root)
    max_commits = _coerce_int(args.get("max_commits", 10), 10, 1, 100)
    pretty = "%H%x1f%an%x1f%ad%x1f%s"
    result = _require_success(
        _run_subprocess(
            ["git", "-C", abs_repo, "log", f"-n{max_commits}", "--date=iso-strict", f"--pretty=format:{pretty}"],
            cwd=abs_repo,
            timeout_s=_coerce_float(args.get("timeout_s", 20), 20.0, 1.0, 120.0),
        ),
        "git log",
    )
    commits = []
    for line in result.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        commits.append(
            {
                "commit": parts[0],
                "author": parts[1],
                "date": parts[2],
                "subject": parts[3],
            }
        )
    return {"repo_path": repo_arg, "count": len(commits), "commits": commits}, {}


def proc_exec(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    command = args.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
        raise ToolError("command must be a non-empty list of strings")
    args = _normalize_workspace_args(args, workspace_root, ["cwd"])
    cwd_arg = args.get("cwd", "workspace:/")
    try:
        cwd = resolve_workspace_path(workspace_root, cwd_arg)
    except SandboxViolation as e:
        raise ToolError(str(e))
    if not os.path.isdir(cwd):
        raise ToolError("cwd must be a directory")

    stdin = args.get("stdin")
    if stdin is not None and not isinstance(stdin, str):
        stdin = str(stdin)
    timeout_s = _coerce_float(args.get("timeout_s", 30), 30.0, 1.0, 300.0)
    max_output_chars = _coerce_int(args.get("max_output_chars", 20_000), 20_000, 256, 200_000)
    env = os.environ.copy()
    raw_env = args.get("env")
    if raw_env is not None:
        if not isinstance(raw_env, dict):
            raise ToolError("env must be an object")
        for key, value in raw_env.items():
            env[str(key)] = str(value)

    try:
        result = _run_subprocess(command, cwd=cwd, timeout_s=timeout_s, stdin=stdin, env=env)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        timed_out = False
        returncode = int(result.returncode)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", errors="replace")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", errors="replace")
        timed_out = True
        returncode = -1

    stdout_truncated = len(stdout) > max_output_chars
    stderr_truncated = len(stderr) > max_output_chars
    return {
        "command": command,
        "cwd": cwd_arg,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": _truncate_text(stdout, max_output_chars),
        "stderr": _truncate_text(stderr, max_output_chars),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }, {}


def image_analyse(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    image_ref = args.get("image_ref", "") or args.get("path", "")
    question = args.get("question") or args.get("prompt")
    if question is not None and not isinstance(question, str):
        question = str(question)
    if not isinstance(image_ref, str) or not image_ref:
        raise ToolError("image_ref is required")
    if image_ref.startswith("file:"):
        abs_path = image_ref[len("file:") :]
        if not abs_path:
            raise ToolError("Invalid file: path")
        rel = os.path.relpath(abs_path, workspace_root)
        image_ref = f"workspace:/{rel}"
    elif not image_ref.startswith("workspace:/"):
        image_ref = f"workspace:/{image_ref.lstrip('/')}"
    try:
        resolve_workspace_path(workspace_root, image_ref)
    except SandboxViolation as e:
        raise ToolError(str(e))
    result = analyze(image_ref, question=question if question else None)
    if isinstance(result, dict) and result.get("ok") is False:
        error = result.get("error")
        err_io: Dict[str, Any] = {}
        raw_output = result.get("raw_output")
        if isinstance(raw_output, str) and raw_output:
            err_io["raw_output"] = raw_output
        if isinstance(error, str) and error:
            raise ToolError(error, io=err_io)
        raise ToolError("Image analysis failed", io=err_io)

    io: Dict[str, Any] = {}
    if isinstance(result, dict):
        raw_output = result.pop("raw_output", None)
        raw_payload = result.pop("raw_payload", None)
        if isinstance(raw_output, str) and raw_output:
            io["raw_output"] = raw_output
        if isinstance(raw_payload, dict):
            io["raw_payload"] = raw_payload
    return result, io


def video_analyse(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    video_ref = args.get("video_ref", "") or args.get("video_path", "") or args.get("path", "")
    question = args.get("question") or args.get("prompt")
    if question is not None and not isinstance(question, str):
        question = str(question)
    if not isinstance(video_ref, str) or not video_ref:
        raise ToolError("video_ref is required")
    if video_ref.startswith("file:"):
        abs_path = video_ref[len("file:") :]
        if not abs_path:
            raise ToolError("Invalid file: path")
        rel = os.path.relpath(abs_path, workspace_root)
        video_ref = f"workspace:/{rel}"
    else:
        video_ref = _normalize_workspace_path_arg(video_ref, workspace_root)
        if not isinstance(video_ref, str) or not video_ref.startswith("workspace:/"):
            raise ToolError("video_ref must resolve inside the workspace")
    try:
        resolve_workspace_path(workspace_root, video_ref)
    except SandboxViolation as e:
        raise ToolError(str(e))

    start_s = args.get("start_s")
    end_s = args.get("end_s")
    try:
        if start_s is not None:
            start_s = float(start_s)
        if end_s is not None:
            end_s = float(end_s)
    except (TypeError, ValueError):
        raise ToolError("start_s and end_s must be numbers")

    result = analyze_video(video_ref, question=question if question else None, start_s=start_s, end_s=end_s)
    if isinstance(result, dict) and result.get("ok") is False:
        error = result.get("error")
        if isinstance(error, str) and error:
            raise ToolError(error)
        raise ToolError("Video analysis failed")
    return result, {}


def _vision_broker_time_scope(value: Any) -> str:
    normalized = str(value or "now").strip().lower()
    if normalized in {"now", "latest", "stable", "recent", "watch"}:
        return normalized
    raise ToolError("time_scope must be one of now, latest, stable, recent, watch")


def _vision_analysis_mode(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"auto", "image", "video"}:
        return normalized
    raise ToolError("analysis_mode must be one of auto, image, video")


def _select_evenly(items: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if count >= len(items):
        return list(items)
    if count <= 1:
        return [items[-1]]
    selected: List[Dict[str, Any]] = []
    used = set()
    max_index = len(items) - 1
    for idx in range(count):
        position = round((idx * max_index) / float(max(1, count - 1)))
        position = max(0, min(max_index, position))
        if position in used:
            continue
        used.add(position)
        selected.append(items[position])
    if selected and selected[-1] is not items[-1]:
        selected[-1] = items[-1]
    return selected or [items[-1]]


def _vision_candidate_frames_from_manifest(
    manifest: Dict[str, Any],
    *,
    time_scope: str,
    lookback_s: float,
    watch_started_ts: float | None = None,
) -> List[Dict[str, Any]]:
    latest = manifest.get("latest")
    stable = manifest.get("stable")
    buffer_payload = manifest.get("buffer") or {}
    buffer_frames = buffer_payload.get("frames") if isinstance(buffer_payload, dict) else []
    if not isinstance(buffer_frames, list):
        buffer_frames = []
    updated_ts = float(manifest.get("updated_ts") or 0.0)

    if time_scope == "latest":
        return [dict(latest)] if isinstance(latest, dict) else []
    if time_scope == "stable":
        return [dict(stable)] if isinstance(stable, dict) else []
    if time_scope == "now":
        candidates = [item for item in [latest, stable] if isinstance(item, dict)]
        if len(candidates) == 2:
            latest_ts = float(candidates[0].get("ts") or 0.0)
            stable_ts = float(candidates[1].get("ts") or 0.0)
            if stable_ts + 1.0 < latest_ts:
                candidates = [candidates[0]]
            else:
                candidates = [max(candidates, key=lambda item: float(item.get("ts") or 0.0))]
        return [dict(item) for item in candidates[:1]]

    candidates: List[Dict[str, Any]] = []
    if time_scope == "recent":
        cutoff = updated_ts - lookback_s
        for item in buffer_frames:
            if not isinstance(item, dict):
                continue
            if float(item.get("ts") or 0.0) >= cutoff:
                candidates.append(dict(item))
    elif time_scope == "watch":
        cutoff = float(watch_started_ts or updated_ts)
        for item in buffer_frames:
            if not isinstance(item, dict):
                continue
            if float(item.get("ts") or 0.0) >= cutoff:
                candidates.append(dict(item))
        if isinstance(latest, dict):
            latest_ts = float(latest.get("ts") or 0.0)
            if latest_ts >= cutoff and all(item.get("frame_id") != latest.get("frame_id") for item in candidates):
                candidates.append(dict(latest))
    if not candidates and time_scope == "recent" and isinstance(latest, dict):
        candidates = [dict(latest)]
    return candidates


def _vision_selected_frames_from_candidates(candidates: List[Dict[str, Any]], sample_count: int) -> List[Dict[str, Any]]:
    return _select_evenly(candidates, sample_count) if candidates else []


def _vision_scene_signature(scene: Any) -> Tuple[Any, ...]:
    if not isinstance(scene, dict):
        return ()
    signature = scene.get("signature")
    if isinstance(signature, list):
        flattened = []
        for item in signature:
            if isinstance(item, list):
                flattened.append(tuple(item))
            else:
                flattened.append(item)
        return tuple(flattened)
    if isinstance(signature, tuple):
        return signature
    objects = scene.get("objects")
    if isinstance(objects, list):
        compact = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            try:
                count = int(item.get("count", 1))
            except (TypeError, ValueError):
                count = 1
            compact.append((label, count))
        return tuple(compact)
    summary = str(scene.get("summary") or "").strip()
    return (summary,) if summary else ()


def _vision_watch_ready(candidate_frames: List[Dict[str, Any]]) -> bool:
    if len(candidate_frames) < 2:
        return False
    first = candidate_frames[0]
    last = candidate_frames[-1]
    first_sig = _vision_scene_signature(first.get("scene"))
    last_sig = _vision_scene_signature(last.get("scene"))
    return bool(first_sig and last_sig and first_sig != last_sig)


def _vision_abs_path_from_ref(image_ref: str, workspace_root: str) -> str:
    if image_ref.startswith("workspace:/"):
        return resolve_workspace_path(workspace_root, image_ref)
    if image_ref.startswith("file:"):
        path = image_ref[len("file:") :].strip()
        if not path:
            raise ToolError("Invalid file path for broker frame")
        return path
    if not image_ref:
        raise ToolError("Broker frame image_ref is empty")
    return resolve_workspace_path(workspace_root, f"workspace:/{image_ref.lstrip('/')}")


def _vision_clip_export_config(settings: Any) -> Tuple[str, bool]:
    vision_cfg = getattr(settings, "vision", {}) or {}
    export_dir = str(vision_cfg.get("broker_clip_export_dir", "") or "").strip()
    if not export_dir:
        broker_dir = str(vision_cfg.get("broker_dir", "") or "").strip()
        if not broker_dir:
            broker_dir = os.path.join(settings.data_dir, "vision_broker")
        export_dir = os.path.join(broker_dir, "clips")
    keep = _coerce_bool(vision_cfg.get("broker_keep_exported_clips", False), False)
    return export_dir, keep


def _vision_clip_fps(candidate_frames: List[Dict[str, Any]]) -> float:
    timestamps = [float(item.get("ts") or 0.0) for item in candidate_frames if float(item.get("ts") or 0.0) > 0.0]
    deltas = []
    for left, right in zip(timestamps, timestamps[1:]):
        delta = right - left
        if delta > 0.0:
            deltas.append(delta)
    if not deltas:
        return 2.0
    avg_delta = sum(deltas) / float(len(deltas))
    if avg_delta <= 0.0:
        return 2.0
    fps = 1.0 / avg_delta
    return max(1.0, min(8.0, fps))


def _vision_export_clip_from_frames(
    candidate_frames: List[Dict[str, Any]],
    *,
    settings: Any,
    workspace_root: str,
) -> Tuple[str, str, bool]:
    if len(candidate_frames) < 2:
        raise ToolError("At least two broker frames are required to export a clip")
    try:
        import cv2
    except ImportError as e:
        raise ToolError(f"opencv-python is required for broker clip export: {e}") from e

    export_dir, keep = _vision_clip_export_config(settings)
    os.makedirs(export_dir, exist_ok=True)
    if keep:
        abs_path = os.path.join(export_dir, f"vision_clip_{int(time.time() * 1000)}.avi")
        cleanup = False
    else:
        fd, abs_path = tempfile.mkstemp(prefix="vision_clip_", suffix=".avi", dir=export_dir)
        os.close(fd)
        cleanup = True

    frame_paths = [_vision_abs_path_from_ref(str(item.get("image_ref") or ""), workspace_root) for item in candidate_frames]
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ToolError("Unable to read the first broker frame while exporting clip")
    height, width = first_frame.shape[:2]
    fps = _vision_clip_fps(candidate_frames)

    writer = cv2.VideoWriter(abs_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(abs_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    if not writer.isOpened():
        raise ToolError("Unable to initialize video writer for broker clip export")

    timestamps = [float(item.get("ts") or 0.0) for item in candidate_frames]
    deltas = [delta for delta in (right - left for left, right in zip(timestamps, timestamps[1:])) if delta > 0.0]
    default_delta = (sum(deltas) / float(len(deltas))) if deltas else max(0.25, 1.0 / fps)
    frames_written = 0
    try:
        for index, frame_path in enumerate(frame_paths):
            image = cv2.imread(frame_path)
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            current_ts = timestamps[index]
            next_ts = timestamps[index + 1] if index + 1 < len(timestamps) else current_ts + default_delta
            duration = max(default_delta, next_ts - current_ts)
            repeat_count = max(1, int(round(duration * fps)))
            for _ in range(repeat_count):
                writer.write(image)
                frames_written += 1
    finally:
        writer.release()
    if frames_written <= 0:
        raise ToolError("Broker clip export produced an empty video")
    return workspace_ref_for_path(abs_path, workspace_root), abs_path, cleanup


def _vision_analyze_selected_frames(selected_frames: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    if not selected_frames:
        raise ToolError("No broker frames were available for observation")
    prompt = build_frame_prompt(question or None, source_label="live camera frame")
    base_ts = float(selected_frames[0].get("ts") or 0.0)
    analyzed_frames: List[Dict[str, Any]] = []
    for item in selected_frames:
        image_ref = str(item.get("image_ref") or "").strip()
        if not image_ref:
            continue
        frame_result = analyze(image_ref, question=prompt)
        if isinstance(frame_result, dict) and frame_result.get("ok") is False:
            frame_summary = str(frame_result.get("summary") or "Frame analysis failed.").strip()
            frame_answer = str(frame_result.get("answer") or "").strip()
            frame_text = ""
            frame_objects: List[Any] = []
        else:
            frame_summary = str(frame_result.get("summary") or "").strip()
            frame_answer = str(frame_result.get("answer") or "").strip()
            frame_text = str(frame_result.get("text") or "").strip()
            frame_objects = frame_result.get("objects") or []
        analyzed_frames.append(
            {
                "frame_id": str(item.get("frame_id") or ""),
                "image_ref": image_ref,
                "timestamp_s": round(float(item.get("ts") or 0.0) - base_ts, 3),
                "absolute_ts": round(float(item.get("ts") or 0.0), 3),
                "summary": frame_summary,
                "answer": frame_answer,
                "text": frame_text,
                "objects": frame_objects,
                "scene": dict(item.get("scene") or {}) if isinstance(item.get("scene"), dict) else {},
            }
        )
    if not analyzed_frames:
        raise ToolError("No broker frames could be analyzed")
    if len(analyzed_frames) == 1:
        only = analyzed_frames[0]
        return {
            "summary": only["summary"],
            "answer": only["answer"] or only["summary"],
            "text": only["text"],
            "objects": only["objects"],
            "frames": analyzed_frames,
        }
    summary = summarize_frame_sequence(analyzed_frames)
    answer = answer_from_frame_sequence(
        question,
        summary,
        analyzed_frames,
        evidence_label="live camera",
        duration_s=round(max(0.0, analyzed_frames[-1]["timestamp_s"]), 3),
    )
    return {
        "summary": summary,
        "answer": answer,
        "text": collect_frame_text(analyzed_frames),
        "objects": aggregate_frame_objects(analyzed_frames),
        "frames": analyzed_frames,
    }


def _vision_analyze_video_frames(
    candidate_frames: List[Dict[str, Any]],
    question: str,
    *,
    settings: Any,
    workspace_root: str,
) -> Dict[str, Any]:
    video_ref, abs_path, cleanup = _vision_export_clip_from_frames(
        candidate_frames,
        settings=settings,
        workspace_root=workspace_root,
    )
    try:
        result = analyze_video(
            video_ref,
            question=question if question else None,
            transcription_enabled_override=False,
        )
        if isinstance(result, dict) and result.get("ok") is False:
            error = str(result.get("error") or "Video analysis failed").strip() or "Video analysis failed"
            raise ToolError(error)
        if not isinstance(result, dict):
            raise ToolError("Video analysis returned an invalid payload")
        return {
            "summary": str(result.get("summary") or "").strip(),
            "answer": str(result.get("answer") or "").strip(),
            "text": str(result.get("text") or "").strip(),
            "objects": result.get("objects") or [],
            "frames": result.get("keyframes") or [],
            "video_ref": video_ref if not cleanup else "",
            "duration_s": float(result.get("duration_s") or 0.0),
            "scenes": result.get("scenes") or [],
            "events": result.get("events") or [],
        }
    finally:
        if cleanup and abs_path and os.path.exists(abs_path):
            try:
                os.unlink(abs_path)
            except OSError:
                pass


def vision_observe(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    settings = load_settings()
    question = args.get("question") or args.get("prompt") or ""
    if question is not None and not isinstance(question, str):
        question = str(question)
    question = str(question or "").strip()
    time_scope = _vision_broker_time_scope(args.get("time_scope", "now"))
    requested_analysis_mode = _vision_analysis_mode(args.get("analysis_mode", "auto"))
    lookback_s = _coerce_float(args.get("lookback_s"), 8.0, 0.5, 60.0)
    max_duration_s = _coerce_float(args.get("max_duration_s"), 8.0, 1.0, 30.0)
    sample_count = _coerce_int(args.get("sample_count"), 4, 1, 12)

    manifest = load_vision_broker_manifest_for_settings(settings)
    if not manifest:
        raise ToolError("Live vision broker manifest is unavailable. Start the local vision sense first.")

    candidate_frames = _vision_candidate_frames_from_manifest(
        manifest,
        time_scope=time_scope,
        lookback_s=lookback_s,
    )
    if time_scope == "watch" and not candidate_frames:
        started_ts = float(manifest.get("updated_ts") or time.time())
        deadline = time.time() + max_duration_s
        while time.time() < deadline:
            time.sleep(0.25)
            manifest = load_vision_broker_manifest_for_settings(settings)
            candidate_frames = _vision_candidate_frames_from_manifest(
                manifest,
                time_scope="watch",
                lookback_s=lookback_s,
                watch_started_ts=started_ts,
            )
            if _vision_watch_ready(candidate_frames):
                break
    if not candidate_frames:
        raise ToolError("No broker frames were available for the requested time scope")

    analysis_mode = requested_analysis_mode
    if analysis_mode == "auto":
        analysis_mode = "video" if time_scope in {"recent", "watch"} and len(candidate_frames) >= 2 else "image"

    analyzed: Dict[str, Any]
    if analysis_mode == "video":
        try:
            analyzed = _vision_analyze_video_frames(
                candidate_frames,
                question,
                settings=settings,
                workspace_root=workspace_root,
            )
        except ToolError:
            if requested_analysis_mode != "auto":
                raise
            analysis_mode = "image"
            selected_frames = _vision_selected_frames_from_candidates(candidate_frames, sample_count)
            analyzed = _vision_analyze_selected_frames(selected_frames, question)
        else:
            selected_frames = candidate_frames
    else:
        selected_frames = _vision_selected_frames_from_candidates(candidate_frames, sample_count)
        analyzed = _vision_analyze_selected_frames(selected_frames, question)

    primary_scene = candidate_frames[-1].get("scene") if isinstance(candidate_frames[-1].get("scene"), dict) else {}
    result = {
        "ok": True,
        "time_scope": time_scope,
        "analysis_mode": analysis_mode,
        "summary": analyzed["summary"],
        "answer": analyzed["answer"],
        "text": analyzed["text"],
        "objects": analyzed["objects"],
        "frames": analyzed["frames"],
        "scene": dict(primary_scene),
        "sampling": {
            "candidate_frame_count": len(candidate_frames),
            "candidate_frame_ids": [str(item.get("frame_id") or "") for item in candidate_frames],
            "selected_frame_count": len(selected_frames),
            "selected_frame_ids": [str(item.get("frame_id") or "") for item in selected_frames],
            "lookback_s": lookback_s,
            "max_duration_s": max_duration_s,
            "sample_count": sample_count,
            "manifest_updated_ts": float(manifest.get("updated_ts") or 0.0),
        },
    }
    video_ref = str(analyzed.get("video_ref") or "").strip()
    if video_ref:
        result["video_ref"] = video_ref
    return result, {}


def _denorm_ui_coords(value: Any, width: int, height: int) -> None:
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
        if isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)) and not isinstance(y, bool):
            value["x"] = (x / 1000.0) * width
            value["y"] = (y / 1000.0) * height
        conf = value.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            conf_f = float(conf)
            if conf_f < 0.0:
                conf_f = 0.0
            elif conf_f > 1.0:
                conf_f = 1.0
            value["confidence"] = conf_f
        for item in value.values():
            _denorm_ui_coords(item, width, height)
    elif isinstance(value, list):
        for item in value:
            _denorm_ui_coords(item, width, height)


def ui_predict_coords(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    image_ref = args.get("image_ref", "") or args.get("path", "")
    query = args.get("query") or args.get("label") or args.get("target") or args.get("question") or args.get("prompt")
    if query is not None and not isinstance(query, str):
        query = str(query)
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query is required")
    if not isinstance(image_ref, str) or not image_ref:
        raise ToolError("image_ref is required")
    if image_ref.startswith("file:"):
        abs_path = image_ref[len("file:") :]
        if not abs_path:
            raise ToolError("Invalid file: path")
        rel = os.path.relpath(abs_path, workspace_root)
        image_ref = f"workspace:/{rel}"
    elif not image_ref.startswith("workspace:/"):
        image_ref = f"workspace:/{image_ref.lstrip('/')}"
    try:
        resolve_workspace_path(workspace_root, image_ref)
    except SandboxViolation as e:
        raise ToolError(str(e))
    settings = load_settings()
    width = args.get("width") or args.get("screen_width") or settings.models.get("ui_grounding_screen_width", 1920)
    height = args.get("height") or args.get("screen_height") or settings.models.get("ui_grounding_screen_height", 1080)
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        raise ToolError("width and height must be integers")
    if width <= 0 or height <= 0:
        raise ToolError("width and height must be > 0")

    result = uground_predict(image_ref, query=query)
    if isinstance(result, dict) and result.get("ok") is False:
        error = result.get("error")
        err_io: Dict[str, Any] = {}
        raw_output = result.get("raw_output")
        if isinstance(raw_output, str) and raw_output:
            err_io["raw_output"] = raw_output
        if isinstance(error, str) and error:
            raise ToolError(error, io=err_io)
        raise ToolError("UI grounding failed", io=err_io)

    io: Dict[str, Any] = {}
    if isinstance(result, dict):
        payload = result.get("json")
        if isinstance(payload, dict):
            original_payload = copy.deepcopy(payload)
            _denorm_ui_coords(payload, width, height)
            result["json"] = payload
            result["raw_payload"] = original_payload
        raw_output = result.pop("raw_output", None)
        raw_payload = result.pop("raw_payload", None)
        if isinstance(raw_output, str) and raw_output:
            io["raw_output"] = raw_output
        if isinstance(raw_payload, dict):
            io["raw_payload"] = raw_payload
    return result, io


def ui_screenshot(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    region = args.get("region")
    if region and isinstance(region, dict):
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        width = int(region.get("width", 0))
        height = int(region.get("height", 0))
        if width <= 0 or height <= 0:
            raise ToolError("region.width and region.height must be > 0")
        bbox = (x, y, width, height)
    else:
        bbox = None

    ts = int(time.time() * 1000)
    workspace_path, abs_path = _resolve_output_workspace_path(
        workspace_root,
        args.get("path"),
        f"workspace:/screenshots/screenshot_{ts}.png",
    )
    try:
        result = DesktopAdapter().capture_screen(abs_path, region=bbox)
    except DesktopAdapterError as e:
        raise ToolError(str(e))
    return {"image_ref": workspace_path, "width": result["width"], "height": result["height"]}, {}


def ui_click(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    x = args.get("x")
    y = args.get("y")
    if x is None or y is None:
        raise ToolError("x and y are required")
    button = args.get("button", "left")
    clicks = int(args.get("clicks", 1))
    interval = float(args.get("interval", 0.0))
    try:
        return DesktopAdapter().click(int(x), int(y), button=button, clicks=clicks, interval=interval), {}
    except DesktopAdapterError as e:
        raise ToolError(str(e))


def ui_move(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    x = args.get("x")
    y = args.get("y")
    if x is None or y is None:
        raise ToolError("x and y are required")
    duration = float(args.get("duration", 0.0))
    try:
        return DesktopAdapter().move(int(x), int(y), duration=duration), {}
    except DesktopAdapterError as e:
        raise ToolError(str(e))


def ui_drag(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    start = args.get("start")
    end = args.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise ToolError("start and end are required objects")
    duration = float(args.get("duration", 0.0))
    button = args.get("button", "left")
    try:
        return DesktopAdapter().drag(start, end, duration=duration, button=button), {}
    except DesktopAdapterError as e:
        raise ToolError(str(e))


def ui_type(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    text = args.get("text", "")
    if not isinstance(text, str) or not text:
        raise ToolError("text is required")
    interval = float(args.get("interval", 0.0))
    try:
        return DesktopAdapter().type_text(text, interval=interval), {}
    except DesktopAdapterError as e:
        raise ToolError(str(e))


def ui_hotkey(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    keys = args.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ToolError("keys must be a non-empty list")
    try:
        return DesktopAdapter().hotkey(keys), {}
    except DesktopAdapterError as e:
        raise ToolError(str(e))


def ui_scroll(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    amount = int(args.get("amount", 0))
    try:
        return DesktopAdapter().scroll(amount), {}
    except DesktopAdapterError as e:
        raise ToolError(str(e))


def ui_focus_window(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    title = args.get("title", "")
    if not isinstance(title, str) or not title:
        raise ToolError("title is required")
    try:
        return DesktopAdapter().focus_window(title), {}
    except DesktopAdapterError as e:
        raise ToolError(str(e))


def ui_list_applications(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manager = EmbodimentManager()
    return manager.list_launchable_applications(), {}


def ui_open_application(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    app_id = args.get("app_id", "")
    if not isinstance(app_id, str) or not app_id.strip():
        raise ToolError("app_id is required")
    manager = EmbodimentManager()
    try:
        return manager.launch_desktop_application(app_id), {}
    except DesktopAdapterError as e:
        raise ToolError(str(e))


def embodiment_describe_host(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manager = EmbodimentManager()
    return manager.describe_host(), {}


def embodiment_list_capabilities(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manager = EmbodimentManager()
    return manager.list_capabilities(), {}


def embodiment_list_fallbacks(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manager = EmbodimentManager()
    return manager.list_fallbacks(), {}


def embodiment_get_state(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manager = EmbodimentManager()
    return manager.get_state(), {}


def embodiment_screen_capture(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return ui_screenshot(args, workspace_root)


def embodiment_pointer_click(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return ui_click(args, workspace_root)


def embodiment_keyboard_type(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return ui_type(args, workspace_root)


def embodiment_window_focus(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return ui_focus_window(args, workspace_root)


def embodiment_camera_capture(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    camera_index = int(args.get("camera_index", 0))
    width = args.get("width")
    height = args.get("height")
    if width is not None:
        width = int(width)
    if height is not None:
        height = int(height)
    ts = int(time.time() * 1000)
    workspace_path, abs_path = _resolve_output_workspace_path(
        workspace_root,
        args.get("path"),
        f"workspace:/camera/frame_{ts}.jpg",
    )
    try:
        result = OpenCvCameraAdapter(camera_index=camera_index).capture_frame(
            abs_path,
            width=width,
            height=height,
        )
    except CameraAdapterError as e:
        raise ToolError(str(e))
    return {
        "image_ref": workspace_path,
        "width": result["width"],
        "height": result["height"],
        "camera_index": result["camera_index"],
    }, {}


def embodiment_phone_screen_capture(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    serial = args.get("serial")
    ts = int(time.time() * 1000)
    workspace_path, abs_path = _resolve_output_workspace_path(
        workspace_root,
        args.get("path"),
        f"workspace:/phone/screenshot_{ts}.png",
    )
    try:
        result = AdbPhoneAdapter(serial=serial).capture_screen(abs_path)
    except PhoneAdapterError as e:
        raise ToolError(str(e))
    return {
        "image_ref": workspace_path,
        "width": result["width"],
        "height": result["height"],
        "serial": result.get("serial"),
    }, {}


def embodiment_phone_tap(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    x = args.get("x")
    y = args.get("y")
    if x is None or y is None:
        raise ToolError("x and y are required")
    try:
        return AdbPhoneAdapter(serial=args.get("serial")).tap(int(x), int(y)), {}
    except PhoneAdapterError as e:
        raise ToolError(str(e))


def embodiment_phone_swipe(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    start = args.get("start")
    end = args.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise ToolError("start and end are required objects")
    duration_ms = args.get("duration_ms")
    if duration_ms is not None:
        duration_ms = int(duration_ms)
    try:
        return AdbPhoneAdapter(serial=args.get("serial")).swipe(start, end, duration_ms=duration_ms), {}
    except PhoneAdapterError as e:
        raise ToolError(str(e))


def embodiment_phone_type(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    text = args.get("text", "")
    if not isinstance(text, str) or not text:
        raise ToolError("text is required")
    try:
        return AdbPhoneAdapter(serial=args.get("serial")).type_text(text), {}
    except PhoneAdapterError as e:
        raise ToolError(str(e))


def embodiment_phone_launch_app(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    package = args.get("package", "")
    activity = args.get("activity")
    if not isinstance(package, str) or not package:
        raise ToolError("package is required")
    if activity is not None and not isinstance(activity, str):
        raise ToolError("activity must be a string")
    try:
        return AdbPhoneAdapter(serial=args.get("serial")).launch_app(package, activity=activity), {}
    except PhoneAdapterError as e:
        raise ToolError(str(e))


def embodiment_robot_get_state(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        return RosCliRobotAdapter().get_state(), {}
    except RobotAdapterError as e:
        raise ToolError(str(e))


def embodiment_robot_move_joint(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    joint = args.get("joint", "")
    position = args.get("position")
    velocity = args.get("velocity")
    if not isinstance(joint, str) or not joint:
        raise ToolError("joint is required")
    if position is None:
        raise ToolError("position is required")
    try:
        return RosCliRobotAdapter().move_joint(joint, float(position), velocity=float(velocity) if velocity is not None else None), {}
    except RobotAdapterError as e:
        raise ToolError(str(e))


def embodiment_robot_set_gripper(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state = args.get("state", "")
    width = args.get("width")
    force = args.get("force")
    if not isinstance(state, str) or not state:
        raise ToolError("state is required")
    try:
        return RosCliRobotAdapter().set_gripper(
            state,
            width=float(width) if width is not None else None,
            force=float(force) if force is not None else None,
        ), {}
    except RobotAdapterError as e:
        raise ToolError(str(e))


def tool_registry() -> Dict[str, Dict[str, Any]]:
    registry = {
        "sys.time": {"fn": sys_time, "tier": 0},
        "sys.list_tools": {"fn": sys_list_tools, "tier": 0},
        "math.eval": {"fn": math_eval, "tier": 0},
        "math.sympy": {"fn": math_sympy, "tier": 0},
        "fs.read_file": {"fn": fs_read_file, "tier": 0},
        "fs.write_file": {"fn": fs_write_file, "tier": 1},
        "fs.list_dir": {"fn": fs_list_dir, "tier": 0},
        "fs.stat": {"fn": fs_stat, "tier": 0},
        "fs.find": {"fn": fs_find, "tier": 0},
        "fs.search_text": {"fn": fs_search_text, "tier": 0},
        "fs.delete": {"fn": fs_delete, "tier": 1},
        "fs.move": {"fn": fs_move, "tier": 1},
        "fs.copy": {"fn": fs_copy, "tier": 1},
        "fs.mkdir": {"fn": fs_mkdir, "tier": 1},
        "data.json_query": {"fn": data_json_query, "tier": 0},
        "data.csv_read": {"fn": data_csv_read, "tier": 0},
        "archive.zip_create": {"fn": archive_zip_create, "tier": 1},
        "archive.zip_extract": {"fn": archive_zip_extract, "tier": 1},
        "image.analyse": {"fn": image_analyse, "tier": 0},
        "video.analyse": {"fn": video_analyse, "tier": 0},
        "vision.observe": {"fn": vision_observe, "tier": 0},
        "ui.predict_coords": {"fn": ui_predict_coords, "tier": 0},
        "net.search": {"fn": net_search, "tier": 1},
        "http.request": {"fn": http_request, "tier": 2},
        "google.status": {"fn": google_status, "tier": 0},
        "google.authorize": {"fn": google_authorize, "tier": 1},
        "google.disconnect": {"fn": google_disconnect, "tier": 1},
        "google.gmail.list_messages": {"fn": google_gmail_list_messages, "tier": 1},
        "google.gmail.get_message": {"fn": google_gmail_get_message, "tier": 1},
        "google.gmail.create_draft": {"fn": google_gmail_create_draft, "tier": 2},
        "google.gmail.send_message": {"fn": google_gmail_send_message, "tier": 2},
        "google.calendar.list_events": {"fn": google_calendar_list_events, "tier": 1},
        "google.calendar.create_event": {"fn": google_calendar_create_event, "tier": 2},
        "google.docs.get_document": {"fn": google_docs_get_document, "tier": 1},
        "google.docs.create_document": {"fn": google_docs_create_document, "tier": 2},
        "google.docs.append_text": {"fn": google_docs_append_text, "tier": 2},
        "git.status": {"fn": git_status, "tier": 0},
        "git.diff": {"fn": git_diff, "tier": 0},
        "git.log": {"fn": git_log, "tier": 0},
        "proc.exec": {"fn": proc_exec, "tier": 2},
        "ui.screenshot": {"fn": ui_screenshot, "tier": 2},
        "ui.click": {"fn": ui_click, "tier": 2},
        "ui.move": {"fn": ui_move, "tier": 2},
        "ui.drag": {"fn": ui_drag, "tier": 2},
        "ui.type": {"fn": ui_type, "tier": 2},
        "ui.hotkey": {"fn": ui_hotkey, "tier": 2},
        "ui.scroll": {"fn": ui_scroll, "tier": 2},
        "ui.focus_window": {"fn": ui_focus_window, "tier": 2},
        "ui.list_applications": {"fn": ui_list_applications, "tier": 0},
        "ui.open_application": {"fn": ui_open_application, "tier": 2},
        "embodiment.describe_host": {"fn": embodiment_describe_host, "tier": 0},
        "embodiment.list_capabilities": {"fn": embodiment_list_capabilities, "tier": 0},
        "embodiment.list_fallbacks": {"fn": embodiment_list_fallbacks, "tier": 0},
        "embodiment.get_state": {"fn": embodiment_get_state, "tier": 0},
        "embodiment.screen_capture": {"fn": embodiment_screen_capture, "tier": 2},
        "embodiment.pointer_click": {"fn": embodiment_pointer_click, "tier": 2},
        "embodiment.keyboard_type": {"fn": embodiment_keyboard_type, "tier": 2},
        "embodiment.window_focus": {"fn": embodiment_window_focus, "tier": 2},
        "embodiment.camera_capture": {"fn": embodiment_camera_capture, "tier": 1},
        "embodiment.phone_screen_capture": {"fn": embodiment_phone_screen_capture, "tier": 2},
        "embodiment.phone_tap": {"fn": embodiment_phone_tap, "tier": 2},
        "embodiment.phone_swipe": {"fn": embodiment_phone_swipe, "tier": 2},
        "embodiment.phone_type": {"fn": embodiment_phone_type, "tier": 2},
        "embodiment.phone_launch_app": {"fn": embodiment_phone_launch_app, "tier": 2},
        "embodiment.robot_get_state": {"fn": embodiment_robot_get_state, "tier": 0},
        "embodiment.robot_move_joint": {"fn": embodiment_robot_move_joint, "tier": 2},
        "embodiment.robot_set_gripper": {"fn": embodiment_robot_set_gripper, "tier": 2},
    }
    registry.update(load_generated_registry())
    return merge_runtime_tool_metadata(registry)


def load_generated_registry(reload_module: bool = False) -> Dict[str, Dict[str, Any]]:
    try:
        import tool_runtime.generated_tools as generated_tools
    except ImportError:
        return {}
    if reload_module:
        try:
            generated_tools = importlib.reload(generated_tools)
        except Exception:
            return {}
    registry = getattr(generated_tools, "generated_registry", None)
    if not registry:
        return {}
    try:
        result = registry()
        if not isinstance(result, dict):
            return {}
        normalized: Dict[str, Dict[str, Any]] = {}
        for tool_id, spec in result.items():
            if not isinstance(tool_id, str) or not isinstance(spec, dict):
                continue
            item = dict(spec)
            item["generated"] = True
            normalized[tool_id] = item
        return normalized
    except Exception:
        return {}
