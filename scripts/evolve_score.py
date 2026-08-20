import argparse
import ast
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunOutcome:
    ok: bool
    value: Any
    error_type: str
    error_message: str
    elapsed_ms: float


@dataclass
class SourceResult:
    source: str
    ok: bool
    score: float
    passed: bool
    details: str
    cases_total: int
    cases_used: int
    passed_cases: int
    correctness: float
    perf_ratio: float
    baseline_median_ms: float
    candidate_median_ms: float
    object_ids: List[str]
    failures: List[str]


@dataclass
class TargetAnalysis:
    kind: str
    object_ids: List[str]
    skip_macro: bool
    reason: str
    has_runtime_dependencies: bool
    input_param_count: int


@dataclass
class InjectionResult:
    ok: bool
    changed: bool
    details: str


def _emit(score: float, details: str, passed: bool, extra: Optional[Dict[str, Any]] = None) -> int:
    payload: Dict[str, Any] = {
        "score": max(0.0, min(1.0, float(score))),
        "details": details,
    }
    if extra:
        payload.update(extra)
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    return 0 if passed else 1


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _json_compatible(value: Any, depth: int = 0) -> Tuple[Any, bool]:
    if depth > 8:
        return None, False
    if value is None or isinstance(value, (bool, int, float, str)):
        return value, True
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            converted, ok = _json_compatible(item, depth + 1)
            if not ok:
                return None, False
            out.append(converted)
        return out, True
    if isinstance(value, tuple):
        out: List[Any] = []
        for item in value:
            converted, ok = _json_compatible(item, depth + 1)
            if not ok:
                return None, False
            out.append(converted)
        return {"__tuple__": out}, True
    if isinstance(value, dict):
        out_dict: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return None, False
            converted, ok = _json_compatible(item, depth + 1)
            if not ok:
                return None, False
            out_dict[key] = converted
        return out_dict, True
    return None, False


def _safe_repr(value: Any, max_chars: int = 200) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _values_match(a: Any, b: Any) -> bool:
    left, left_ok = _json_compatible(a)
    right, right_ok = _json_compatible(b)
    if left_ok and right_ok:
        return left == right
    if type(a) is not type(b):
        return False
    return _safe_repr(a, max_chars=2000) == _safe_repr(b, max_chars=2000)


def _restore_replayable(value: Any) -> Any:
    if isinstance(value, dict) and "__tuple__" in value and isinstance(value["__tuple__"], list):
        return tuple(_restore_replayable(item) for item in value["__tuple__"])
    if isinstance(value, list):
        return [_restore_replayable(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_replayable(item) for key, item in value.items()}
    return value


def _restore_call_payload(payload: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    args = payload.get("args", [])
    kwargs = payload.get("kwargs", {})
    if not isinstance(args, list):
        args = []
    if not isinstance(kwargs, dict):
        kwargs = {}
    restored_args = [_restore_replayable(item) for item in args]
    restored_kwargs = {str(key): _restore_replayable(val) for key, val in kwargs.items()}
    return restored_args, restored_kwargs


def _module_from_target_rel(target_rel: str) -> str:
    normalized = _normalize_path(target_rel).lstrip("./")
    if normalized.startswith("src/") and normalized.endswith(".py"):
        module_rel = normalized[len("src/") : -3]
        return module_rel.replace("/", ".")
    return ""


def _build_parent_map(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    parents: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _line_span(node: ast.AST) -> Tuple[int, int]:
    start = int(getattr(node, "lineno", 0) or 0)
    end = int(getattr(node, "end_lineno", start) or start)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            starts = [int(getattr(dec, "lineno", start) or start) for dec in decorators]
            if starts:
                start = min(start, min(starts))
    return start, end


def _callable_input_param_count(node: ast.AST, is_method: bool) -> int:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0
    args = node.args
    count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    if args.vararg is not None:
        count += 1
    if args.kwarg is not None:
        count += 1
    if is_method and args.args:
        first = args.args[0].arg
        if first in {"self", "cls"}:
            count -= 1
    return max(0, count)


def _has_runtime_dependencies(node: ast.AST) -> bool:
    marker_names = {
        "os",
        "time",
        "random",
        "uuid",
        "subprocess",
        "socket",
        "requests",
        "pathlib",
        "json",
        "open",
        "load_settings",
        "load_system_entity",
    }
    marker_attrs = {
        "os.path",
        "time.time",
        "time.perf_counter",
        "random.random",
        "uuid.uuid4",
        "subprocess.run",
        "socket.gethostbyname",
    }
    for child in ast.walk(node):
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            return True
        if isinstance(child, ast.Name) and child.id in marker_names:
            return True
        if isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name):
                root = child.value.id
                candidate = f"{root}.{child.attr}"
                if root in marker_names or candidate in marker_attrs:
                    return True
                if root == "os" and child.attr == "path":
                    return True
            if isinstance(child.value, ast.Attribute) and isinstance(child.value.value, ast.Name):
                root = child.value.value.id
                candidate = f"{root}.{child.value.attr}"
                if root in marker_names or candidate in marker_attrs:
                    return True
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id in marker_names:
                return True
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                root = func.value.id
                if root in marker_names:
                    return True
    return False


def _find_smallest_enclosing_node(
    tree: ast.AST,
    start_line: int,
    end_line: int,
) -> Optional[ast.AST]:
    if start_line <= 0:
        return None
    if end_line <= 0:
        end_line = start_line
    candidates: List[Tuple[int, int, ast.AST]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        node_start, node_end = _line_span(node)
        if node_start <= start_line and node_end >= end_line:
            candidates.append((node_start, node_end, node))
    if not candidates:
        fallback: List[Tuple[int, int, ast.AST]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            node_start, node_end = _line_span(node)
            if abs(node_start - start_line) <= 2:
                fallback.append((node_start, node_end, node))
        if not fallback:
            return None
        fallback.sort(key=lambda item: (abs(item[0] - start_line), item[1] - item[0], item[0]))
        return fallback[0][2]
    candidates.sort(key=lambda item: (item[1] - item[0], item[0]))
    return candidates[0][2]


def _qual_parts_for_node(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> List[str]:
    parts: List[str] = []
    current: Optional[ast.AST] = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(current.name)
        current = parents.get(current)
    parts.reverse()
    return parts


def _object_id_for_node(module_name: str, node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> str:
    parts = _qual_parts_for_node(node, parents)
    if not parts:
        return ""
    return module_name + "." + ".".join(parts)


def _is_traceable_method_name(name: str) -> bool:
    if name in {"__init__", "__call__"}:
        return True
    if name.startswith("__") and name.endswith("__"):
        return False
    if name.startswith("_"):
        return False
    return True


def _collect_class_callable_object_ids(
    module_name: str,
    class_node: ast.ClassDef,
    parents: Dict[ast.AST, ast.AST],
) -> Tuple[List[str], int, bool]:
    object_ids: List[str] = []
    best_input_count = 0
    has_runtime = False
    for item in class_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_traceable_method_name(item.name):
            continue
        obj_id = _object_id_for_node(module_name, item, parents)
        if not obj_id:
            continue
        object_ids.append(obj_id)
        input_count = _callable_input_param_count(item, is_method=True)
        best_input_count = max(best_input_count, input_count)
        if _has_runtime_dependencies(item):
            has_runtime = True
    deduped = sorted(set(object_ids))
    return deduped, best_input_count, has_runtime


def _resolve_target_analysis(
    repo_root: str,
    target_rel: str,
    start_line: int,
    end_line: int,
    explicit_object_ids: List[str],
    macro_policy: str,
    auto_skip_no_input: bool,
) -> TargetAnalysis:
    if macro_policy == "never":
        return TargetAnalysis(
            kind="policy",
            object_ids=list(explicit_object_ids),
            skip_macro=True,
            reason="macro_policy_never",
            has_runtime_dependencies=False,
            input_param_count=0,
        )

    if explicit_object_ids:
        return TargetAnalysis(
            kind="explicit",
            object_ids=list(explicit_object_ids),
            skip_macro=False,
            reason="explicit_object_ids",
            has_runtime_dependencies=False,
            input_param_count=0,
        )

    module_name = _module_from_target_rel(target_rel)
    if not module_name:
        return TargetAnalysis(
            kind="module",
            object_ids=[],
            skip_macro=(macro_policy == "auto"),
            reason="target_not_in_src_module",
            has_runtime_dependencies=False,
            input_param_count=0,
        )

    target_abs = os.path.join(repo_root, _normalize_path(target_rel))
    if not os.path.exists(target_abs):
        return TargetAnalysis(
            kind="missing",
            object_ids=[],
            skip_macro=True,
            reason="target_file_missing",
            has_runtime_dependencies=False,
            input_param_count=0,
        )
    try:
        with open(target_abs, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return TargetAnalysis(
            kind="parse_error",
            object_ids=[],
            skip_macro=(macro_policy == "auto"),
            reason="target_parse_failed",
            has_runtime_dependencies=False,
            input_param_count=0,
        )

    parents = _build_parent_map(tree)
    node = _find_smallest_enclosing_node(tree, start_line=start_line, end_line=end_line)
    if node is None:
        return TargetAnalysis(
            kind="module",
            object_ids=[],
            skip_macro=(macro_policy == "auto"),
            reason="module_level_target",
            has_runtime_dependencies=False,
            input_param_count=0,
        )

    if isinstance(node, ast.ClassDef):
        class_ids, input_count, has_runtime = _collect_class_callable_object_ids(
            module_name=module_name, class_node=node, parents=parents
        )
        if not class_ids:
            return TargetAnalysis(
                kind="class",
                object_ids=[],
                skip_macro=(macro_policy == "auto"),
                reason="class_has_no_traceable_methods",
                has_runtime_dependencies=has_runtime,
                input_param_count=input_count,
            )
        if macro_policy == "auto" and auto_skip_no_input and input_count <= 0 and not has_runtime:
            return TargetAnalysis(
                kind="class",
                object_ids=class_ids,
                skip_macro=True,
                reason="class_methods_no_input_and_no_runtime_dependencies",
                has_runtime_dependencies=has_runtime,
                input_param_count=input_count,
            )
        return TargetAnalysis(
            kind="class",
            object_ids=class_ids,
            skip_macro=False,
            reason="class_traceable_methods_resolved",
            has_runtime_dependencies=has_runtime,
            input_param_count=input_count,
        )

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        object_id = _object_id_for_node(module_name, node, parents)
        is_method = isinstance(parents.get(node), ast.ClassDef)
        input_count = _callable_input_param_count(node, is_method=is_method)
        has_runtime = _has_runtime_dependencies(node)
        if macro_policy == "auto" and auto_skip_no_input and input_count <= 0 and not has_runtime:
            return TargetAnalysis(
                kind="method" if is_method else "function",
                object_ids=[object_id] if object_id else [],
                skip_macro=True,
                reason="no_input_flow_and_no_runtime_dependencies",
                has_runtime_dependencies=has_runtime,
                input_param_count=input_count,
            )
        return TargetAnalysis(
            kind="method" if is_method else "function",
            object_ids=[object_id] if object_id else [],
            skip_macro=False,
            reason="callable_resolved",
            has_runtime_dependencies=has_runtime,
            input_param_count=input_count,
        )

    return TargetAnalysis(
        kind="unknown",
        object_ids=[],
        skip_macro=(macro_policy == "auto"),
        reason="unsupported_target_node",
        has_runtime_dependencies=False,
        input_param_count=0,
    )


def _decorator_matches(decorator: ast.AST, expected_name: str) -> bool:
    if isinstance(decorator, ast.Name):
        return decorator.id == expected_name
    if isinstance(decorator, ast.Attribute):
        return decorator.attr == expected_name
    if isinstance(decorator, ast.Call):
        return _decorator_matches(decorator.func, expected_name)
    return False


def _find_node_by_object_id(
    tree: ast.AST,
    parents: Dict[ast.AST, ast.AST],
    module_name: str,
    object_id: str,
) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        node_object_id = _object_id_for_node(module_name, node, parents)
        if node_object_id == object_id:
            return node
    return None


def _insert_line(lines: List[str], line_no: int, text: str) -> None:
    idx = max(0, min(len(lines), line_no - 1))
    lines.insert(idx, text)


def _ensure_trace_import(
    lines: List[str],
    tree: ast.AST,
    needed_names: List[str],
) -> bool:
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("from common.evolve_trace import"):
            continue
        before_comment, comment_sep, comment = line.partition("#")
        body = before_comment.strip()
        parts = body.split("import", 1)
        if len(parts) != 2:
            continue
        imported = [part.strip() for part in parts[1].split(",") if part.strip()]
        changed = False
        for name in needed_names:
            if name not in imported:
                imported.append(name)
                changed = True
        if not changed:
            return False
        new_line = f"from common.evolve_trace import {', '.join(imported)}"
        if comment_sep:
            new_line += f" #{comment.strip()}"
        lines[idx] = new_line + "\n"
        return True

    insert_line = 1
    if tree.body:
        first = tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            insert_line = int(getattr(first, "end_lineno", first.lineno) or first.lineno) + 1
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_line = max(insert_line, int(getattr(node, "end_lineno", node.lineno) or node.lineno) + 1)

    import_line = f"from common.evolve_trace import {', '.join(needed_names)}\n"
    _insert_line(lines, insert_line, import_line)
    return True


def _ensure_trace_injection(
    repo_root: str,
    target_rel: str,
    target_start_line: int,
    target_end_line: int,
    target_analysis: TargetAnalysis,
    explicit_object_ids: List[str],
) -> InjectionResult:
    if target_analysis.kind in {"module", "missing", "parse_error", "unknown", "policy"}:
        return InjectionResult(ok=True, changed=False, details=f"injection_skipped:{target_analysis.kind}")

    target_abs = os.path.join(repo_root, _normalize_path(target_rel))
    if not os.path.exists(target_abs):
        return InjectionResult(ok=False, changed=False, details=f"target_file_missing:{target_abs}")

    try:
        with open(target_abs, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception as e:
        return InjectionResult(ok=False, changed=False, details=f"target_parse_failed:{e}")

    parents = _build_parent_map(tree)
    module_name = _module_from_target_rel(target_rel)
    if not module_name:
        return InjectionResult(ok=False, changed=False, details="module_name_resolution_failed")

    node: Optional[ast.AST] = None
    decorator_name = ""
    decorator_payload = ""
    needed_imports: List[str] = []

    if target_analysis.kind in {"function", "method"}:
        node = _find_smallest_enclosing_node(tree, start_line=target_start_line, end_line=target_end_line)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return InjectionResult(ok=False, changed=False, details="callable_node_not_found_for_injection")
        object_id = _object_id_for_node(module_name, node, parents)
        if not object_id and explicit_object_ids:
            object_id = explicit_object_ids[0]
        if not object_id:
            return InjectionResult(ok=False, changed=False, details="object_id_resolution_failed")
        decorator_name = "trace_calls"
        decorator_payload = f'@trace_calls("{object_id}")\n'
        needed_imports = ["trace_calls"]
    elif target_analysis.kind == "class":
        node = _find_smallest_enclosing_node(tree, start_line=target_start_line, end_line=target_end_line)
        if not isinstance(node, ast.ClassDef):
            if explicit_object_ids:
                prefix = ".".join(explicit_object_ids[0].split(".")[:-1])
                node = _find_node_by_object_id(tree, parents, module_name, prefix)
            if not isinstance(node, ast.ClassDef):
                return InjectionResult(ok=False, changed=False, details="class_node_not_found_for_injection")
        class_prefix = _object_id_for_node(module_name, node, parents)
        if not class_prefix:
            return InjectionResult(ok=False, changed=False, details="class_object_id_resolution_failed")
        decorator_name = "trace_class_calls"
        decorator_payload = f'@trace_class_calls("{class_prefix}")\n'
        needed_imports = ["trace_class_calls"]
    else:
        return InjectionResult(ok=True, changed=False, details=f"injection_not_required:{target_analysis.kind}")

    if node is None:
        return InjectionResult(ok=False, changed=False, details="node_resolution_failed")

    lines = source.splitlines(keepends=True)
    changed = False

    has_decorator = any(_decorator_matches(dec, decorator_name) for dec in getattr(node, "decorator_list", []))
    if not has_decorator:
        insert_line = int(getattr(node, "lineno", 1) or 1)
        decorator_list = getattr(node, "decorator_list", [])
        if decorator_list:
            insert_line = min(int(getattr(dec, "lineno", insert_line) or insert_line) for dec in decorator_list)
        indent = " " * int(getattr(node, "col_offset", 0) or 0)
        _insert_line(lines, insert_line, indent + decorator_payload)
        changed = True

    import_changed = _ensure_trace_import(lines, tree, needed_imports)
    changed = changed or import_changed
    if not changed:
        return InjectionResult(ok=True, changed=False, details="trace_injection_already_present")

    try:
        with open(target_abs, "w", encoding="utf-8") as f:
            f.write("".join(lines))
    except Exception as e:
        return InjectionResult(ok=False, changed=False, details=f"trace_injection_write_failed:{e}")
    return InjectionResult(ok=True, changed=True, details="trace_injection_applied")


def _wait_for_trace_cases(
    trace_path: str,
    target_rel: str,
    object_ids: List[str],
    min_cases: int,
    collect_window_s: int,
    poll_interval_s: float,
) -> Tuple[bool, int]:
    required = max(1, int(min_cases))
    deadline = time.time() + max(0, int(collect_window_s))
    while True:
        cases, _ = _read_trace_cases(
            trace_path=trace_path,
            target_rel=target_rel,
            object_ids=object_ids,
            limit=max(required, 10000),
        )
        if len(cases) >= required:
            return True, len(cases)
        if time.time() >= deadline:
            return False, len(cases)
        time.sleep(max(0.1, float(poll_interval_s)))


def _resolve_data_path(path: str, repo_root: str, run_dir: str, prefer_repo: bool = False) -> str:
    if os.path.isabs(path):
        return path
    if prefer_repo:
        return os.path.join(repo_root, path)
    candidate = os.path.join(run_dir, path)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(repo_root, path)


def _normalize_case(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    object_id = str(entry.get("object_id", "") or "").strip()
    if not object_id:
        return None
    call = entry.get("call", {})
    if not isinstance(call, dict):
        return None
    args = call.get("args", [])
    kwargs = call.get("kwargs", {})
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        return None
    _, args_ok = _json_compatible(args)
    _, kwargs_ok = _json_compatible(kwargs)
    if not args_ok or not kwargs_ok:
        return None
    return {
        "object_id": object_id,
        "call": {"args": args, "kwargs": kwargs},
    }


def _dedupe_cases(cases: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for case in cases:
        key = json.dumps(case, ensure_ascii=True, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _read_trace_cases(
    trace_path: str,
    target_rel: str,
    object_ids: List[str],
    limit: int,
) -> Tuple[List[Dict[str, Any]], int]:
    if not os.path.exists(trace_path):
        return [], 0
    total_events = 0
    cases: List[Dict[str, Any]] = []
    target_norm = _normalize_path(target_rel).lstrip("./")
    allow = set(object_ids)
    with open(trace_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("kind") != "trace_case" or not bool(event.get("replayable", False)):
                continue
            total_events += 1
            object_id = str(event.get("object_id", "") or "")
            source_path = _normalize_path(str(event.get("source_path", "") or "")).lstrip("./")
            if allow:
                if object_id not in allow:
                    continue
            elif target_norm:
                if source_path != target_norm and not source_path.endswith("/" + target_norm):
                    continue
            payload = event.get("call")
            if not isinstance(payload, dict):
                continue
            case = _normalize_case({"object_id": object_id, "call": payload})
            if not case:
                continue
            cases.append(case)
    return _dedupe_cases(cases, limit=limit), total_events


def _read_llm_cases(
    scenarios_path: str,
    object_ids: List[str],
    limit: int,
) -> Tuple[List[Dict[str, Any]], int]:
    if not os.path.exists(scenarios_path):
        return [], 0
    try:
        with open(scenarios_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], 0
    entries: Any = []
    if isinstance(data, dict):
        entries = data.get("scenarios")
        if entries is None:
            entries = data.get("cases", [])
    elif isinstance(data, list):
        entries = data
    if not isinstance(entries, list):
        return [], 0
    allow = set(object_ids)
    cases: List[Dict[str, Any]] = []
    for entry in entries:
        case = _normalize_case(entry)
        if not case:
            continue
        if allow and case["object_id"] not in allow:
            continue
        cases.append(case)
    return _dedupe_cases(cases, limit=limit), len(entries)


def _purge_repo_modules(repo_roots: List[str]) -> None:
    normalized: List[str] = []
    for root in repo_roots:
        normalized.append(os.path.abspath(os.path.join(root, "src")))
    for module_name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        module_abs = os.path.abspath(module_file)
        for src_root in normalized:
            if module_abs == src_root or module_abs.startswith(src_root + os.sep):
                sys.modules.pop(module_name, None)
                break


def _resolve_object(module_root: str, object_id: str, purge_roots: List[str]) -> Tuple[Optional[Any], str]:
    parts = [part for part in object_id.split(".") if part]
    if len(parts) < 2:
        return None, "invalid_object_id"

    module_path = ""
    attr_parts: List[str] = []
    for split_idx in range(len(parts) - 1, 0, -1):
        module_candidate = ".".join(parts[:split_idx])
        module_file = os.path.join(module_root, "src", *module_candidate.split(".")) + ".py"
        if os.path.exists(module_file):
            module_path = module_file
            attr_parts = parts[split_idx:]
            break
    if not module_path or not attr_parts:
        return None, f"module_not_found_for:{object_id}"

    module_name = "_evolve_macro_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        return None, f"spec_failed:{object_id}"
    module = importlib.util.module_from_spec(spec)

    src_path = os.path.join(module_root, "src")
    path_added = False
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        path_added = True
    _purge_repo_modules(purge_roots)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"import_failed:{object_id}:{e}"
    finally:
        if path_added and src_path in sys.path:
            sys.path.remove(src_path)

    current: Any = module
    for attr in attr_parts:
        if not hasattr(current, attr):
            return None, f"attr_missing:{object_id}:{attr}"
        current = getattr(current, attr)
    if not callable(current):
        return None, f"not_callable:{object_id}"
    return current, ""


def _run_callable(func: Any, args: List[Any], kwargs: Dict[str, Any]) -> RunOutcome:
    started = time.perf_counter()
    try:
        value = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RunOutcome(ok=True, value=value, error_type="", error_message="", elapsed_ms=elapsed_ms)
    except Exception as e:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RunOutcome(
            ok=False,
            value=None,
            error_type=type(e).__name__,
            error_message=_safe_repr(e, max_chars=500),
            elapsed_ms=elapsed_ms,
        )


def _score_components(correctness: float, perf_ratio: float, correctness_weight: float, perf_weight: float) -> float:
    perf_component = max(0.0, min(1.0, perf_ratio))
    cw = max(0.0, correctness_weight)
    pw = max(0.0, perf_weight)
    total = cw + pw
    if total <= 0:
        return correctness
    return ((cw * correctness) + (pw * perf_component)) / total


def _evaluate_cases(
    source: str,
    cases: List[Dict[str, Any]],
    repo_root: str,
    run_dir: str,
    min_cases: int,
    min_correctness: float,
    min_perf_ratio: float,
    strict_output: bool,
    correctness_weight: float,
    perf_weight: float,
) -> SourceResult:
    object_cache: Dict[Tuple[str, str], Any] = {}
    failures: List[str] = []
    baseline_times: List[float] = []
    candidate_times: List[float] = []
    passed_cases = 0
    min_cases = max(1, int(min_cases))

    if len(cases) < min_cases:
        return SourceResult(
            source=source,
            ok=False,
            score=0.0,
            passed=False,
            details=f"insufficient_cases:{len(cases)}<{min_cases}",
            cases_total=len(cases),
            cases_used=0,
            passed_cases=0,
            correctness=0.0,
            perf_ratio=0.0,
            baseline_median_ms=0.0,
            candidate_median_ms=0.0,
            object_ids=[],
            failures=[],
        )

    object_ids_seen = set()
    for idx, case in enumerate(cases, start=1):
        object_id = str(case.get("object_id", "") or "")
        call = case.get("call", {})
        if not isinstance(call, dict):
            failures.append(f"case{idx}:invalid_payload")
            continue
        call_args, call_kwargs = _restore_call_payload(call)
        object_ids_seen.add(object_id)

        baseline_key = (repo_root, object_id)
        candidate_key = (run_dir, object_id)

        baseline_callable = object_cache.get(baseline_key)
        if baseline_callable is None:
            baseline_callable, baseline_err = _resolve_object(
                repo_root, object_id, purge_roots=[repo_root, run_dir]
            )
            if baseline_callable is None:
                failures.append(f"case{idx}:baseline_resolve_failed:{baseline_err}")
                continue
            object_cache[baseline_key] = baseline_callable

        candidate_callable = object_cache.get(candidate_key)
        if candidate_callable is None:
            candidate_callable, candidate_err = _resolve_object(
                run_dir, object_id, purge_roots=[repo_root, run_dir]
            )
            if candidate_callable is None:
                failures.append(f"case{idx}:candidate_resolve_failed:{candidate_err}")
                continue
            object_cache[candidate_key] = candidate_callable

        baseline = _run_callable(baseline_callable, call_args, call_kwargs)
        candidate = _run_callable(candidate_callable, call_args, call_kwargs)
        baseline_times.append(baseline.elapsed_ms)
        candidate_times.append(candidate.elapsed_ms)

        case_pass = baseline.ok == candidate.ok
        if case_pass:
            if baseline.ok and candidate.ok and strict_output:
                case_pass = _values_match(baseline.value, candidate.value)
            elif not baseline.ok and not candidate.ok:
                case_pass = baseline.error_type == candidate.error_type

        if case_pass:
            passed_cases += 1
        else:
            if baseline.ok and candidate.ok:
                failures.append(f"case{idx}:output_mismatch")
            elif baseline.ok and not candidate.ok:
                failures.append(f"case{idx}:candidate_failed:{candidate.error_type}")
            elif not baseline.ok and candidate.ok:
                failures.append(f"case{idx}:candidate_succeeded_baseline_failed")
            else:
                failures.append(f"case{idx}:exception_mismatch:{baseline.error_type}!={candidate.error_type}")

    used_cases = len(baseline_times)
    if used_cases < min_cases:
        return SourceResult(
            source=source,
            ok=False,
            score=0.0,
            passed=False,
            details=f"insufficient_executed_cases:{used_cases}<{min_cases}",
            cases_total=len(cases),
            cases_used=used_cases,
            passed_cases=passed_cases,
            correctness=0.0,
            perf_ratio=0.0,
            baseline_median_ms=0.0,
            candidate_median_ms=0.0,
            object_ids=sorted(object_ids_seen),
            failures=failures[:10],
        )

    baseline_median_ms = statistics.median(baseline_times)
    candidate_median_ms = statistics.median(candidate_times)
    perf_ratio = 0.0 if candidate_median_ms <= 0 else baseline_median_ms / candidate_median_ms
    correctness = passed_cases / max(1, used_cases)
    score = _score_components(correctness, perf_ratio, correctness_weight, perf_weight)
    passed = correctness >= min_correctness and perf_ratio >= min_perf_ratio

    details = (
        f"cases={used_cases},correctness={correctness:.4f},"
        f"baseline_median_ms={baseline_median_ms:.4f},"
        f"candidate_median_ms={candidate_median_ms:.4f},"
        f"perf_ratio={perf_ratio:.4f}"
    )
    return SourceResult(
        source=source,
        ok=True,
        score=score,
        passed=passed,
        details=details,
        cases_total=len(cases),
        cases_used=used_cases,
        passed_cases=passed_cases,
        correctness=correctness,
        perf_ratio=perf_ratio,
        baseline_median_ms=baseline_median_ms,
        candidate_median_ms=candidate_median_ms,
        object_ids=sorted(object_ids_seen),
        failures=failures[:10],
    )


def _source_to_dict(result: SourceResult) -> Dict[str, Any]:
    return {
        "source": result.source,
        "ok": result.ok,
        "score": round(result.score, 6),
        "passed": result.passed,
        "details": result.details,
        "cases_total": result.cases_total,
        "cases_used": result.cases_used,
        "passed_cases": result.passed_cases,
        "correctness": round(result.correctness, 6),
        "perf_ratio": round(result.perf_ratio, 6),
        "baseline_median_ms": round(result.baseline_median_ms, 6),
        "candidate_median_ms": round(result.candidate_median_ms, 6),
        "object_ids": result.object_ids,
        "failures": result.failures,
    }


def _autogenerate_llm_scenarios(
    repo_root: str,
    target_rel: str,
    target_start_line: int,
    target_end_line: int,
    object_ids: List[str],
    output_path: str,
    max_cases: int,
    provider: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
    enable_thinking: bool,
    timeout_s: int,
) -> Optional[str]:
    script_path = os.path.join(repo_root, "scripts", "evolve_generate_macro_scenarios.py")
    if not os.path.exists(script_path):
        return f"generator_not_found:{script_path}"
    command = [
        sys.executable,
        script_path,
        "--target",
        target_rel,
        "--target-start-line",
        str(target_start_line),
        "--target-end-line",
        str(target_end_line),
        "--repo-root",
        repo_root,
        "--max-cases",
        str(max_cases),
        "--output",
        output_path,
    ]
    if object_ids:
        command.extend(["--object-ids", ",".join(object_ids)])
    if provider:
        command.extend(["--provider", provider])
    if model:
        command.extend(["--model", model])
    command.extend(["--temperature", str(temperature)])
    command.extend(["--max-new-tokens", str(max_new_tokens)])
    if enable_thinking:
        command.append("--enable-thinking")

    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_s)),
        )
    except subprocess.TimeoutExpired:
        return "generator_timeout"
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}"
        return f"generator_failed:{detail[-500:]}"
    return None


def _neutral_or_fail(policy: str, reason: str, extra: Dict[str, Any]) -> int:
    if policy == "neutral":
        payload = dict(extra)
        payload["status"] = "neutral_fallback"
        payload["reason"] = reason
        return _emit(0.5, f"neutral_fallback:{reason}", True, extra=payload)
    payload = dict(extra)
    payload["status"] = "source_missing"
    payload["reason"] = reason
    return _emit(0.0, reason, False, extra=payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scenario scorer with macro/micro/hybrid modes.")
    parser.add_argument("--mode", default="hybrid", choices=["macro", "micro", "hybrid", "trace", "llm"])
    parser.add_argument("--macro-policy", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--auto-skip-no-input", default="true", choices=["true", "false"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--inject-tracing-if-missing", default="true", choices=["true", "false"])
    parser.add_argument("--target", required=True, help="Target file path relative to repo root.")
    parser.add_argument("--target-start-line", type=int, default=0)
    parser.add_argument("--target-end-line", type=int, default=0)
    parser.add_argument("--repo-root", required=True, help="Baseline repository root.")
    parser.add_argument("--run-dir", default="", help="Candidate repository root. Default: cwd.")
    parser.add_argument("--object-ids", default="", help="Comma-separated callable ids.")
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--min-correctness", type=float, default=1.0)
    parser.add_argument("--min-perf-ratio", type=float, default=0.95)
    parser.add_argument("--strict-output", action="store_true")
    parser.add_argument("--correctness-weight", type=float, default=0.7)
    parser.add_argument("--perf-weight", type=float, default=0.3)
    parser.add_argument("--missing-source-policy", choices=["fail", "neutral"], default="neutral")

    parser.add_argument("--trace-file", default="evolve/scenarios/traces/current.jsonl")
    parser.add_argument("--trace-sample-limit", type=int, default=100)
    parser.add_argument("--trace-collect-window-s", type=int, default=0)
    parser.add_argument("--trace-poll-interval-s", type=float, default=5.0)

    parser.add_argument("--llm-scenarios-file", dest="llm_scenarios_file", default="evolve/scenarios/llm/current.json")
    parser.add_argument("--micro-scenarios-file", dest="llm_scenarios_file")
    parser.add_argument("--llm-sample-limit", dest="llm_sample_limit", type=int, default=100)
    parser.add_argument("--micro-sample-limit", dest="llm_sample_limit", type=int)
    parser.add_argument("--llm-autogenerate", dest="llm_autogenerate", action="store_true")
    parser.add_argument("--micro-autogenerate", dest="llm_autogenerate", action="store_true")
    parser.add_argument("--llm-force-regenerate", dest="llm_force_regenerate", action="store_true")
    parser.add_argument("--micro-force-regenerate", dest="llm_force_regenerate", action="store_true")
    parser.add_argument("--llm-max-cases", dest="llm_max_cases", type=int, default=8)
    parser.add_argument("--micro-max-cases", dest="llm_max_cases", type=int)
    parser.add_argument("--llm-provider", dest="llm_provider", default="auto", choices=["auto", "ollama", "hf"])
    parser.add_argument("--micro-provider", dest="llm_provider", choices=["auto", "ollama", "hf"])
    parser.add_argument("--llm-model", dest="llm_model", default="")
    parser.add_argument("--micro-model", dest="llm_model")
    parser.add_argument("--llm-temperature", dest="llm_temperature", type=float, default=0.1)
    parser.add_argument("--micro-temperature", dest="llm_temperature", type=float)
    parser.add_argument("--llm-max-new-tokens", dest="llm_max_new_tokens", type=int, default=1000)
    parser.add_argument("--micro-max-new-tokens", dest="llm_max_new_tokens", type=int)
    parser.add_argument("--llm-enable-thinking", dest="llm_enable_thinking", action="store_true")
    parser.add_argument("--micro-enable-thinking", dest="llm_enable_thinking", action="store_true")
    parser.add_argument("--llm-timeout-s", dest="llm_timeout_s", type=int, default=180)
    parser.add_argument("--micro-timeout-s", dest="llm_timeout_s", type=int)

    parser.add_argument("--macro-weight", type=float, default=0.5)
    parser.add_argument("--micro-weight", type=float, default=0.5)
    parser.add_argument("--trace-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--llm-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hybrid-require-both", action="store_true")
    args = parser.parse_args()

    mode = str(args.mode).lower().strip()
    if mode == "trace":
        mode = "macro"
    elif mode == "llm":
        mode = "micro"
    if args.trace_weight is not None:
        args.macro_weight = args.trace_weight
    if args.llm_weight is not None:
        args.micro_weight = args.llm_weight

    repo_root = os.path.abspath(args.repo_root)
    run_dir = os.path.abspath(args.run_dir or os.getcwd())
    target_rel = _normalize_path(args.target)
    auto_skip_no_input = str(args.auto_skip_no_input).lower() == "true"
    explicit_object_ids = [part.strip() for part in args.object_ids.split(",") if part.strip()]
    target_analysis = _resolve_target_analysis(
        repo_root=repo_root,
        target_rel=target_rel,
        start_line=int(args.target_start_line),
        end_line=int(args.target_end_line),
        explicit_object_ids=explicit_object_ids,
        macro_policy=str(args.macro_policy),
        auto_skip_no_input=auto_skip_no_input,
    )
    object_ids = list(target_analysis.object_ids)

    if not object_ids and explicit_object_ids:
        object_ids = list(explicit_object_ids)

    analysis_payload = {
        "kind": target_analysis.kind,
        "skip_macro": target_analysis.skip_macro,
        "reason": target_analysis.reason,
        "object_ids": target_analysis.object_ids,
        "has_runtime_dependencies": target_analysis.has_runtime_dependencies,
        "input_param_count": target_analysis.input_param_count,
    }

    if target_analysis.skip_macro:
        return _emit(
            0.5,
            f"macro_skipped:{target_analysis.reason}",
            True,
            extra={
                "mode": mode,
                "macro_policy": args.macro_policy,
                "object_ids": object_ids,
                "target_analysis": analysis_payload,
                "status": "macro_skipped",
                "reason": target_analysis.reason,
            },
        )

    if args.prepare_only:
        if mode not in {"macro", "hybrid"}:
            return _emit(
                1.0,
                "prepare_not_required_for_mode",
                True,
                extra={
                    "mode": mode,
                    "macro_policy": args.macro_policy,
                    "object_ids": object_ids,
                    "target_analysis": analysis_payload,
                    "status": "prepare_skipped",
                },
            )

        inject_enabled = str(args.inject_tracing_if_missing).lower() == "true"
        injection_result = InjectionResult(ok=True, changed=False, details="trace_injection_disabled")
        if inject_enabled:
            injection_result = _ensure_trace_injection(
                repo_root=repo_root,
                target_rel=target_rel,
                target_start_line=int(args.target_start_line),
                target_end_line=int(args.target_end_line),
                target_analysis=target_analysis,
                explicit_object_ids=explicit_object_ids,
            )
            if not injection_result.ok:
                return _neutral_or_fail(
                    args.missing_source_policy,
                    f"trace_injection_failed:{injection_result.details}",
                    {
                        "mode": mode,
                        "macro_policy": args.macro_policy,
                        "object_ids": object_ids,
                        "target_analysis": analysis_payload,
                        "injection": {
                            "ok": injection_result.ok,
                            "changed": injection_result.changed,
                            "details": injection_result.details,
                        },
                    },
                )

        trace_path = _resolve_data_path(args.trace_file, repo_root=repo_root, run_dir=run_dir)
        collect_window_s = max(0, int(args.trace_collect_window_s))
        min_cases_required = max(1, int(args.min_cases))
        if collect_window_s > 0:
            ready, seen_cases = _wait_for_trace_cases(
                trace_path=trace_path,
                target_rel=target_rel,
                object_ids=object_ids,
                min_cases=min_cases_required,
                collect_window_s=collect_window_s,
                poll_interval_s=float(args.trace_poll_interval_s),
            )
            if not ready:
                return _neutral_or_fail(
                    args.missing_source_policy,
                    f"trace_collection_insufficient:{seen_cases}<{min_cases_required}",
                    {
                        "mode": mode,
                        "macro_policy": args.macro_policy,
                        "object_ids": object_ids,
                        "target_analysis": analysis_payload,
                        "trace_file": trace_path,
                        "required_cases": min_cases_required,
                        "seen_cases": seen_cases,
                        "injection": {
                            "ok": injection_result.ok,
                            "changed": injection_result.changed,
                            "details": injection_result.details,
                        },
                    },
                )

        return _emit(
            1.0,
            "prepare_ok",
            True,
            extra={
                "mode": mode,
                "macro_policy": args.macro_policy,
                "object_ids": object_ids,
                "target_analysis": analysis_payload,
                "trace_file": trace_path,
                "required_cases": min_cases_required,
                "collect_window_s": collect_window_s,
                "injection": {
                    "ok": injection_result.ok,
                    "changed": injection_result.changed,
                    "details": injection_result.details,
                },
                "status": "prepare_ok",
            },
        )

    source_results: Dict[str, SourceResult] = {}
    source_case_counts: Dict[str, int] = {}
    source_case_totals: Dict[str, int] = {}
    source_errors: Dict[str, str] = {}

    if mode in {"macro", "hybrid"}:
        trace_path = _resolve_data_path(args.trace_file, repo_root=repo_root, run_dir=run_dir)
        trace_cases, trace_total = _read_trace_cases(
            trace_path=trace_path,
            target_rel=target_rel,
            object_ids=object_ids,
            limit=max(1, int(args.trace_sample_limit)),
        )
        source_case_counts["macro"] = len(trace_cases)
        source_case_totals["macro"] = trace_total
        if trace_cases:
            source_results["macro"] = _evaluate_cases(
                source="macro",
                cases=trace_cases,
                repo_root=repo_root,
                run_dir=run_dir,
                min_cases=int(args.min_cases),
                min_correctness=float(args.min_correctness),
                min_perf_ratio=float(args.min_perf_ratio),
                strict_output=bool(args.strict_output),
                correctness_weight=float(args.correctness_weight),
                perf_weight=float(args.perf_weight),
            )
        else:
            source_errors["macro"] = f"no_trace_cases:{trace_path}"

    if mode in {"micro", "hybrid"}:
        llm_path = _resolve_data_path(
            args.llm_scenarios_file,
            repo_root=repo_root,
            run_dir=run_dir,
            prefer_repo=True,
        )
        if args.llm_force_regenerate and os.path.exists(llm_path):
            try:
                os.unlink(llm_path)
            except OSError:
                pass
        if (args.llm_autogenerate and not os.path.exists(llm_path)) or args.llm_force_regenerate:
            os.makedirs(os.path.dirname(llm_path), exist_ok=True)
            gen_error = _autogenerate_llm_scenarios(
                repo_root=repo_root,
                target_rel=target_rel,
                target_start_line=int(args.target_start_line),
                target_end_line=int(args.target_end_line),
                object_ids=object_ids,
                output_path=llm_path,
                max_cases=max(1, int(args.llm_max_cases)),
                provider=str(args.llm_provider),
                model=str(args.llm_model),
                temperature=float(args.llm_temperature),
                max_new_tokens=max(32, int(args.llm_max_new_tokens)),
                enable_thinking=bool(args.llm_enable_thinking),
                timeout_s=max(1, int(args.llm_timeout_s)),
            )
            if gen_error:
                source_errors["micro_generate"] = gen_error

        llm_cases, llm_total = _read_llm_cases(
            scenarios_path=llm_path,
            object_ids=object_ids,
            limit=max(1, int(args.llm_sample_limit)),
        )
        source_case_counts["micro"] = len(llm_cases)
        source_case_totals["micro"] = llm_total
        if llm_cases:
            source_results["micro"] = _evaluate_cases(
                source="micro",
                cases=llm_cases,
                repo_root=repo_root,
                run_dir=run_dir,
                min_cases=int(args.min_cases),
                min_correctness=float(args.min_correctness),
                min_perf_ratio=float(args.min_perf_ratio),
                strict_output=bool(args.strict_output),
                correctness_weight=float(args.correctness_weight),
                perf_weight=float(args.perf_weight),
            )
        else:
            source_errors["micro"] = f"no_micro_cases:{llm_path}"

    extra: Dict[str, Any] = {
        "mode": mode,
        "macro_policy": args.macro_policy,
        "object_ids": object_ids,
        "target_analysis": analysis_payload,
        "source_case_counts": source_case_counts,
        "source_case_totals": source_case_totals,
        "source_errors": source_errors,
        "source_results": {name: _source_to_dict(res) for name, res in source_results.items()},
    }

    if mode == "macro":
        macro_res = source_results.get("macro")
        if not macro_res or not macro_res.ok:
            reason = source_errors.get("macro", "macro_unavailable")
            return _neutral_or_fail(args.missing_source_policy, reason, extra)
        return _emit(macro_res.score, macro_res.details, macro_res.passed, extra=extra)

    if mode == "micro":
        micro_res = source_results.get("micro")
        if not micro_res or not micro_res.ok:
            reason = source_errors.get("micro", source_errors.get("micro_generate", "micro_unavailable"))
            return _neutral_or_fail(args.missing_source_policy, reason, extra)
        return _emit(micro_res.score, micro_res.details, micro_res.passed, extra=extra)

    macro_res = source_results.get("macro")
    micro_res = source_results.get("micro")
    active: List[Tuple[SourceResult, float]] = []
    if macro_res and macro_res.ok:
        active.append((macro_res, max(0.0, float(args.macro_weight))))
    if micro_res and micro_res.ok:
        active.append((micro_res, max(0.0, float(args.micro_weight))))

    if args.hybrid_require_both and len(active) < 2:
        return _neutral_or_fail(args.missing_source_policy, "hybrid_missing_source", extra)
    if not active:
        reason = source_errors.get("macro") or source_errors.get("micro") or "hybrid_no_sources"
        return _neutral_or_fail(args.missing_source_policy, reason, extra)

    total_weight = sum(weight for _, weight in active)
    if total_weight <= 0:
        total_weight = float(len(active))
        active = [(result, 1.0) for result, _ in active]

    combined_score = sum(result.score * weight for result, weight in active) / total_weight
    combined_correctness = sum(result.correctness * weight for result, weight in active) / total_weight
    combined_perf = sum(result.perf_ratio * weight for result, weight in active) / total_weight
    combined_passed = all(result.passed for result, _ in active)

    detail_parts = []
    for result, _ in active:
        detail_parts.append(f"{result.source}[{result.details}]")
    details = "hybrid:" + ";".join(detail_parts)

    extra["hybrid_used_sources"] = [result.source for result, _ in active]
    extra["correctness"] = round(combined_correctness, 6)
    extra["perf_ratio"] = round(combined_perf, 6)
    return _emit(combined_score, details, combined_passed, extra=extra)


if __name__ == "__main__":
    raise SystemExit(main())
