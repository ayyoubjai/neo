import argparse
import ast
import importlib.util
import json
import os
import statistics
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


def _read_trace_events(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
            if event.get("kind") != "trace_case":
                continue
            if not bool(event.get("replayable", False)):
                continue
            events.append(event)
    return events


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


def _safe_repr(value: Any, max_chars: int = 200) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


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


def _values_match(a: Any, b: Any) -> bool:
    left, left_ok = _json_compatible(a)
    right, right_ok = _json_compatible(b)
    if left_ok and right_ok:
        return left == right
    if type(a) is not type(b):
        return False
    return _safe_repr(a, max_chars=2000) == _safe_repr(b, max_chars=2000)


def _resolve_trace_path(trace_file: str, repo_root: str, run_dir: str) -> str:
    if os.path.isabs(trace_file):
        return trace_file
    candidate = os.path.join(run_dir, trace_file)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(repo_root, trace_file)


def _select_cases(
    events: List[Dict[str, Any]],
    target_rel: str,
    object_ids: List[str],
    sample_limit: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    target_norm = _normalize_path(target_rel).lstrip("./")

    for event in events:
        object_id = str(event.get("object_id", "") or "")
        source_path = _normalize_path(str(event.get("source_path", "") or "")).lstrip("./")
        if object_ids:
            if object_id not in object_ids:
                continue
        elif target_norm:
            if source_path != target_norm and not source_path.endswith("/" + target_norm):
                continue

        call = event.get("call")
        if not isinstance(call, dict):
            continue
        event_key = json.dumps(
            {
                "object_id": object_id,
                "args": call.get("args", []),
                "kwargs": call.get("kwargs", {}),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        if event_key in seen:
            continue
        seen.add(event_key)
        selected.append(event)
        if sample_limit > 0 and len(selected) >= sample_limit:
            break
    return selected


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
            decorator_starts = [int(getattr(dec, "lineno", start) or start) for dec in decorators]
            if decorator_starts:
                start = min(start, min(decorator_starts))
    return start, end


def _smallest_enclosing_object_id(
    repo_root: str,
    target_rel: str,
    start_line: int,
    end_line: int,
) -> str:
    if start_line <= 0:
        return ""
    if end_line <= 0:
        end_line = start_line
    module_name = _module_from_target_rel(target_rel)
    if not module_name:
        return ""
    target_abs = os.path.join(repo_root, _normalize_path(target_rel))
    if not os.path.exists(target_abs):
        return ""
    try:
        with open(target_abs, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return ""

    parents = _build_parent_map(tree)
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
            if node_start >= start_line - 2:
                fallback.append((node_start, node_end, node))
        if not fallback:
            return ""
        fallback.sort(key=lambda item: (abs(item[0] - start_line), item[1] - item[0], item[0]))
        _, _, best = fallback[0]
    else:
        candidates.sort(key=lambda item: (item[1] - item[0], item[0]))
        _, _, best = candidates[0]

    qual_parts: List[str] = []
    current: Optional[ast.AST] = best
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qual_parts.append(current.name)
        current = parents.get(current)
    qual_parts.reverse()
    if not qual_parts:
        return ""
    return module_name + "." + ".".join(qual_parts)


def _purge_repo_modules(repo_roots: List[str]) -> None:
    normalized_roots: List[str] = []
    for root in repo_roots:
        src_root = os.path.abspath(os.path.join(root, "src"))
        normalized_roots.append(src_root)
    for module_name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        module_abs = os.path.abspath(module_file)
        for src_root in normalized_roots:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Macro metric from captured trace scenarios.")
    parser.add_argument("--target", required=True, help="Target file path relative to repo root.")
    parser.add_argument(
        "--target-start-line",
        type=int,
        default=0,
        help="Optional target start line used for auto object resolution.",
    )
    parser.add_argument(
        "--target-end-line",
        type=int,
        default=0,
        help="Optional target end line used for auto object resolution.",
    )
    parser.add_argument("--repo-root", required=True, help="Baseline repository root.")
    parser.add_argument("--run-dir", default="", help="Candidate repository root. Default: cwd.")
    parser.add_argument(
        "--trace-file",
        default="evolve/scenarios/traces/current.jsonl",
        help="Trace JSONL path. Relative paths resolve in run-dir then repo-root.",
    )
    parser.add_argument(
        "--object-ids",
        default="",
        help="Comma-separated object ids to replay. Default: inferred from --target source_path.",
    )
    parser.add_argument("--sample-limit", type=int, default=100, help="Max number of replay cases.")
    parser.add_argument("--min-cases", type=int, default=1, help="Minimum replay cases required.")
    parser.add_argument(
        "--min-correctness",
        type=float,
        default=1.0,
        help="Minimum pass ratio across replay cases.",
    )
    parser.add_argument(
        "--min-perf-ratio",
        type=float,
        default=0.95,
        help="Minimum baseline/candidate median latency ratio.",
    )
    parser.add_argument(
        "--strict-output",
        action="store_true",
        help="Require candidate return value to match baseline value.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    run_dir = os.path.abspath(args.run_dir or os.getcwd())
    target_rel = _normalize_path(args.target)
    object_ids = [part.strip() for part in args.object_ids.split(",") if part.strip()]
    if not object_ids:
        resolved = _smallest_enclosing_object_id(
            repo_root=repo_root,
            target_rel=target_rel,
            start_line=int(args.target_start_line),
            end_line=int(args.target_end_line),
        )
        if resolved:
            object_ids = [resolved]
    sample_limit = max(0, int(args.sample_limit))
    min_cases = max(1, int(args.min_cases))
    min_correctness = max(0.0, min(1.0, float(args.min_correctness)))
    min_perf_ratio = max(0.0, float(args.min_perf_ratio))

    trace_path = _resolve_trace_path(args.trace_file, repo_root, run_dir)
    if not os.path.exists(trace_path):
        return _emit(
            0.0,
            f"trace_file_not_found:{trace_path}",
            False,
            extra={"cases_total": 0, "cases_used": 0},
        )

    try:
        events = _read_trace_events(trace_path)
    except Exception as e:
        return _emit(0.0, f"trace_read_failed:{e}", False, extra={"cases_total": 0, "cases_used": 0})

    selected = _select_cases(events, target_rel=target_rel, object_ids=object_ids, sample_limit=sample_limit)
    if len(selected) < min_cases:
        return _emit(
            0.0,
            f"insufficient_cases:{len(selected)}<{min_cases}",
            False,
            extra={
                "cases_total": len(events),
                "cases_used": len(selected),
                "object_ids": object_ids,
            },
        )

    object_cache: Dict[Tuple[str, str], Any] = {}
    failures: List[str] = []
    baseline_times: List[float] = []
    candidate_times: List[float] = []
    passed_cases = 0

    for idx, event in enumerate(selected, start=1):
        object_id = str(event.get("object_id", "") or "")
        payload = event.get("call", {})
        if not isinstance(payload, dict):
            failures.append(f"case{idx}:invalid_payload")
            continue
        call_args, call_kwargs = _restore_call_payload(payload)

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
            if baseline.ok and candidate.ok and args.strict_output:
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
                failures.append(
                    f"case{idx}:exception_mismatch:{baseline.error_type}!={candidate.error_type}"
                )

    used_cases = len(baseline_times)
    if used_cases < min_cases:
        return _emit(
            0.0,
            f"insufficient_executed_cases:{used_cases}<{min_cases}",
            False,
            extra={
                "cases_total": len(events),
                "cases_used": used_cases,
                "object_ids": object_ids,
                "failures": failures[:10],
            },
        )

    baseline_median_ms = statistics.median(baseline_times)
    candidate_median_ms = statistics.median(candidate_times)
    if candidate_median_ms <= 0.0:
        perf_ratio = 0.0
    else:
        perf_ratio = baseline_median_ms / candidate_median_ms
    correctness = passed_cases / max(1, used_cases)

    perf_component = max(0.0, min(1.0, perf_ratio))
    score = (0.7 * correctness) + (0.3 * perf_component)
    passed = correctness >= min_correctness and perf_ratio >= min_perf_ratio

    details = (
        f"cases={used_cases},correctness={correctness:.4f},"
        f"baseline_median_ms={baseline_median_ms:.4f},"
        f"candidate_median_ms={candidate_median_ms:.4f},"
        f"perf_ratio={perf_ratio:.4f}"
    )
    return _emit(
        score,
        details,
        passed,
        extra={
            "cases_total": len(events),
            "cases_used": used_cases,
            "object_ids": object_ids,
            "passed_cases": passed_cases,
            "correctness": round(correctness, 6),
            "baseline_median_ms": round(baseline_median_ms, 6),
            "candidate_median_ms": round(candidate_median_ms, 6),
            "perf_ratio": round(perf_ratio, 6),
            "failures": failures[:10],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
