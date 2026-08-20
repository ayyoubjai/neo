import argparse
import difflib
import importlib.util
import json
import os
import py_compile
import sys
from typing import Dict, Tuple


def _emit(score: float, details: str, passed: bool) -> int:
    payload = {
        "score": max(0.0, min(1.0, float(score))),
        "details": details,
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    return 0 if passed else 1


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _baseline_text(target_rel: str, baseline_root: str) -> str:
    if not baseline_root:
        return ""
    base_path = os.path.join(baseline_root, target_rel)
    if not os.path.exists(base_path):
        return ""
    return _read_text(base_path)


def _metric_syntax(target: str) -> Tuple[float, str, bool]:
    try:
        py_compile.compile(target, doraise=True)
        return 1.0, "syntax_ok", True
    except py_compile.PyCompileError as e:
        return 0.0, f"syntax_error: {e}", False
    except Exception as e:
        return 0.0, f"syntax_check_failed: {e}", False


def _metric_import(target: str, repo_root: str) -> Tuple[float, str, bool]:
    target_path = os.path.abspath(target)
    if not os.path.exists(target_path):
        return 0.0, "target_not_found", False
    module_name = "_evolve_metric_module"
    try:
        spec = importlib.util.spec_from_file_location(module_name, target_path)
        if not spec or not spec.loader:
            return 0.0, "import_spec_failed", False
        module = importlib.util.module_from_spec(spec)
        src_path = os.path.join(repo_root, "src")
        path_added = False
        if os.path.isdir(src_path) and src_path not in sys.path:
            sys.path.insert(0, src_path)
            path_added = True
        try:
            spec.loader.exec_module(module)
        finally:
            if path_added and src_path in sys.path:
                sys.path.remove(src_path)
        return 1.0, "import_ok", True
    except Exception as e:
        return 0.0, f"import_failed: {e}", False


def _metric_safety(target_abs: str, target_rel: str, baseline_root: str) -> Tuple[float, str, bool]:
    current = _read_text(target_abs)
    baseline = _baseline_text(target_rel, baseline_root)
    patterns = [
        ("eval(", 1.0),
        ("exec(", 1.0),
        ("shell=True", 0.75),
        ("subprocess.Popen(", 0.5),
        ("os.system(", 0.75),
    ]
    risk = 0.0
    hits = []
    for token, weight in patterns:
        delta = current.count(token) - baseline.count(token)
        if delta > 0:
            risk += delta * weight
            hits.append(f"{token}:{delta}")
    score = max(0.0, 1.0 - (risk * 0.35))
    passed = risk <= 0.0
    details = "no_new_risky_patterns" if not hits else "new_risky_patterns=" + ",".join(hits)
    return score, details, passed


def _metric_readability(target_abs: str) -> Tuple[float, str, bool]:
    lines = _read_text(target_abs).splitlines()
    total = max(1, len(lines))
    long_lines = sum(1 for line in lines if len(line.rstrip("\n")) > 120)
    tab_lines = sum(1 for line in lines if "\t" in line)
    trailing_ws = sum(1 for line in lines if line.rstrip(" \t") != line)
    penalty = (long_lines * 1.0 + tab_lines * 0.6 + trailing_ws * 0.5) / total
    score = max(0.0, 1.0 - penalty)
    passed = score >= 0.5
    details = f"long_lines={long_lines},tabs={tab_lines},trailing_ws={trailing_ws}"
    return score, details, passed


def _metric_minimal_diff(target_abs: str, target_rel: str, baseline_root: str) -> Tuple[float, str, bool]:
    baseline = _baseline_text(target_rel, baseline_root)
    if not baseline:
        return 0.5, "baseline_missing", True
    current = _read_text(target_abs)
    old = baseline.splitlines()
    new = current.splitlines()
    matcher = difflib.SequenceMatcher(a=old, b=new)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    ratio = changed / max(1, len(old))
    score = max(0.0, 1.0 - min(1.0, ratio * 3.0))
    details = f"changed_lines={changed},baseline_lines={len(old)},ratio={ratio:.4f}"
    return score, details, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight metric runner for evolve_improve.")
    parser.add_argument("--metric", required=True, choices=["syntax", "import", "safety", "readability", "minimal_diff"])
    parser.add_argument("--target", required=True, help="Relative target path from repo root.")
    parser.add_argument("--repo-root", default="", help="Original repo root path.")
    parser.add_argument("--baseline-root", default="", help="Baseline repo root path for diffs.")
    args = parser.parse_args()

    target_arg = args.target
    target_path = target_arg if os.path.isabs(target_arg) else os.path.abspath(target_arg)
    if not os.path.exists(target_path):
        return _emit(0.0, "target_not_found", False)

    repo_root = args.repo_root or os.getcwd()
    target_rel = target_arg.replace("\\", "/")
    if os.path.isabs(target_arg):
        try:
            target_rel = os.path.relpath(target_path, repo_root).replace("\\", "/")
        except ValueError:
            target_rel = os.path.basename(target_path)
    baseline_root = args.baseline_root or args.repo_root

    if args.metric == "syntax":
        score, details, passed = _metric_syntax(target_path)
    elif args.metric == "import":
        score, details, passed = _metric_import(target_path, repo_root)
    elif args.metric == "safety":
        score, details, passed = _metric_safety(target_path, target_rel, baseline_root)
    elif args.metric == "readability":
        score, details, passed = _metric_readability(target_path)
    else:
        score, details, passed = _metric_minimal_diff(target_path, target_rel, baseline_root)
    return _emit(score, details, passed)


if __name__ == "__main__":
    raise SystemExit(main())
