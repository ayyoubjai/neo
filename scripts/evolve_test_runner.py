#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _normalize_target(target: str) -> str:
    normalized = target.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


_UNITTEST_RUNNER = """
import json
import sys
import unittest
from pathlib import Path

report_path, start_dir, pattern = sys.argv[1:]
program = unittest.main(
    module=None,
    argv=["unittest", "discover", "-s", start_dir, "-p", pattern],
    exit=False,
)
result = program.result
counts = {
    "tests_run": result.testsRun,
    "skipped": len(result.skipped),
    "expected_failures": len(result.expectedFailures),
}
counts["executed_tests"] = max(
    0, counts["tests_run"] - counts["skipped"] - counts["expected_failures"]
)
Path(report_path).write_text(json.dumps(counts), encoding="utf-8")
sys.exit(0 if result.wasSuccessful() and counts["executed_tests"] > 0 else 1)
"""


def _output_tail(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return (output or "")[-4000:]


def _target_module(target: str) -> str:
    normalized = _normalize_target(target)
    if normalized.startswith("src/"):
        normalized = normalized[4:]
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    return normalized.replace("/", ".")


def _discover_tests(repo_root: Path, target: str, max_files: int) -> List[Path]:
    tests_root = repo_root / "tests"
    if not tests_root.is_dir():
        return []

    normalized = _normalize_target(target)
    module = _target_module(normalized)
    stem = Path(normalized).stem.lower()
    module_leaf = module.rsplit(".", 1)[-1].lower()
    package = module.split(".", 1)[0].lower() if module else ""
    scored: List[Tuple[int, str, Path]] = []

    for test_path in sorted(tests_root.rglob("test*.py")):
        rel = test_path.relative_to(repo_root).as_posix()
        name = test_path.stem.lower()
        try:
            content = test_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        score = 0
        if name == f"test_{stem}" or name == f"test_{module_leaf}":
            score += 100
        if stem and re.search(rf"(?:^|_){re.escape(stem)}(?:_|$)", name):
            score += 35
        if module and re.search(rf"\b(?:from|import)\s+{re.escape(module)}\b", content):
            score += 90
        elif module and module in content:
            score += 55
        if normalized and normalized in content:
            score += 50
        if package and package not in {"src", "main", "common"} and package in name:
            score += 10

        if score > 0:
            scored.append((score, rel, test_path))

    scored.sort(key=lambda item: (-item[0], item[1]))
    limit = max(1, int(max_files))
    return [path for _, _, path in scored[:limit]]


def _run_test_file(repo_root: Path, test_path: Path, timeout_s: int) -> Dict[str, object]:
    relative = test_path.relative_to(repo_root)
    pattern = relative.name
    started = time.monotonic()
    env = os.environ.copy()
    import_roots = [str(repo_root / "src"), str(repo_root)]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        import_roots.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(import_roots)
    try:
        with tempfile.TemporaryDirectory(prefix="evolve-test-") as tmp:
            report_path = Path(tmp) / "result.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _UNITTEST_RUNNER,
                    str(report_path),
                    str(relative.parent),
                    pattern,
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                env=env,
                timeout=max(1, int(timeout_s)),
                check=False,
            )
            counts = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        passed = proc.returncode == 0 and counts.get("executed_tests", 0) > 0
        result = {
            "path": relative.as_posix(),
            "passed": passed,
            "returncode": proc.returncode,
            "duration_s": round(time.monotonic() - started, 3),
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            **counts,
        }
        if not counts:
            result["error"] = "test process did not produce execution evidence"
        elif counts["executed_tests"] == 0:
            result["error"] = "no regression tests executed (empty, skipped, or expected failures only)"
        return result
    except subprocess.TimeoutExpired as exc:
        return {
            "path": relative.as_posix(),
            "passed": False,
            "returncode": None,
            "duration_s": round(time.monotonic() - started, 3),
            "stdout": _output_tail(exc.stdout),
            "stderr": _output_tail(exc.stderr),
            "error": f"timeout after {timeout_s}s",
        }


def _emit(payload: Dict[str, object], passed: bool) -> int:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run focused regression tests for an evolution target.")
    parser.add_argument("--target", required=True, help="Target path relative to the repository root.")
    parser.add_argument("--repo-root", default=".", help="Candidate repository root.")
    parser.add_argument("--max-files", type=int, default=8, help="Maximum number of focused test files.")
    parser.add_argument("--timeout-s", type=int, default=120, help="Total timeout across selected tests.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    target = _normalize_target(args.target)
    target_path = (repo_root / target).resolve()
    try:
        target_path.relative_to(repo_root)
    except ValueError:
        return _emit(
            {"score": 0.0, "status": "missing_evidence", "details": "target escapes repository root"},
            False,
        )
    if not target_path.is_file():
        return _emit(
            {"score": 0.0, "status": "missing_evidence", "details": f"target not found: {target}"},
            False,
        )

    target = target_path.relative_to(repo_root).as_posix()
    selected = _discover_tests(repo_root, target, max_files=args.max_files)
    if not selected:
        return _emit(
            {
                "score": 0.0,
                "status": "missing_evidence",
                "details": f"no focused regression tests found for {target}",
                "selected_tests": [],
            },
            False,
        )

    deadline = time.monotonic() + max(1, int(args.timeout_s))
    results: List[Dict[str, object]] = []
    for test_path in selected:
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            results.append(
                {
                    "path": test_path.relative_to(repo_root).as_posix(),
                    "passed": False,
                    "error": "timeout",
                }
            )
            break
        result = _run_test_file(repo_root, test_path, remaining)
        results.append(result)
        if not bool(result.get("passed", False)):
            break

    passed = len(results) == len(selected) and all(bool(result.get("passed", False)) for result in results)
    passed_count = sum(1 for result in results if bool(result.get("passed", False)))
    details = f"focused_tests={passed_count}/{len(selected)}"
    return _emit(
        {
            "score": 1.0 if passed else 0.0,
            "status": "passed" if passed else "failed",
            "details": details,
            "selected_tests": [path.relative_to(repo_root).as_posix() for path in selected],
            "results": results,
        },
        passed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
