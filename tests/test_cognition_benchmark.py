import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("benchmark_cognition", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_cognition.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_reported_verification_does_not_hide_wrong_answer():
    case = {"acceptance_checks": [{"id": "total"}], "answer_contains": ["42"]}
    payload = {"text": "43", "completion": {"status": "verified", "checks": [{"id": "total", "passed": True}]}}
    score = benchmark.score_result(case, payload)
    assert score["false_success"]
    assert not score["success"]


def test_missing_expected_check_cannot_count_as_success():
    case = {"acceptance_checks": [{"id": "first"}, {"id": "second"}], "answer_contains": ["42"]}
    payload = {"text": "42", "completion": {"status": "verified", "checks": [{"id": "first", "passed": True}]}}
    assert benchmark.score_result(case, payload)["false_success"]
