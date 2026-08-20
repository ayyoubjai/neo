from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestrator.main import Orchestrator


class _FakeTools:
    def __init__(self) -> None:
        self._tool = {
            "tool_id": "ui.screenshot",
            "name": "Screenshot",
            "description": "Capture a screenshot.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "capabilities": [],
        }

    def get_tool(self, tool_id: str):
        if tool_id == "ui.screenshot":
            return dict(self._tool)
        return None

    def list_active(self):
        return [dict(self._tool)]


class AsiSourcePatternTests(unittest.IsolatedAsyncioTestCase):
    def _orchestrator(self) -> Orchestrator:
        orchestrator = Orchestrator()
        orchestrator._tools = _FakeTools()
        orchestrator._asi_recursion_limit = 4
        return orchestrator

    def test_normalize_source_pattern_derives_inputs_from_signature(self) -> None:
        orchestrator = self._orchestrator()
        pattern = {
            "pattern_id": "repeat_capture",
            "name": "Repeat Capture",
            "when_to_use": ["Capture a path multiple times."],
            "source": (
                "def repeat_capture(path, n: int = 2):\n"
                "    return path\n"
            ),
        }

        normalized = orchestrator._normalize_asi_pattern(pattern)

        self.assertEqual(
            normalized["inputs_needed"],
            [
                {"name": "path", "type": "text", "required": True},
                {"name": "n", "type": "int", "required": False, "default": 2},
            ],
        )

    def test_validate_source_pattern_accepts_looped_tool_calls(self) -> None:
        orchestrator = self._orchestrator()
        pattern = {
            "pattern_id": "repeat_capture",
            "name": "Repeat Capture",
            "when_to_use": ["Capture a screenshot several times."],
            "source": (
                "def repeat_capture(path, n: int):\n"
                "    results = []\n"
                "    for i in range(n):\n"
                "        results.append(ui.screenshot(path=path))\n"
                "    return results\n"
            ),
        }

        errors = orchestrator._validate_asi_pattern(
            pattern,
            ["ui.screenshot"],
            {"ui.screenshot": orchestrator._tools.get_tool("ui.screenshot")},
        )

        self.assertEqual(errors, [])

    def test_validate_source_pattern_accepts_comprehensions_and_helpers(self) -> None:
        orchestrator = self._orchestrator()
        pattern = {
            "pattern_id": "index_paths",
            "name": "Index Paths",
            "when_to_use": ["Build an index for a list of paths."],
            "source": (
                "def index_paths(paths):\n"
                "    return {idx: path for idx, path in enumerate(paths)}\n"
            ),
        }

        errors = orchestrator._validate_asi_pattern(
            pattern,
            ["ui.screenshot"],
            {"ui.screenshot": orchestrator._tools.get_tool("ui.screenshot")},
        )

        self.assertEqual(errors, [])

    def test_validate_asi_pattern_rejects_step_backed_shape(self) -> None:
        orchestrator = self._orchestrator()
        pattern = {
            "pattern_id": "step_capture",
            "name": "Step Capture",
            "when_to_use": ["Capture a screenshot."],
            "steps": [
                {
                    "op": "tool",
                    "tool_id": "ui.screenshot",
                    "args": {"path": "{{path}}"},
                    "save_as": "shot",
                },
                {"op": "return", "template": "{{shot.result.path}}"},
            ],
        }

        errors = orchestrator._validate_asi_pattern(
            pattern,
            ["ui.screenshot"],
            {"ui.screenshot": orchestrator._tools.get_tool("ui.screenshot")},
        )

        self.assertIn("step-backed ASI patterns are no longer supported", errors)





    def test_find_asi_pattern_resolves_active_tool_as_pattern(self) -> None:
        orchestrator = self._orchestrator()

        pattern = orchestrator._find_asi_pattern("ui.screenshot")

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern["pattern_id"], "ui.screenshot")
        self.assertEqual(pattern["tool_id"], "ui.screenshot")
        self.assertEqual(orchestrator._asi_pattern_form(pattern), "tool")

    def test_normalize_asi_pattern_ref_accepts_call_syntax(self) -> None:
        orchestrator = self._orchestrator()

        self.assertEqual(orchestrator._normalize_asi_pattern_ref("sys.time()"), "sys.time")
        self.assertEqual(
            orchestrator._normalize_asi_pattern_ref("pat.ui.screenshot(path: str = None)"),
            "ui.screenshot",
        )






    def test_tool_to_asi_pattern_handles_missing_required_list(self) -> None:
        orchestrator = self._orchestrator()

        pattern = orchestrator._tool_to_asi_pattern(
            {
                "tool_id": "search.web",
                "name": "Web Search",
                "description": "Search the web.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                },
                "capabilities": [],
            }
        )

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern["pattern_id"], "search.web")
        self.assertEqual(
            pattern["inputs_needed"],
            [{"name": "query", "type": "text", "required": False, "description": "Search query"}],
        )



    async def test_execute_tool_backed_pattern_directly(self) -> None:
        orchestrator = self._orchestrator()
        tool_calls = []

        async def _fake_execute_tool_action(tool_id: str, args, trace_id: str, thought: str = ""):
            tool_calls.append({"tool_id": tool_id, "args": dict(args), "trace_id": trace_id})
            return {
                "thought": thought,
                "action": {"tool_id": tool_id, "args": dict(args)},
                "observation": {
                    "status": "APPROVED",
                    "result": {"path": args.get("path"), "trace_id": trace_id},
                    "error": "",
                },
            }

        orchestrator._execute_tool_action = _fake_execute_tool_action  # type: ignore[method-assign]
        pattern = orchestrator._find_asi_pattern("ui.screenshot")

        result = await orchestrator._execute_asi_pattern_runtime(
            pattern or {},
            {"path": "/tmp/direct.png"},
            "trace-direct-tool",
            {"remaining": 3},
            user_request="take a screenshot",
            depth=0,
        )

        self.assertEqual(result["type"], "final")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["tool_id"], "ui.screenshot")
        self.assertEqual(tool_calls[0]["args"]["path"], "/tmp/direct.png")
        self.assertEqual(result["value"]["path"], "/tmp/direct.png")

    async def test_execute_source_pattern_loops_tool_n_times(self) -> None:
        orchestrator = self._orchestrator()
        tool_calls = []

        async def _fake_execute_tool_action(tool_id: str, args, trace_id: str, thought: str = ""):
            tool_calls.append({"tool_id": tool_id, "args": dict(args), "trace_id": trace_id})
            return {
                "thought": thought,
                "action": {"tool_id": tool_id, "args": dict(args)},
                "observation": {
                    "status": "APPROVED",
                    "result": {"path": args.get("path"), "trace_id": trace_id},
                    "error": "",
                },
            }

        orchestrator._execute_tool_action = _fake_execute_tool_action  # type: ignore[method-assign]
        pattern = orchestrator._normalize_asi_pattern(
            {
                "pattern_id": "repeat_capture",
                "name": "Repeat Capture",
                "when_to_use": ["Capture a screenshot several times."],
                "source": (
                    "def repeat_capture(path, n: int):\n"
                    "    results = []\n"
                    "    for i in range(n):\n"
                    "        results.append(ui.screenshot(path=path))\n"
                    "    return results\n"
                ),
            }
        )

        result = await orchestrator._execute_asi_pattern_runtime(
            pattern,
            {"path": "/tmp/example.png", "n": 3},
            "trace-1",
            {"remaining": 10},
            user_request="capture three screenshots",
            depth=0,
        )

        self.assertEqual(result["type"], "final")
        self.assertEqual(len(tool_calls), 3)
        self.assertEqual(len(result["value"]), 3)
        self.assertEqual(tool_calls[0]["args"]["path"], "/tmp/example.png")
        tool_trace = [item for item in result["trace"] if item.get("tool_id") == "ui.screenshot"]
        self.assertEqual(len(tool_trace), 3)

    async def test_execute_source_pattern_supports_comprehensions(self) -> None:
        orchestrator = self._orchestrator()
        tool_calls = []

        async def _fake_execute_tool_action(tool_id: str, args, trace_id: str, thought: str = ""):
            tool_calls.append({"tool_id": tool_id, "args": dict(args), "trace_id": trace_id})
            return {
                "thought": thought,
                "action": {"tool_id": tool_id, "args": dict(args)},
                "observation": {
                    "status": "APPROVED",
                    "result": {"path": args.get("path"), "trace_id": trace_id},
                    "error": "",
                },
            }

        orchestrator._execute_tool_action = _fake_execute_tool_action  # type: ignore[method-assign]
        pattern = orchestrator._normalize_asi_pattern(
            {
                "pattern_id": "capture_many",
                "name": "Capture Many",
                "when_to_use": ["Capture multiple screenshots."],
                "source": (
                    "def capture_many(paths):\n"
                    "    return [ui.screenshot(path=path).result.path for path in paths]\n"
                ),
            }
        )

        result = await orchestrator._execute_asi_pattern_runtime(
            pattern,
            {"paths": ["/tmp/a.png", "/tmp/b.png"]},
            "trace-comp",
            {"remaining": 10},
            user_request="capture screenshots for each path",
            depth=0,
        )

        self.assertEqual(result["type"], "final")
        self.assertEqual(result["value"], ["/tmp/a.png", "/tmp/b.png"])
        self.assertEqual(len(tool_calls), 2)

    async def test_execute_source_pattern_supports_tuple_unpacking_and_dict_comprehensions(self) -> None:
        orchestrator = self._orchestrator()
        pattern = orchestrator._normalize_asi_pattern(
            {
                "pattern_id": "index_paths",
                "name": "Index Paths",
                "when_to_use": ["Build an index for a list of paths."],
                "source": (
                    "def index_paths(paths):\n"
                    "    return {idx: path for idx, path in enumerate(paths)}\n"
                ),
            }
        )

        result = await orchestrator._execute_asi_pattern_runtime(
            pattern,
            {"paths": ["alpha", "beta", "gamma"]},
            "trace-dict-comp",
            {"remaining": 10},
            user_request="index these paths",
            depth=0,
        )

        self.assertEqual(result["type"], "final")
        self.assertEqual(result["value"], {0: "alpha", 1: "beta", 2: "gamma"})

    async def test_execute_source_pattern_can_branch_on_tool_result_wrapper(self) -> None:
        orchestrator = self._orchestrator()

        async def _fake_execute_tool_action(tool_id: str, args, trace_id: str, thought: str = ""):
            if args.get("path") == "/tmp/fail.png":
                return {
                    "thought": thought,
                    "action": {"tool_id": tool_id, "args": dict(args)},
                    "observation": {
                        "status": "ERROR",
                        "result": None,
                        "error": "capture failed",
                    },
                }
            return {
                "thought": thought,
                "action": {"tool_id": tool_id, "args": dict(args)},
                "observation": {
                    "status": "APPROVED",
                    "result": {"path": args.get("path"), "trace_id": trace_id},
                    "error": "",
                },
            }

        orchestrator._execute_tool_action = _fake_execute_tool_action  # type: ignore[method-assign]
        pattern = orchestrator._normalize_asi_pattern(
            {
                "pattern_id": "capture_paths",
                "name": "Capture Paths",
                "when_to_use": ["Capture a set of paths while tolerating failures."],
                "source": (
                    "def capture_paths(paths):\n"
                    "    results = []\n"
                    "    for idx, path in enumerate(paths):\n"
                    "        shot = ui.screenshot(path=path)\n"
                    "        if shot.ok:\n"
                    "            results.append(f\"{idx}:{shot.result.path}\")\n"
                    "        else:\n"
                    "            results.append(f\"{idx}:{shot.error}\")\n"
                    "    return results\n"
                ),
            }
        )

        result = await orchestrator._execute_asi_pattern_runtime(
            pattern,
            {"paths": ["/tmp/ok.png", "/tmp/fail.png"]},
            "trace-tool-wrapper",
            {"remaining": 10},
            user_request="capture paths and keep going on failure",
            depth=0,
        )

        self.assertEqual(result["type"], "final")
        self.assertEqual(result["value"], ["0:/tmp/ok.png", "1:capture failed"])
        tool_trace = [item for item in result["trace"] if item.get("tool_id") == "ui.screenshot"]
        self.assertEqual(len(tool_trace), 2)
        self.assertEqual(tool_trace[0]["status"], "success")
        self.assertEqual(tool_trace[1]["status"], "error")

    async def test_execute_source_pattern_can_call_other_patterns_as_functions(self) -> None:
        orchestrator = self._orchestrator()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            patterns_path = tmp_path / "asi_patterns.json"
            patterns_path.write_text(
                json.dumps(
                    [
                        {
                            "pattern_id": "child_greet",
                            "name": "Child Greet",
                            "when_to_use": ["Create a greeting."],
                            "source": (
                                "def child_greet(name):\n"
                                "    return f\"Hello {name}\"\n"
                            ),
                        },
                        {
                            "pattern_id": "parent_greet",
                            "name": "Parent Greet",
                            "when_to_use": ["Call another pattern."],
                            "source": (
                                "def parent_greet(name):\n"
                                "    reply = pat.child_greet(name=name)\n"
                                "    return reply.value\n"
                            ),
                        },
                    ],
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            orchestrator._asi_patterns_path = str(patterns_path)
            orchestrator._asi_patterns_cache = None
            orchestrator._asi_patterns_mtime = None

            parent = orchestrator._find_asi_pattern("parent_greet")
            self.assertIsNotNone(parent)

            result = await orchestrator._execute_asi_pattern_runtime(
                parent or {},
                {"name": "Ada"},
                "trace-2",
                {"remaining": 10},
                user_request="say hello to Ada",
                depth=0,
            )

            self.assertEqual(result["type"], "final")
            self.assertEqual(result["text"], "Hello Ada")

    async def test_execute_source_pattern_wraps_child_pattern_clarify_results(self) -> None:
        orchestrator = self._orchestrator()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            patterns_path = tmp_path / "asi_patterns.json"
            patterns_path.write_text(
                json.dumps(
                    [
                        {
                            "pattern_id": "child_requires_name",
                            "name": "Child Requires Name",
                            "when_to_use": ["Need a name before greeting."],
                            "source": (
                                "def child_requires_name(name):\n"
                                "    return f\"Hello {name}\"\n"
                            ),
                        },
                        {
                            "pattern_id": "parent_handles_clarify",
                            "name": "Parent Handles Clarify",
                            "when_to_use": ["Inspect a child pattern result wrapper."],
                            "source": (
                                "def parent_handles_clarify():\n"
                                "    reply = pat.child_requires_name()\n"
                                "    if reply.ok:\n"
                                "        return reply.value\n"
                                "    return reply.question\n"
                            ),
                        },
                    ],
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            orchestrator._asi_patterns_path = str(patterns_path)
            orchestrator._asi_patterns_cache = None
            orchestrator._asi_patterns_mtime = None

            parent = orchestrator._find_asi_pattern("parent_handles_clarify")
            self.assertIsNotNone(parent)

            result = await orchestrator._execute_asi_pattern_runtime(
                parent or {},
                {},
                "trace-pattern-wrapper",
                {"remaining": 10},
                user_request="handle a missing input from a child pattern",
                depth=0,
            )

            self.assertEqual(result["type"], "final")
            self.assertIn("name", result["text"])

    def test_load_asi_patterns_ignores_step_backed_library_entries(self) -> None:
        orchestrator = self._orchestrator()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            patterns_path = tmp_path / "asi_patterns.json"
            patterns_path.write_text(
                json.dumps(
                    [
                        {
                            "pattern_id": "step_capture",
                            "name": "Step Capture",
                            "when_to_use": ["Capture a screenshot."],
                            "inputs_needed": [{"name": "path", "type": "string", "required": True}],
                            "steps": [
                                {
                                    "op": "tool",
                                    "tool_id": "ui.screenshot",
                                    "args": {"path": "{{path}}"},
                                    "save_as": "shot",
                                },
                                {"op": "return", "template": "{{shot.result.path}}"},
                            ],
                        },
                        {
                            "pattern_id": "source_capture",
                            "name": "Source Capture",
                            "when_to_use": ["Capture a screenshot via source."],
                            "source": (
                                "def source_capture(path):\n"
                                "    shot = ui.screenshot(path=path)\n"
                                "    return shot.result.path\n"
                            ),
                        },
                    ],
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            orchestrator._asi_patterns_path = str(patterns_path)
            orchestrator._asi_patterns_cache = None
            orchestrator._asi_patterns_mtime = None

            patterns = orchestrator._load_asi_patterns()

            self.assertEqual([pattern["pattern_id"] for pattern in patterns], ["source_capture"])
            self.assertIsNone(orchestrator._find_asi_pattern("step_capture"))
            self.assertIsNotNone(orchestrator._find_asi_pattern("source_capture"))


if __name__ == "__main__":
    unittest.main()
