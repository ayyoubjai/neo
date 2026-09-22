from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orchestrator.tool_selector import ToolRegistry
from tool_runtime.tools import (
    archive_zip_create,
    archive_zip_extract,
    data_csv_read,
    data_json_query,
    fs_find,
    fs_search_text,
    fs_stat,
    git_diff,
    git_log,
    git_status,
    http_request,
    net_search,
    proc_exec,
    sys_list_tools,
    tool_registry,
)


class _FakeHttpResponse:
    def __init__(self, body: bytes, *, status: int = 200, content_type: str = "application/json; charset=utf-8") -> None:
        self._body = body
        self.status = status
        self.headers = {"Content-Type": content_type}

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._body
        return self._body[:size]

    def close(self) -> None:
        return

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class GeneralToolTests(unittest.TestCase):
    def test_search_engine_failures_are_not_successful_empty_results(self) -> None:
        from tool_runtime.tools import _net_search_searxng, ToolError
        body = b'{"results": [], "unresponsive_engines": [["brave", "timeout"]]}'
        with patch("tool_runtime.tools.urllib.request.urlopen", return_value=_FakeHttpResponse(body)):
            with self.assertRaisesRegex(ToolError, "engine failures.*brave.*timeout"):
                _net_search_searxng("query", 5, "http://localhost", 10)

    def test_search_preserves_results_when_some_engines_fail(self) -> None:
        from tool_runtime.tools import _net_search_searxng
        body = b'{"results": [{"title": "Found", "url": "https://example.com"}], "unresponsive_engines": [["brave", "timeout"]]}'
        with patch("tool_runtime.tools.urllib.request.urlopen", return_value=_FakeHttpResponse(body)):
            result = _net_search_searxng("query", 5, "http://localhost", 10)
        self.assertEqual(result[0]["title"], "Found")

    def test_search_rejects_malformed_payload(self) -> None:
        from tool_runtime.tools import _net_search_searxng, ToolError
        for body in [b'[]', b'{}', b'{"results": null}']:
            with self.subTest(body=body), patch(
                "tool_runtime.tools.urllib.request.urlopen", return_value=_FakeHttpResponse(body)
            ):
                with self.assertRaises(ToolError):
                    _net_search_searxng("query", 5, "http://localhost", 10)

    def test_runtime_tool_registry_contains_general_tools(self) -> None:
        registry = tool_registry()
        self.assertIn("sys.list_tools", registry)
        self.assertIn("fs.stat", registry)
        self.assertIn("fs.find", registry)
        self.assertIn("fs.search_text", registry)
        self.assertIn("data.json_query", registry)
        self.assertIn("data.csv_read", registry)
        self.assertIn("archive.zip_create", registry)
        self.assertIn("archive.zip_extract", registry)
        self.assertIn("http.request", registry)
        self.assertIn("git.status", registry)
        self.assertIn("git.diff", registry)
        self.assertIn("git.log", registry)
        self.assertIn("proc.exec", registry)

    def test_metadata_registry_contains_new_permissions(self) -> None:
        registry = ToolRegistry()
        self.assertEqual(registry.get_tool("sys.list_tools")["required_permissions"], ["tier0"])
        self.assertEqual(registry.get_tool("archive.zip_create")["required_permissions"], ["tier1"])
        self.assertEqual(registry.get_tool("proc.exec")["required_permissions"], ["tier2"])

    def test_sys_list_tools_reports_new_entries(self) -> None:
        result, io = sys_list_tools({}, str(REPO_ROOT))
        self.assertEqual(io, {})
        tool_ids = {item["tool_id"] for item in result["tools"]}
        self.assertIn("http.request", tool_ids)
        self.assertIn("proc.exec", tool_ids)
        self.assertGreaterEqual(result["count"], len(tool_ids))

    def test_fs_and_data_tools_work_on_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes").mkdir()
            (root / "notes" / "sample.txt").write_text("alpha\nbeta\n", encoding="utf-8")
            (root / "payload.json").write_text(
                json.dumps({"user": {"name": "atlas", "roles": ["builder"]}}, ensure_ascii=True),
                encoding="utf-8",
            )
            (root / "table.csv").write_text("name,age\natlas,2\n", encoding="utf-8")

            stat_result, _ = fs_stat({"path": "notes/sample.txt"}, tmpdir)
            self.assertTrue(stat_result["exists"])
            self.assertEqual(stat_result["kind"], "file")

            find_result, _ = fs_find({"root": "notes", "pattern": "*.txt"}, tmpdir)
            self.assertEqual(find_result["count"], 1)
            self.assertEqual(find_result["entries"][0]["path"], "workspace:/notes/sample.txt")

            search_result, _ = fs_search_text({"root": "notes", "query": "beta"}, tmpdir)
            self.assertEqual(search_result["count"], 1)
            self.assertEqual(search_result["matches"][0]["line_number"], 2)

            json_result, _ = data_json_query({"path": "payload.json", "pointer": "/user/name"}, tmpdir)
            self.assertTrue(json_result["found"])
            self.assertEqual(json_result["value"], "atlas")

            csv_result, _ = data_csv_read({"path": "table.csv"}, tmpdir)
            self.assertEqual(csv_result["columns"], ["name", "age"])
            self.assertEqual(csv_result["rows"][0]["name"], "atlas")

    def test_archive_tools_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "folder").mkdir()
            (root / "folder" / "hello.txt").write_text("hello", encoding="utf-8")

            create_result, _ = archive_zip_create(
                {"sources": ["folder"], "output_path": "archives/test.zip"},
                tmpdir,
            )
            self.assertEqual(create_result["entry_count"], 1)
            self.assertTrue((root / "archives" / "test.zip").exists())

            extract_result, _ = archive_zip_extract(
                {"archive_path": "archives/test.zip", "output_dir": "unzipped"},
                tmpdir,
            )
            self.assertEqual(extract_result["extracted_count"], 1)
            self.assertEqual((root / "unzipped" / "folder" / "hello.txt").read_text(encoding="utf-8"), "hello")

    def test_http_request_reads_local_server_and_saves_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "http://example.test/hello"
            payload = json.dumps({"path": "/hello?name=atlas"}, ensure_ascii=True).encode("utf-8")
            with patch("tool_runtime.tools.urllib.request.urlopen", return_value=_FakeHttpResponse(payload)):
                result, _ = http_request(
                    {
                        "url": url,
                        "params": {"name": "atlas"},
                        "save_to_path": "downloads/echo.json",
                    },
                    tmpdir,
                )

            self.assertEqual(result["status_code"], 200)
            self.assertIn("name=atlas", result["body_text"])
            self.assertEqual(result["saved_to_path"], "workspace:/downloads/echo.json")
            saved = Path(tmpdir) / "downloads" / "echo.json"
            self.assertTrue(saved.exists())

    def test_net_search_uses_configured_safe_search_level(self) -> None:
        settings = SimpleNamespace(
            search={
                "provider": "searxng",
                "searxng_url": "http://127.0.0.1:8081",
                "safe_search": 2,
                "timeout_s": 20,
            }
        )
        response = _FakeHttpResponse(b'{"results": []}')
        with (
            patch("tool_runtime.tools.load_settings", return_value=settings),
            patch("tool_runtime.tools.urllib.request.urlopen", return_value=response) as urlopen,
        ):
            result, io = net_search({"query": "security test"}, str(REPO_ROOT))

        request_url = urlopen.call_args.args[0]
        query = parse_qs(urlsplit(request_url).query)
        self.assertEqual(query["safesearch"], ["2"])
        self.assertEqual(result["results"], [])
        self.assertEqual(io, {})

    def test_net_search_defaults_invalid_safe_search_to_moderate(self) -> None:
        settings = SimpleNamespace(
            search={
                "provider": "searxng",
                "searxng_url": "http://127.0.0.1:8081",
                "safe_search": "invalid",
                "timeout_s": 20,
            }
        )
        response = _FakeHttpResponse(b'{"results": []}')
        with (
            patch("tool_runtime.tools.load_settings", return_value=settings),
            patch("tool_runtime.tools.urllib.request.urlopen", return_value=response) as urlopen,
        ):
            net_search({"query": "security test"}, str(REPO_ROOT))

        request_url = urlopen.call_args.args[0]
        query = parse_qs(urlsplit(request_url).query)
        self.assertEqual(query["safesearch"], ["1"])

    def test_git_tools_report_repo_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmpdir, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmpdir, check=True)
            (root / "tracked.txt").write_text("one\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=tmpdir, check=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=tmpdir, check=True, capture_output=True, text=True)
            (root / "tracked.txt").write_text("one\ntwo\n", encoding="utf-8")

            status_result, _ = git_status({}, tmpdir)
            self.assertEqual(status_result["change_count"], 1)
            self.assertEqual(status_result["changes"][0]["path"], "tracked.txt")

            diff_result, _ = git_diff({}, tmpdir)
            self.assertIn("+two", diff_result["diff"])

            log_result, _ = git_log({"max_commits": 5}, tmpdir)
            self.assertEqual(log_result["count"], 1)
            self.assertEqual(log_result["commits"][0]["subject"], "initial")

    def test_proc_exec_runs_command_without_shell(self) -> None:
        result, _ = proc_exec({"command": ["python3", "-c", "print('hello')"]}, str(REPO_ROOT))
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["timed_out"])
        self.assertIn("hello", result["stdout"])


if __name__ == "__main__":
    unittest.main()
