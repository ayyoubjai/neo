import json
import tempfile
import unittest
from pathlib import Path


import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SCRIPTS_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import local_code_agent


class LocalCodeAgentTests(unittest.TestCase):
    def test_resolve_repo_path_blocks_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            with self.assertRaises(RuntimeError):
                local_code_agent._resolve_repo_path(repo, "../outside.txt")

    def test_apply_edit_requires_unique_snippet_without_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            target = repo / "sample.py"
            target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                local_code_agent._apply_edit(
                    repo,
                    {"path": "sample.py", "before": "value = 1", "after": "value = 2"},
                )

            changed = local_code_agent._apply_edit(
                repo,
                {"path": "sample.py", "before": "value = 1", "after": "value = 2", "occurrence": 2},
            )

            self.assertEqual(changed, "sample.py")
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\nvalue = 2\n")

    def test_select_snippets_uses_soul_files_and_task_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / "src").mkdir()
            (repo / "soul" / "meta").mkdir(parents=True)
            (repo / "src" / "alpha.py").write_text("def auth_login():\n    return True\n", encoding="utf-8")
            (repo / "src" / "beta.py").write_text("def billing():\n    return False\n", encoding="utf-8")
            rows = [
                {"path": "src/beta.py"},
                {"path": "src/alpha.py"},
            ]
            with (repo / "soul" / "meta" / "files.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            snippets = local_code_agent._select_snippets(
                repo,
                repo / "soul",
                "fix auth login behavior",
                explicit_files=[],
                max_files=1,
            )

            self.assertEqual(len(snippets), 1)
            self.assertEqual(snippets[0].path, "src/alpha.py")

    def test_normalize_candidate_defaults_missing_lists(self) -> None:
        candidate = local_code_agent._normalize_candidate({"summary": 123, "edits": None})

        self.assertEqual(candidate["summary"], "")
        self.assertEqual(candidate["edits"], [])
        self.assertEqual(candidate["files"], [])


if __name__ == "__main__":
    unittest.main()
