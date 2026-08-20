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

from common.soul_mirror import SoulConfig, load_config, sync_soul


class SoulMirrorTests(unittest.TestCase):
    def test_sync_builds_metadata_and_python_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src").mkdir()
            (repo_root / "docs").mkdir()
            (repo_root / "src" / "app.py").write_text(
                "import os\n"
                "from helper import run_helper\n\n"
                "VALUE = 3\n\n"
                "class Runner:\n"
                "    def go(self) -> None:\n"
                "        return None\n\n"
                "def main() -> int:\n"
                "    return run_helper()\n\n"
                "if __name__ == \"__main__\":\n"
                "    raise SystemExit(main())\n",
                encoding="utf-8",
            )
            (repo_root / "src" / "helper.py").write_text(
                "def run_helper() -> int:\n"
                "    return 1\n",
                encoding="utf-8",
            )
            (repo_root / "docs" / "notes.md").write_text("# Notes\n", encoding="utf-8")
            (repo_root / "data").mkdir()
            (repo_root / "data" / "secret.txt").write_text("skip me\n", encoding="utf-8")
            (repo_root / "large.txt").write_text("x" * 400, encoding="utf-8")

            config = SoulConfig(
                include=("src", "docs", "data", "large.txt"),
                exclude=("data/**",),
                max_file_bytes=256,
                output_dir="soul",
            )

            result = sync_soul(repo_root, config)

            self.assertTrue(result.changed)
            self.assertEqual(result.manifest["file_count"], 3)
            self.assertEqual(result.manifest["python_file_count"], 2)
            self.assertEqual(result.manifest["skipped_count"], 1)
            self.assertEqual(result.manifest["symbol_count"], 5)
            self.assertEqual(result.manifest["internal_dependency_edge_count"], 1)
            self.assertEqual(result.manifest["entrypoint_count"], 1)
            self.assertTrue((repo_root / "soul" / "code" / "src" / "app.py").exists())
            self.assertTrue((repo_root / "soul" / "code" / "src" / "helper.py").exists())
            self.assertTrue((repo_root / "soul" / "code" / "docs" / "notes.md").exists())
            self.assertFalse((repo_root / "soul" / "code" / "data" / "secret.txt").exists())

            python_index = _read_jsonl(repo_root / "soul" / "meta" / "python_index.jsonl")
            self.assertEqual(len(python_index), 2)
            self.assertEqual(python_index[0]["path"], "src/app.py")
            self.assertEqual(python_index[0]["functions"], ["main"])
            self.assertEqual(python_index[0]["constants"], ["VALUE"])
            self.assertEqual(python_index[0]["classes"], [{"name": "Runner", "methods": ["go"]}])

            symbols = _read_jsonl(repo_root / "soul" / "meta" / "symbols.jsonl")
            self.assertEqual(
                [(row["qualname"], row["kind"]) for row in symbols],
                [
                    ("VALUE", "constant"),
                    ("Runner", "class"),
                    ("Runner.go", "method"),
                    ("main", "function"),
                    ("run_helper", "function"),
                ],
            )

            module_graph = _read_json(repo_root / "soul" / "meta" / "module_graph.json")
            self.assertEqual(len(module_graph["edges"]), 1)
            self.assertEqual(module_graph["edges"][0]["source_module"], "app")
            self.assertEqual(module_graph["edges"][0]["target_module"], "helper")

            entrypoints = _read_jsonl(repo_root / "soul" / "meta" / "entrypoints.jsonl")
            self.assertEqual(entrypoints, [
                {
                    "path": "src/app.py",
                    "module": "app",
                    "role": "python_main",
                    "language": "python",
                    "has_main_guard": True,
                    "has_main_function": True,
                    "uses_argparse": False,
                    "module_docstring": "",
                }
            ])

            views = _read_json(repo_root / "soul" / "meta" / "views.json")
            self.assertEqual(views["runtime"]["config_files"], [])
            self.assertEqual(views["runtime"]["service_entrypoints"], [])
            self.assertEqual([entry["path"] for entry in views["runtime"]["cli_entrypoints"]], [])
            self.assertEqual(views["hotspots"]["largest_files"][0]["path"], "src/app.py")
            area_summaries = _read_json(repo_root / "soul" / "meta" / "area_summaries.json")
            self.assertEqual(area_summaries["schema_version"], 3)
            self.assertEqual(area_summaries["areas"][0]["area"], "docs")
            self.assertEqual(area_summaries["areas"][1]["area"], "src")
            self.assertIn("runtime code area", area_summaries["areas"][1]["summary"])
            self.assertIn("app", area_summaries["areas"][1]["responsibilities"])
            self.assertTrue((repo_root / "soul" / "meta" / "overview.md").exists())

            skipped = _read_jsonl(repo_root / "soul" / "meta" / "skipped.jsonl")
            self.assertEqual(skipped, [{"path": "large.txt", "reason": "max_file_bytes", "size_bytes": 400}])

    def test_sync_is_incremental_and_removes_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src").mkdir()
            (repo_root / "src" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
            config = SoulConfig(include=("src",), exclude=(), output_dir="soul")

            first = sync_soul(repo_root, config)
            second = sync_soul(repo_root, config)

            self.assertTrue(first.changed)
            self.assertFalse(second.changed)
            self.assertTrue((repo_root / "soul" / "code" / "src" / "a.py").exists())

            (repo_root / "src" / "a.py").unlink()
            (repo_root / "src" / "b.py").write_text("VALUE = 2\n", encoding="utf-8")

            third = sync_soul(repo_root, config)
            self.assertTrue(third.changed)
            self.assertEqual(third.removed_files, 1)
            self.assertFalse((repo_root / "soul" / "code" / "src" / "a.py").exists())
            self.assertTrue((repo_root / "soul" / "code" / "src" / "b.py").exists())

    def test_sync_resolves_relative_import_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "pkg").mkdir(parents=True)
            (repo_root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (repo_root / "src" / "pkg" / "a.py").write_text(
                "from .b import helper\n\n"
                "def use() -> int:\n"
                "    return helper()\n",
                encoding="utf-8",
            )
            (repo_root / "src" / "pkg" / "b.py").write_text(
                "def helper() -> int:\n"
                "    return 1\n",
                encoding="utf-8",
            )
            config = SoulConfig(include=("src",), exclude=(), output_dir="soul")

            sync_soul(repo_root, config)

            module_graph = _read_json(repo_root / "soul" / "meta" / "module_graph.json")
            self.assertEqual(
                [(edge["source_module"], edge["target_module"]) for edge in module_graph["edges"]],
                [("pkg.a", "pkg.b")],
            )

    def test_area_summaries_describe_cross_area_subsystems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "src" / "orchestrator").mkdir(parents=True)
            (repo_root / "src" / "common").mkdir(parents=True)
            (repo_root / "config").mkdir()
            (repo_root / "src" / "orchestrator" / "main.py").write_text(
                "from common.util import log\n"
                "from orchestrator.planner import build_plan\n\n"
                "def main() -> int:\n"
                "    log(build_plan())\n"
                "    return 0\n\n"
                "if __name__ == \"__main__\":\n"
                "    raise SystemExit(main())\n",
                encoding="utf-8",
            )
            (repo_root / "src" / "orchestrator" / "planner.py").write_text(
                "def build_plan() -> str:\n"
                "    return \"plan\"\n",
                encoding="utf-8",
            )
            (repo_root / "src" / "common" / "util.py").write_text(
                "def log(value: str) -> None:\n"
                "    return None\n",
                encoding="utf-8",
            )
            (repo_root / "config" / "settings.json").write_text("{\"mode\": \"test\"}\n", encoding="utf-8")
            config = SoulConfig(include=("src", "config"), exclude=(), output_dir="soul")

            sync_soul(repo_root, config)

            area_summaries = _read_json(repo_root / "soul" / "meta" / "area_summaries.json")
            areas = {item["area"]: item for item in area_summaries["areas"]}

            orchestrator = areas["src/orchestrator"]
            self.assertEqual(orchestrator["kind"], "runtime subsystem")
            self.assertIn("planner", orchestrator["responsibilities"])
            self.assertIn("src/common", orchestrator["depends_on"])
            self.assertIn("service entrypoint", orchestrator["summary"])

            common = areas["src/common"]
            self.assertEqual(common["kind"], "runtime code area")
            self.assertIn("src/orchestrator", common["depended_on_by"])

            config_area = areas["config"]
            self.assertEqual(config_area["kind"], "configuration surface")
            self.assertIn("settings", config_area["dominant_terms"])

    def test_load_config_preserves_dot_prefixed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "soul.json"
            config_path.write_text(
                json.dumps(
                    {
                        "output_dir": "soul",
                        "include": ["./src"],
                        "exclude": [".git/**", ".venv/**", "./data/**"],
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            self.assertEqual(config.include, ("src",))
            self.assertEqual(config.exclude, (".git/**", ".venv/**", "data/**"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    unittest.main()
