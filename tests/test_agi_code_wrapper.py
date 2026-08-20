import tempfile
import unittest
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import agi_code


class AgiCodeWrapperTests(unittest.TestCase):
    def test_build_agent_argv_injects_current_directory_as_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            argv = agi_code.build_agent_argv(["implement", "x"], cwd)

            self.assertEqual(argv[2], "--repo-root")
            self.assertEqual(Path(argv[3]), cwd.resolve())
            self.assertEqual(argv[4:], ["implement", "x"])

    def test_build_agent_argv_preserves_explicit_repo_root(self) -> None:
        argv = agi_code.build_agent_argv(["--repo-root", "/tmp/project", "implement", "x"], Path("/ignored"))

        self.assertEqual(argv[2:], ["--repo-root", "/tmp/project", "implement", "x"])

    def test_build_agent_argv_preserves_equals_style_repo_root(self) -> None:
        argv = agi_code.build_agent_argv(["--repo-root=/tmp/project", "implement", "x"], Path("/ignored"))

        self.assertEqual(argv[2:], ["--repo-root=/tmp/project", "implement", "x"])


if __name__ == "__main__":
    unittest.main()
