from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP_PATH = REPO_ROOT / "scripts" / "setup_release.py"
SPEC = importlib.util.spec_from_file_location("setup_release", SETUP_PATH)
assert SPEC and SPEC.loader
setup_release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup_release
SPEC.loader.exec_module(setup_release)


class ReleaseSetupTests(unittest.TestCase):
    def test_llama_swap_config_keeps_relative_vision_paths(self) -> None:
        config = setup_release._llama_swap_config(
            llama_server="llama-server",
            model_dir=Path("/models"),
            fast_model="fast.gguf",
            reasoning_model="reasoning.gguf",
            embedding_model="embed.gguf",
            vision_model="vision.gguf",
            vision_model_path="vision/model.gguf",
            vision_mmproj="mmproj.gguf",
            vision_mmproj_path="vision/mmproj.gguf",
            gpu_layers=999,
            context_size=8192,
        )
        self.assertIn("/models/vision/model.gguf", config)
        self.assertIn("/models/vision/mmproj.gguf", config)

    def test_env_update_preserves_existing_searxng_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / ".env.example"
            environment = root / ".env"
            template.write_text("SEARXNG_SECRET=\nTELEGRAM_BOT_TOKEN=\n", encoding="utf-8")
            environment.write_text("SEARXNG_SECRET=keep-this-secret\nTELEGRAM_BOT_TOKEN=\n", encoding="utf-8")
            with patch.object(setup_release, "ENV_TEMPLATE", template), patch.object(setup_release, "ENV_LOCAL", environment):
                setup_release._ensure_env({"TELEGRAM_BOT_TOKEN": "hidden-token"})
            contents = environment.read_text(encoding="utf-8")
            self.assertIn("SEARXNG_SECRET=keep-this-secret", contents)
            self.assertIn("TELEGRAM_BOT_TOKEN=hidden-token", contents)
