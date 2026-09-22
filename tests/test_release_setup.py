from __future__ import annotations

import importlib.util
import json
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
    def test_google_setup_choices_validate_before_writing(self) -> None:
        self.assertFalse(setup_release._google_settings("skip", "")["enabled"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = root / "desktop.json"
            payload = {"installed": {"client_id": "test.apps.googleusercontent.com",
                       "client_secret": "test", "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                       "token_uri": "https://oauth2.googleapis.com/token"}}
            client.write_text(json.dumps(payload))
            cfg = setup_release._google_settings("own", str(client))
            self.assertTrue(cfg["enabled"])
            self.assertFalse(cfg["auto_google_authorize"])
            with patch.object(setup_release, "ROOT", root):
                with self.assertRaises(ValueError):
                    setup_release._google_settings("neo", "")
                bundled = root / "config/google_oauth/neo_desktop.json"
                bundled.parent.mkdir(parents=True)
                bundled.write_text(json.dumps(payload))
                self.assertEqual(setup_release._google_settings("neo", "")["oauth_client_source"], "neo")
            payload["installed"]["token_uri"] = "https://example.com/token"
            client.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                setup_release._google_settings("own", str(client))

    def test_firewall_plan_rejects_unsafe_or_ambiguous_inputs(self) -> None:
        for bridge, uplink, subnet in [
            ('br0"; flush ruleset', "eth0", "172.18.0.0/16"),
            ("", "eth0", "172.18.0.0/16"),
            ("br0", "br0", "172.18.0.0/16"),
            ("br0", "eth0", "0.0.0.0/0"),
            ("br0", "eth0", "172.18.0.2/16"),
        ]:
            with self.subTest(bridge=bridge, uplink=uplink, subnet=subnet):
                with self.assertRaises(ValueError):
                    setup_release._firewall_plan(bridge, uplink, subnet)

    def test_server_executable_normalizes_paths_but_preserves_commands(self) -> None:
        self.assertEqual(setup_release._server_executable("llama-server"), "llama-server")
        self.assertEqual(
            setup_release._server_executable('"./llama build/bin/llama-server"'),
            str(Path("./llama build/bin/llama-server").resolve()),
        )
        self.assertEqual(
            setup_release._server_executable("~/llama.cpp/build/bin/llama-server"),
            str(Path.home() / "llama.cpp/build/bin/llama-server"),
        )
        with self.assertRaises(ValueError):
            setup_release._server_executable(" ")

    def test_llama_router_config_keeps_relative_vision_paths(self) -> None:
        config = setup_release._llama_router_config(
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

    def test_llamarouter_provider_and_local_config_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "config" / "settings.example.json").write_text('{"interface": {"mode": "text"}, "models": {}, "orchestrator": {}}\n', encoding="utf-8")
            (root / ".env.example").write_text("SEARXNG_SECRET=\n", encoding="utf-8")

            with (
                patch.object(setup_release, "ROOT", root),
                patch.object(setup_release, "SETTINGS_TEMPLATE", root / "config" / "settings.example.json"),
                patch.object(setup_release, "SETTINGS_LOCAL", root / "config" / "settings.local.json"),
                patch.object(setup_release, "LLAMA_ROUTER_LOCAL", root / "config" / "llama-router.local.yaml"),
                patch.object(setup_release, "ENV_TEMPLATE", root / ".env.example"),
                patch.object(setup_release, "ENV_LOCAL", root / ".env"),
            ):
                exit_code = setup_release.main(
                    [
                        "--provider",
                        "llama-router",
                        "--model-dir",
                        str(root / "models"),
                        "--fast-model",
                        "fast.gguf",
                        "--reasoning-model",
                        "reasoning.gguf",
                        "--embedding-model",
                        "embed.gguf",
                        "--llama-server",
                        str(root / "llama build" / "llama-server"),
                    ]
                )
                self.assertEqual(exit_code, 0)
                # settings.local.json should be created and record the models dir
                self.assertTrue((root / "config" / "settings.local.json").exists())
                settings = json.loads((root / "config" / "settings.local.json").read_text(encoding="utf-8"))
                self.assertEqual(settings.get("models", {}).get("models_dir"), str((root / "models").resolve()))
                self.assertEqual(settings["models"]["llama_server"], str(root / "llama build" / "llama-server"))
                self.assertFalse((root / "config/firewall.local.nft").exists())
                with patch("subprocess.run") as run, patch("builtins.print") as output:
                    exit_code = setup_release.main([
                        "--provider", "ollama", "--force", "--prepare-firewall-plan",
                        "--firewall-bridge", "br-test", "--firewall-uplink", "eth0",
                        "--firewall-subnet", "172.18.0.0/16",
                    ])
                self.assertEqual(exit_code, 0)
                run.assert_not_called()
                plan = (root / "config/firewall.local.nft").read_text()
                self.assertIn('iifname "br-test" oifname "eth0" ip saddr 172.18.0.0/16', plan)
                self.assertIn('tcp dport { 80, 443 }', plan)
                self.assertIn('ct state { established, related }', plan)
                statements = [line for line in plan.splitlines() if not line.startswith("#")]
                self.assertEqual(len(statements), 2)
                self.assertTrue(all(line.startswith("add rule inet filter forward ") for line in statements))
                printed = "\n".join(str(call.args[0]) for call in output.call_args_list)
                self.assertIn("NOT APPLIED", printed)
                self.assertIn("public IP address", printed)
                self.assertIn("If you approve these exact changes", printed)

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
