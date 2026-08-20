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

from adaptive_agent.device_probe import DeviceProfile, classify_device
from adaptive_agent.gene_catalog import Gene
from adaptive_agent.materializer import bundle_ids_for_profile, materialize_profile, pack_dna
from adaptive_agent.purpose import infer_purpose
from adaptive_agent.resolver import PurposeProfile, resolve_genes
from adaptive_agent.runner import build_llama_swap_command, build_llamacpp_command, build_model_server_command
from adaptive_agent.settings_apply import apply_profile_settings
from adaptive_agent.wizard import build_profile


def _device(device_class: str, *, android: bool = False, termux: bool = False, ram_mb: int = 8000) -> DeviceProfile:
    return DeviceProfile(
        os_name="posix",
        platform_system="linux",
        machine="x86_64",
        cpu_count=8,
        ram_mb=ram_mb,
        storage_free_mb=20000,
        is_termux=termux,
        is_android=android,
        device_class=device_class,
    )


class AdaptiveAgentTests(unittest.TestCase):
    def test_classify_android_phone_by_ram(self) -> None:
        self.assertEqual(
            classify_device(is_android=True, ram_mb=2500, cpu_count=8, storage_free_mb=10000),
            "phone_tiny",
        )
        self.assertEqual(
            classify_device(is_android=True, ram_mb=5000, cpu_count=8, storage_free_mb=10000),
            "phone_small",
        )

    def test_resolver_selects_best_matching_model_and_tools(self) -> None:
        genes = [
            Gene("cog.system2", "cognition", 30, ["coding"], {"device_classes": ["pc_medium"]}, {"mode": "system2"}),
            Gene("model.small", "model", 10, ["coding"], {"device_classes": ["pc_small"]}, {"model_path": "a.gguf"}),
            Gene("model.medium", "model", 20, ["coding"], {"device_classes": ["pc_medium"]}, {"model_path": "b.gguf"}),
            Gene("tool.notes", "tool", 5, ["*"], {"device_classes": ["*"]}, {}),
            Gene("tool.phone", "tool", 50, ["coding"], {"android": True}, {}),
        ]
        plan = resolve_genes(genes, _device("pc_medium"), PurposeProfile("coding", "direct"))
        self.assertEqual(plan.cognition["id"], "cog.system2")
        self.assertEqual(plan.model["id"], "model.medium")
        self.assertEqual([item["id"] for item in plan.tools], ["tool.notes"])
        self.assertIn("android required", [item["reason"] for item in plan.disabled])

    def test_resolver_can_force_cognition_mode_when_available(self) -> None:
        genes = [
            Gene("cog.system1", "cognition", 10, ["coding"], {"device_classes": ["pc_medium"]}, {"mode": "system1"}),
            Gene("cog.system2", "cognition", 20, ["coding"], {"device_classes": ["pc_medium"]}, {"mode": "system2"}),
        ]
        plan = resolve_genes(
            genes,
            _device("pc_medium"),
            PurposeProfile("coding", "direct", cognition_mode="system1"),
        )
        self.assertEqual(plan.cognition["id"], "cog.system1")

    def test_network_and_background_require_explicit_profile_flags(self) -> None:
        genes = [
            Gene("tool.net", "tool", 10, ["research"], {"network": True}, {}),
            Gene("service.bg", "service", 10, ["research"], {"background": True}, {}),
        ]
        blocked = resolve_genes(genes, _device("pc_medium"), PurposeProfile("research", "direct"))
        self.assertEqual(blocked.tools, [])
        self.assertEqual(blocked.services, [])

        allowed = resolve_genes(
            genes,
            _device("pc_medium"),
            PurposeProfile("research", "direct", allow_network=True, allow_background=True),
        )
        self.assertEqual([item["id"] for item in allowed.tools], ["tool.net"])
        self.assertEqual([item["id"] for item in allowed.services], ["service.bg"])

    def test_runner_builds_llamacpp_command(self) -> None:
        command = build_llamacpp_command(
            {
                "model": {
                    "payload": {
                        "model_path": "models/test.gguf",
                        "server_binary": "llama-server",
                        "options": {"ctx_size": 2048, "threads": 2, "port": 8081},
                    }
                }
            }
        )
        self.assertEqual(
            command,
            ["llama-server", "-m", "models/test.gguf", "-c", "2048", "-t", "2", "--port", "8081"],
        )

    def test_runner_builds_llama_swap_command_for_service_gene(self) -> None:
        profile = {
            "services": [
                {
                    "id": "service.model.llama_swap",
                    "payload": {
                        "server_kind": "llama_swap",
                        "binary": "llama-swap",
                        "config_path": "config/config.yaml",
                        "listen": "localhost:8080",
                    },
                }
            ]
        }
        self.assertEqual(
            build_llama_swap_command(profile),
            ["llama-swap", "--config", "config/config.yaml", "--listen", "localhost:8080"],
        )
        self.assertEqual(build_model_server_command(profile)[0], "llama-swap")

    def test_infer_purpose_from_free_form_goal(self) -> None:
        self.assertEqual(infer_purpose("run on my phone in termux"), "phone_companion")
        self.assertEqual(infer_purpose("debug and edit python repositories"), "coding")
        self.assertEqual(infer_purpose("search papers and summarize knowledge"), "research")

    def test_build_profile_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "profile.json"
            data = build_profile(purpose="assistant", personality="concise", output_path=output)
            self.assertTrue(output.exists())
            self.assertEqual(data["purpose"]["purpose"], "assistant")
            self.assertIn("device_class", data["device"])

    def test_bundle_ids_are_collected_from_selected_genes(self) -> None:
        profile = {
            "cognition": {"payload": {"bundles": ["cognition_basic"]}},
            "model": {"payload": {"bundles": ["model_runtime"]}},
            "tools": [{"payload": {"bundles": ["tool_runtime_basic"]}}],
            "services": [{"payload": {"bundles": ["llama_swap"]}}],
            "prompts": [{"payload": {"bundles": ["prompts"]}}],
        }
        self.assertEqual(
            bundle_ids_for_profile(profile),
            ["core", "model_runtime", "cognition_basic", "tool_runtime_basic", "llama_swap", "prompts"],
        )

    def test_apply_profile_settings_merges_cognition_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = Path(tmpdir) / "profile.json"
            settings = Path(tmpdir) / "settings.json"
            profile.write_text(
                json.dumps(
                    {
                        "device": {"device_class": "pc_small"},
                        "purpose": {"purpose": "coding"},
                        "cognition": {
                            "id": "cognition.system0_system1",
                            "payload": {
                                "mode": "system1",
                                "systems_enabled": ["SYSTEM0", "SYSTEM1"],
                                "settings_patch": {
                                    "orchestrator": {"cognition_system3_enabled": False},
                                    "autonomy": {"requested_mode": "SYSTEM1"},
                                },
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            settings.write_text('{"orchestrator": {}, "autonomy": {}}\n', encoding="utf-8")
            applied = apply_profile_settings(profile_path=profile, settings_path=settings)
            saved = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual(applied["cognition_mode"], "system1")
            self.assertEqual(saved["autonomy"]["requested_mode"], "SYSTEM1")
            self.assertFalse(saved["orchestrator"]["cognition_system3_enabled"])

    def test_materialize_profile_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            root.mkdir()
            (root / "src").mkdir()
            (root / "src" / "core.py").write_text("x = 1\n", encoding="utf-8")
            profile = Path(tmpdir) / "profile.json"
            profile.write_text(
                '{"model": {"payload": {"bundles": ["tiny"]}}, "tools": [], "services": [], "prompts": []}',
                encoding="utf-8",
            )
            bundles = Path(tmpdir) / "bundles.json"
            bundles.write_text(
                '{"bundles": [{"id": "core", "paths": []}, {"id": "tiny", "paths": ["src/**"]}]}',
                encoding="utf-8",
            )
            target = Path(tmpdir) / "cell"
            result = materialize_profile(
                profile_path=profile,
                target_path=target,
                bundle_path=bundles,
                source_path=root,
            )
            self.assertEqual(result.files_written, 1)
            self.assertTrue((target / "src" / "core.py").exists())

    def test_pack_dna_writes_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            root.mkdir()
            (root / "decoder.py").write_text("print('decode')\n", encoding="utf-8")
            bundles = Path(tmpdir) / "bundles.json"
            bundles.write_text('{"bundles": [{"id": "core", "paths": ["decoder.py"]}]}', encoding="utf-8")
            output = Path(tmpdir) / "dna.zip"
            pack_dna(output_path=output, repo_root=root, bundle_path=bundles)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
