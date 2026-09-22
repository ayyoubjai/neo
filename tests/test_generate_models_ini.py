import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_models_ini.py"
SPEC = importlib.util.spec_from_file_location("generate_models_ini", SCRIPT)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class GenerateModelsIniTests(unittest.TestCase):
    def test_memory_policy_bounds_context_and_keeps_projector(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'models.ini'
            generator.write_ini({'models': {
                'models_dir': '/models', 'text_model': 'text.gguf',
                'embedding_model': 'embed.gguf', 'vision_model': 'vision.gguf',
                'vision_mmproj': 'mmproj.gguf', 'serialize_model_requests': True,
                'model_context_size': 8192,
            }}, output)
            config = generator.configparser.ConfigParser()
            config.read_string('[DEFAULT]\n' + output.read_text())
            self.assertEqual(config['text.gguf']['parallel'], '1')
            self.assertEqual(config['text.gguf']['ctx-size'], '8192')
            self.assertEqual(config['vision.gguf']['mmproj'], '/models/mmproj.gguf')
            self.assertEqual(config['vision.gguf']['image-min-tokens'], '1024')
            self.assertEqual(config['embed.gguf']['device'], 'none')
            self.assertEqual(config['embed.gguf']['n-gpu-layers'], '0')

    def test_preserves_projector_and_context_across_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'models.ini'
            output.write_text('version = 1\n\n[vision.gguf]\nmodel = /models/vision.gguf\nmmproj = /models/mmproj.gguf\nctx-size = 8192\n')
            settings = {'models': {'models_dir': '/models', 'vision_model': 'vision.gguf'}}
            generator.write_ini(settings, output)
            first = output.read_text()
            generator.write_ini(settings, output)
            self.assertEqual(first, output.read_text())
            self.assertIn('mmproj = /models/mmproj.gguf', first)
            self.assertIn('ctx-size = 8192', first)

    def test_explicit_projector_survives_fresh_generation_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'vision.gguf').touch()
            (root / 'mmproj.gguf').touch()
            output = root / 'models.ini'
            settings = {'models': {'models_dir': tmp, 'vision_model': 'vision', 'vision_mmproj': 'mmproj.gguf'}}
            generator.write_ini(settings, output)
            first = output.read_text()
            self.assertEqual(first.count(f'mmproj = {root / "mmproj.gguf"}'), 2)
            self.assertNotIn('[mmproj.gguf]', first)
            settings['models']['vision_mmproj'] = '/other/projector.gguf'
            generator.write_ini(settings, output)
            self.assertEqual(output.read_text().count('mmproj = /other/projector.gguf'), 2)

    def test_writes_llama_server_presets_from_selected_models(self) -> None:
        settings = {
            "models": {
                "models_dir": "/models",
                "embedding_model": "embed.gguf",
                "router_model": "fast.gguf",
                "text_model": "reasoning.gguf",
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "models.ini"
            # Call the pure writer directly; load_settings is covered by the
            # setup flow and this test focuses on the INI contract.
            generator.write_ini(settings, output)

            content = output.read_text(encoding="utf-8")
            self.assertIn("version = 1", content)
            self.assertIn("[embed.gguf]\nmodel = /models/embed.gguf\nembeddings = true", content)
            self.assertIn("[fast.gguf]\nmodel = /models/fast.gguf", content)
            self.assertIn("[reasoning.gguf]\nmodel = /models/reasoning.gguf", content)
            self.assertNotIn("embedding_model =", content)
            self.assertNotIn("router_model =", content)
            self.assertNotIn("type =", content)


if __name__ == "__main__":
    unittest.main()
