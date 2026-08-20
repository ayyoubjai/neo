from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model_server.image_embed_model import embed_image


class ImageEmbedModelTests(unittest.TestCase):
    def test_embed_image_is_deterministic_for_same_input(self) -> None:
        image_b64 = base64.b64encode(b"object-red").decode("ascii")

        first_vec, first_model = embed_image(image_b64=image_b64)
        second_vec, second_model = embed_image(image_b64=image_b64)

        self.assertEqual(first_vec, second_vec)
        self.assertEqual(first_model, second_model)
        self.assertTrue(first_model)

    def test_embed_image_distinguishes_different_inputs(self) -> None:
        red_b64 = base64.b64encode(b"object-red").decode("ascii")
        blue_b64 = base64.b64encode(b"object-blue").decode("ascii")

        red_vec, red_model = embed_image(image_b64=red_b64)
        blue_vec, blue_model = embed_image(image_b64=blue_b64)

        self.assertNotEqual(red_vec, blue_vec)
        self.assertEqual(red_model, blue_model)

    def test_embed_image_reports_load_error_for_invalid_base64(self) -> None:
        vec, model = embed_image(image_b64="%%%")

        self.assertEqual(vec, [])
        self.assertEqual(model, "image-load-error")

    def test_embed_image_can_load_from_file_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "crop.jpg"
            image_path.write_bytes(b"object-red-file")

            vec, model = embed_image(image_ref=str(image_path))

            self.assertTrue(vec)
            self.assertTrue(model)


if __name__ == "__main__":
    unittest.main()
