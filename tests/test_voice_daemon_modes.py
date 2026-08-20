from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_daemon.main import VoiceDaemon


class VoiceDaemonModeTests(unittest.TestCase):
    def _daemon(self) -> VoiceDaemon:
        daemon = VoiceDaemon.__new__(VoiceDaemon)
        daemon._interface_cfg = {"senses": ["audio", "vision"]}
        return daemon

    def test_text_mode_resolves_to_text(self) -> None:
        daemon = self._daemon()
        self.assertEqual(daemon._resolve_requested_senses("text", None), ["text"])

    def test_local_mode_reads_configured_senses(self) -> None:
        daemon = self._daemon()
        self.assertEqual(daemon._resolve_requested_senses("local", None), ["audio", "vision"])

    def test_invalid_mode_is_rejected(self) -> None:
        daemon = self._daemon()
        with self.assertRaisesRegex(ValueError, "Unsupported local mode"):
            daemon._resolve_requested_senses("cmd", None)

    def test_text_and_audio_cannot_share_stdin(self) -> None:
        daemon = self._daemon()
        with self.assertRaisesRegex(ValueError, "cannot be active together"):
            daemon._resolve_requested_senses("local", ["text", "audio"])

    def test_text_interface_disables_tts_in_effective_voice_config(self) -> None:
        daemon = self._daemon()

        effective = daemon._build_effective_voice_cfg(
            {
                "tts_enabled": True,
                "permission_prompt_tts_enabled": True,
                "sound_enabled": True,
            },
            "text",
        )

        self.assertFalse(effective["tts_enabled"])
        self.assertFalse(effective["permission_prompt_tts_enabled"])
        self.assertTrue(effective["sound_enabled"])


if __name__ == "__main__":
    unittest.main()
