from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_daemon.output_manager import VoiceOutputManager


class VoiceOutputManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_tts_primes_echo_filter(self) -> None:
        manager = VoiceOutputManager(
            {
                "tts_enabled": True,
                "echo_filter_enabled": True,
            }
        )
        manager._tts_queue = asyncio.Queue()

        await manager.enqueue_tts("Hello 2")

        queued = await asyncio.wait_for(manager._tts_queue.get(), timeout=1.0)
        self.assertEqual(queued, "Hello 2")
        self.assertIsNone(manager.filter_echo_text("hello two", segment_id=1))

    def test_should_suppress_capture_tracks_active_and_tail_window(self) -> None:
        manager = VoiceOutputManager(
            {
                "suppress_mic_during_tts": True,
                "tts_suppress_ms": 500,
            }
        )

        manager._tts_active = True
        self.assertTrue(manager.should_suppress_capture(10.0))

        manager._tts_active = False
        manager._tts_suppress_until = 10.5
        self.assertTrue(manager.should_suppress_capture(10.25))
        self.assertFalse(manager.should_suppress_capture(10.75))

    async def test_activation_and_sleep_notifications_enqueue_debounced_sounds(self) -> None:
        manager = VoiceOutputManager({"sound_enabled": True})
        manager._sound_queue = asyncio.Queue()

        manager.notify_activation(10.0)
        manager.notify_activation(10.1)
        manager.notify_sleep(11.0)

        self.assertEqual(await asyncio.wait_for(manager._sound_queue.get(), timeout=1.0), "activate")
        self.assertEqual(await asyncio.wait_for(manager._sound_queue.get(), timeout=1.0), "sleep")
        self.assertTrue(manager._sound_queue.empty())


if __name__ == "__main__":
    unittest.main()
