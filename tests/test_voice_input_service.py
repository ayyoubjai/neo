from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice_daemon.input_service import VoiceInputService


class _FakeStream:
    def __init__(self, device):
        self.device = device


class _FakeSoundDevice:
    def __init__(self) -> None:
        self.calls = []
        self.devices = [
            {"name": "Output", "max_input_channels": 0, "hostapi": 0},
            {"name": "Mic One", "max_input_channels": 1, "hostapi": 0},
        ]

    def query_devices(self, device=None):
        if device is None:
            return self.devices
        return self.devices[device]

    def query_hostapis(self, index):
        return {"name": f"Host {index}"}

    def RawInputStream(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError("primary device failed")
        return _FakeStream(kwargs.get("device"))


class _FakeOpenWakeWord:
    __file__ = "/tmp/fake_openwakeword/__init__.py"


class VoiceInputServiceTests(unittest.TestCase):
    def test_open_input_stream_falls_back_to_first_available_input_device(self) -> None:
        sd = _FakeSoundDevice()
        service = VoiceInputService({"input_device": "missing mic"}, debug_audio=False, audio_debug_log_path=None)

        stream = service.open_input_stream(sd, sample_rate=16000, frame_samples=480, callback=lambda *args: None)

        self.assertIsNotNone(stream)
        self.assertEqual(stream.device, 1)
        self.assertEqual(sd.calls[1]["device"], 1)

    def test_predict_wake_falls_back_to_float_input_and_triggers(self) -> None:
        class _FakeWakeModel:
            def predict(self, value):
                if value == "int16":
                    raise RuntimeError("bad dtype")
                return {"atlas": [0.2, 0.8]}

        service = VoiceInputService({}, debug_audio=False, audio_debug_log_path=None)

        pred = service.predict_wake(_FakeWakeModel(), "int16", "float32", wake_use_float=None, threshold=0.7)

        self.assertEqual(pred, {"triggered": True, "name": "atlas", "use_float": True})

    def test_init_wake_model_prefers_named_model_init_when_supported(self) -> None:
        calls = []

        class _FakeWakeModel:
            def __init__(self, wakeword_models=None, inference_framework=None):
                calls.append(
                    {
                        "wakeword_models": wakeword_models,
                        "inference_framework": inference_framework,
                    }
                )

        service = VoiceInputService({}, debug_audio=False, audio_debug_log_path=None)

        model = service.init_wake_model(_FakeOpenWakeWord(), _FakeWakeModel, ["atlas"])

        self.assertIsInstance(model, _FakeWakeModel)
        self.assertEqual(
            calls[0],
            {
                "wakeword_models": ["atlas"],
                "inference_framework": "onnx",
            },
        )


if __name__ == "__main__":
    unittest.main()
