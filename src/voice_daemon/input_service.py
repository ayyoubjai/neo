import inspect
import os
import time
from typing import Any, Dict, List, Mapping, Optional

from common.jsonlog import append_jsonl


class VoiceInputService:
    def __init__(
        self,
        voice_cfg: Mapping[str, Any],
        *,
        debug_audio: bool,
        audio_debug_log_path: Optional[str],
    ):
        self._voice_cfg = voice_cfg
        self._debug_audio = debug_audio
        self._audio_debug_log_path = audio_debug_log_path

    def init_wake_model(self, openwakeword, wake_model_cls, wake_models: List[Any]):
        wake_models = wake_models or []
        params = inspect.signature(wake_model_cls.__init__).parameters
        supports_names = "wakeword_models" in params
        supports_paths = "wakeword_model_paths" in params
        supports_framework = "inference_framework" in params

        name_list: List[str] = []
        path_list: List[str] = []
        for item in wake_models:
            if isinstance(item, str) and os.path.isfile(item):
                path_list.append(item)
            elif isinstance(item, str):
                name_list.append(item)

        if name_list and supports_names:
            model = self._build_wake_model(
                wake_model_cls, supports_names, supports_paths, supports_framework, name_list, False
            )
            if model is not None:
                return model

        onnx_map = self._discover_wakeword_model_paths(openwakeword)
        if onnx_map and name_list:
            for name in list(name_list):
                normalized = name.replace(" ", "_")
                if normalized in onnx_map:
                    path_list.append(onnx_map[normalized])
                    name_list.remove(name)
                elif name in onnx_map:
                    path_list.append(onnx_map[name])
                    name_list.remove(name)
                else:
                    prefix = self._find_prefix_match(onnx_map, normalized)
                    if prefix:
                        path_list.append(onnx_map[prefix])
                        name_list.remove(name)

        for name in name_list:
            print(f"[voice] Unknown wakeword model: {name}")

        if not onnx_map and (supports_paths or supports_names):
            print("[voice] ONNX models not found. Run openwakeword.utils.download_models().")

        if path_list:
            model = self._build_wake_model(
                wake_model_cls, supports_names, supports_paths, supports_framework, path_list, True
            )
            if model is not None:
                return model

        if not wake_models and onnx_map:
            all_paths = list(onnx_map.values())
            model = self._build_wake_model(
                wake_model_cls, supports_names, supports_paths, supports_framework, all_paths, True
            )
            if model is not None:
                return model

        get_pretrained = getattr(openwakeword, "get_pretrained_model_paths", None)
        if not wake_models and callable(get_pretrained):
            model = self._build_wake_model(
                wake_model_cls, supports_names, supports_paths, supports_framework, get_pretrained(), True
            )
            if model is not None:
                return model

        model = self._build_wake_model(wake_model_cls, supports_names, supports_paths, supports_framework, [], False)
        if model is not None:
            return model

        return None

    def open_input_stream(self, sd, sample_rate: int, frame_samples: int, callback):
        device_setting = self._voice_cfg.get("input_device")
        device = None
        if isinstance(device_setting, int):
            device = device_setting
        elif isinstance(device_setting, str) and device_setting.strip():
            device = self._find_device_by_name(sd, device_setting.strip())
            if device is None:
                print(f"[voice] Input device name not found: {device_setting}")
        try:
            return sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=frame_samples,
                dtype="int16",
                channels=1,
                callback=callback,
                device=device,
            )
        except Exception as e:
            fallback = self._first_input_device(sd)
            if fallback is None or fallback == device:
                print(f"[voice] Failed to open input device: {e}")
                self._print_input_devices(sd)
                return None
            try:
                print(f"[voice] Falling back to input device {fallback}")
                return sd.RawInputStream(
                    samplerate=sample_rate,
                    blocksize=frame_samples,
                    dtype="int16",
                    channels=1,
                    callback=callback,
                    device=fallback,
                )
            except Exception as fallback_error:
                print(f"[voice] Failed to open fallback device: {fallback_error}")
                self._print_input_devices(sd)
                return None

    def predict_wake(self, wake_model, audio_int16, audio_float, wake_use_float: Optional[bool], threshold: float):
        try:
            if wake_use_float is None:
                try:
                    preds = wake_model.predict(audio_int16)
                    use_float = False
                except Exception:
                    preds = wake_model.predict(audio_float)
                    use_float = True
            elif wake_use_float:
                preds = wake_model.predict(audio_float)
                use_float = True
            else:
                preds = wake_model.predict(audio_int16)
                use_float = False
        except Exception:
            return None

        triggered = False
        trigger_name = None
        if isinstance(preds, dict):
            for name, value in preds.items():
                score = self._wake_score(value)
                if score >= threshold:
                    triggered = True
                    trigger_name = name
                    break
        return {"triggered": triggered, "name": trigger_name, "use_float": use_float}

    def log_audio_debug(self, avg_level: float, is_speech: bool, capture_active: bool) -> None:
        if not self._debug_audio:
            return
        msg = "[voice] audio level={:.4f} speech={} capturing={}".format(avg_level, is_speech, capture_active)
        print(msg, flush=True)
        if not self._audio_debug_log_path:
            return
        append_jsonl(
            self._audio_debug_log_path,
            {
                "ts": time.time(),
                "avg_level": avg_level,
                "speech": is_speech,
                "capturing": capture_active,
            },
        )

    def print_stream_info(self, sd, stream) -> None:
        try:
            device = stream.device
            if isinstance(device, (list, tuple)):
                device = device[0]
            info = sd.query_devices(device)
            print(
                "[voice] Using input device: {name} ({host})".format(
                    name=info.get("name"),
                    host=sd.query_hostapis(info.get("hostapi", 0)).get("name"),
                )
            )
        except Exception:
            pass

    def _build_wake_model(
        self,
        wake_model_cls,
        supports_names: bool,
        supports_paths: bool,
        supports_framework: bool,
        items: List[str],
        items_are_paths: bool,
    ):
        if items_are_paths and supports_paths:
            try:
                if supports_framework:
                    return wake_model_cls(wakeword_model_paths=items, inference_framework="onnx")
                return wake_model_cls(wakeword_model_paths=items)
            except Exception as e:
                print(f"[voice] Wake model init failed (paths): {e}")
        if not items_are_paths and supports_names:
            try:
                if supports_framework:
                    return wake_model_cls(wakeword_models=items, inference_framework="onnx")
                return wake_model_cls(wakeword_models=items)
            except Exception as e:
                print(f"[voice] Wake model init failed (names): {e}")
        try:
            if supports_framework:
                return wake_model_cls(inference_framework="onnx")
            return wake_model_cls()
        except Exception as e:
            print(f"[voice] Wake model init failed (fallback): {e}")
        return None

    def _discover_wakeword_model_paths(self, openwakeword) -> Dict[str, str]:
        pkg_path = getattr(openwakeword, "__file__", None)
        if not pkg_path:
            return {}
        base_dir = os.path.dirname(pkg_path)
        models_dir = os.path.join(base_dir, "resources", "models")
        if not os.path.isdir(models_dir):
            return {}

        onnx_map: Dict[str, str] = {}
        for filename in os.listdir(models_dir):
            if filename.endswith(".onnx"):
                name = filename[: -len(".onnx")]
                path = os.path.join(models_dir, filename)
                onnx_map[name] = path
                alias = self._strip_version_suffix(name)
                if alias and alias not in onnx_map:
                    onnx_map[alias] = path
        return onnx_map

    def _strip_version_suffix(self, name: str) -> Optional[str]:
        for marker in ("_v", "-v"):
            idx = name.find(marker)
            if idx > 0:
                return name[:idx]
        return None

    def _find_prefix_match(self, mapping: Dict[str, str], name: str) -> Optional[str]:
        for key in mapping.keys():
            if key.startswith(name + "_") or key.startswith(name + "-") or key.startswith(name + "."):
                return key
        return None

    def _first_input_device(self, sd) -> Optional[int]:
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        for idx, info in enumerate(devices):
            if info.get("max_input_channels", 0) > 0:
                return idx
        return None

    def _find_device_by_name(self, sd, name: str) -> Optional[int]:
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        target = name.lower()
        for idx, info in enumerate(devices):
            if target in str(info.get("name", "")).lower():
                return idx
        return None

    def _print_input_devices(self, sd) -> None:
        try:
            devices = sd.query_devices()
        except Exception:
            print("[voice] Unable to query audio devices.")
            return
        print("[voice] Available input devices:")
        for idx, info in enumerate(devices):
            if info.get("max_input_channels", 0) > 0:
                print(f"  [{idx}] {info.get('name')} ({info.get('max_input_channels')} ch)")

    def _wake_score(self, value: Any) -> float:
        try:
            if hasattr(value, "max"):
                return float(value.max())
            if isinstance(value, (list, tuple)):
                return float(max(value)) if value else 0.0
            return float(value)
        except Exception:
            return 0.0
