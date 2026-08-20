import asyncio
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from typing import Any, Dict, List, Mapping, Optional

from common.jsonlog import append_jsonl


class VoiceOutputManager:
    _ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    _TEENS = (
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    _TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    _ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    _SYMBOL_WORDS = {
        "@": "alt",
        "&": "and",
        "#": "hash",
        "%": "percent",
        "$": "dollar",
        "+": "plus",
        "=": "equals",
        "/": "slash",
        "\\": "backslash",
        "*": "star",
    }

    def __init__(self, voice_cfg: Mapping[str, Any]):
        self._voice_cfg = voice_cfg
        self._tts_enabled = bool(voice_cfg.get("tts_enabled", False))
        self._tts_rate = voice_cfg.get("tts_rate")
        self._tts_volume = voice_cfg.get("tts_volume")
        self._tts_voice = voice_cfg.get("tts_voice")
        self._tts_backend_override = voice_cfg.get("tts_backend")
        self._suppress_mic_during_tts = bool(voice_cfg.get("suppress_mic_during_tts", True))
        self._tts_suppress_ms = int(voice_cfg.get("tts_suppress_ms", 500))
        self._permission_prompt_tts_enabled = bool(voice_cfg.get("permission_prompt_tts_enabled", True))
        self._echo_filter_enabled = bool(voice_cfg.get("echo_filter_enabled", True))
        self._echo_filter_window_s = float(voice_cfg.get("echo_filter_window_s", 12.0))
        self._echo_match_threshold = float(voice_cfg.get("echo_match_threshold", 0.85))
        self._echo_log_path = voice_cfg.get("echo_log_path")
        self._log_echo = bool(self._echo_log_path)
        self._sound_enabled = bool(voice_cfg.get("sound_enabled", False))
        self._sound_backend_override = voice_cfg.get("sound_backend")
        self._sound_activation_freq = float(voice_cfg.get("activation_sound_freq", 880.0))
        self._sound_sleep_freq = float(voice_cfg.get("sleep_sound_freq", 440.0))
        self._sound_duration_ms = int(voice_cfg.get("sound_duration_ms", 180))
        self._sound_volume = float(voice_cfg.get("sound_volume", 0.2))
        self._sound_output_device = voice_cfg.get("sound_output_device")

        self._sd = None
        self._np = None
        self._tts_queue: Optional[asyncio.Queue] = None
        self._sound_queue: Optional[asyncio.Queue] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._sound_task: Optional[asyncio.Task] = None
        self._tts_backend: Optional[str] = None
        self._tts_warned = False
        self._sound_warned = False
        self._last_activation_sound = 0.0
        self._last_sleep_sound = 0.0
        self._sound_debounce_s = 0.5
        self._tts_active = False
        self._tts_suppress_until = 0.0
        self._last_tts_time = 0.0
        self._tts_engine = None
        self._tts_process: Optional[subprocess.Popen] = None
        self._echo_active = False
        self._echo_index = 0
        self._echo_tts_norm: List[str] = []
        self._last_user_text = ""
        self._last_user_time = 0.0

        if self._log_echo and self._echo_log_path:
            os.makedirs(os.path.dirname(self._echo_log_path), exist_ok=True)

    @property
    def permission_prompt_tts_enabled(self) -> bool:
        return self._permission_prompt_tts_enabled

    def start(self, *, sd=None, np=None) -> None:
        self._sd = sd
        self._np = np
        loop = asyncio.get_running_loop()
        if self._tts_enabled and self._tts_task is None:
            self._tts_queue = asyncio.Queue(maxsize=10)
            self._tts_task = loop.create_task(self._tts_worker())
        if self._sound_enabled and self._sound_task is None:
            self._sound_queue = asyncio.Queue(maxsize=10)
            self._sound_task = loop.create_task(self._sound_worker())

    async def stop(self) -> None:
        tasks = [task for task in (self._tts_task, self._sound_task) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tts_task = None
        self._sound_task = None
        self._tts_queue = None
        self._sound_queue = None
        self._tts_active = False
        self._tts_suppress_until = 0.0
        self._sd = None
        self._np = None

    async def enqueue_tts(self, text: str) -> None:
        if not self._tts_enabled or not self._tts_queue:
            return
        cleaned = text.strip(" \t")
        if not cleaned:
            return
        tts_text = self._sanitize_tts_text(cleaned)
        if not tts_text:
            return
        spoken_text = self.normalize_spoken_text(tts_text)
        if not spoken_text:
            return
        self._last_tts_time = time.monotonic()
        if self._echo_filter_enabled:
            tokens = spoken_text.split()
            self._echo_tts_norm = [token.lower() for token in tokens]
            self._echo_index = 0
            self._echo_active = bool(tokens)
        else:
            self._reset_echo_state()
        self.log_echo_debug(
            "tts_queued",
            {
                "text": cleaned,
                "tts_text": tts_text,
                "spoken_text": spoken_text,
                "tokens": list(self._echo_tts_norm),
                "token_count": len(self._echo_tts_norm),
                "echo_active": self._echo_active,
                "echo_index": self._echo_index,
            },
        )
        try:
            self._tts_queue.put_nowait(tts_text)
        except asyncio.QueueFull:
            pass

    def enqueue_sound(self, kind: str) -> None:
        if not self._sound_enabled or not self._sound_queue:
            return
        try:
            self._sound_queue.put_nowait(kind)
        except asyncio.QueueFull:
            pass

    def notify_activation(self, now: float) -> None:
        if now - self._last_activation_sound < self._sound_debounce_s:
            return
        self._last_activation_sound = now
        self.enqueue_sound("activate")

    def notify_sleep(self, now: float) -> None:
        if now - self._last_sleep_sound < self._sound_debounce_s:
            return
        self._last_sleep_sound = now
        self.enqueue_sound("sleep")

    def should_suppress_capture(self, now: float) -> bool:
        return self._suppress_mic_during_tts and (self._tts_active or now < self._tts_suppress_until)

    def record_user_turn(self, text: str, at: float) -> None:
        self._last_user_text = text
        self._last_user_time = at

    def normalize_spoken_text(self, text: str) -> str:
        if not text:
            return ""
        tokens: List[str] = []
        current: List[str] = []
        current_kind: Optional[str] = None

        def flush_current() -> None:
            nonlocal current_kind
            if not current:
                return
            token = "".join(current)
            if current_kind == "digit":
                tokens.extend(self._digits_to_words(token))
            else:
                tokens.extend(self._normalize_word_token(token))
            current.clear()
            current_kind = None

        for ch in text:
            if ch.isdigit():
                if current_kind != "digit":
                    flush_current()
                    current_kind = "digit"
                current.append(ch)
                continue
            if ch.isalpha():
                if ord(ch) >= 128:
                    greek = self._greek_letter_name(ch)
                    if greek:
                        flush_current()
                        tokens.append(greek)
                        continue
                if current_kind != "alpha":
                    flush_current()
                    current_kind = "alpha"
                current.append(ch)
                continue
            if ch in {",", "_"} and current_kind == "digit":
                continue
            flush_current()
            mapped = self._SYMBOL_WORDS.get(ch)
            if mapped:
                tokens.extend(mapped.split())
        flush_current()
        return " ".join(tokens).strip()

    def log_echo_debug(self, event: str, payload: Dict[str, Any]) -> None:
        if not self._log_echo or not self._echo_log_path:
            return
        append_jsonl(
            self._echo_log_path,
            {
                "ts": time.time(),
                "event": event,
                **payload,
            },
        )

    def filter_echo_text(self, text: str, segment_id: Optional[int] = None) -> Optional[str]:
        if not self._echo_filter_enabled or not self._echo_active:
            return text
        if self._echo_filter_window_s > 0:
            if time.monotonic() - self._last_tts_time > self._echo_filter_window_s:
                self.log_echo_debug(
                    "echo_window_expired",
                    {
                        "segment_id": segment_id,
                        "echo_index": self._echo_index,
                    },
                )
                self._reset_echo_state()
                return text
        stt_tokens = self._tokenize_echo_text(text)
        if not stt_tokens:
            self.log_echo_debug(
                "echo_stt_empty",
                {"segment_id": segment_id, "echo_index": self._echo_index},
            )
            return text
        if not self._echo_tts_norm or self._echo_index >= len(self._echo_tts_norm):
            self.log_echo_debug(
                "echo_no_tts_remaining",
                {"segment_id": segment_id, "echo_index": self._echo_index},
            )
            self._reset_echo_state()
            return text
        stt_norm = [token.lower() for token in stt_tokens]
        remaining = self._echo_tts_norm[self._echo_index:]
        compare_len = min(len(stt_norm), len(remaining))
        if compare_len == 0:
            self.log_echo_debug(
                "echo_no_compare",
                {"segment_id": segment_id, "echo_index": self._echo_index},
            )
            self._reset_echo_state()
            return text
        match_count = 0
        for idx in range(compare_len):
            if stt_norm[idx] == remaining[idx]:
                match_count += 1
        match_ratio = match_count / len(stt_norm)
        if match_ratio < self._echo_match_threshold:
            self.log_echo_debug(
                "echo_deviation",
                {
                    "segment_id": segment_id,
                    "match_ratio": match_ratio,
                    "threshold": self._echo_match_threshold,
                    "echo_index": self._echo_index,
                    "stt_text": text,
                    "stt_tokens": stt_norm,
                    "tts_tokens": remaining[:compare_len],
                },
            )
            self._reset_echo_state()
            return text
        echo_text = text.strip() or " ".join(stt_tokens).strip()
        if echo_text:
            print(f"\n[echo] {echo_text}\n", flush=True)
            self.log_echo_debug(
                "echo_match",
                {
                    "segment_id": segment_id,
                    "match_ratio": match_ratio,
                    "echo_index": self._echo_index,
                    "stt_tokens": stt_norm,
                    "tts_tokens": remaining[:compare_len],
                    "echo_text": echo_text,
                },
            )
        self._echo_index += len(stt_norm)
        if self._echo_index >= len(self._echo_tts_norm):
            self._reset_echo_state()
        return None

    async def _tts_worker(self) -> None:
        if not self._tts_queue:
            return
        while True:
            text = await self._tts_queue.get()
            if text is None:
                break
            self._tts_active = True
            try:
                await asyncio.to_thread(self._speak_blocking, text)
            finally:
                self._tts_active = False
                if self._suppress_mic_during_tts and self._tts_suppress_ms > 0:
                    self._tts_suppress_until = time.monotonic() + (self._tts_suppress_ms / 1000.0)

    async def _sound_worker(self) -> None:
        if not self._sound_queue:
            return
        while True:
            kind = await self._sound_queue.get()
            if kind is None:
                break
            await asyncio.to_thread(self._play_sound_blocking, kind)

    def _resolve_tts_backend(self) -> str:
        if self._tts_backend:
            return self._tts_backend
        override = str(self._tts_backend_override or "").strip().lower()
        if override:
            self._tts_backend = override
            return self._tts_backend
        try:
            import pyttsx3  # noqa: F401

            self._tts_backend = "pyttsx3"
            return self._tts_backend
        except Exception:
            pass
        if sys.platform == "win32":
            if shutil.which("powershell"):
                self._tts_backend = "powershell"
                return self._tts_backend
            if shutil.which("pwsh"):
                self._tts_backend = "pwsh"
                return self._tts_backend
        if sys.platform == "darwin" and shutil.which("say"):
            self._tts_backend = "say"
            return self._tts_backend
        if shutil.which("espeak"):
            self._tts_backend = "espeak"
            return self._tts_backend
        self._tts_backend = "none"
        return self._tts_backend

    def _resolve_sound_backend(self) -> str:
        override = str(self._sound_backend_override or "").strip().lower()
        if override and override != "auto":
            return override
        if sys.platform == "win32":
            return "winsound"
        if self._sd is not None and self._np is not None:
            return "sounddevice"
        return "none"

    def _speak_blocking(self, text: str) -> None:
        if not self._tts_enabled:
            return
        cleaned = text.strip()
        if not cleaned:
            return
        backend = self._resolve_tts_backend()
        if backend == "none":
            return
        try:
            if backend == "pyttsx3":
                import pyttsx3

                engine = pyttsx3.init()
                self._tts_engine = engine
                if self._tts_rate:
                    try:
                        engine.setProperty("rate", int(self._tts_rate))
                    except Exception:
                        pass
                if self._tts_volume is not None:
                    try:
                        engine.setProperty("volume", float(self._tts_volume))
                    except Exception:
                        pass
                if self._tts_voice:
                    try:
                        engine.setProperty("voice", str(self._tts_voice))
                    except Exception:
                        pass
                try:
                    engine.say(cleaned)
                    engine.runAndWait()
                finally:
                    self._tts_engine = None
                return
            if backend in ("powershell", "pwsh"):
                exe = "powershell" if backend == "powershell" else "pwsh"
                cmd = [
                    exe,
                    "-NoProfile",
                    "-Command",
                    "Add-Type -AssemblyName System.Speech;"
                    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                    "$t=[Console]::In.ReadToEnd();"
                    "if ($t) { $s.Speak($t) }",
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
                self._tts_process = proc
                try:
                    proc.communicate(cleaned)
                finally:
                    self._tts_process = None
                return
            if backend == "say":
                proc = subprocess.Popen(["say", cleaned])
                self._tts_process = proc
                try:
                    proc.wait()
                finally:
                    self._tts_process = None
                return
            if backend == "espeak":
                proc = subprocess.Popen(["espeak", cleaned])
                self._tts_process = proc
                try:
                    proc.wait()
                finally:
                    self._tts_process = None
                return
        except Exception:
            pass
        if not self._tts_warned:
            self._tts_warned = True
            print("[voice] TTS backend unavailable.", flush=True)

    def _play_sound_blocking(self, kind: str) -> None:
        if not self._sound_enabled:
            return
        freq = self._sound_activation_freq if kind == "activate" else self._sound_sleep_freq
        duration_ms = max(20, int(self._sound_duration_ms))
        duration_s = duration_ms / 1000.0
        backend = self._resolve_sound_backend()
        if backend == "sounddevice" and self._sd is not None and self._np is not None:
            try:
                rate = int(self._voice_cfg.get("sample_rate", 16000))
                samples = int(rate * duration_s)
                if samples <= 0:
                    return
                t = self._np.linspace(0, duration_s, samples, endpoint=False)
                wave = self._np.sin(2 * self._np.pi * float(freq) * t) * float(self._sound_volume)
                self._sd.play(wave.astype("float32"), samplerate=rate, device=self._sound_output_device)
                self._sd.wait()
                return
            except Exception:
                if sys.platform != "win32":
                    pass
        if backend in ("winsound", "sounddevice") and sys.platform == "win32":
            try:
                import winsound

                winsound.Beep(int(freq), duration_ms)
                return
            except Exception:
                pass
        if not self._sound_warned:
            self._sound_warned = True
            print("[voice] Sound notification unavailable.", flush=True)

    def _sanitize_tts_text(self, text: str) -> str:
        if not text:
            return ""
        output: List[str] = []
        for ch in text:
            if self._is_emoji(ch):
                continue
            mapped = self._SYMBOL_WORDS.get(ch)
            if mapped:
                output.append(" ")
                output.append(mapped)
                output.append(" ")
                continue
            if ch.isprintable():
                output.append(ch)
        sanitized = "".join(output)
        return " ".join(sanitized.split()).strip()

    def _is_emoji(self, ch: str) -> bool:
        if not ch:
            return False
        code = ord(ch)
        if code == 0x200D:
            return True
        if 0xFE00 <= code <= 0xFE0F:
            return True
        if 0x1F1E6 <= code <= 0x1F1FF:
            return True
        if 0x1F3FB <= code <= 0x1F3FF:
            return True
        if 0x2600 <= code <= 0x26FF:
            return True
        if 0x2700 <= code <= 0x27BF:
            return True
        if 0x2300 <= code <= 0x23FF:
            return True
        if 0x1F300 <= code <= 0x1F5FF:
            return True
        if 0x1F600 <= code <= 0x1F64F:
            return True
        if 0x1F680 <= code <= 0x1F6FF:
            return True
        if 0x1F700 <= code <= 0x1F77F:
            return True
        if 0x1F780 <= code <= 0x1F7FF:
            return True
        if 0x1F800 <= code <= 0x1F8FF:
            return True
        if 0x1F900 <= code <= 0x1F9FF:
            return True
        if 0x1FA00 <= code <= 0x1FA6F:
            return True
        if 0x1FA70 <= code <= 0x1FAFF:
            return True
        return False

    def _normalize_word_token(self, token: str) -> List[str]:
        if not token:
            return []
        if self._is_roman_numeral(token):
            value = self._roman_to_int(token)
            if value is not None:
                return self._number_to_words(value)
        return [token.lower()]

    def _greek_letter_name(self, ch: str) -> Optional[str]:
        name = unicodedata.name(ch, "")
        if "GREEK" not in name or "LETTER" not in name:
            return None
        letter = name.split("LETTER", 1)[1].strip()
        if "WITH" in letter:
            letter = letter.split("WITH", 1)[0].strip()
        if not letter:
            return None
        return letter.split()[-1].lower()

    def _is_roman_numeral(self, token: str) -> bool:
        if len(token) < 2:
            return False
        if not token.isascii():
            return False
        if token.upper() != token:
            return False
        for ch in token:
            if ch not in self._ROMAN_MAP:
                return False
        return True

    def _roman_to_int(self, token: str) -> Optional[int]:
        total = 0
        prev = 0
        for ch in reversed(token.upper()):
            value = self._ROMAN_MAP.get(ch)
            if value is None:
                return None
            if value < prev:
                total -= value
            else:
                total += value
                prev = value
        return total

    def _digits_to_words(self, token: str) -> List[str]:
        if not token:
            return []
        if len(token) > 1 and token.startswith("0"):
            return [self._ONES[int(ch)] for ch in token]
        try:
            value = int(token)
        except ValueError:
            return []
        if value >= 1000000:
            return [self._ONES[int(ch)] for ch in token]
        return self._number_to_words(value)

    def _number_to_words(self, value: int) -> List[str]:
        if value == 0:
            return ["zero"]
        if value < 0:
            return ["minus"] + self._number_to_words(-value)
        if value >= 1000000:
            return [self._ONES[int(ch)] for ch in str(value)]
        tokens: List[str] = []
        if value >= 1000:
            thousands = value // 1000
            tokens.extend(self._number_to_words(thousands))
            tokens.append("thousand")
            value %= 1000
            if value == 0:
                return tokens
        if value >= 100:
            hundreds = value // 100
            tokens.append(self._ONES[hundreds])
            tokens.append("hundred")
            value %= 100
            if value == 0:
                return tokens
        if value >= 20:
            tens = value // 10
            tokens.append(self._TENS[tens])
            value %= 10
            if value == 0:
                return tokens
        if value >= 10:
            tokens.append(self._TEENS[value - 10])
            return tokens
        if value > 0:
            tokens.append(self._ONES[value])
        return tokens

    def _tokenize_echo_text(self, text: str) -> List[str]:
        normalized = self.normalize_spoken_text(text)
        if not normalized:
            return []
        return normalized.split()

    def _reset_echo_state(self) -> None:
        self._echo_active = False
        self._echo_index = 0
        self._echo_tts_norm = []
