import argparse
import os
import re
import time
import wave

import sounddevice as sd


def _normalize_phrase(phrase: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", phrase.strip().lower()).strip("_")
    return cleaned or "wakeword"


def _write_wav(path: str, audio, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Record wakeword samples.")
    parser.add_argument("--phrase", required=True, help="Wake phrase (used for folder naming).")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to record.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Seconds per sample.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate.")
    parser.add_argument("--device", type=int, default=None, help="Audio device index.")
    parser.add_argument(
        "--kind",
        choices=["wakeword", "negative"],
        default="wakeword",
        help="Record wakeword or negative samples.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to data/wakeword/<phrase>/<kind>.",
    )
    args = parser.parse_args()

    phrase_id = _normalize_phrase(args.phrase)
    out_dir = args.out_dir or os.path.join("data", "wakeword", phrase_id, args.kind)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[wakeword] Recording {args.count} samples to {out_dir}")
    for idx in range(args.count):
        input(f"[wakeword] Press Enter to record sample {idx + 1}/{args.count}...")
        print("[wakeword] Recording...")
        frames = int(args.seconds * args.sample_rate)
        audio = sd.rec(frames, samplerate=args.sample_rate, channels=1, dtype="int16", device=args.device)
        sd.wait()
        ts = int(time.time() * 1000)
        filename = f"{args.kind}_{ts}_{idx + 1:03d}.wav"
        path = os.path.join(out_dir, filename)
        _write_wav(path, audio, args.sample_rate)
        print(f"[wakeword] Saved {path}")

    print("[wakeword] Done.")


if __name__ == "__main__":
    main()
