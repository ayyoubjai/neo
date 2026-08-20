import argparse
import os
import re

import pyttsx3


def _normalize_phrase(phrase: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", phrase.strip().lower()).strip("_")
    return cleaned or "wakeword"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTS wakeword samples.")
    parser.add_argument("--phrase", required=True, help="Wake phrase to synthesize.")
    parser.add_argument("--out-dir", default="", help="Output directory for TTS wav files.")
    parser.add_argument("--max-voices", type=int, default=5, help="Max number of voices to use.")
    parser.add_argument("--voice-filter", default="", help="Only use voices that contain this substring.")
    parser.add_argument("--rate", type=int, default=None, help="Optional speech rate override.")
    parser.add_argument("--volume", type=float, default=None, help="Optional volume override (0.0-1.0).")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="Optional variants. If omitted, uses the phrase as-is.",
    )
    args = parser.parse_args()

    phrase_id = _normalize_phrase(args.phrase)
    out_dir = args.out_dir or os.path.join("data", "wakeword", phrase_id, "tts")
    os.makedirs(out_dir, exist_ok=True)

    engine = pyttsx3.init()
    if args.rate is not None:
        engine.setProperty("rate", int(args.rate))
    if args.volume is not None:
        engine.setProperty("volume", float(args.volume))

    voices = engine.getProperty("voices") or []
    if args.voice_filter:
        voices = [v for v in voices if args.voice_filter.lower() in (v.name or "").lower()]

    if not voices:
        print("[wakeword] No voices found.")
        return

    variants = args.variants or [args.phrase]
    used = 0
    for vidx, voice in enumerate(voices):
        engine.setProperty("voice", voice.id)
        for tidx, text in enumerate(variants):
            filename = f"tts_voice{vidx:02d}_var{tidx:02d}.wav"
            path = os.path.join(out_dir, filename)
            engine.save_to_file(text, path)
        engine.runAndWait()
        used += 1
        if used >= args.max_voices:
            break

    print(f"[wakeword] Generated TTS samples in {out_dir}")


if __name__ == "__main__":
    main()
