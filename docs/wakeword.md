# Wakeword Data Collection (Small Real + TTS Bootstrap)

This project uses `openwakeword` for wakeword detection. The voice daemon can load custom wakeword models by path via `voice.wakeword_models` in `config/settings.json`.

This guide helps you collect a **small real dataset** and **bootstrap with TTS** so you can train your own wakeword model externally.

## 1) Collect real samples

Record a small set of real samples (10–50 is fine to start):

```bash
python scripts/wakeword_collect.py --phrase "Hey Atlas" --count 20 --seconds 2.0
```

This writes WAV files to:

```
data/wakeword/hey_atlas/wakeword
```

## 2) Collect negative samples

Record negatives (ambient speech/noise without the wake phrase):

```bash
python scripts/wakeword_collect.py --phrase "Hey Atlas" --count 20 --seconds 2.0 --kind negative
```

Output:

```
data/wakeword/hey_atlas/negative
```

## 3) Bootstrap with TTS

Generate synthetic samples using system TTS voices:

```bash
python scripts/wakeword_bootstrap_tts.py --phrase "Hey Atlas" --max-voices 5
```

Output:

```
data/wakeword/hey_atlas/tts
```

You can add variants:

```bash
python scripts/wakeword_bootstrap_tts.py --phrase "Hey Atlas" --variants "Hey Atlas" "Hey Atlas!" "Hey, Atlas"
```

## 4) Train a wakeword model

Training is done outside this repo using `openwakeword` training utilities.
Use your real + TTS + negative audio as the dataset.

When you have a trained model file (e.g., ONNX), add its path to:

```json
{
  "voice": {
    "wakeword_models": ["path/to/your_model.onnx"]
  }
}
```

Restart the voice daemon to load the new wakeword.

## Notes

- Keep samples short (1–2s) and consistent.
- Mix multiple speakers for generalization.
- TTS is only a supplement—real samples matter most.
