# System Bootstrap Guide

## Overview

The `system_bootstrap.py` script provides an interactive CLI for setting up your AI system across different configurations. It handles:

1. **System Identity** - Name and aliases
2. **Purpose Definition** - What your system is designed to do
3. **Personality Generation** - LLM-powered expansion of purpose into detailed personality
4. **Interaction Mode** - Whether to use voice/TTS for prompts or text-only

## Quick Start

### Windows
```powershell
.\scripts\system_bootstrap.ps1
```

### Linux / macOS
```bash
bash scripts/system_bootstrap.sh
```

### Manual Python
```bash
python scripts/system_bootstrap.py
```

## Modes of Operation

### Text Mode (Default)
- Questions presented as text
- Answers provided via keyboard input
- Reliable, no audio dependencies

### Voice Mode
- Questions read aloud via TTS
- Answers captured via voice
- Requires audio hardware and packages

**Enable voice mode by setting:**
```json
{
  "bootstrap": {
    "interaction_mode": "voice"
  }
}
```

In `config/settings.json`

## What The Script Does

### 1. Detects Mode Setting
Checks `settings.bootstrap.interaction_mode` to determine text or voice interaction

### 2. Collects System Identity
- Asks for system name (e.g., "Neo", "Assistant", "Jarvis")
- Collects aliases (comma-separated)

### 3. Gathers Purpose
- Prompts for a description of the system's role
- Example: "A helpful coding assistant specialized in Python"

### 4. Expands Personality via LLM
- Sends the purpose to a language model
- Generates a comprehensive personality profile including:
  - Core values and principles
  - Communication style and tone
  - Decision-making approach
  - Strengths and capability focus
  - User interaction guidelines

### 5. Configures Interaction Mode
- In text mode: Option to enable voice/TTS for system runtime
- In voice mode: Already uses voice for prompts

### 6. Saves Configuration

Creates/Updates three files:

**`config/system_entity.json`** - System identity
```json
{
  "mem_id": "system.object",
  "name": "Your System Name",
  "aliases": ["alias1", "alias2"],
  "properties": {
    "description": "Custom configured AI system",
    "version": "0.1.0",
    "configured_at": "2026-03-29T..."
  },
  "pins": ["IMMUTABLE_CORE_REFERENCE"]
}
```

**`config/personality.json`** - Personality profile (NEW)
```json
{
  "mem_id": "system.personality",
  "original_purpose": "User's original purpose description",
  "expanded_personality": "Full personality profile from LLM",
  "created_at": "2026-03-29T...",
  "version": "0.1.0"
}
```

**`config/settings.json`** - Updates voice and bootstrap settings
- `voice.tts_enabled`: true/false
- `bootstrap.interaction_mode`: "text" or "voice"
- `bootstrap.voice_tts_enabled`: true/false  
- `bootstrap.voice_timeout_s`: Timeout for voice capture (default 30s)
- `bootstrap.voice_silence_threshold_ms`: Silence threshold (default 1000ms)
- `bootstrap.configured_at`: timestamp

## Voice Mode Features

### Requirements
To use voice mode, install optional dependencies:
```bash
pip install pyttsx3 SpeechRecognition
```

### Configuration
Override defaults in `settings.json`:
```json
{
  "bootstrap": {
    "interaction_mode": "voice",
    "voice_timeout_s": 30,
    "voice_sample_rate": 16000,
    "voice_silence_threshold_ms": 1000,
    "auto_confirm": false,
    "review_before_save": true
  }
}
```

### How It Works
1. **Question Read Aloud** - Questions are spoken using TTS (text-to-speech)
2. **Voice Capture** - Microphone listens for spoken answer
3. **Speech Recognition** - Google Speech Recognition (free, no API key needed)
4. **Text Fallback** - If voice recognition fails, switches to keyboard input
5. **Confirmation** - User sees interpreted text before confirming

### Audio Flow
```
Script → TTS Engine → Speaker
   ↓
User → Microphone → STT Engine (Google)
   ↓
Text Transcription → Display & Confirm
```

## Features

✓ **Text and Voice modes** - Choose your interaction style
✓ **TTS-powered prompts** - Questions read aloud in voice mode
✓ **STT voice capture** - Answers via microphone
✓ **Graceful fallbacks** - Voice → text if device unavailable
✓ **LLM-powered personality** - Expands brief purpose into detailed personality
✓ **Settings-driven** - All behavior configurable in `settings.json`
✓ **Reversible** - Can re-run anytime to reconfigure
✓ **Summary review** - Shows configuration before saving
✓ **Timestamped** - Tracks when config was created

## Integration with System

Once configured:

1. **Personality** is injected into LLM context for system requests
2. **Name/Aliases** are used in system prompts to identify the assistant
3. **Voice Setting** controls whether bootstrap-phase questions use TTS/STT
4. Can be re-run anytime to update configuration

## Troubleshooting

### Voice Mode Not Working

**Issue:** Script falls back to text mode even though `interaction_mode` is "voice"

**Solutions:**
- Install dependencies: `pip install pyttsx3 SpeechRecognition`
- Check microphone is connected and working
- Verify audio permissions on your OS
- Test voice separately: `python -m speech_recognition`

### LLM Not Available
If the script can't reach the LLM service:
- Falls back to using the original purpose as personality
- You can manually edit `config/personality.json` later
- Re-run the script once LLM is available to regenerate

### Path Issues
- Ensure you run from the AGI root directory or a subdirectory
- Script automatically finds `config/` directory

### Python Import Errors
- Make sure Python environment is properly configured
- Check that dependencies in `requirements.txt` are installed

## Re-running Bootstrap

You can run the script multiple times:
- Each run overwrites previous configuration
- Useful for testing different personalities
- Can switch between text and voice modes

Example workflow:
```bash
# Initial setup with text
.\scripts\system_bootstrap.ps1

# Later, switch to voice mode
# (first update settings.json bootstrap.interaction_mode to "voice")
.\scripts\system_bootstrap.ps1
```

## Example Scenarios

### Scenario 1: Voice-First Coding Assistant
- Set `bootstrap.interaction_mode: "voice"`
- Name: "CodeBot"
- Aliases: "cb, coder, assistant"
- Purpose: "A specialized Python development assistant that helps with debugging, optimization, and best practices"
- Result: Voice prompts for setup, personality emphasizing technical accuracy

### Scenario 2: Text-based General Purpose Helper
- Set `bootstrap.interaction_mode: "text"`
- Name: "Atlas"
- Aliases: "helper, assistant"
- Purpose: "A general knowledge assistant that helps with research, learning, and problem-solving across all domains"
- Result: Text-based setup, personality emphasizing breadth and curiosity

### Scenario 3: Hands-Free Tool Agent
- Set `bootstrap.interaction_mode: "voice"`
- Set `bootstrap.auto_confirm: true` (for continuous operation)
- Name: "Executor"
- Aliases: "tool, executor"
- Purpose: "An efficient command execution system that handles system tasks and tool integration with minimal overhead"
- Result: Voice-guided setup perfect for hands-free operation

## Advanced: Manual Editing

After bootstrap, you can manually edit configuration files:

### Edit Personality
Edit `config/personality.json` to refine the personality after LLM expansion.

### Edit System Name
Edit `config/system_entity.json` to change name/aliases (but keep `mem_id` and `pins`).

### Edit Voice Settings
Modify `config/settings.json` voice and bootstrap sections for fine-tuning:
- Timeout values for voice capture
- TTS rate and volume
- Silence thresholds

## File Preservation

The script preserves:
- `mem_id` and `pins` in system_entity (required for system integrity)
- Existing voice and bootstrap settings while updating only relevant fields
- Creation timestamps for audit trail

## Environment Variables

Optional environment variables to control behavior:

```bash
# Force text mode even if configured for voice
BOOTSTRAP_FORCE_TEXT=1 ./scripts/system_bootstrap.ps1

# Set custom timeout for voice capture
BOOTSTRAP_VOICE_TIMEOUT=60 ./scripts/system_bootstrap.ps1

# Verbose logging
BOOTSTRAP_DEBUG=1 ./scripts/system_bootstrap.ps1
```

## Future Enhancements

Possible additions:
- Personality versioning (store multiple personalities, switch between them)
- Personality refinement after generation (user feedback loop)
- Export/import configurations for sharing setups
- Custom TTS voice selection
- Multi-language support for voice prompts
- Personality traits quantification (for multi-dimensional personality modeling)
- Audio quality settings for different environments
