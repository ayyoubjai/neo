#!/usr/bin/env python3
"""
Interactive system bootstrap script for AGI.
Sets up system entity name/aliases and generates personality through LLM expansion.
Supports both text and voice (TTS/STT) interaction modes.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from common.config import load_settings
from adaptive_agent.materializer import DEFAULT_DNA_PATH, DEFAULT_TARGET_PATH, materialize_profile
from adaptive_agent.purpose import infer_purpose
from adaptive_agent.wizard import DEFAULT_OUTPUT_PATH, build_profile
from src.model_server.text_model import generate as generate_text

# Optional voice support
try:
    import pyttsx3  # Text-to-speech fallback
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import speech_recognition as sr  # Speech-to-text
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False


CONFIG_DIR = Path(__file__).parent.parent / "config"
SYSTEM_ENTITY_PATH = CONFIG_DIR / "system_entity.json"
PERSONALITY_PATH = CONFIG_DIR / "personality.json"
SETTINGS_PATH = CONFIG_DIR / "settings.json"


# ============================================================================
# Voice I/O Support
# ============================================================================

_tts_engine = None
_stt_recognizer = None


def _init_tts() -> Any:
    """Initialize text-to-speech engine."""
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    
    if not TTS_AVAILABLE:
        return None
    
    try:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty('rate', 150)  # Slightly slower for clarity
        return _tts_engine
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize TTS: {e}")
        return None


def _init_stt() -> Any:
    """Initialize speech-to-text recognizer."""
    global _stt_recognizer
    if _stt_recognizer is not None:
        return _stt_recognizer
    
    if not STT_AVAILABLE:
        return None
    
    try:
        _stt_recognizer = sr.Recognizer()
        return _stt_recognizer
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize STT: {e}")
        return None


def speak(text: str) -> bool:
    """
    Speak text using TTS.
    
    Args:
        text: Text to speak
        
    Returns:
        True if successful, False otherwise
    """
    engine = _init_tts()
    if engine is None:
        return False
    
    try:
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"⚠️  TTS error: {e}")
        return False


def listen_for_voice_input(timeout_s: int = 30, silence_threshold_ms: int = 1000) -> Optional[str]:
    """
    Capture voice input from microphone using STT.
    
    Args:
        timeout_s: Timeout for listening in seconds
        silence_threshold_ms: Silence threshold in milliseconds
        
    Returns:
        Transcribed text or None if failed/timeout
    """
    recognizer = _init_stt()
    if recognizer is None:
        return None
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("🎤 Listening... (speak now)")
            
            # Set timeout
            audio = recognizer.listen(source, timeout=timeout_s, phrase_time_limit=timeout_s)
        
        print("🔄 Processing speech...")
        
        # Try to use Google Speech Recognition (free, doesn't require API key)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand speech. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
            
    except sr.RequestError as e:
        print(f"❌ Microphone error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error during voice input: {e}")
        return None


def get_bootstrap_config() -> Dict[str, Any]:
    """Get bootstrap configuration from settings."""
    try:
        settings_dict = load_settings()
        
        # handle both dictionary and object-like access
        if isinstance(settings_dict, dict):
            bootstrap_config = settings_dict.get("bootstrap", {})
        else:
            # If it's an object with attributes
            bootstrap_config = getattr(settings_dict, "bootstrap", {})
            if not isinstance(bootstrap_config, dict):
                bootstrap_config = {}
        
        return bootstrap_config
    except Exception:
        return {}


def should_use_voice_mode() -> bool:
    """Check if voice mode is enabled in settings."""
    config = get_bootstrap_config()
    mode = config.get("interaction_mode", "text")
    
    if mode == "voice":
        if not (TTS_AVAILABLE and STT_AVAILABLE):
            print("⚠️  Voice mode requested but TTS/STT not available. Falling back to text mode.")
            print("   Install: pip install pyttsx3 SpeechRecognition")
            return False
        return True
    
    return False


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}\n")


def read_input(prompt: str, default: Optional[str] = None, use_voice: bool = False) -> str:
    """
    Read user input with optional voice support.
    
    Args:
        prompt: Input prompt text
        default: Default value if user provides no input
        use_voice: Whether to use voice input
        
    Returns:
        User input or default value
    """
    # Print the prompt
    if default:
        full_prompt = f"{prompt} [{default}]:"
    else:
        full_prompt = f"{prompt}:"
    
    print(full_prompt)
    
    if use_voice:
        # Speak the prompt
        speak(prompt)
        
        # Capture voice input
        config = get_bootstrap_config()
        timeout_s = config.get("voice_timeout_s", 30)
        silence_threshold_ms = config.get("voice_silence_threshold_ms", 1000)
        
        voice_input = listen_for_voice_input(timeout_s, silence_threshold_ms)
        
        if voice_input:
            print(f"✓ Understood: {voice_input}")
            return voice_input.strip()
        else:
            # Fallback to text input
            print("Falling back to text input...")
            return text_input(f"  {prompt}", default)
    else:
        return text_input(f"  {prompt}", default)


def text_input(prompt: str, default: Optional[str] = None) -> str:
    """Read text input from user."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    value = input(full_prompt).strip()
    return value if value else (default or "")


def read_aliases(use_voice: bool = False) -> list:
    """
    Interactively read aliases for the system.
    
    Args:
        use_voice: Whether to use voice input
        
    Returns:
        List of aliases
    """
    if use_voice:
        speak("Please enter aliases for your system, comma-separated. For example: neo, laptop, assistant")
    
    print("Enter aliases for your system (comma-separated, or press Enter to skip):")
    print("  Example: neo, laptop, assistant, system")
    aliases_input = input("  Aliases: ").strip()
    
    if not aliases_input:
        return []
    
    aliases = [alias.strip() for alias in aliases_input.split(",") if alias.strip()]
    return aliases


def read_yes_no(prompt: str, default: bool = False, use_voice: bool = False) -> bool:
    """
    Read a yes/no response from user.
    
    Args:
        prompt: Question to ask
        default: Default answer if no input
        use_voice: Whether to use voice input
        
    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    
    if use_voice:
        speak(prompt)
    
    print(f"{prompt} [{default_str}]: ", end="", flush=True)
    response = input().strip().lower()
    
    if response in ("y", "yes"):
        return True
    elif response in ("n", "no"):
        return False
    else:
        return default


def expand_personality_via_llm(purpose: str) -> str:
    """
    Call LLM to expand the system purpose into a full personality.
    
    Args:
        purpose: Brief description of the system's purpose
        
    Returns:
        Expanded personality description
    """
    print("\n⏳ Expanding purpose into personality through LLM...")
    print("   (This may take a moment...)\n")
    
    expansion_prompt = f"""You are a personality designer for AI systems. 

A user has described their system's purpose as:
"{purpose}"

Based on this purpose, generate a comprehensive personality profile that includes:
1. Core values and principles
2. Communication style and tone
3. Decision-making approach
4. Strengths and capabilities focus
5. How they should interact with users

Write the personality as a cohesive narrative that captures the essence of what kind of assistant this system should be. Make it vivid and specific, not generic.

Personality Profile:
"""
    
    try:
        context = {
            "summary": "System personality expansion",
            "memory": [],
            "trace_id": "bootstrap_personality",
            "turn_id": "0"
        }
        
        # Try to generate with the configured model
        personality = generate_text(expansion_prompt, context)
        
        if not personality or len(personality) < 50:
            print("⚠️  Warning: LLM returned short/empty response. You may want to edit the personality later.")
            return purpose  # Fallback to original purpose
        
        return personality.strip()
        
    except Exception as e:
        print(f"⚠️  Error calling LLM: {e}")
        print("   Falling back to original purpose as personality.\n")
        return purpose


def save_system_entity(name: str, aliases: list) -> None:
    """Save system entity configuration."""
    entity = {
        "mem_id": "system.object",
        "name": name,
        "aliases": aliases,
        "properties": {
            "description": "Custom configured AI system",
            "version": "0.1.0",
            "configured_at": datetime.now().isoformat()
        },
        "pins": ["IMMUTABLE_CORE_REFERENCE"]
    }
    
    with open(SYSTEM_ENTITY_PATH, "w") as f:
        json.dump(entity, f, indent=2)
    
    print(f"✓ System entity saved to {SYSTEM_ENTITY_PATH}")


def save_personality(purpose: str, personality: str) -> None:
    """Save personality configuration."""
    personality_config = {
        "mem_id": "system.personality",
        "original_purpose": purpose,
        "expanded_personality": personality,
        "created_at": datetime.now().isoformat(),
        "version": "0.1.0"
    }
    
    with open(PERSONALITY_PATH, "w") as f:
        json.dump(personality_config, f, indent=2)
    
    print(f"✓ Personality saved to {PERSONALITY_PATH}")


def configure_tts_settings(use_tts: bool) -> None:
    """Update TTS and bootstrap settings in settings.json."""
    try:
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        
        # Ensure voice section exists
        if "voice" not in settings:
            settings["voice"] = {}
        
        settings["voice"]["tts_enabled"] = use_tts
        
        # Ensure bootstrap section exists
        if "bootstrap" not in settings:
            settings["bootstrap"] = {}
        
        # Update bootstrap settings based on TTS preference
        if use_tts:
            settings["bootstrap"]["interaction_mode"] = "voice"
        else:
            settings["bootstrap"]["interaction_mode"] = "text"
        
        settings["bootstrap"]["voice_tts_enabled"] = use_tts
        settings["bootstrap"]["configured_at"] = datetime.now().isoformat()
        
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
        
        tts_mode = "enabled (voice)" if use_tts else "disabled (text-only)"
        print(f"✓ TTS interaction mode: {tts_mode}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not update TTS settings: {e}")


def read_cognition_mode(use_voice: bool = False) -> str:
    print("Choose cognition mode:")
    print("  auto    - choose by device capacity")
    print("  system0 - fastest/smallest")
    print("  system1 - system0 + light planning")
    print("  system2 - system0 + system1 + deeper reasoning")
    print("  full    - all systems including system3 peers")
    if use_voice:
        speak("Choose cognition mode. Press enter for auto.")
    value = text_input("  Cognition mode", "auto").strip().lower()
    if value in {"auto", "system0", "system1", "system2", "system3", "full"}:
        return value
    print("Unknown cognition mode; using auto.")
    return "auto"


def configure_adaptive_runtime(
    purpose: str,
    personality: str,
    cognition_mode: str = "auto",
    use_voice: bool = False,
) -> None:
    """Create the active adaptive profile and optionally materialize selected bundles."""
    print("\nStep 7: Adaptive Runtime")
    print("-" * 40)
    if use_voice:
        speak("Creating the adaptive runtime profile.")
    adaptive_purpose = infer_purpose(purpose)
    print(f"Inferred adaptive purpose: {adaptive_purpose}")
    try:
        profile = build_profile(
            purpose=adaptive_purpose,
            personality=personality,
            output_path=DEFAULT_OUTPUT_PATH,
            cognition_mode=cognition_mode,
        )
        print(f"✓ Adaptive profile saved to {DEFAULT_OUTPUT_PATH}")
        cognition_id = profile["cognition"]["id"] if profile.get("cognition") else "none"
        model_id = profile["model"]["id"] if profile.get("model") else "none"
        tool_ids = []
        for gene in profile.get("tools", []):
            tool_ids.extend((gene.get("payload") or {}).get("tool_ids", []))
        print(f"  Device class: {profile['device']['device_class']}")
        print(f"  Cognition gene: {cognition_id}")
        print(f"  Model gene: {model_id}")
        print(f"  Tool IDs: {', '.join(tool_ids) if tool_ids else 'none'}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create adaptive profile: {e}")
        return

    if read_yes_no("Materialize relevant system parts now?", True, use_voice):
        try:
            result = materialize_profile(
                profile_path=DEFAULT_OUTPUT_PATH,
                target_path=DEFAULT_TARGET_PATH,
                source_path=DEFAULT_DNA_PATH,
            )
            print(f"✓ Materialized {result.files_written} files into {result.target}")
            print(f"  Bundles: {', '.join(result.bundle_ids)}")
        except Exception as e:
            print(f"⚠️  Warning: Could not materialize adaptive runtime: {e}")


def show_summary(name: str, aliases: list, purpose: str, personality: str, use_tts: bool, use_voice: bool = False) -> None:
    """Display a summary of the configuration."""
    print_header("Configuration Summary")
    
    print(f"System Name: {name}")
    print(f"Aliases: {', '.join(aliases) if aliases else '(none)'}")
    print(f"\nPurpose:\n  {purpose}")
    print(f"\nInteraction Mode: {'Voice + Text' if use_tts else 'Text-only'}")
    print(f"\nPersonality Profile:\n")
    
    # Print personality with nice formatting
    personality_lines = personality.split("\n")
    for line in personality_lines[:10]:  # Show first 10 lines
        if line.strip():
            print(f"  {line}")
    
    if len(personality_lines) > 10:
        print(f"  ... ({len(personality_lines) - 10} more lines)")
    
    if use_voice:
        speak("Review complete. Ready to save.")


def main() -> None:
    """Main bootstrap flow."""
    print_header("AGI System Bootstrap")
    print("This tool will help you configure your AI system.\n")
    
    # Detect voice mode
    use_voice = should_use_voice_mode()
    
    if use_voice:
        print("🎤 Voice Mode: Questions will be spoken and answers can be voice-captured\n")
        speak("Starting system bootstrap. Voice mode enabled.")
    else:
        print("📝 Text Mode: Enter responses as text\n")
    
    # Step 1: System name
    print("Step 1: System Identity")
    print("-" * 40)
    name = read_input("What is your system's name?", "system", use_voice)
    
    # Step 2: Aliases
    print("\nStep 2: Aliases")
    print("-" * 40)
    aliases = read_aliases(use_voice)
    
    # Step 3: Purpose
    print("\nStep 3: System Purpose")
    print("-" * 40)
    print("Describe the purpose and role of your system.")
    print("Example: 'A helpful coding assistant that specializes in Python and provides explanations'")
    purpose = read_input("What is your system's purpose?", "A helpful and intelligent assistant", use_voice)
    
    # Step 4: LLM personality expansion
    print("\nStep 4: Personality Generation")
    print("-" * 40)
    if use_voice:
        speak("Expanding your system's purpose into a personality profile. This may take a moment.")
    personality = expand_personality_via_llm(purpose)
    
    # Step 5: TTS preference (only if not already using voice)
    print("\nStep 5: Interaction Mode")
    print("-" * 40)
    if use_voice:
        print("(Running in voice mode)")
        speak("Voice mode is already enabled for this session.")
        final_use_tts = True
    else:
        print("Should the system use voice/TTS for interaction prompts?")
        final_use_tts = read_yes_no("Enable voice interaction?", False, False)

    print("\nStep 6: Cognition Mode")
    print("-" * 40)
    cognition_mode = read_cognition_mode(use_voice)
    
    # Show summary before saving
    show_summary(name, aliases, purpose, personality, final_use_tts, use_voice)
    
    # Confirm before saving
    print("\n" + "=" * 60)
    if read_yes_no("Save this configuration?", True, use_voice):
        save_system_entity(name, aliases)
        save_personality(purpose, personality)
        configure_tts_settings(final_use_tts)
        configure_adaptive_runtime(purpose, personality, cognition_mode, use_voice)
        
        print_header("✓ Bootstrap Complete!")
        print("Your system has been configured and is ready to use.")
        print(f"\nConfiguration files:")
        print(f"  - {SYSTEM_ENTITY_PATH}")
        print(f"  - {PERSONALITY_PATH}")
        print(f"  - {SETTINGS_PATH}")
        print(f"  - {DEFAULT_OUTPUT_PATH}")
        
        if use_voice:
            speak("Bootstrap complete. Your system is configured and ready to use.")
    else:
        print("\n✗ Configuration not saved. Exiting without changes.")
        if use_voice:
            speak("Bootstrap cancelled.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Bootstrap cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error during bootstrap: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
