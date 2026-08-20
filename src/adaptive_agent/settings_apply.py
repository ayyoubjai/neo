from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from adaptive_agent.wizard import DEFAULT_OUTPUT_PATH


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.json"


def apply_profile_settings(
    *,
    profile_path: str | Path = DEFAULT_OUTPUT_PATH,
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
) -> Dict[str, Any]:
    profile = _load_json(profile_path)
    settings = _load_json(settings_path)
    cognition = profile.get("cognition") or {}
    payload = cognition.get("payload") or {}
    patch = payload.get("settings_patch") or {}
    _deep_merge(settings, patch)
    settings.setdefault("adaptive_agent", {})
    settings["adaptive_agent"].update(
        {
            "active_profile_path": str(profile_path),
            "device_class": (profile.get("device") or {}).get("device_class", ""),
            "purpose": (profile.get("purpose") or {}).get("purpose", ""),
            "cognition_gene": cognition.get("id", ""),
            "cognition_mode": payload.get("mode", ""),
            "systems_enabled": payload.get("systems_enabled", []),
        }
    )
    Path(settings_path).write_text(json.dumps(settings, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return settings["adaptive_agent"]


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(target: Dict[str, Any], patch: Dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply adaptive profile settings to config/settings.json.")
    parser.add_argument("--profile", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH))
    args = parser.parse_args(argv)
    applied = apply_profile_settings(profile_path=args.profile, settings_path=args.settings)
    print(f"Applied adaptive settings: cognition={applied.get('cognition_mode', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

