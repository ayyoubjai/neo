from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from adaptive_agent.device_probe import probe_device
from adaptive_agent.gene_catalog import DEFAULT_CATALOG_PATH, load_gene_catalog
from adaptive_agent.resolver import PurposeProfile, resolve_genes


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "config" / "adaptive_agent" / "active_profile.json"


def _ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def build_profile(
    *,
    purpose: str,
    personality: str,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    allow_network: bool = False,
    allow_background: bool = False,
    offline_first: bool = True,
    cognition_mode: str = "auto",
) -> Dict[str, Any]:
    device = probe_device(str(Path(output_path).resolve().parent))
    purpose_profile = PurposeProfile(
        purpose=purpose,
        personality=personality,
        offline_first=offline_first,
        allow_network=allow_network,
        allow_background=allow_background,
        cognition_mode=cognition_mode,
    )
    plan = resolve_genes(load_gene_catalog(catalog_path), device, purpose_profile)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = plan.to_dict()
    output.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an adaptive local-agent profile.")
    parser.add_argument("--purpose", choices=["coding", "assistant", "research", "automation", "phone_companion"], help="Agent purpose.")
    parser.add_argument("--personality", help="Short personality/system behavior description.")
    parser.add_argument("--allow-network", action="store_true", help="Allow genes that need network access.")
    parser.add_argument("--allow-background", action="store_true", help="Allow background services.")
    parser.add_argument(
        "--cognition-mode",
        default="auto",
        choices=["auto", "system0", "system1", "system2", "system3", "full"],
        help="Override adaptive cognition mode. Default auto chooses by device class.",
    )
    parser.add_argument("--online-first", action="store_true", help="Prefer online behavior over offline-first behavior.")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH), help="Path to gene catalog JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to write the active profile JSON.")
    
    # Interactive Mode Arguments
    parser.add_argument("--interactive", action="store_true", help="Run the interactive LLM-based initialization wizard.")
    parser.add_argument("--model", default="Llama-3.1-8B-Instruct-Q4_K_M.gguf", help="The LLaMA model to use for the interactive wizard.")
    parser.add_argument("--auto-inject-tools", action="store_true", help="Automatically inject generated tools into the runtime without prompting.")
    
    args = parser.parse_args(argv)

    if args.interactive:
        from adaptive_agent.interactive_wizard import run_interactive_setup
        config_data = run_interactive_setup(args.model)
        
        purpose = config_data.get("purpose", args.purpose or "assistant")
        personality = config_data.get("personality", args.personality or "pragmatic, concise")
        tools_needed = config_data.get("tools_needed", [])
        
        config_dir = Path(args.output).resolve().parents[1]
        system_entity_path = config_dir / "system_entity.json"
        
        if system_entity_path.exists():
            with open(system_entity_path, "r", encoding="utf-8") as f:
                entity_data = json.load(f)
        else:
            entity_data = {}
            
        entity_data["name"] = config_data.get("name", "Assistant")
        entity_data["aliases"] = config_data.get("aliases", [])
        
        with open(system_entity_path, "w", encoding="utf-8") as f:
            json.dump(entity_data, f, indent=2)
            
        personality_path = config_dir / "personality.json"
        with open(personality_path, "w", encoding="utf-8") as f:
            json.dump({"expanded_personality": personality}, f, indent=2)
            
        print(f"\n[+] Updated system persona and identity configurations.")
    else:
        purpose = args.purpose or _ask("Purpose: coding, assistant, research, automation, phone_companion", "assistant")
        personality = args.personality or _ask("Personality", "pragmatic, concise, local-first")
        tools_needed = []
    data = build_profile(
        purpose=purpose,
        personality=personality,
        output_path=args.output,
        catalog_path=args.catalog,
        allow_network=args.allow_network,
        allow_background=args.allow_background,
        offline_first=not args.online_first,
        cognition_mode=args.cognition_mode,
    )
    model_id = data["model"]["id"] if data.get("model") else "none"
    cognition_id = data["cognition"]["id"] if data.get("cognition") else "none"
    print(f"Wrote adaptive profile to {args.output}")
    print(f"Device class: {data['device']['device_class']}")
    print(f"Selected cognition gene: {cognition_id}")
    print(f"Selected model gene: {model_id}")
    print(f"Enabled tools: {len(data['tools'])}")
    
    if args.interactive and tools_needed:
        device_class = data['device']['device_class']
        if device_class in ["pc_medium", "pc_strong"]:
            print(f"\n[+] Capable device detected ({device_class}). Starting Tool Generation Stage...")
            from tool_runtime.tool_generator import generate_tools
            generate_tools(tools_needed, auto_inject=args.auto_inject_tools)
        else:
            print(f"\n[-] Device class '{device_class}' is too constrained for on-device tool generation. Skipping tool generation.")
            
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
