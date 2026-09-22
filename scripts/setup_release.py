#!/usr/bin/env python3
"""Create ignored, machine-local configuration for the focused public release.

The script deliberately never modifies a checked-in configuration file. It
creates `config/settings.local.json`, optionally `config/llama-router.local.yaml`,
and an `.env` file from their safe templates.
"""

from __future__ import annotations

import argparse
import getpass
import json
import ipaddress
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_TEMPLATE = ROOT / "config" / "settings.example.json"
SETTINGS_LOCAL = ROOT / "config" / "settings.local.json"
LLAMA_ROUTER_LOCAL = ROOT / "config" / "llama-router.local.yaml"
ENV_TEMPLATE = ROOT / ".env.example"
ENV_LOCAL = ROOT / ".env"
VALID_INTERFACES = {"text", "audio", "vision", "local", "telegram", "whatsapp"}


def _google_settings(mode: str, client_path: str) -> dict:
    if mode == "skip":
        return {"enabled": False, "oauth_client_source": "skip", "auto_google_authorize": False}
    path = (ROOT / "config" / "google_oauth" / "neo_desktop.json" if mode == "neo"
            else Path(client_path).expanduser().resolve())
    if mode == "own" and not client_path.strip():
        raise ValueError("--google-client-json is required with --google own.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        installed = payload.get("installed", {})
        if not installed.get("client_id") or not installed.get("client_secret"):
            raise ValueError("Expected a Desktop app OAuth JSON with client_id and client_secret.")
        if installed.get("auth_uri") != "https://accounts.google.com/o/oauth2/auth" or installed.get("token_uri") != "https://oauth2.googleapis.com/token":
            raise ValueError("OAuth JSON must use Google's official authorization and token endpoints.")
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(f"Google {mode} client is unavailable or invalid: {path}. {exc}") from exc
    return {"enabled": True, "oauth_client_source": mode,
            "client_secrets_path": str(path), "auto_google_authorize": False}


def _print_google_privacy_notice() -> None:
    print("\nGoogle is optional: skip, use Neo's Desktop OAuth app, or supply your own.\n"
          "Neo's app identifies the project to Google and depends on its continued availability.\n"
          "Your own client avoids that project dependency. Neither option hides activity from Google.\n"
          "Authorization happens later in your browser, separately for each requested capability.\n"
          "Credentials are stored in your local system keyring; API requests go directly to Google.\n"
          "Retrieved content can enter model prompts and local conversation logs; review your model\n"
          "provider and logging settings before accessing private data. Never share tokens with support.\n"
          "Setup does not connect an account or grant permissions.\n")


def _print_search_privacy_notice() -> None:
    print(
        "\nWeb search and firewall notice:\n"
        "SearXNG runs locally, but sends search queries to external search engines.\n"
        "Those engines can see the queries and your public IP address. Local hosting does not provide anonymity.\n"
        "Search results and tool activity may be saved in this project's local logs.\n"
        "Most installations need no firewall changes. Setup and startup never modify your firewall automatically.\n"
        "If Docker cannot reach search engines, you may optionally prepare a narrowly scoped firewall plan.\n"
        "Review its exact rules before applying it separately with administrator privileges.\n"
    )


def _firewall_plan(bridge: str, uplink: str, subnet: str) -> str:
    """Prepare additive rules for the specific nftables chain diagnosed in this project."""
    for interface in (bridge, uplink):
        if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,15}", interface):
            raise ValueError("Firewall interfaces must be explicit Linux interface names (1-15 characters).")
    if bridge == uplink:
        raise ValueError("Docker bridge and internet interface must be different.")
    network = ipaddress.IPv4Network(subnet)
    if network.prefixlen == 0:
        raise ValueError("Use the Docker subnet, not the entire internet (0.0.0.0/0).")
    return (
        "# REVIEW BEFORE APPLYING: optional Linux nftables web-search repair.\n"
        "# Requires an existing inet filter forward chain; not a general firewall configuration.\n"
        "# Permits ALL containers on the specified bridge/subnet to make HTTP/HTTPS connections.\n"
        "# Allows replies to established connections; does not publish inbound services.\n"
        "# No flush, default-policy change, or boot-time persistence. Apply once to avoid duplicates.\n"
        f'add rule inet filter forward iifname "{bridge}" oifname "{uplink}" '
        f'ip saddr {network} tcp dport {{ 80, 443 }} counter accept comment "neo-web-out"\n'
        f'add rule inet filter forward iifname "{uplink}" oifname "{bridge}" '
        f'ip daddr {network} ct state {{ established, related }} counter accept comment "neo-web-replies"\n'
    )


def _server_executable(value: str) -> str:
    """Keep PATH commands portable; anchor explicit paths at setup time."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        raise ValueError("--llama-server must name an executable.")
    path = Path(value).expanduser()
    if path.is_absolute() or "/" in value or "\\" in value or value.startswith("~"):
        return str(path.resolve())
    return value


def _ask(prompt: str, default: str) -> str:
    if not sys.stdin.isatty():
        return default
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def _ask_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    while True:
        value = _ask(f"{prompt} ({'/'.join(choices)})", default).lower()
        if value in choices:
            return value
        print(f"Choose one of: {', '.join(choices)}")


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y or n.")


def _model_id(value: str, field: str) -> str:
    raw = value.strip().replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a filename relative to --model-dir.")
    name = path.name
    if not name or name in {".", ".."} or "\n" in name or "\r" in name:
        raise ValueError(f"{field} must be a model filename, not a path or URL.")
    return name


def _interfaces(value: str) -> list[str]:
    selected: list[str] = []
    for item in value.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in VALID_INTERFACES:
            raise ValueError(f"Unsupported interface '{name}'. Choose from {', '.join(sorted(VALID_INTERFACES))}.")
        if name not in selected:
            selected.append(name)
    if not selected:
        return ["text"]
    if len(selected) > 1 and any(item not in {"telegram", "whatsapp"} for item in selected):
        raise ValueError("Only Telegram and WhatsApp may be combined in one launch.")
    return selected


def _local_senses(value: str) -> list[str]:
    selected: list[str] = []
    for item in value.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in {"text", "audio", "vision"}:
            raise ValueError("--local-senses accepts only text, audio, and vision.")
        if name not in selected:
            selected.append(name)
    if not selected:
        return ["text"]
    if "text" in selected and "audio" in selected:
        raise ValueError("Text and audio cannot both be local senses because they share stdin.")
    return selected


def _yaml_command(command: str) -> str:
    return command.replace("\n", " ").strip()


def _llama_router_config(
    *,
    llama_server: str,
    model_dir: Path,
    fast_model: str,
    reasoning_model: str,
    embedding_model: str,
    vision_model: str,
    vision_model_path: str,
    vision_mmproj: str,
    vision_mmproj_path: str,
    gpu_layers: int,
    context_size: int,
) -> str:
    models: list[tuple[str, str, int]] = [
        (fast_model, f"${{llama}} --model {model_dir / fast_model} --ctx-size {context_size}", 300),
    ]
    if reasoning_model != fast_model:
        models.append(
            (reasoning_model, f"${{llama}} --model {model_dir / reasoning_model} --ctx-size {context_size}", 600)
        )
    if vision_model:
        models.append(
            (
                vision_model,
                f"${{llama}} --model {model_dir / vision_model_path} --mmproj {model_dir / vision_mmproj_path} --ctx-size {context_size}",
                600,
            )
        )
    models.append(
        (embedding_model, f"${{llama}} --model {model_dir / embedding_model} --embeddings --ctx-size 2048", 0)
    )

    lines = [
        "# Generated for llama.cpp router mode.",
        "healthCheckTimeout: 180",
        "globalTTL: 900",
        "startPort: 5800",
        "",
        "macros:",
        '  "llama": >',
        f"    {_yaml_command(llama_server)} --port ${{PORT}}",
        "",
        "models:",
    ]
    for name, command, ttl in models:
        lines.extend([f'  "{name}":', "    cmd: >", f"      {_yaml_command(command)}", f"    ttl: {ttl}"])
    chat_models = [fast_model]
    if reasoning_model != fast_model:
        chat_models.append(reasoning_model)
    if vision_model:
        chat_models.append(vision_model)
    lines.extend(["", "groups:", "  always_on:", "    swap: false", "    members:", f'      - "{embedding_model}"', "  chat:", "    swap: true", "    members:"])
    lines.extend(f'      - "{model}"' for model in chat_models)
    return "\n".join(lines) + "\n"


def _ensure_env(updates: dict[str, str] | None = None) -> None:
    created = not ENV_LOCAL.exists()
    content = ENV_TEMPLATE.read_text(encoding="utf-8") if created else ENV_LOCAL.read_text(encoding="utf-8")
    secret_line = f"SEARXNG_SECRET={secrets.token_urlsafe(32)}"
    lines = content.splitlines()
    updates = {key: value for key, value in (updates or {}).items() if value}
    found_updates: set[str] = set()
    secret_found = False
    for index, line in enumerate(lines):
        key = line.split("=", 1)[0] if "=" in line else ""
        if key in updates:
            lines[index] = f"{key}={updates[key]}"
            found_updates.add(key)
        if line.startswith("SEARXNG_SECRET="):
            if line.split("=", 1)[1].strip():
                secret_found = True
            else:
                lines[index] = secret_line
                secret_found = True
    if not secret_found:
        lines.extend(["", "# Local SearXNG service", secret_line])
    for key, value in updates.items():
        if key not in found_updates:
            lines.append(f"{key}={value}")
    content = "\n".join(lines) + "\n"
    ENV_LOCAL.write_text(content, encoding="utf-8")
    if created:
        print("Created .env from .env.example. Add messaging credentials before enabling Telegram or WhatsApp.")


def _replace_models(settings: dict, fast_model: str, reasoning_model: str, embedding_model: str, vision_model: str) -> None:
    models = settings["models"]
    models.update(
        {
            "router_model": fast_model,
            "text_model": reasoning_model,
            "embedding_model": embedding_model,
            "vision_model": vision_model,
        }
    )
    cognition = settings["orchestrator"]
    for key in ("cognition_system0_model", "cognition_route_model", "cognition_init_model"):
        cognition[key] = fast_model
    for key in (
        "cognition_system1_model",
        "cognition_system2_model",
        "cognition_repair_model",
        "cognition_judge_model",
    ):
        cognition[key] = reasoning_model


def _interactive_values(args: argparse.Namespace) -> dict[str, str]:
    """Ask only for local choices; secrets use no-echo input and stay in .env."""
    print("\nAGI core setup — all generated files are local and ignored by Git.\n")
    args.provider = _ask_choice("Local model backend", ("llama-router", "ollama"), args.provider)
    args.interfaces = _ask(
        "Interfaces (comma-separated: text, audio, vision, local, telegram, whatsapp)",
        args.interfaces,
    )
    selected_interfaces = _interfaces(args.interfaces)
    if selected_interfaces == ["local"]:
        args.local_senses = _ask("Local senses (comma-separated: text, audio, vision)", args.local_senses)
    args.fast_model = _ask("Fast model for routing and System 0", args.fast_model)
    args.reasoning_model = _ask("Reasoning model for Systems 1 and 2", args.reasoning_model)
    args.embedding_model = _ask("Embedding model", args.embedding_model)

    configure_vision = _ask_yes_no("Configure an optional image-analysis model", bool(args.vision_model))
    if configure_vision:
        args.vision_model = _ask("Vision model (relative to model directory for llama-router)", args.vision_model or "vision-model.gguf")
        if args.provider == "llama-router":
            args.vision_mmproj = _ask("Matching mmproj file", args.vision_mmproj or "mmproj.gguf")
    else:
        args.vision_model = ""
        args.vision_mmproj = ""

    if args.provider == "llama-router":
        args.model_dir = _ask("Directory containing your GGUF files", args.model_dir or str(Path.home() / "models"))
        default_server = args.llama_server or ("llama-server.exe" if os.name == "nt" else "llama-server")
        args.llama_server = _ask("llama-server executable", default_server)
        args.context_size = int(_ask("Context size", str(args.context_size)))

    updates: dict[str, str] = {}
    if "telegram" in selected_interfaces and _ask_yes_no("Configure Telegram now", True):
        token = getpass.getpass("Telegram bot token (input hidden; leave empty to edit .env later): ").strip()
        chat_ids = _ask("Allowed Telegram chat IDs (comma-separated; blank permits configured defaults)", "")
        if token:
            updates["TELEGRAM_BOT_TOKEN"] = token
        if chat_ids:
            updates["TELEGRAM_ALLOWED_CHAT_IDS"] = chat_ids
    if "whatsapp" in selected_interfaces:
        print("WhatsApp will show a QR code on first launch; authentication stays in data/whatsapp_auth.")
    args.google = _ask_choice("Google integration", ("skip", "neo", "own"), args.google)
    if args.google == "own":
        args.google_client_json = _ask("Path to your downloaded Desktop OAuth JSON", args.google_client_json)
    print("Firewall plans apply only to Linux hosts with an existing nftables inet filter forward chain.")
    args.prepare_firewall_plan = _ask_yes_no("Prepare an optional firewall plan for review (does NOT apply it)", False)
    if args.prepare_firewall_plan:
        args.firewall_bridge = _ask("Docker bridge interface (from docker network inspect / ip link)", args.firewall_bridge)
        args.firewall_uplink = _ask("Internet interface (from ip route)", args.firewall_uplink)
        args.firewall_subnet = _ask("Docker IPv4 subnet (from docker network inspect)", args.firewall_subnet)
    return updates


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create portable local configuration for AGI core.")
    parser.add_argument("--provider", choices=("llama-router", "ollama"), default="llama-router")
    parser.add_argument("--interactive", action="store_true", help="Prompt for backend, models, interfaces, and optional Telegram credentials")
    parser.add_argument("--interfaces", default="text", help="Comma-separated: text, audio, vision, local, telegram, whatsapp")
    parser.add_argument("--local-senses", default="text", help="Comma-separated local senses when --interfaces local is selected")
    parser.add_argument("--fast-model", default="qwen2.5-3b-instruct-q4_k_m.gguf")
    parser.add_argument("--reasoning-model", default="qwen2.5-7b-instruct-q4_k_m.gguf")
    parser.add_argument("--embedding-model", default="nomic-embed-text-v1.5-q4_k_m.gguf")
    parser.add_argument("--vision-model", default="", help="Optional multimodal GGUF/Ollama model")
    parser.add_argument("--vision-mmproj", default="", help="Required with --vision-model for llama-router")
    parser.add_argument("--model-dir", default="", help="Directory containing GGUF files (required for llama-router)")
    parser.add_argument("--llama-server", default="llama-server")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument("--force", action="store_true", help="Replace existing local configuration files")
    parser.add_argument("--prepare-firewall-plan", action="store_true", help="Generate a review-only nftables plan; never applies firewall changes")
    parser.add_argument("--firewall-bridge", default="", help="Explicit Docker bridge interface for the optional plan")
    parser.add_argument("--firewall-uplink", default="", help="Explicit internet interface for the optional plan")
    parser.add_argument("--firewall-subnet", default="", help="Explicit Docker IPv4 CIDR subnet for the optional plan")
    parser.add_argument("--google", choices=("skip", "neo", "own"), default="skip")
    parser.add_argument("--google-client-json", default="", help="Desktop OAuth JSON path for --google own")
    args = parser.parse_args(argv)
    _print_search_privacy_notice()
    _print_google_privacy_notice()

    env_updates: dict[str, str] = {}
    if args.interactive:
        if not sys.stdin.isatty():
            parser.error("--interactive requires a terminal.")
        try:
            env_updates = _interactive_values(args)
        except ValueError as error:
            parser.error(str(error))

    try:
        google_settings = _google_settings(args.google, args.google_client_json)
        firewall_plan = None
        if args.prepare_firewall_plan:
            if not sys.platform.startswith("linux"):
                raise ValueError("The optional nftables firewall plan is for Linux hosts only.")
            firewall_plan = _firewall_plan(args.firewall_bridge, args.firewall_uplink, args.firewall_subnet)
        interfaces = _interfaces(args.interfaces)
        local_senses = _local_senses(args.local_senses)
        fast_model = _model_id(args.fast_model, "--fast-model")
        reasoning_model = _model_id(args.reasoning_model, "--reasoning-model")
        embedding_model = _model_id(args.embedding_model, "--embedding-model")
        vision_model = _model_id(args.vision_model, "--vision-model") if args.vision_model else ""
        vision_mmproj = _model_id(args.vision_mmproj, "--vision-mmproj") if args.vision_mmproj else ""
        vision_model_path = args.vision_model.strip().replace("\\", "/")
        vision_mmproj_path = args.vision_mmproj.strip().replace("\\", "/")
        if args.provider == "llama-router" and not args.model_dir:
            if not sys.stdin.isatty():
                raise ValueError("--model-dir is required when --provider llama-router is used non-interactively.")
            args.model_dir = _ask("Directory containing your GGUF models", str(Path.home() / "models"))
        if args.provider == "llama-router" and bool(vision_model) != bool(vision_mmproj):
            raise ValueError("--vision-model and --vision-mmproj must be supplied together for llama-router.")
        if args.provider == "llama-router":
            args.llama_server = _server_executable(args.llama_server)
    except ValueError as error:
        parser.error(str(error))

    if SETTINGS_LOCAL.exists() and not args.force:
        parser.error(f"{SETTINGS_LOCAL.relative_to(ROOT)} exists. Re-run with --force to replace it.")
    if args.provider == "llama-router" and LLAMA_ROUTER_LOCAL.exists() and not args.force:
        parser.error(f"{LLAMA_ROUTER_LOCAL.relative_to(ROOT)} exists. Re-run with --force to replace it.")
    settings = json.loads(SETTINGS_TEMPLATE.read_text(encoding="utf-8"))
    settings["google"] = google_settings
    settings["orchestrator"]["auto_approve_all"] = False
    settings["interface"]["mode"] = interfaces[0] if len(interfaces) == 1 else interfaces
    if interfaces[0] == "local":
        settings["interface"]["senses"] = local_senses
    settings["models"]["provider"] = "llamacpp" if args.provider == "llama-router" else "ollama"
    settings["models"]["ollama_host"] = args.ollama_host.rstrip("/")
    _replace_models(settings, fast_model, reasoning_model, embedding_model, vision_model)
    if args.provider == "llama-router":
        model_dir = Path(args.model_dir).expanduser().resolve()
        settings["models"]["models_dir"] = str(model_dir)
        settings["models"]["llama_server"] = args.llama_server
    SETTINGS_LOCAL.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    _ensure_env(env_updates)
    print(f"Created {SETTINGS_LOCAL.relative_to(ROOT)}")
    if firewall_plan is not None:
        plan_path = ROOT / "config" / "firewall.local.nft"
        plan_path.write_text(firewall_plan, encoding="utf-8")
        print("\nProposed firewall changes (NOT APPLIED):\n" + firewall_plan)
        print("Review config/firewall.local.nft. If you approve these exact changes, run from this project:")
        print("  sudo nft -c -f config/firewall.local.nft && sudo nft -f config/firewall.local.nft")
        print("Persistence requires a separate, deliberate edit to your system firewall configuration.")
        print("To undo: sudo nft -a list chain inet filter forward")
        print("Then delete the two rules marked neo-web-out / neo-web-replies by handle:")
        print("  sudo nft delete rule inet filter forward handle <handle>")
    if args.provider == "llama-router":
        print("To run the local router manually, start your server like:")
        print(f"  {args.llama_server} --models-dir {model_dir} --host 127.0.0.1 --port 8080")
        print("Or run ./start.sh to have the project start the server for you.")
        # Attempt to generate a models.ini preset file for llama-server
        try:
            gen = ROOT / "scripts" / "generate_models_ini.py"
            if gen.exists():
                import subprocess

                preset_out = ROOT / 'config' / 'models.ini'
                subprocess.run([sys.executable, str(gen), "--output", str(preset_out)], check=False)
                print(f"Wrote {preset_out}")
        except Exception:
            # Non-fatal; setup should continue even if generation fails.
            pass
    else:
        print("Pull the selected models with `ollama pull <model>` before starting the assistant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
