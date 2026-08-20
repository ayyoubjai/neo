from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from adaptive_agent.wizard import DEFAULT_OUTPUT_PATH


def load_active_profile(path: str | Path = DEFAULT_OUTPUT_PATH) -> Dict[str, Any]:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def selected_model_server(profile: Dict[str, Any]) -> Dict[str, Any]:
    for service in profile.get("services", []) or []:
        payload = service.get("payload") or {}
        if payload.get("server_kind") in {"llama_cpp", "llama_swap"}:
            return service
    return {
        "id": "service.model.llama_cpp.default",
        "payload": {"server_kind": "llama_cpp"},
    }


def build_llamacpp_command(profile: Dict[str, Any]) -> List[str]:
    model = profile.get("model") or {}
    payload = model.get("payload") or {}
    binary = os.environ.get("LLAMA_CPP_SERVER") or payload.get("server_binary") or "llama-server"
    model_path = payload.get("model_path")
    if not model_path:
        raise ValueError("Selected model gene does not define payload.model_path")

    options = payload.get("options") or {}
    command = [str(binary), "-m", str(model_path)]
    if "ctx_size" in options:
        command.extend(["-c", str(options["ctx_size"])])
    if "threads" in options:
        command.extend(["-t", str(options["threads"])])
    if "port" in options:
        command.extend(["--port", str(options["port"])])
    return command


def build_llama_swap_command(profile: Dict[str, Any]) -> List[str]:
    service = selected_model_server(profile)
    payload = service.get("payload") or {}
    binary = os.environ.get("LLAMA_SWAP_BINARY") or payload.get("binary") or "llama-swap"
    config = payload.get("config_path") or "config/config.yaml"
    listen = payload.get("listen") or "localhost:8080"
    return [str(binary), "--config", str(config), "--listen", str(listen)]


def build_model_server_command(profile: Dict[str, Any]) -> List[str]:
    service = selected_model_server(profile)
    payload = service.get("payload") or {}
    if payload.get("server_kind") == "llama_swap":
        return build_llama_swap_command(profile)
    return build_llamacpp_command(profile)


def explain_profile(profile: Dict[str, Any]) -> str:
    model = profile.get("model") or {}
    cognition = profile.get("cognition") or {}
    cognition_payload = cognition.get("payload") or {}
    purpose = profile.get("purpose") or {}
    tools = [item["id"] for item in profile.get("tools", [])]
    services = [item["id"] for item in profile.get("services", [])]
    server = selected_model_server(profile)
    return "\n".join(
        [
            f"device_class={profile.get('device', {}).get('device_class', 'unknown')}",
            f"purpose={purpose.get('purpose', 'unknown')}",
            f"personality={purpose.get('personality', '')}",
            f"cognition={cognition.get('id', 'none')}:{cognition_payload.get('mode', '')}",
            f"model={model.get('id', 'none')}",
            f"model_server={server.get('id', 'none')}",
            f"tools={', '.join(tools) if tools else 'none'}",
            f"services={', '.join(services) if services else 'none'}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the active adaptive local-agent profile.")
    parser.add_argument("--profile", default=str(DEFAULT_OUTPUT_PATH), help="Path to active profile JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected runtime without launching it.")
    args = parser.parse_args(argv)

    profile = load_active_profile(args.profile)
    print(explain_profile(profile))
    try:
        command = build_model_server_command(profile)
    except ValueError as e:
        print(str(e))
        print("Run the wizard with a purpose supported on this device, or add a matching model gene.")
        return 2
    if args.dry_run:
        print(shlex.join(command))
        return 0
    service = selected_model_server(profile)
    if (service.get("payload") or {}).get("server_kind") != "llama_swap":
        model_path = Path(command[2]).expanduser()
        if not model_path.exists():
            print(f"Model file is missing: {model_path}")
            print("Place the GGUF model there or edit the selected gene payload.model_path.")
            print("Command that would run:")
            print(shlex.join(command))
            return 2
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
