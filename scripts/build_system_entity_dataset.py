import argparse
import json
from pathlib import Path


def _load_entity(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(entity: dict) -> list[dict]:
    name = entity.get("name", "system")
    aliases = entity.get("aliases", [])
    props = entity.get("properties", {}) if isinstance(entity.get("properties"), dict) else {}
    desc = props.get("description", "")
    version = props.get("version", "")
    mem_id = entity.get("mem_id", "")
    pins = entity.get("pins", [])

    system_prompt = (
        "You are a concise assistant that answers questions about the system entity. "
        "Use only the provided system info."
    )

    return [
        {
            "system": system_prompt,
            "prompt": "What is the system name and its aliases?\n",
            "response": f"The system name is {name}. Aliases: {', '.join(aliases) if aliases else 'None'}.",
        },
        {
            "system": system_prompt,
            "prompt": "Describe the system and its version.\n",
            "response": f"{desc} Version {version}." if desc or version else "No description or version available.",
        },
        {
            "system": system_prompt,
            "prompt": "What is the system mem_id and pins?\n",
            "response": f"mem_id: {mem_id}. Pins: {', '.join(pins) if pins else 'None'}.",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a small SFT dataset from system_entity.json.")
    parser.add_argument(
        "--entity",
        default="config/system_entity.json",
        help="Path to system_entity.json (default: config/system_entity.json).",
    )
    parser.add_argument(
        "--out",
        default="evolve/datasets/sft_general.jsonl",
        help="Output JSONL path (default: evolve/datasets/sft_general.jsonl).",
    )
    args = parser.parse_args()

    entity_path = Path(args.entity)
    if not entity_path.exists():
        raise SystemExit(f"[error] entity file not found: {entity_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_dataset(_load_entity(entity_path))
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=True) + "\n")

    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
