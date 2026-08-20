from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from adaptive_agent.wizard import DEFAULT_OUTPUT_PATH
from adaptive_agent.wizard import build_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_PATH = REPO_ROOT / "config" / "adaptive_agent" / "bundles.json"
DEFAULT_TARGET_PATH = REPO_ROOT / "runtime" / "adaptive_cell"
DEFAULT_DNA_PATH = REPO_ROOT / "dist" / "adaptive_agent_dna.zip"


@dataclass
class MaterializeResult:
    target: str
    bundle_ids: List[str]
    files_written: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "bundle_ids": list(self.bundle_ids),
            "files_written": self.files_written,
        }


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def bundle_ids_for_profile(profile: Dict[str, Any]) -> List[str]:
    bundle_ids = ["core"]
    for section in ("model", "cognition"):
        item = profile.get(section)
        if isinstance(item, dict):
            bundle_ids.extend((item.get("payload") or {}).get("bundles", []))
    for section in ("tools", "services", "prompts"):
        for item in profile.get(section, []) or []:
            bundle_ids.extend((item.get("payload") or {}).get("bundles", []))
    return _dedupe(bundle_ids)


def pack_dna(
    *,
    output_path: str | Path = DEFAULT_DNA_PATH,
    repo_root: str | Path = REPO_ROOT,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
) -> Path:
    root = Path(repo_root).resolve()
    bundles = load_json(bundle_path)
    paths = _all_bundle_paths(bundles)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _expand_paths(root, paths)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, path in enumerate(files, 1):
            _print_progress("packing dna", index, len(files))
            zf.write(path, path.relative_to(root).as_posix())
        print()
    return output


def materialize_profile(
    *,
    profile_path: str | Path = DEFAULT_OUTPUT_PATH,
    target_path: str | Path = DEFAULT_TARGET_PATH,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    source_path: str | Path = DEFAULT_DNA_PATH,
) -> MaterializeResult:
    profile = load_json(profile_path)
    bundles = load_json(bundle_path)
    selected_bundle_ids = bundle_ids_for_profile(profile)
    paths = _paths_for_bundle_ids(bundles, selected_bundle_ids)
    target = Path(target_path)
    source = Path(source_path)
    target.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(
            f"DNA source not found: {source}. Create it with 'python3 scripts/adaptive_agent_dna.py pack' "
            "or pass --source to an existing DNA zip."
        )

    if source.is_file() and source.suffix.lower() == ".zip":
        files_written = _materialize_from_zip(source, target, paths)
    else:
        files_written = _materialize_from_directory(source.resolve(), target, paths)

    manifest = {
        "source": str(source),
        "profile": str(profile_path),
        "bundle_ids": selected_bundle_ids,
        "files_written": files_written,
    }
    (target / "adaptive_cell_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return MaterializeResult(str(target), selected_bundle_ids, files_written)


def _paths_for_bundle_ids(bundles: Dict[str, Any], bundle_ids: Iterable[str]) -> List[str]:
    by_id = {item["id"]: item for item in bundles.get("bundles", [])}
    paths: List[str] = []
    for bundle_id in bundle_ids:
        item = by_id.get(bundle_id)
        if not item:
            continue
        paths.extend(item.get("paths", []))
    return _dedupe(paths)


def _all_bundle_paths(bundles: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for item in bundles.get("bundles", []):
        paths.extend(item.get("paths", []))
    return _dedupe(paths)


def _expand_paths(root: Path, path_patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in path_patterns:
        if pattern.endswith("/**"):
            base = root / pattern[:-3]
            if base.is_dir():
                files.extend(path for path in base.rglob("*") if path.is_file())
            elif base.is_file():
                files.append(base)
            continue
        matches = list(root.glob(pattern))
        for match in matches:
            if match.is_dir():
                files.extend(path for path in match.rglob("*") if path.is_file())
            elif match.is_file():
                files.append(match)
    return sorted({path.resolve() for path in files if _is_inside(path.resolve(), root)})


def _materialize_from_directory(source: Path, target: Path, path_patterns: Iterable[str]) -> int:
    files = _expand_paths(source, path_patterns)
    for index, path in enumerate(files, 1):
        rel = path.relative_to(source)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        _print_progress("materializing", index, len(files))
    if files:
        print()
    return len(files)


def _materialize_from_zip(source: Path, target: Path, path_patterns: Iterable[str]) -> int:
    written = 0
    with zipfile.ZipFile(source, "r") as zf:
        names = [name for name in zf.namelist() if _zip_name_matches(name, path_patterns)]
        for index, name in enumerate(names, 1):
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            written += 1
            _print_progress("decoding dna", index, len(names))
    if names:
        print()
    return written


def _zip_name_matches(name: str, path_patterns: Iterable[str]) -> bool:
    clean = name.strip("/")
    for pattern in path_patterns:
        if fnmatch.fnmatch(clean, pattern) or clean.startswith(pattern.rstrip("/") + "/"):
            return True
    return False


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _print_progress(label: str, current: int, total: int) -> None:
    step = max(1, total // 40) if total else 1
    if current != total and current % step != 0:
        return
    width = 28
    ratio = current / total if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label}: [{bar}] {current}/{total}", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack or materialize adaptive-agent DNA bundles.")
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="Create a compressed DNA zip from all known bundles.")
    pack.add_argument("--output", default=str(DEFAULT_DNA_PATH))
    pack.add_argument("--bundles", default=str(DEFAULT_BUNDLE_PATH))

    decode = sub.add_parser("decode", help="Decode selected bundles from compressed DNA for an active profile.")
    decode.add_argument("--profile", default=str(DEFAULT_OUTPUT_PATH))
    decode.add_argument("--target", default=str(DEFAULT_TARGET_PATH))
    decode.add_argument("--bundles", default=str(DEFAULT_BUNDLE_PATH))
    decode.add_argument("--source", default=str(DEFAULT_DNA_PATH), help="DNA zip path, or repo directory for development.")

    install = sub.add_parser("install", help="Create a profile and decode relevant bundles from compressed DNA.")
    install.add_argument("--purpose", choices=["coding", "assistant", "research", "automation", "phone_companion"], default="assistant")
    install.add_argument("--personality", default="pragmatic, concise, local-first")
    install.add_argument(
        "--cognition-mode",
        default="auto",
        choices=["auto", "system0", "system1", "system2", "system3", "full"],
    )
    install.add_argument("--allow-network", action="store_true")
    install.add_argument("--allow-background", action="store_true")
    install.add_argument("--profile", default=str(DEFAULT_OUTPUT_PATH))
    install.add_argument("--target", default=str(DEFAULT_TARGET_PATH))
    install.add_argument("--bundles", default=str(DEFAULT_BUNDLE_PATH))
    install.add_argument("--source", default=str(DEFAULT_DNA_PATH), help="Compressed DNA zip to decode from.")

    args = parser.parse_args(argv)
    if args.command == "pack":
        path = pack_dna(output_path=args.output, bundle_path=args.bundles)
        print(f"Wrote DNA archive: {path}")
        return 0

    if args.command == "install":
        profile = build_profile(
            purpose=args.purpose,
            personality=args.personality,
            output_path=args.profile,
            allow_network=args.allow_network,
            allow_background=args.allow_background,
            cognition_mode=args.cognition_mode,
        )
        print(f"Wrote adaptive profile to {args.profile}")
        print(f"Device class: {profile['device']['device_class']}")
        if profile.get("cognition"):
            print(f"Cognition: {profile['cognition']['id']}")

    result = materialize_profile(
        profile_path=args.profile,
        target_path=args.target,
        bundle_path=args.bundles,
        source_path=args.source,
    )
    print(f"Materialized {result.files_written} files into {result.target}")
    print(f"Bundles: {', '.join(result.bundle_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
