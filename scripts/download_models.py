import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from common.config import load_settings


def _is_local(model_id: str, workspace_root: str) -> bool:
    if not model_id:
        return True
    if os.path.isabs(model_id) or model_id.startswith("."):
        return True

    candidate = os.path.join(workspace_root, model_id)
    return os.path.exists(candidate)


def _target_dir(base_dir: str, model_id: str) -> str:
    safe = model_id.replace("/", "_").replace(":", "_")
    return os.path.join(base_dir, safe)


def _has_model_artifacts(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for name in ("config.json", "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"):
        if os.path.exists(os.path.join(path, name)):
            return True
    try:
        return any(os.scandir(path))
    except OSError:
        return False


def _looks_like_hf_repo(model_id: str) -> bool:
    if not model_id or os.path.isabs(model_id) or model_id.startswith("."):
        return False
    if "/" in model_id:
        return True
    if ":" in model_id:
        return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HF models from config/settings.json or explicit IDs.")
    parser.add_argument(
        "--local-dir",
        default="models",
        help="Directory to store downloaded models (default: ./models)",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=["router", "text", "vision", "embedding"],
        help="Download only a specific model type (can be repeated)",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        help="Explicit HF repo id to download (can be repeated).",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Optional HF revision/branch/tag to download.",
    )
    args = parser.parse_args()

    settings = load_settings()
    models = {}
    if args.model_id:
        models = {f"explicit_{idx}": model_id for idx, model_id in enumerate(args.model_id, start=1)}
    else:
        provider = str(settings.models.get("provider", "")).lower()
        if provider != "hf":
            print(f"[warn] provider is '{provider}', expected hf. Continuing anyway.")
        models = {
            "router": settings.models.get("router_model", ""),
            "text": settings.models.get("text_model", ""),
            "vision": settings.models.get("vision_model", ""),
            "embedding": settings.models.get("embedding_model", ""),
        }
        if args.only:
            models = {k: v for k, v in models.items() if k in set(args.only)}

    base_dir = os.path.abspath(args.local_dir)
    os.makedirs(base_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[error] huggingface_hub not available: {e}")
        print("Install it with: pip install -U huggingface_hub")
        return 1

    for kind, model_id in models.items():
        if not model_id:
            print(f"[skip] {kind}: empty model id")
            continue
        if _is_local(model_id, settings.workspace_root):
            print(f"[skip] {kind}: local path detected ({model_id})")
            continue
        if not args.model_id and not _looks_like_hf_repo(model_id):
            print(f"[skip] {kind}: not a HF repo id ({model_id})")
            continue
        out_dir = _target_dir(base_dir, model_id)
        if _has_model_artifacts(out_dir):
            print(f"[skip] {kind}: already downloaded ({out_dir})")
            continue
        print(f"[download] {kind}: {model_id} -> {out_dir}")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=out_dir,
                local_dir_use_symlinks=False,
                revision=args.revision or None,
            )
        except Exception as e:
            print(f"[error] {kind}: {e}")

    print("[done] model download finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
