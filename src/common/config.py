import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Settings:
    workspace_root: str
    data_dir: str
    rpc: Dict[str, Any]
    interface: Dict[str, Any]
    voice: Dict[str, Any]
    vision: Dict[str, Any]
    video: Dict[str, Any]
    telegram: Dict[str, Any]
    google: Dict[str, Any]
    tool: Dict[str, Any]
    search: Dict[str, Any]
    orchestrator: Dict[str, Any]
    evolve: Dict[str, Any]
    models: Dict[str, Any]
    autonomy: Dict[str, Any] = field(default_factory=dict)


DEFAULT_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "settings.json",
)
EXAMPLE_SETTINGS_PATH = os.path.join(
    os.path.dirname(DEFAULT_SETTINGS_PATH),
    "settings.example.json",
)


def resolve_settings_path(path: str | None = None) -> str:
    """Return the active settings file without requiring users to edit tracked config.

    A command-line caller may provide a path explicitly. Otherwise an
    ``AGI_SETTINGS_PATH`` environment variable wins, followed by the ignored
    ``config/settings.local.json`` file. The checked-in ``settings.json`` is a
    backwards-compatible fallback for existing installations.
    """
    if path:
        return os.path.abspath(path)
    configured_path = os.environ.get("AGI_SETTINGS_PATH", "").strip()
    if configured_path:
        return os.path.abspath(configured_path)
    local_path = os.path.join(os.path.dirname(DEFAULT_SETTINGS_PATH), "settings.local.json")
    if os.path.exists(local_path):
        return local_path
    if not os.path.exists(DEFAULT_SETTINGS_PATH) and os.path.exists(EXAMPLE_SETTINGS_PATH):
        return EXAMPLE_SETTINGS_PATH
    return DEFAULT_SETTINGS_PATH


def _repo_root(settings_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(settings_path), ".."))


def _normalize_path(path_value: str, base_root: str) -> str:
    if not path_value:
        return base_root
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_root, path_value))


def _remap_path(path_value: str, raw_root: str, resolved_root: str, base_root: str) -> str:
    if not path_value:
        return resolved_root
    if os.path.isabs(path_value):
        if raw_root and os.path.isabs(raw_root):
            raw_root_norm = os.path.normcase(os.path.normpath(raw_root))
            path_norm = os.path.normcase(os.path.normpath(path_value))
            if path_norm == raw_root_norm or path_norm.startswith(raw_root_norm + os.sep):
                rel = os.path.relpath(path_value, raw_root)
                return os.path.abspath(os.path.join(resolved_root, rel))
        return path_value
    return os.path.abspath(os.path.join(base_root, path_value))


def load_settings(path: str | None = None) -> Settings:
    path = resolve_settings_path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    base_root = _repo_root(path)
    raw_workspace = raw.get("workspace_root", "")
    workspace_root = _normalize_path(raw_workspace, base_root)
    if os.path.isabs(raw_workspace) and not os.path.exists(raw_workspace):
        workspace_root = base_root

    raw_data_dir = raw.get("data_dir", "")
    data_dir = _remap_path(raw_data_dir, raw_workspace, workspace_root, base_root)

    evolve = dict(raw.get("evolve", {}))
    evolve_candidate = evolve.get("candidate_dir", "")
    evolve["candidate_dir"] = _remap_path(evolve_candidate, raw_workspace, workspace_root, base_root)
    for key in (
        "auto_run_config_path",
        "auto_run_log_path",
        "scheduler_state_path",
        "scheduler_lock_path",
        "run_lock_path",
        "resource_disk_path",
    ):
        evolve_path = evolve.get(key)
        if evolve_path:
            evolve[key] = _remap_path(evolve_path, raw_workspace, workspace_root, base_root)

    voice = dict(raw.get("voice", {}))
    transcript_path = voice.get("transcript_log_path")
    if transcript_path:
        voice["transcript_log_path"] = _remap_path(transcript_path, raw_workspace, workspace_root, base_root)
    audio_debug_path = voice.get("audio_debug_log_path")
    if audio_debug_path:
        voice["audio_debug_log_path"] = _remap_path(audio_debug_path, raw_workspace, workspace_root, base_root)

    vision = dict(raw.get("vision", {}))
    scene_log_path = vision.get("scene_log_path")
    if scene_log_path:
        vision["scene_log_path"] = _remap_path(scene_log_path, raw_workspace, workspace_root, base_root)
    broker_dir = vision.get("broker_dir")
    if broker_dir:
        vision["broker_dir"] = _remap_path(broker_dir, raw_workspace, workspace_root, base_root)
    broker_clip_export_dir = vision.get("broker_clip_export_dir")
    if broker_clip_export_dir:
        vision["broker_clip_export_dir"] = _remap_path(
            broker_clip_export_dir,
            raw_workspace,
            workspace_root,
            base_root,
        )
    crop_temp_dir = vision.get("object_memory_crop_temp_dir")
    if crop_temp_dir:
        vision["object_memory_crop_temp_dir"] = _remap_path(crop_temp_dir, raw_workspace, workspace_root, base_root)
    export_dir = vision.get("object_memory_export_dir")
    if export_dir:
        vision["object_memory_export_dir"] = _remap_path(export_dir, raw_workspace, workspace_root, base_root)

    video = dict(raw.get("video", {}))
    sampled_frame_dir = video.get("sampled_frame_dir")
    if sampled_frame_dir:
        video["sampled_frame_dir"] = _remap_path(sampled_frame_dir, raw_workspace, workspace_root, base_root)
    audio_extract_dir = video.get("audio_extract_dir")
    if audio_extract_dir:
        video["audio_extract_dir"] = _remap_path(audio_extract_dir, raw_workspace, workspace_root, base_root)

    google = dict(raw.get("google", {}))
    client_secrets_path = google.get("client_secrets_path")
    if client_secrets_path:
        google["client_secrets_path"] = _remap_path(client_secrets_path, raw_workspace, workspace_root, base_root)

    rpc = raw.get("rpc") or {}
    return Settings(
        workspace_root=workspace_root,
        data_dir=data_dir,
        rpc=rpc,
        interface=raw.get("interface", {}),
        voice=voice,
        vision=vision,
        video=video,
        telegram=raw.get("telegram", {}),
        google=google,
        tool=raw["tool"],
        search=raw.get("search", {}),
        orchestrator=raw.get("orchestrator", {}),
        evolve=evolve,
        models=raw.get("models", {}),
        autonomy=raw.get("autonomy", {}),
    )
