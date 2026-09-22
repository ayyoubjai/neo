#!/usr/bin/env python3
"""
Generate a simple models.ini from config/settings*.json

This writes an INI file with a small set of presets derived from the
settings file so `llama-server --models-preset ./models.ini` can use it.

llama-server treats every section as a model preset.  Therefore the
application's model-role names must not be written as INI options: the
selected filenames are used as both the preset names and model values.
"""
import argparse
import configparser
import json
import os
from pathlib import Path
import sys


def load_settings():
    configured_path = os.environ.get('AGI_SETTINGS_PATH', '').strip()
    if configured_path:
        return json.loads(Path(configured_path).read_text(encoding='utf-8'))
    for p in (Path('config/settings.local.json'), Path('config/settings.json'), Path('config/settings.example.json')):
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return {}


def write_ini(settings: dict, out_path: Path, models_dir_override=None):
    models = settings.get('models', {})
    configured = {
        'embedding': str(models.get('embedding_model', '')).strip(),
        'router': str(models.get('router_model', '')).strip(),
        'text': str(models.get('text_model', '')).strip(),
        'vision': str(models.get('vision_model', '')).strip(),
    }
    configured_dir = str(models_dir_override or models.get('models_dir') or '').strip()
    models_dir = Path(configured_dir).expanduser() if configured_dir else Path('models')
    if not models_dir.is_absolute():
        models_dir = (out_path.parent.parent / (models_dir or 'models')).resolve()
    files = sorted(
        p for p in models_dir.glob('*.gguf')
        if not p.name.lower().startswith('mmproj')
    )

    def resolve_model(value):
        if not value:
            return None
        for path in files:
            if value == path.name or value == path.stem:
                return path.name
        return value

    selected = {role: resolve_model(value) for role, value in configured.items()}
    presets = {}
    # Publish every discovered model, not just the three role-selected ones.
    for path in files:
        filename = path.name
        presets[filename] = {'model': str(models_dir / filename), 'embedding': False}
    # Keep the IDs used by settings as aliases, while pointing them at the
    # actual discovered filenames (important when settings omit .gguf).
    for role, value in configured.items():
        filename = selected[role]
        if value and filename:
            presets[value] = {'model': str(models_dir / filename), 'embedding': role == 'embedding'}
            if filename in presets and role == 'embedding':
                presets[filename]['embedding'] = True

    # Keep model-specific CLI options edited in the preset file. The leading
    # version declaration is outside a section in llama.cpp's INI format.
    existing = configparser.ConfigParser(interpolation=None)
    if out_path.exists():
        existing.read_string('[DEFAULT]\n' + out_path.read_text(encoding='utf-8'))
    options_by_model = {}
    for section in existing.sections():
        options = {key: value for key, value in existing.items(section)
                   if key not in {'version', 'model', 'embeddings'}}
        model_path = existing.get(section, 'model', fallback='')
        if model_path:
            options_by_model[model_path] = options

    vision_path = selected.get('vision')
    projector = str(models.get('vision_mmproj') or '').strip()
    if projector and not vision_path:
        raise ValueError('models.vision_mmproj requires models.vision_model')
    if projector:
        projector = str(models_dir / Path(projector).expanduser())

    lines = ['version = 1', '']

    # Each section is a llama-server model preset.  Do not add project-level
    # keys such as router_model or embedding_model here; those belong in
    # settings.local.json and are not llama-server CLI arguments.
    for name, preset in presets.items():
        lines.append(f'[{name}]')
        lines.append(f"model = {preset['model']}")
        if preset['embedding']:
            lines.append('embeddings = true')
        options = dict(options_by_model.get(preset['model'], {}))
        if existing.has_section(name) and existing.get(name, 'model', fallback='') == preset['model']:
            options.update({key: value for key, value in existing.items(name)
                            if key not in {'version', 'model', 'embeddings'}})
        if projector and preset['model'] == str(models_dir / vision_path):
            options['mmproj'] = projector
            # llama.cpp warns that Qwen-VL grounding becomes unreliable below
            # 1024 visual tokens. Keep this configurable for other VLMs.
            options.setdefault('image-min-tokens', str(models.get('vision_image_min_tokens', 1024)))
        if models.get('serialize_model_requests', False):
            # One request at a time needs one KV-cache slot. Bound context
            # allocation instead of allowing each model's training maximum.
            options['parallel'] = '1'
            options.setdefault('ctx-size', str(models.get('model_context_size', 8192)))
            options.setdefault('n-gpu-layers', 'auto')
            options['load-on-startup'] = 'false'
            if preset['embedding']:
                options['device'] = 'none'
                options['n-gpu-layers'] = '0'
                options['ctx-size'] = '2048'
        lines.extend(f'{key} = {value}' for key, value in options.items())
        lines.append('')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='config/models.ini', help='Output INI path')
    parser.add_argument('--models-dir', help='Directory containing GGUF models (overrides models.models_dir)')
    args = parser.parse_args()

    settings = load_settings()
    out = Path(args.output)
    try:
        write_ini(settings, out, args.models_dir)
    except Exception as e:
        print('ERROR writing models.ini:', e, file=sys.stderr)
        sys.exit(2)

    print(f'Wrote models preset to {out}')


if __name__ == '__main__':
    main()
