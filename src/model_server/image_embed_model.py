from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
import os
from typing import List, Tuple

from common.config import load_settings
from tool_runtime.sandbox import SandboxViolation, resolve_workspace_path


def _load_image_bytes(image_ref: str = "", image_b64: str = "") -> bytes:
    if image_b64:
        return base64.b64decode(image_b64.encode("ascii"), validate=True)
    if not image_ref:
        raise FileNotFoundError("Image reference is empty")
    if image_ref.startswith("workspace:/"):
        settings = load_settings()
        path = resolve_workspace_path(settings.workspace_root, image_ref)
    elif image_ref.startswith("file:"):
        path = image_ref[len("file:") :]
    else:
        path = image_ref
    if not os.path.exists(path):
        raise FileNotFoundError("Image not found")
    with open(path, "rb") as f:
        return f.read()


def _hash_embedding(raw: bytes, dim: int = 32) -> List[float]:
    digest = hashlib.sha256(raw).digest()
    values = []
    for b in digest[:dim]:
        values.append((b / 255.0) * 2.0 - 1.0)
    return values


def _histogram_embedding(raw: bytes) -> List[float]:
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return []
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        width, height = image.size
        resized = image.resize((64, 64))
        arr = np.asarray(resized, dtype=np.float32)
    except Exception:
        return []

    features: List[float] = []
    for channel_index in range(3):
        channel = arr[:, :, channel_index]
        hist, _ = np.histogram(channel, bins=16, range=(0, 256))
        hist = hist.astype(np.float32)
        total = float(hist.sum()) or 1.0
        features.extend((hist / total).tolist())
        channel /= 255.0
        features.append(float(channel.mean()))
        features.append(float(channel.std()))

    aspect_ratio = float(width) / float(max(height, 1))
    size_ratio = min(1.0, float(width * height) / float(1920 * 1080))
    features.append(min(4.0, aspect_ratio) / 4.0)
    features.append(size_ratio)

    norm = math.sqrt(sum(value * value for value in features))
    if norm > 0:
        features = [float(value / norm) for value in features]
    return features


def embed_image(image_ref: str = "", image_b64: str = "") -> Tuple[List[float], str]:
    try:
        raw = _load_image_bytes(image_ref=image_ref, image_b64=image_b64)
    except (FileNotFoundError, SandboxViolation, ValueError, binascii.Error):
        return [], "image-load-error"

    histogram = _histogram_embedding(raw)
    if histogram:
        return histogram, "rgb-hist-v1"
    return _hash_embedding(raw), "sha256-image-v1"
