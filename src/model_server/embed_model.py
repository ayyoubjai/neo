import hashlib
from typing import List, Tuple

from common.config import load_settings
from common.error_log import log_exception
from model_server.ollama_client import OllamaError, embed as ollama_embed, provider
from model_server.llamacpp_client import LlamacppError, embed as llamacpp_embed


def embed_text(text: str, dim: int = 32) -> Tuple[List[float], str]:
    settings = load_settings()
    model = settings.models.get("embedding_model", "")
    current_provider = provider()
    if current_provider == "llamacpp" and model:
        try:
            resp = llamacpp_embed(text, model)
            vec = resp.get("embedding", [])
            if vec:
                return vec, f"llamacpp:{model}"
        except LlamacppError as e:
            log_exception(
                "embedding_error",
                e,
                {"provider": "llamacpp", "model": model},
            )

    if model:
        try:
            resp = ollama_embed(text, model)
            vec = resp.get("embedding", [])
            if vec:
                return vec, f"ollama:{model}"
        except OllamaError as e:
            log_exception(
                "embedding_error",
                e,
                {"provider": "ollama", "model": model},
            )

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for b in digest[:dim]:
        vals.append((b / 255.0) * 2.0 - 1.0)
    return vals, "sha256-text-v1"
