from typing import Optional

from model_server.hf_client import generate_text as hf_generate_text
from model_server.llamacpp_client import generate as llamacpp_generate
from model_server.ollama_client import generate as ollama_generate


SUPPORTED_LOCAL_PROVIDERS = ("ollama", "hf", "llamacpp")


def generate_local_text(
    prompt: str,
    model: str,
    *,
    provider: str,
    temperature: float = 0.1,
    max_new_tokens: Optional[int] = None,
    system: str = "",
    response_format: Optional[str] = None,
) -> str:
    """Generate text through any supported local model backend."""
    normalized_provider = str(provider or "").strip().lower()
    if normalized_provider == "hf":
        return hf_generate_text(
            prompt,
            model,
            temperature=float(temperature),
            max_new_tokens=max_new_tokens,
        )

    options = {"temperature": float(temperature)}
    if max_new_tokens is not None:
        options["num_predict"] = int(max_new_tokens)

    if normalized_provider == "ollama":
        return ollama_generate(
            prompt,
            model,
            system=system or None,
            options=options,
            response_format=response_format,
        )
    if normalized_provider == "llamacpp":
        return llamacpp_generate(
            prompt,
            model,
            system=system or None,
            options=options,
            response_format=response_format,
        )
    supported = ", ".join(SUPPORTED_LOCAL_PROVIDERS)
    provider_label = normalized_provider or "<empty>"
    raise ValueError(f"unsupported local model provider: {provider_label}; expected one of {supported}")
