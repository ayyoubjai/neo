from functools import lru_cache
from typing import Any, Dict, Optional

from common.config import load_settings


class HfError(Exception):
    pass


def _resolve_dtype(dtype_value: str):
    if not dtype_value or dtype_value == "auto":
        return "auto"
    try:
        import torch
    except Exception:
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(dtype_value).lower(), "auto")


def _load_hf_settings() -> Dict[str, Any]:
    settings = load_settings()
    models = settings.models
    return {
        "device_map": models.get("hf_device_map", "auto"),
        "dtype": _resolve_dtype(str(models.get("hf_dtype", "auto"))),
        "trust_remote_code": bool(models.get("hf_trust_remote_code", True)),
        "max_new_tokens": int(models.get("hf_max_new_tokens", 512)),
        "vision_max_new_tokens": int(models.get("hf_vision_max_new_tokens", 512)),
        "router_max_new_tokens": int(models.get("hf_router_max_new_tokens", 128)),
    }


@lru_cache(maxsize=4)
def _load_text_model(model_id: str, device_map: str, dtype: object, trust_remote_code: bool):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise HfError(f"transformers not available: {e}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        raise HfError(str(e))


@lru_cache(maxsize=2)
def _load_vision_model(model_id: str, device_map: str, dtype: object, trust_remote_code: bool):
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except Exception as e:
        raise HfError(f"transformers not available: {e}")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        return processor, model
    except Exception as e:
        raise HfError(str(e))


def generate_text(prompt: str, model_id: str, temperature: float = 0.2, max_new_tokens: Optional[int] = None) -> str:
    cfg = _load_hf_settings()
    if max_new_tokens is None:
        max_new_tokens = cfg["max_new_tokens"]
    tokenizer, model = _load_text_model(
        model_id, cfg["device_map"], cfg["dtype"], cfg["trust_remote_code"]
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        do_sample = temperature and temperature > 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(do_sample),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        outputs = model.generate(**inputs, **gen_kwargs)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            return decoded[len(prompt) :].lstrip()
        return decoded
    except Exception as e:
        raise HfError(str(e))


def generate_router(prompt: str, model_id: str) -> str:
    cfg = _load_hf_settings()
    return generate_text(prompt, model_id, temperature=0.0, max_new_tokens=cfg["router_max_new_tokens"])


def generate_vision(
    prompt: str,
    model_id: str,
    image_bytes: bytes,
    max_new_tokens: Optional[int] = None,
) -> str:
    cfg = _load_hf_settings()
    if max_new_tokens is None:
        max_new_tokens = cfg["vision_max_new_tokens"]
    processor, model = _load_vision_model(
        model_id, cfg["device_map"], cfg["dtype"], cfg["trust_remote_code"]
    )
    try:
        from PIL import Image
        import io
    except Exception as e:
        raise HfError(f"PIL not available: {e}")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        if decoded:
            return decoded[0]
        return ""
    except Exception as e:
        raise HfError(str(e))
