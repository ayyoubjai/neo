import json
import re

with open(r'c:\AGI\config\settings.json', 'r') as f:
    settings = json.load(f)

def replace_model(model_name):
    if not isinstance(model_name, str):
        return model_name
    if "qwen3.6" in model_name.lower():
        return "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    if "qwen3" in model_name.lower() or "qwen" in model_name.lower():
        return "Qwen3.5-4B-UD-Q4_K_XL.gguf"
    return model_name

def process_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            process_dict(v)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], str):
                    v[i] = replace_model(v[i])
        elif isinstance(v, str):
            if "model" in k.lower() or k.lower() in ("proposers", "critics"):
                d[k] = replace_model(v)

process_dict(settings)

with open(r'c:\AGI\config\settings.json', 'w') as f:
    json.dump(settings, f, indent=2)
