import json
import re
import os

with open(r'c:\AGI\config\settings.json', 'r') as f:
    text = f.read()

# Fix the corrupted JSON string by extracting the first valid JSON object
# The file has a duplicate tail. We can find the last '}' of the main object.
# The main object ends at line 439 in the corrupted file, which has a single '}'.
try:
    # A simple way to fix it is to decode using JSONDecoder
    decoder = json.JSONDecoder()
    settings, idx = decoder.raw_decode(text)
except Exception as e:
    print("Failed to decode json:", e)
    import sys
    sys.exit(1)

def replace_model(model_name):
    if not isinstance(model_name, str):
        return model_name
    if "qwen3.6" in model_name.lower():
        return "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    if "qwen" in model_name.lower():
        return "Qwen3.5-4B-UD-Q4_K_XL.gguf"
    if "nomic" in model_name.lower():
        return "Qwen3.5-4B-UD-Q4_K_XL.gguf"
    return model_name

def process_dict(d):
    for k, v in list(d.items()):
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

print("Settings successfully fixed and updated.")
