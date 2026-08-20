import json
import urllib.request
import urllib.error

payloads = [
    {"model": "ggml-vocab-qwen35", "prompt": "Hello", "stream": False},
    {"model": "ggml-vocab-qwen35", "prompt": "Hello", "stream": False, "system": "System prompt"},
    {"model": "ggml-vocab-qwen35", "prompt": "Hello", "stream": False, "response_format": {"type": "json_object"}},
]

for i, p in enumerate(payloads):
    try:
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request('http://127.0.0.1:8080/completion', data=data, headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req)
        print(f"Payload {i}: OK")
    except urllib.error.HTTPError as e:
        print(f"Payload {i}: HTTP {e.code} {e.reason} - {e.read().decode()}")
    except Exception as e:
        print(f"Payload {i}: {e}")

try:
    data = json.dumps({"model": "ggml-vocab-bert-bge", "content": "Hello"}).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:8080/embedding', data=data, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req)
    print("Embedding: OK")
except urllib.error.HTTPError as e:
    print(f"Embedding: HTTP {e.code} {e.reason} - {e.read().decode()}")
except Exception as e:
    print(f"Embedding: {e}")
