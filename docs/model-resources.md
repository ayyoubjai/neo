# GPU and CPU model placement

Neo's launchers start a managed pair of llama.cpp services:

- The router on `models.llamacpp_host` keeps at most
  `models.router_max_models` models loaded (default: 1). On a model switch,
  llama.cpp evicts the previous model before loading its replacement.
- A dedicated CPU embedding server listens at `models.embedding_host`
  (configured as `http://127.0.0.1:8082`). It uses `--device none` and
  `--n-gpu-layers 0`. Embedding requests do not change the GPU router's
  resident model. `models.embedding_threads` defaults to 4.

With `models.serialize_model_requests: true`, Neo takes a cross-process lock
for each router request. This prevents another Neo request from switching
models during an active generation. Background work waits for the current
request; it is not a priority queue. The lock is scoped to the endpoint and
Neo's data directory. External clients and separate data directories are not
coordinated by this lock. Do not send concurrent requests from such clients
to a router with a single-model limit.

The current model stays loaded between requests. Only a request for a different
model triggers eviction. Reloading costs time and loses the KV cache; Neo's
conversation and tool results remain in its own storage/context.

Generated presets default to one slot, automatic GPU offloading, and
`models.model_context_size` (8192 tokens). Explicit existing `ctx-size` or
offloading overrides in `models.ini` are preserved. More context uses more
memory; requests exceeding the configured capacity need a larger context or
shorter prompt. Automatic GPU offloading retains CPU fallback for models too
large for available VRAM. This policy does not force oversized models onto GPU.

Projector mappings are preserved. Models are loaded on demand, not all at
startup. The managed router uses the generated presets only, avoiding alternate
directory-discovered aliases with different memory settings.

Restart the model stack with `./start.sh`, or `./start.ps1` on Windows.
If a server is already running, the launcher leaves it alone: stop the old
server first to apply this policy. Close any standalone llama CLI sessions
holding VRAM as well. An existing external server must be configured separately
with the same residency policy and an embedding service on the configured port.
To use one external endpoint for everything, remove `embedding_host`; embeddings
then share the router and can cause additional model swaps.

The supervisor stops its owned services if either exits. It checks CPU-service
health before launching the GPU router. Linux startup writes both services'
output to `data/llama-server.log`; look for `models_max limit reached` when a
model is evicted and compare its generation rate with the earlier run.
Queue waits have a separate `model_queue_timeout_s` (600 seconds); request
timeouts still use the existing model settings. RPC/caller timeouts must also
accommodate queuing and cold model loads.

The policy improves memory availability; it does not fix repetitive generation
or malformed model output. Measure the same image and prompt to compare speed.
