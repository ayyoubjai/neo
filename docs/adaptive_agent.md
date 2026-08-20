# Adaptive Agent Bootstrap

This layer is the first implementation of the DNA/cell idea:

- `config/adaptive_agent/genes.json` is the gene catalog.
- `config/adaptive_agent/bundles.json` maps genes to materializable source bundles.
- `scripts/adaptive_agent_wizard.py` probes the current device, asks for purpose/personality, resolves matching genes, and writes `config/adaptive_agent/active_profile.json`.
- `scripts/adaptive_agent_dna.py` packs the compressed DNA archive and decodes only the bundles needed by the active profile.
- `scripts/adaptive_agent_start_model.py` reads the active profile and launches either `llama-server` or `llama-swap`.
- `scripts/adaptive_agent_entrypoint.py` starts the selected model backend and then runs the normal system startup script.

Run the wizard:

```bash
python3 scripts/adaptive_agent_wizard.py --purpose phone_companion --personality "concise, private, local-first"
```

Inspect the runtime without launching:

```bash
python3 scripts/adaptive_agent_start_model.py --dry-run
```

Start the whole system through the adaptive entrypoint:

```bash
python3 scripts/adaptive_agent_entrypoint.py
```

Preview both commands first:

```bash
python3 scripts/adaptive_agent_entrypoint.py --dry-run
```

Pack the full compressed DNA archive:

```bash
python3 scripts/adaptive_agent_dna.py pack --output dist/adaptive_agent_dna.zip
```

Decode only the relevant bundles for this device and purpose:

```bash
python3 scripts/adaptive_agent_dna.py decode --source dist/adaptive_agent_dna.zip --target runtime/adaptive_cell
```

On Termux, set `LLAMA_CPP_SERVER` if your binary is not named `llama-server`:

```bash
export LLAMA_CPP_SERVER="$HOME/llama.cpp/build/bin/llama-server"
```

On stronger machines, the selected service gene uses `llama-swap` by default:

```bash
llama-swap --config config/config.yaml --listen localhost:8080
```

Override the binary when needed:

```bash
export LLAMA_SWAP_BINARY="/path/to/llama-swap"
```

The generated profile deliberately references local GGUF paths such as `models/Qwen3.5-0.8B-Q4_K_M.gguf` or `models/pc-medium.gguf`. Put your GGUF model there or edit `payload.model_path` in the selected model gene.

For weak devices, the model service gene uses one direct `llama-server`. For stronger machines, it uses `llama-swap` with `config/config.yaml`, matching:

```bash
llama-swap --config config/config.yaml --listen localhost:8080
```

The existing system bootstrap now also creates the adaptive profile from the free-form purpose. It infers one of:

- `assistant`
- `coding`
- `research`
- `automation`
- `phone_companion`

Then it can materialize the selected bundles with a progress bar.
