# Core release setup

This release runs locally and keeps credentials, model paths, sessions, and
conversation data outside version control. The guided setup creates ignored
machine-local files; do not edit the tracked examples for a personal install.

## 1. Install the application dependencies

Fastest first run (Linux or Windows):

```bash
python scripts/bootstrap_release.py --install
```

This creates a project-local virtual environment and installs requirements. It
uses `npm ci` when Node is available, but it does not install operating-system
software, GPU drivers, Docker, llama.cpp, or llama-swap for you. Check what is
available without changing anything with:

```bash
python scripts/bootstrap_release.py
```

Linux/macOS:

```bash
git clone https://github.com/<your-account>/AGI.git
cd AGI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm ci
```

Windows PowerShell:

```powershell
git clone https://github.com/<your-account>/AGI.git
Set-Location AGI
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm ci
```

`npm ci` is required only for WhatsApp. Audio needs a working microphone and
PortAudio; vision needs a supported camera. `ffmpeg` is recommended for audio
and video messages.

## 2. Choose the local model backend

The simplest configuration path is interactive and works in PowerShell, bash,
and zsh. Telegram secrets are entered with hidden input and written only to
ignored `.env`; WhatsApp is paired later with its QR code.

```bash
python scripts/setup_release.py --interactive
```

PowerShell equivalent:

```powershell
py scripts\bootstrap_release.py --install
.\.venv\Scripts\python.exe scripts\setup_release.py --interactive
```

It asks for the backend, interface(s), cognition model allocation, optional
vision model, and llama.cpp paths. Use the non-interactive commands below when
you want a reproducible installation script.

### Recommended: llama.cpp + llama-swap

Build or install `llama-server` with the hardware backend you need. For a CUDA
build on Linux:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j
```

Install `llama-swap` from its release binary, package manager, or source. On
Windows, its documented package-manager command is:

```powershell
winget install llama-swap
```

Place your downloaded GGUF files in one directory. Then create local
configuration, assigning a fast model to routing/System 0 and a larger model to
Systems 1–2:

```bash
python scripts/setup_release.py \
  --provider llamaswap \
  --model-dir /absolute/path/to/models \
  --llama-server /absolute/path/to/llama-server \
  --fast-model qwen2.5-3b-instruct-q4_k_m.gguf \
  --reasoning-model qwen2.5-7b-instruct-q4_k_m.gguf \
  --embedding-model nomic-embed-text-v1.5-q4_k_m.gguf \
  --interfaces telegram,whatsapp
```

For a vision-language model, pass its model GGUF and matching `mmproj` file:

```bash
python scripts/setup_release.py \
  --provider llamaswap \
  --model-dir /absolute/path/to/models \
  --llama-server /absolute/path/to/llama-server \
  --fast-model qwen2.5-3b-instruct-q4_k_m.gguf \
  --reasoning-model qwen2.5-7b-instruct-q4_k_m.gguf \
  --embedding-model nomic-embed-text-v1.5-q4_k_m.gguf \
  --vision-model Qwen2.5-VL/qwen2.5-vl-7b-instruct-q4_k_m.gguf \
  --vision-mmproj Qwen2.5-VL/mmproj-qwen2.5-vl-7b-f16.gguf \
  --interfaces local --local-senses text,vision --force
```

Start llama-swap in one terminal, then keep it running:

```bash
llama-swap --config config/llama-swap.local.yaml --listen 127.0.0.1:8080
```

For PowerShell, use backticks for line continuations and quote Windows paths:

```powershell
python scripts/setup_release.py --provider llamaswap --model-dir "C:\AI\models" --llama-server "C:\AI\llama.cpp\build\bin\Release\llama-server.exe" --fast-model "qwen2.5-3b-instruct-q4_k_m.gguf" --reasoning-model "qwen2.5-7b-instruct-q4_k_m.gguf" --embedding-model "nomic-embed-text-v1.5-q4_k_m.gguf" --interfaces telegram,whatsapp
llama-swap --config config/llama-swap.local.yaml --listen 127.0.0.1:8080
```

`--vision-model` and `--vision-mmproj` may include paths relative to
`--model-dir`; the filename itself becomes the model ID. The generated files are `config/settings.local.json` and
`config/llama-swap.local.yaml`. Change the `cognition_*_model` values in the
former whenever you want a different cognition allocation. Model IDs must be
identical in both files.

### Alternative: Ollama

```bash
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
python scripts/setup_release.py \
  --provider ollama \
  --fast-model qwen2.5:3b-instruct \
  --reasoning-model qwen2.5:7b-instruct \
  --embedding-model nomic-embed-text \
  --interfaces text
```

Use an Ollama vision-capable model with `--vision-model` if you need image
analysis from Telegram, WhatsApp, or the visual tools.

## 3. Start local web search

SearXNG is private by default and deliberately uses port `8081`; llama-swap
uses `8080`.

```bash
docker compose -f docker-compose.searxng.yml up -d
curl --fail "http://127.0.0.1:8081/search?q=local+llm&format=json"
```

The assistant uses it through the permission-gated `net.search` tool. Do not
publish this container directly to the Internet. If you expose it beyond the
host, add a reverse proxy, authentication, TLS, and rate limiting.

## 4. Enable messaging

The setup script creates `.env` once and never overwrites it. Add only the
platforms you use.

Telegram:

```dotenv
TELEGRAM_BOT_TOKEN=<token-from-BotFather>
TELEGRAM_ALLOWED_CHAT_IDS=<your-numeric-chat-id>
TELEGRAM_RESPONSE_MODE=same
```

WhatsApp uses a linked device. Keep self-chat mode enabled until you have
explicitly reviewed the access model:

```dotenv
WHATSAPP_SELF_CHAT_ONLY=true
WHATSAPP_TRIGGER_PREFIX=/agi
WHATSAPP_QR_IN_TERMINAL=true
```

Run the assistant and scan the QR shown in the terminal with WhatsApp Linked
Devices. The generated authentication state stays under ignored
`data/whatsapp_auth/`.

## 5. Start the assistant

```bash
./scripts/run_all.sh
```

PowerShell:

```powershell
.\scripts\run_all.ps1
```

For local voice, rerun setup with `--interfaces audio`; for local camera
vision, use `--interfaces local --local-senses text,vision`. Do not combine
local text and audio: they both read standard input.

## Publish preparation

Before making a repository public, revoke the existing Google OAuth secret and
WhatsApp linked-device session, then remove all historical copies. Removing
the files in a new commit alone does not remove them from old commits.
