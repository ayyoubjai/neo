Hybrid Voice-First Assistant (MVP)

> For the streamlined local release—llama.cpp + llama-router, cognition model
> selection, Telegram, WhatsApp, voice, vision, and private SearXNG—follow
> [the core setup guide](docs/release-setup.md). It creates local configuration
> files and keeps credentials and runtime data out of Git.

This is a Python MVP scaffold for the hybrid voice-first assistant spec. It uses TCP JSONL for IPC, SQLite for memory/audit, and Ollama-backed model stubs with fallback behavior.

Architecture
- Voice Daemon (VD): local text/audio/vision interface runtime, sends STT events and handles permission responses.
- Telegram Daemon (TG): Telegram long-polling bridge that sends/receives the same orchestrator events.
- Orchestrator (ORCH): routing, policy, planning, memory, tool loop, response synthesis.
- Tool Runtime (TR): sandboxed tool execution with audit logging.
- Model Server (MS): router + text + embedding + vision via Ollama (fallback to stubs if Ollama is unavailable).
- Storage (ST): SQLite memory store + append-only audit log.
- Autonomy Runtime (AR): optional epistemic + power-process background loops backed by Neo4j.

Quickstart
1) Create a virtual environment (no external deps required for text mode):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run all services:

```bash
./scripts/run_all.sh
```

Windows (PowerShell):

```powershell
.\scripts\run_all.ps1
```

Model server (llama-server)
- When using the local `llamacpp` provider the project expects a running
  `llama-server` (router mode). The setup wizard no longer generates a
  `config/llama-router.local.yaml`; instead the chosen models directory is
  recorded in `config/settings.local.json` under `models.models_dir`.
- To start the server manually run:

```bash
llama-server --models-dir ./models --host 127.0.0.1 --port 8080
```

- When you run `./start.sh` (or `start.ps1` on Windows), the script will
  automatically launch `llama-server --models-dir <dir>` for you if
  `models.provider` is set to `llamacpp` in `config/settings.local.json`.
  The script will wait for the server health endpoint before continuing.

- The PowerShell helper `start.ps1` provides equivalent behavior on
  Windows.

Optional autonomy runtime
- The `epistemic` and `power_process` loops are now available as an opt-in managed service.
- Configure them under `autonomy` in `config/settings.json`.
- Enable one or both of `autonomy.epistemic_enabled` / `autonomy.power_process_enabled`, and set `autonomy.start_with_run_all=true` if you want `run_all` to launch them.
- `autonomy.enabled` is a master kill switch; set it to `false` to force the runtime off even if loop flags are enabled.
- The autonomy runtime persists theories, goals, and observations in Neo4j, and fulfills exploration/execution through orchestrator COGNITION mode.
- Start Neo4j first with `docker compose -f docker-compose.neo4j.yml up -d`.
- `autonomy.cognition_timeout_s` controls per-request COGNITION timeout, and `autonomy.permission_policy` controls how autonomy answers tool approval requests when the orchestrator asks.
- You can also launch it directly with `./scripts/run_autonomy_runtime.sh` or the PowerShell equivalent.
- Cognition model overrides live under `orchestrator` in `config/settings.json`: `cognition_system1_model` and `cognition_system2_model`.
- `orchestrator.cognition_system0_enabled` controls the first-pass gate. When `true` (default), System0 may answer simple requests directly; when `false`, normal requests go to the thinking router.
- `orchestrator.cognition_context_sources` controls pre-cognition context retrieval: `semantic`, `memory`, or `both` (default). The retrieved context is prepared once before System0/router execution and reused after escalation.
- Each cognition turn runs the `sys.time` tool once and adds its UTC timestamp to every model prompt; if the tool runtime is unavailable, the orchestrator UTC clock is used as a logged fallback.
- The main thinking router can also use its own override pair: `cognition_route_model` and `cognition_route_options`.
- Per-slot inference controls live next to those model settings, for example `cognition_system1_options`, `cognition_system2_options`, `cognition_route_options`, `cognition_init_options`, `cognition_repair_options`, `cognition_judge_options`, `final_response_critic_options`, and `tool_codegen_critic_options`.
- Use `thinking` and `reasoning_effort` inside those option objects when the selected model/provider supports them.
- System3 has no global model override, but it does have global default options via `cognition_system3_options`; those defaults are merged with each peer's own `options` in `cognition_peer_pools`, with the peer-specific values winning.
- System3 uses only the per-peer `model` values configured in `cognition_peer_pools`.
- System3 selects a winner only when its average review score reaches `cognition_system3_safety_threshold` (default 7.0/10) and the proposal has reviews from at least `cognition_system3_review_quorum` distinct critics (default 2); otherwise it falls back without a winner.
- Cognition now preserves checkpoints across bounded continuation and retries, distinguishes verified outcomes from model-reviewed answers, and quarantines learned artifacts until reviewed. See [cognition reliability](docs/cognition-reliability.md) for acceptance checks, resume, learning review, and System2/System3 evaluation.

Interface mode
- Primary switch is `interface.mode` in `config/settings.json`.
- `interface.mode` can be a single string or a list, for example `["whatsapp", "telegram"]`.
- Valid values are `text`, `audio`, `vision`, `local`, `telegram`, `whatsapp`.
- `text` = local text daemon.
- `audio` = mic/VAD/STT local daemon.
- `vision` = camera-only YOLO local daemon.
- `local` = compose local senses from `interface.senses`, for example `["audio", "vision"]` or `["text", "vision"]`.
- `telegram` = Telegram daemon.
- `whatsapp` = WhatsApp linked-device daemon.
- Multiple interface modes currently support the daemon interfaces `telegram` and `whatsapp`.
- `interface.conversation_mode` controls how multiple daemons share context:
  - `shared` = one conversation/history across the active daemons.
  - `separate` = each daemon keeps its own conversation/history.

Telegram mode
- Create a bot with BotFather and copy the token.
- Fill `.env` (a template is provided in `.env.example`).
- Required: `TELEGRAM_BOT_TOKEN`
- Optional: `TELEGRAM_ALLOWED_CHAT_IDS` (comma-separated) or legacy `TELEGRAM_ALLOWED_CHAT_ID`, `TELEGRAM_API_BASE_URL`, `TELEGRAM_POLL_TIMEOUT_S`, `TELEGRAM_REQUEST_TIMEOUT_S`, `TELEGRAM_MODEL_RPC_TIMEOUT_S`, `TELEGRAM_SEND_TYPING_ACTION`, `TELEGRAM_SEND_USER_UPDATES`, `TELEGRAM_MAX_MESSAGE_CHARS`, `TELEGRAM_RESPONSE_MODE` (`text`|`audio`|`same`), `TELEGRAM_AUDIO_RESPONSE_MAX_CHARS`, `TELEGRAM_TTS_RATE`, `TELEGRAM_TTS_VOLUME`, `TELEGRAM_MAX_DOWNLOAD_BYTES`, `TELEGRAM_TEXT_ATTACHMENT_MAX_CHARS`, `TELEGRAM_PDF_MAX_PAGES`, `TELEGRAM_PDF_MAX_CHARS`, `TELEGRAM_STT_MODEL`, `TELEGRAM_STT_DEVICE`, `TELEGRAM_STT_COMPUTE_TYPE`
- `telegram.allowed_chat_ids` in `config/settings.json` also accepts a JSON array such as `[123456789, 987654321]`.
- If you see `[telegram] getUpdates failed: Telegram network error: [Errno 11001] getaddrinfo failed`, that is a DNS lookup failure before Telegram responds. The usual causes are a bad `TELEGRAM_API_BASE_URL` / `telegram.api_base_url` host, or local DNS/network connectivity that cannot resolve or reach Telegram.
- `TELEGRAM_MODEL_RPC_TIMEOUT_S=0` disables the model RPC timeout entirely.
- `TELEGRAM_RESPONSE_MODE=same` replies in audio only when the user input was audio; otherwise replies in text.
- `TELEGRAM_SEND_USER_UPDATES=false` disables progress messages such as "Transcribing audio...".
- `telegram.unsupported_files_to_agent=true` (in `config/settings.json`) forwards unsupported document types to COMPLEX mode for file analysis instead of rejecting them.
- `run_all` scripts auto-load `.env`.
- Start with `./scripts/run_all.sh` (or PowerShell equivalent) when `interface.mode` is `telegram`.
- Supported Telegram inputs: text, audio messages, image attachments, video attachments, PDFs (text-extraction), and text/code documents.
- Current limitation: scanned/image-only PDFs still need OCR support.
- Commands in Telegram:
  - `/help` and `/status`
  - `/approve <request_id>` or `/deny <request_id>`
  - `+ <text>` to patch the previous turn
  - `/wake` to emit a wake event

WhatsApp mode
- Requires Node.js 20+ and `npm install` in the repo root.
- Uses Baileys as a linked-device bridge for a personal WhatsApp account or WhatsApp Business app account.
- Set `interface.mode` to `whatsapp` in `config/settings.json`.
- Main config lives under `whatsapp` in `config/settings.json`.
- Recommended self-chat setup:
  - Keep `whatsapp.self_chat_only=true`.
  - Send prompts from your personal "message yourself" chat with the configured prefix, default `/agi`.
- Pairing options:
  - QR login: leave `WHATSAPP_PAIRING_PHONE_NUMBER` empty and scan the QR printed in the terminal.
  - Pairing code login: set `WHATSAPP_PAIRING_PHONE_NUMBER` to your number in E.164 format without `+`.
- Important env vars: `WHATSAPP_AUTH_DIR`, `WHATSAPP_INCOMING_DIR`, `WHATSAPP_SESSION_NAME`, `WHATSAPP_TRIGGER_PREFIX`, `WHATSAPP_SELF_CHAT_ONLY`, `WHATSAPP_ALLOWED_CHAT_JIDS`, `WHATSAPP_PAIRING_PHONE_NUMBER`, `WHATSAPP_QR_IN_TERMINAL`, `WHATSAPP_SEND_PRESENCE_UPDATES`, `WHATSAPP_SEND_USER_UPDATES`, `WHATSAPP_UNSUPPORTED_FILES_TO_AGENT`, `WHATSAPP_MAX_MESSAGE_CHARS`, `WHATSAPP_RESPONSE_MODE`, `WHATSAPP_AUDIO_RESPONSE_MAX_CHARS`, `WHATSAPP_TTS_RATE`, `WHATSAPP_TTS_VOLUME`, `WHATSAPP_FFMPEG_BINARY`, `WHATSAPP_MODEL_RPC_TIMEOUT_S`, `WHATSAPP_MAX_DOWNLOAD_BYTES`, `WHATSAPP_TEXT_ATTACHMENT_MAX_CHARS`, `WHATSAPP_VISION_QUESTION_FALLBACK`, `WHATSAPP_PDF_MAX_PAGES`, `WHATSAPP_PDF_MAX_CHARS`, `WHATSAPP_STT_MODEL`, `WHATSAPP_STT_DEVICE`, `WHATSAPP_STT_COMPUTE_TYPE`, `WHATSAPP_PYTHON_BIN`
- `WHATSAPP_MODEL_RPC_TIMEOUT_S=0` disables the model RPC timeout entirely.
- Set `WHATSAPP_TRIGGER_PREFIX=` or `whatsapp.trigger_prefix` to an empty string to disable the prefix requirement entirely.
- `WHATSAPP_RESPONSE_MODE` matches Telegram behavior: `text`, `audio`, or `same`.
- `same` replies in audio only when the user input was audio; other inputs still get text.
- WhatsApp audio responses use local TTS generation and prefer ffmpeg conversion to Opus/Ogg when `ffmpeg` is available.
- Supported WhatsApp inputs: text, audio messages, image attachments, video attachments, PDFs (text extraction), and text/code documents.
- Unsupported WhatsApp files can be forwarded to COMPLEX mode when `whatsapp.unsupported_files_to_agent=true`.
- Current limitation: scanned/image-only PDFs still need OCR support.
- Commands in WhatsApp:
  - `/help` and `/status`
  - `/approve <request_id>` or `/deny <request_id>`
  - `+ <text>` to patch the previous turn
  - `/wake` to emit a wake event
  - `/agi <text>` to submit a turn when using the default trigger prefix

Audio mode (OpenWakeWord + VAD + STT)
- Install audio deps: `pip install -r requirements.txt`
- Set `interface.mode` to `audio` in `config/settings.json` (or `text` for text interaction).
- OpenWakeWord uses built-in models (e.g., `hey_mycroft`) or custom model paths in `voice.wakeword_models`
- OpenWakeWord runs in ONNX mode; ensure models are downloaded with `openwakeword.utils.download_models()`
- `voice.wake_mode` supports `openwakeword`, `stt_prefix`, or `hybrid` (hybrid uses wakeword hints plus STT prefix checks)
- If you want wake-by-name without a custom model, set `voice.wake_mode` to `stt_prefix` (heavier: STT runs on every VAD segment)
- Match `config/system_entity.json` aliases to your chosen wake model if you want consistent naming
- If PortAudio can't find a mic, set `voice.input_device` to a device index or name substring
- Transcripts can be logged to `voice.transcript_log_path` (JSONL) when `voice.log_transcripts` is true; entries include `accepted` and `final_text`
- Enable spoken responses with `voice.tts_enabled` (uses `pyttsx3` if installed, else OS TTS fallbacks)
- Optional TTS tuning: `voice.tts_rate`, `voice.tts_volume`, `voice.tts_voice` (pyttsx3 only)
- Prevent mic pickup during TTS with `voice.suppress_mic_during_tts` and `voice.tts_suppress_ms`
- Echo filtering can ignore assistant TTS captured by the mic via `voice.echo_filter_*` settings
- Echo debugging logs to `voice.echo_log_path` (JSONL with TTS/STT/chunk details)
- Silence filtering uses `voice.no_speech_prob_threshold` (Whisper no_speech_prob average, default 0.6)
- Activation/sleep sounds are controlled by `voice.sound_enabled` and the `activation_sound_*`/`sleep_sound_*` settings
- Use `voice.sound_backend` to force `winsound` (Windows) or `sounddevice` if auto selection is silent
- After a wake, audio stays in conversation mode until a sleep phrase is spoken or `voice.conversation_timeout_s` elapses
- Sleep phrases are configured via `voice.sleep_phrases` (defaults to `["sleep"]`)
- Console user transcripts can be shown with `voice.print_user_transcripts`

Vision sense (YOLO + camera)
- Install vision deps: `pip install -r requirements.txt`
- Vision uses `ultralytics` YOLO plus OpenCV camera capture.
- Run camera-only mode with `interface.mode=vision`, or combine senses with `interface.mode=local` and `interface.senses=["audio","vision"]` or `["text","vision"]`.
- Main settings live under `vision` in `config/settings.json`.
- `vision.model_path` defaults to `yolov8n.pt`.
- `vision.camera_index` selects the camera device.
- `vision.capture_interval_ms`, `vision.conf_threshold`, `vision.iou_threshold`, and `vision.max_det` control detection cadence and filtering.
- `vision.track_enabled` switches YOLO from plain detect mode to `track()` mode and exposes per-object `track_id`s in the local runtime.
- `vision.tracker_config`, `vision.track_persist`, `vision.track_stable_frames`, and `vision.track_max_missing_frames` control tracker behavior and when a live track becomes a stable local entity.
- `vision.min_stable_frames` and `vision.scene_cooldown_s` debounce scene-change noise before a scene becomes the active shared scene.
- `vision.object_memory_enabled` compares stable tracked objects against saved `object` memories using vision appearance embeddings plus lightweight local features.
- `vision.object_memory_auto_store` creates a new `object` memory when a stable track does not match anything durable; keep it off until the matcher is tuned for your camera.
- `vision.object_memory_match_top_k`, `vision.object_memory_match_threshold`, `vision.object_memory_match_margin`, and `vision.object_memory_label_gate` control the durable object matcher.
- `vision.object_memory_embed_transport` controls how tracked-object crops are sent to `model.EmbedImage`: use `temp_file` for the local same-machine setup, or `base64` if the model server cannot read local temp files.
- `vision.object_memory_crop_max_width`, `vision.object_memory_crop_max_height`, and `vision.object_memory_crop_jpeg_quality` control crop resizing and JPEG compression before the appearance-embedding request is sent.
- `vision.object_memory_crop_temp_dir` and `vision.object_memory_crop_keep_temp_files` control where temp-file transport writes crops and whether those files are deleted after embedding.
- `vision.object_memory_export_created_entities` writes each newly created object memory under `data/vision_entities/<mem_id>/` with `crop.jpg` and `memory.json`.
- `vision.object_memory_export_dir` controls the base export folder for those created entities.
- `vision.preview_enabled` opens an annotated live preview instead of raw camera output.
- `vision.preview_show_boxes`, `vision.preview_show_labels`, `vision.preview_show_confidence`, and `vision.preview_show_summary` control the overlay content.
- `vision.broker_enabled` writes bounded live-camera artifacts under `data/vision_broker/` so tool-based OCR and video analysis can inspect the active vision stream without reopening the camera.
- `vision.broker_latest_interval_ms`, `vision.broker_buffer_frame_interval_ms`, `vision.broker_buffer_retention_s`, and `vision.broker_buffer_max_frames` control how often broker frames are persisted and how much recent history is retained.
- `vision.broker_clip_export_dir` and `vision.broker_keep_exported_clips` control where multi-frame observations export temporary clips and whether those clips are preserved after analysis.
- When a tracked object resolves to durable memory, the preview overlay appends the saved object name to the box label, for example `#7 cup -> blue_cup`.
- Press `q` or `Esc` in the preview window to close the live preview loop.
- Stable scenes are published into the local runtime and injected into explicit scene queries such as "what do you see?" when audio/text is active with vision.
- When tracking is enabled, the shared scene context also carries tracked entities such as `#7 cup` or `#12 vision.object.laptop`.
- Optional background scene-change submission is controlled by `vision.auto_submit_scene_changes`.
- Scene changes can be logged to `vision.scene_log_path`.

Image tool (`image.analyse`)
- `image.analyse` is the image-analysis tool used by the agent tool runtime.
- Desktop capture, Linux/Windows dependencies, and Wayland limitations are documented in [Desktop tools](docs/desktop-tools.md).

Model runtime options
- See [GPU and CPU model placement](docs/model-resources.md) for model swapping, CPU embeddings, and memory settings.
- Default text generation uses `models.text_model` with optional `models.text_model_options`.
- Vision analysis uses `models.vision_model` with optional `models.vision_model_options`.
- UI grounding uses `models.ui_grounding_model` with optional `models.ui_grounding_model_options`.
- When using Ollama, `thinking` is sent as a `think` request flag and `reasoning_effort` is forwarded in the model options payload.

Live vision observation tool (`vision.observe`)
- `vision.observe` reads from the live broker created by the local vision sense.
- Use `time_scope="now"|"latest"|"stable"` for single-frame inspection.
- Use `time_scope="recent"` to inspect the broker's recent bounded history.
- Use `time_scope="watch"` to wait up to `max_duration_s` for new broker frames or a scene change, then analyze what happened.
- `analysis_mode="auto"` chooses image analysis for single-frame cases and video analysis for multi-frame `recent`/`watch` cases; `analysis_mode="image"` or `"video"` can force the path.

Video tool (`video.analyse`)
- `video.analyse` is a tool for `AGENT`/`COMPLEX`, not a local sense.
- It analyzes a video by adaptively sampling representative frames across the duration, adding scene-change frames, extracting and transcribing the audio track, then running frame-level image analysis and aggregating the result.
- Main settings live under `video` in `config/settings.json`.
- `video.min_baseline_frames`, `video.max_baseline_frames`, `video.min_spacing_s`, `video.max_spacing_s`, `video.duration_growth_divisor`, and `video.duration_growth_weight` control duration-aware baseline sampling.
- `video.scene_ratio`, `video.min_scene_frames`, `video.max_scene_frames`, `video.scene_probe_scale`, and `video.scene_change_threshold` control scene-change refinement.
- `video.max_total_frames` caps analysis cost even for long videos.
- `video.frame_jpeg_quality`, `video.sampled_frame_dir`, and `video.keep_sampled_frames` control extracted keyframe files.
- `video.transcription_enabled`, `video.ffmpeg_binary`, `video.audio_extract_sample_rate_hz`, `video.audio_extract_dir`, `video.keep_audio_extracts`, `video.transcript_max_chars`, `video.stt_model`, `video.stt_device`, and `video.stt_compute_type` control the transcription path.

3) Ensure Ollama has the models in `config/settings.json` (examples):

```bash
ollama pull llama3.2:1b
ollama pull llama3.1:8b
ollama pull llava:7b
ollama pull nomic-embed-text
```

Text usage
- Type a normal line to send an STTFinal event.
- Use '+ <text>' to patch the last turn (merge window simulation).
- Use '/approve <request_id>' or '/deny <request_id>' to answer permission requests.
- Wake word support uses `config/system_entity.json` (name and aliases). Typing the name/alias alone emits a WakeEvent; typing it as a prefix strips it and sends the remainder.

Tool usage patterns
- Tool retrieval uses `orchestrator.tool_retrieval_k` as a maximum, not a quota. Only tools with cosine similarity at least `orchestrator.tool_retrieval_min_similarity` are returned (default `0.3`, an initial cutoff to tune for your embedding model). Zero matches is valid. Set the cutoff to `-1.0` to restore unfiltered top-K ranking. `TOOL_RETRIEVAL`, `RETRIEVED_TOOLS`, and `TOP_TOOL_SCORES` in `data/human_record.log` show the decision. On embedding exceptions, a logged fallback supplies at most K active tools without similarity filtering. Disabling retrieval still exposes the active catalog; an independent LLM selector may add tools beyond this retrieval cap.
- Time: "what time is it"
- Math: "calc: 2+2"
- Read: "read workspace:/README.md"
- Write: "write workspace:/notes/todo.md :: buy milk"

High-level tool generation
- The single-tool `create_tool` path still exists internally, but there is now a higher-level generator pipeline for turning a broad JSON spec into one or more runtime tools.
- Run it with:

```bash
python3 scripts/generate_tools_from_spec.py --spec path/to/tool_spec.json
```

- Minimal spec shape:

```json
{
  "request": "Create tools for querying local SQLite databases and exporting results.",
  "namespace_prefix": "data.sqlite",
  "max_tools": 3,
  "default_permissions": ["tier1"],
  "blacklist_libraries": [],
  "constraints": {
    "allow_network": false
  }
}
```

- The pipeline plans a tool blueprint first, then runs the existing codegen/critic flow per tool, skips duplicate tool ids idempotently, and persists a JSON run record under `data/tool_generation_runs/`.
- Generated tools can compose installed tools with `call_tool("tool.id", args)`; nested calls are cycle-checked and cannot exceed the parent permission tier.
- No code libraries or calls are blocked by default. Add `blacklist_libraries`, `blacklist_modules`, or `blacklist_calls` to a generation spec, or configure `create_tool_blocked_modules` / `create_tool_blocked_calls` globally.
- Non-standard-library dependencies returned by code generation are installed with the active Python interpreter before registration. Installation failures or blacklisted dependencies abort that tool.

Neo Code
- `scripts/neo-code` runs a persistent inspect → edit → check → continue/repair session with direct local-model or cognition reasoning.
- From the target project (with this installation's `scripts` directory on `PATH`):

```bash
neo-code "implement the requested change" --mode cognition --apply \
  --test-command "python3 -m pytest -q" --max-steps 24
```

- Successful checks feed back into the session; completion requires an explicit final response, completed plan milestones, and current verification.
- `--allow-exec` enables model-selected setup/build commands and managed background servers. `--allow-network` enables the shared runtime's search and HTTP tools. Commands run with your process permissions, without an OS sandbox.
- Resume using the printed checkpoint path; the plan, transcript, notes, and execution options are retained:

```bash
neo-code --resume data/neo_code/sessions/<session-id>.json --max-steps 24
```

- Without `--apply`, source changes are review-only. Apply the exact reviewed patch with `--candidate path/to/candidate.json --apply --test-command "..."`.
- Windows launchers: `neo-code.ps1` and `neo-code.cmd`. See [Neo Code usage and architecture](docs/neo-code.md) for tools, permissions, session behavior, and limits.

Google Workspace integration
- Google is disabled for new installations until explicitly enabled. Interactive setup offers `skip`, `neo`, and `own`; see [Google setup and privacy](docs/google-privacy.md).
- For your own client, use setup's `--google own --google-client-json /path/to/desktop.json`, or set `google.enabled=true` and `google.client_secrets_path` in `config/settings.local.json`. Older installations without an `enabled` setting retain their existing behavior.
- Authorize one bundle at a time with `google.authorize`.
- Supported bundles: `gmail_send`, `gmail_compose`, `gmail_readonly`, `calendar_readonly`, `calendar_events`, `docs_readonly`, `docs_edit`.
- Draft creation requires separate authorization of `gmail_compose`; a `gmail_send` token cannot create drafts. Google's compose scope permits managing drafts and sending mail, so it is not a draft-only permission. Sending through `google.gmail.send_message` continues to use the narrower `gmail_send` bundle.
- Credentials are stored in the OS keyring, not in the repo.
- `orchestrator.auto_approve_all=true` allows Google tools to run without confirmation after OAuth is granted; `false` keeps the normal approval flow.
- `google.auto_google_authorize=true` makes the tool runtime ensure Google authorization during startup. Use `google.auto_google_authorize_bundles` to limit which bundles are auto-authorized; if the list is empty, startup attempts all supported bundles.

Files of interest
- config/settings.json: ports, paths, and model provider + names
- config/settings.json: also carries Google OAuth settings under `google`
- config/reflection.json: thresholds and output settings for generated self-reflection dossiers
- config/reflection_improve.json: backend and scope settings for isolated reflection candidates
- config/soul.json: scope and exclusions for the generated `soul/` mirror
- config/tool_registry.json: tool definitions
- config/router_config.json: evolvable router thresholds
- config/system_entity.json: system identity (name, aliases, properties)
- config/settings.json: `autonomy` controls the optional Neo4j-backed, COGNITION-driven epistemic and power-process runtime
- evolve/run_evolve.py: generates candidate configs in evolve/candidates/
- scripts/reflection_dossier.py: builds ranked reflection dossiers from `soul` plus runtime logs
- scripts/reflection_prepare_candidate.py: prepares an isolated candidate workspace from the top reflection dossier
- scripts/neo-code, scripts/neo-code.ps1, scripts/neo-code.cmd: portable Neo Code launchers
- scripts/local_code_agent.py: Neo Code controller using `soul`, persistent sessions, and direct or cognition reasoning
- scripts/soul_mirror.py: builds and watches the `soul/` analysis mirror with symbol and dependency indexes
- scripts/train_models.py: trains router/SFT models from logs (writes manifest.json)
- scripts/package_ollama.py: writes Ollama Modelfile + manifest (optional ollama create)
- data/storage.db + data/audit.log: memory + audit log output

Notes
- All tool writes emit audit records in data/audit.log.
- Evolution writes only to evolve/candidates/ and does not touch production configs.
- `python scripts/soul_mirror.py sync` builds `soul/code/` plus metadata for LLM-oriented analysis, including symbols, internal module edges, entrypoints, area summaries, and architecture views.
- `python scripts/reflection_dossier.py` builds read-only self-critique dossiers under `reflection/runs/` from `soul` plus local logs.
- `python scripts/reflection_prepare_candidate.py` turns the selected reflection dossier into a constrained proposal plus an isolated candidate worktree under `reflection/candidates/`.
