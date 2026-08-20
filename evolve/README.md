Evolve (MVP)

This folder contains staged evolution artifacts only. No production files are edited here.

- run_evolve.py generates a candidate router config into evolve/candidates/
- candidate artifacts are JSON with metrics placeholders and STAGED status

Promotion is intentionally manual for this MVP.

Training (logs -> datasets -> models)

- scripts/build_training_datasets.py builds JSONL datasets from data/record.log into evolve/datasets/
- scripts/train_models.py trains router or SFT models (requires requirements-train.txt and torch)
  - writes manifest.json next to outputs
  - supports LoRA + optional merge to full weights
- scripts/package_ollama.py writes an Ollama Modelfile + manifest (optionally runs ollama create/--quantize)

Improvement loop (code evolution MVP)

- evolve/improve_config.json defines targets, generator, metrics, and selection weights
- generator can run multiple LLM calls (runs + candidates_per_run)
- optional context probe + retrieval (context.enabled) to enrich LLM prompt
- optional multi-generation evolution (evolution.generations > 1) with parent seeding
- metric commands can use scripts/evolve_metric_runner.py (syntax/import/safety/readability/minimal_diff)
- optional project profile (evolve/project_profile.json) can define repo-specific commands while keeping the runner generic
  - profile.mode=replace uses only profile metrics
  - profile.mode=append adds profile metrics on top of config metrics
  - required profile checks run once as baseline preflight before candidate generation
  - optional benchmark metrics run warmups + repeated runs and score by baseline/candidate median ratio
- execution.backend=git_worktree creates one isolated git branch/worktree per candidate under evolve/runs/
  - baseline snapshot repo is stored at execution.evolution_repo_dir
  - reset state by deleting evolve/runs/ and execution.evolution_repo_dir
- execution.copy_mode is compatibility-only; runner forces copy-only sandboxing to protect main source
- the runner aborts if `src/` changes during an evolution run
- reporting.dashboard=true writes an HTML dashboard via scripts/evolve_report.py
- reporting.dashboard=false writes evolve/runs/<run_id>/evolution.log text summary
- scripts/evolve_improve.py runs the loop:
  - generates candidates (via generator command or evolve/candidates/inbox JSON)
  - for generation > 1, feeds promoted/top prior candidates back as seed_candidates
  - evaluates metrics (command-based by default)
  - stages top candidates into evolve/candidates/
- scripts/evolve_generate_llm.py is the default generator (uses settings.models.provider + text_model)
  - enable thinking by default via /think (see evolve/improve_config.json)
- scripts/evolve_probe_llm.py is the default context probe (requests needed symbols/files)
- scripts/evolve_capture_trace.py captures replayable real-call scenarios into evolve/scenarios/traces/*.jsonl
- tracing helpers live in `src/common/evolve_trace.py`:
  - `@trace_calls("module.func")` for single callables
  - `@trace_class_calls("module.Class")` for class constructor/public methods
- scripts/evolve_generate_macro_scenarios.py creates JSON-only LLM scenarios into evolve/scenarios/llm/*.json
- scripts/evolve_score.py computes scenario score with mode switch:
  - `--mode macro`: evaluate real trace scenarios only
  - `--mode micro`: evaluate generated scenarios only
  - `--mode hybrid`: weighted combination of macro + micro
  - `--macro-policy auto|always|never` controls when macro is applied
  - in `auto`, module-level or no-input/no-runtime callables are marked `macro_skipped` (neutral score)
  - `--prepare-only` prepares macro sources before evolution (trace injection + optional trace collection wait)
  - `scripts/evolve_improve.py` auto-runs `--prepare-only` once per target for macro metrics
  - returns JSON `score/details` for `parse_json` metric integration
