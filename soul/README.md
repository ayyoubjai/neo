# Soul

`Soul` is a generated mirror of the codebase optimized for LLM analysis.

What it contains after a sync:
- `code/`: a clean mirrored copy of the selected source files.
- `meta/manifest.json`: scope, counts, and digest for the current mirror.
- `meta/files.jsonl`: per-file metadata, hashes, and line counts.
- `meta/python_index.jsonl`: top-level Python imports, classes, functions, and constants.
- `meta/symbols.jsonl`: symbol-level index with kinds, qualified names, and line spans.
- `meta/module_graph.json`: internal Python dependency graph between mirrored modules.
- `meta/entrypoints.jsonl`: detected service mains, CLI scripts, and launcher scripts.
- `meta/area_summaries.json`: deterministic architecture summaries for each subsystem/area.
- `meta/views.json`: task-oriented runtime, area, and hotspot views.
- `meta/overview.md`: readable architecture snapshot generated from the mirror.
- `meta/tree.txt`: a stable text tree of the mirrored files.

How to use it:

```bash
python scripts/soul_mirror.py sync
python scripts/soul_mirror.py watch --interval 2
```

Scope and exclusions are controlled by `config/soul.json`.
