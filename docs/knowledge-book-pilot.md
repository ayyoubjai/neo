# Coyne book ingestion pilot

## Repeat with fresh model output

### Comparison modes and required entities

The pilot accepts `--comparison-mode per-claim` (default), `batched`, or `whole-set`.
Per-claim uses one retrieved candidate set per claim. Batched uses `--batch-size 6`
focal claims per call plus their retrieved candidates from the full population.
Whole-set supplies all compact claims and allows connections in either direction.
Groups over `--comparison-max-chars 12000` split into smaller groups with retrieved
outside candidates. If even one focal group is too large, the run stops with advice
to lower the candidate limit or adjust the budget. This is a character heuristic,
not exact model token accounting. All modes retain the same semantic validation.
Group prompts prioritize at most 40 connections per call; they are not exhaustive.
Larger responses may still fail or truncate. Use smaller batches in that case.

```bash
PYTHONPATH=src .venv/bin/python -m knowledge.pilot \
  --resume data/knowledge/fresh-pilot-20260912-154956-755465 \
  --stage compare --comparison-mode batched --batch-size 6
```

Use `--comparison-mode whole-set` for a whole-set attempt. Resume inherits saved
mode and budget options when omitted. Changing comparison settings archives prior
connections as `comparisons-<signature>.json` and rebuilds the active comparison
checkpoints while reusing extraction and embeddings.

For a fresh extraction, `--entities auto` leaves entity output optional (default).
`--entities required` explicitly asks for supported named referents for every claim
and rejects claims with an omitted entities field. An empty list is allowed when no
entities are supported; the option cannot guarantee semantic completeness or force
invented entities. Changing this policy on an already extracted run requires a new
run, so existing optional outputs are never mistaken for required extraction.

From the repository root, with the configured local model server running:

```bash
PYTHONPATH=src .venv/bin/python -m knowledge.pilot \
  "tmp_data/Why Evolution Is True (Jerry A. Coyne) (z-library.sk, 1lib.sk, z-lib.sk).pdf" \
  --first-page 24 --last-page 25
```

For separate stages, add `--stage extract` to that command. It extracts **all** selected
pages and exports a graph before any embedding or comparison calls. Then continue with:

```bash
PYTHONPATH=src .venv/bin/python -m knowledge.pilot \
  --resume data/knowledge/fresh-pilot-<timestamp> --stage embed
PYTHONPATH=src .venv/bin/python -m knowledge.pilot \
  --resume data/knowledge/fresh-pilot-<timestamp> --stage compare
```

`--stage all` (the default) runs all three stages in order. Each stage ensures preceding
stages are complete. Repeating `--resume` skips completed work; failed transport calls
are retried. Validation rejections remain recorded instead of triggering unlimited
retries. Per-call progress prints every 15 seconds, and elapsed time prints on completion.
Use `--timeout 600` to set the per-call deadline, or `--candidate-limit 6` to reduce
comparison candidates (default 12). Smaller candidate sets can miss relationships.

Identical claim IDs are deduplicated while all supporting passages and quotes are
preserved. Identity includes source, text, speaker, stance, scope, and claim type.
Paraphrases and opposing positions remain separate. Embeddings do not merge claims.
`embeddings.json` caches normalized vectors with model, endpoint, and content identity.
The configured `models.embedding_model` and embedding endpoint must be available.
Embedding models/files replaced under the same name require a fresh cache/run.

Comparison searches the entire extracted population, including later pages. Candidate
ranking combines the existing concept/word ranking with positive cosine similarity,
using reciprocal rank fusion. This is a retrieval heuristic, not a truth or equivalence
judgment. Accepted connections with identical IDs are deduplicated. Reciprocal arrows
are retained as distinct proposals. Candidate IDs and completed work are checkpointed.

Existing interrupted pilot folders can also be resumed: their original report is backed
up as `before-staged-resume-report.json`, and exact matching completed extraction
requests are reused. Comparisons are rebuilt against the full population. The checkpoint
and partial explorer are saved after each completed extraction/comparison, including
when the run exits through an interruption. Raw response files remain unchanged.

Each invocation creates a new `data/knowledge/fresh-pilot-<timestamp>/` directory.
Open its `explorer.html`. Requests, raw responses, the report, source transcript,
and graph JSON are retained. This uses the configured text model's direct JSON
endpoint with the current extraction/comparison functions. Neighboring pages supply
context only. No cached claims, semantic corrections, proposition groups, or Neo4j
data are imported. All claims await review. It tests the PDF-to-graph stage; it does
not run epistemic investigations or the standard cognition orchestration route.
Failures remain in `report.json`; a nonzero exit indicates issues to inspect.

Deleting `data/knowledge` removes exports and test artifacts, **not** Neo4j data.
The reviewed pilot remains in Docker volume `neo-coyne-graph-data`. A new collection
alone does not isolate source claim identities in that database. The fresh command
avoids database reuse entirely, so no deletion is necessary to repeat this experiment.

## Reviewed baseline

**Current: assistant semantic review applied.** The pilot retains 45 source claims,
with 34 reviewed abstract concepts, 13 named entities, 16 active proposed connections,
and one shared proposition grouping two paraphrases. Thirteen original links were
retired, not deleted. All claims still require extraction review; no truth assessment
has been performed.

The original graph and explorer are saved under
`data/knowledge/coyne/before-semantic-review/`. The explicit correction manifest is
`data/knowledge/coyne/semantic-review.json`, authored by `assistant:codex`. This was a
source-based assistant review, **not** a fresh automatic extraction success. The
inspector exposes original claims, corrections, contextual notes and retired links.
Enable Entities or Propositions in the whole graph to inspect the new layers.

The eight incorrect `quotes` stances were corrected to `endorses` in the interpretation
layer. The two between-species rate paraphrases share one proposition; the distinct
within-species temporal variation claim remains separate. Context-dependent references
were clarified, including the “two tenets” reference using the continuation on PDF page
26. The rhetorical design question remains flagged for interpretive review.

The review artifacts below describe the current snapshot. The findings and run
history further below document the earlier extraction and its original counts.


Input: the 330-page PDF of *Why Evolution Is True* by Jerry A. Coyne in `tmp_data`.
This is a **partial extraction test**, not a complete chapter/book graph or a truth assessment.
The selected material is physical PDF pages 24–25 (printed pages 3–4), in Chapter 1.
Page 25 initially failed the configured cognition client's JSON/schema validation; a later direct-JSON retry succeeded in extracting review candidates.

## Review artifacts

Open `data/knowledge/coyne/pilot-explorer.html` in a browser. The default Whole graph
view shows all 45 claims and 16 active proposed connections, including 32 disconnected
components. Enable Concepts to add the 34 reviewed topic nodes, Entities for 13 named
referents, or Propositions for the shared paraphrase group. Enable Source documents to
show their common book. Pan, zoom, or Fit all, and select a claim to see its excerpt.
The Sources, Concepts, and Assessments list views provide alternative groupings.
This remains an offline snapshot of the two-page pilot, not the entire book.

- `pilot-review-report.json`: original extraction candidates and unresolved rejections; predates semantic corrections.
- `pilot-graph.json`: current interpreted claims, active connections, original provenance, and semantic review history.
- `semantic-review.json`: explicit assistant corrections and relationship decisions.
- `semantic-review-result.json`: applied review identifier and current graph counts.
- `pilot-source.txt`: the two selected pages, preserving their extracted layout.
- `review-sample.json`: original extraction sample for human ratings; predates semantic corrections and has not been overwritten.
- `pilot-initial-extraction.json`: the initial failed extraction, preserved for comparison.
- `pdf-preflight.json`: file/page checks and reader comparison.
- `logs/`: local model, service, and ingestion diagnostics.

The graph collection is `coyne-chapter1-pilot`, stored in the dedicated persistent
Docker volume `neo-coyne-graph-data`, served by container `neo-coyne-graph` on local
Bolt port 17688. It is separate from the previous disposable test graph.

```bash
PYTHONPATH=src EPISTEMIC_NEO4J_URI=bolt://127.0.0.1:17688 \
  .venv/bin/python -m knowledge.inspect --collection coyne-chapter1-pilot
```

All pilot claims are marked for extraction review. No investigations, truth assessments,
or promotions to world-model theories have been run.

## Findings from the real model run

1. Default pypdf extraction joins some words. Poppler's `pdftotext -layout` preserves
   spacing better for this file. The first page is an image cover, not missing body text.
   The pilot adapter uses Poppler, preserves the original PDF hash, and identifies every
   passage by its physical page and transcript. This is not a successful unmodified
   full-book `knowledge.ingest` run.
2. Strict quote matching originally rejected all eleven first-page candidates. Layout
   recovery now maps whitespace, ligatures and line-wrap hyphens back to a unique exact
   original substring. It does not repair wording, ambiguous occurrences, or proprietary
   font digits. Both the proposed quote and recovered original are preserved.
3. The formatting model changed `endorses` to `endorse`. Explicit singular stance
   variants are now canonicalized without changing the source's position.
4. The first extraction omitted the central paragraph defining evolution's six
   components. A focused pass was added; the initial omission remains a coverage failure.
5. The cognition client unnecessarily rewrote already-structured JSON in another model
   turn. Valid structured results are now retained, with normal schema validation and
   evidence handling. Extraction prompts explicitly request JSON and whole-passage coverage.
6. Page 25 did not yield a valid claims array. The absence of accepted claims there is
   an extraction failure, not evidence that the page contains no ideas.

These fixes improve traceability, not demonstrated semantic accuracy. Historical
attribution, empirical-versus-value classification, rhetorical questions, and completeness
still need review. Broader import should follow that review and a reliable second-page run.

## Fallback and semantic review

The focused recovery uses the **same configured local text model** through its native
JSON-constrained endpoint, bypassing the failing cognition formatting loop. It still
uses `extract_passage`, `validate_claim`, `compare_claims`, and `KnowledgeGraph`.
Raw responses are retained as `direct-*.json`; `recover-direct.py` is the pilot adapter.
This demonstrates graph construction with an alternative model adapter, not a successful
standard-orchestrator run for the whole book.

The focused output incorrectly labels several statements in the narrator's own summary
as `quotes`. It appears to confuse quoting an excerpt in the output with the source
quoting another speaker. Those model outputs are retained for review, not silently
corrected. The extraction prompt has been clarified for subsequent runs. All pilot
claims still require human extraction review before investigation.

## Completed two-page pilot

The final export contains **45 claims, 79 concept nodes, and 20 proposed claim
connections**, covering physical PDF pages 24–25 (printed pages 3–4). Six candidate
claims still fail source validation; rejected relationship proposals are also retained
in `pilot-review-report.json` under `relationship_issues`. Only selected focal claims
were compared, so the connection set is not exhaustive.

Every exported excerpt was checked as an exact substring of its corresponding Poppler
page transcript. All 45 claims require extraction review; there are no investigations
or assessments. This verifies traceability, not the correctness of each paraphrase.
Eight records from the earlier focused pass still carry questionable `quotes` stances.
The later page-25 pass used the clarified stance prompt.

Useful claims to search in the explorer include “six components”, “genetic change”,
“most (but not all)”, and “Gradualism does not mean”. The latter preserves the distinction
between gradual change over generations and a constant rate of change.

The full 330-page book has **not** been imported. These results support reviewing and
fixing extraction quality before scaling up. The normal importer still needs integration
of the Poppler adapter, reliable structured generation, and resumable large-document
checkpoints; this pilot uses explicit local scripts for those tasks.
