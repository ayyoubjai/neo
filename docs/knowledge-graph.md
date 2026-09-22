# Import a library of ideas and investigate its claims

The library supports TXT, Markdown, EPUB, and PDF files with optional local OCR. It preserves
source claims separately from the agent's own assessments. Ingestion can suggest
connections and possible contradictions; it never promotes imported statements to
world-model `Theory` nodes or treats their repetition as verification.

## Start the services

From the repository root, with the local models configured:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
docker compose -f docker-compose.neo4j.yml up -d
./start.sh
```

Use a second terminal for the commands below, activating the same virtual environment
and setting `PYTHONPATH` there too. The importer requires the existing cognition
orchestrator and `neo4j` driver. PDF extraction uses the existing `pypdf` dependency.
Neo4j connection settings use `EPISTEMIC_NEO4J_URI`, `EPISTEMIC_NEO4J_USER`, and
`EPISTEMIC_NEO4J_PASSWORD`. Export those in the command terminal if they differ from
the defaults. These commands do not require the background autonomy loops to be enabled.

## Import documents

```bash
python -m knowledge.ingest ./books --collection personal-library
```

Directories are scanned recursively for supported extensions. Explicit unsupported
files, undecodable text, encrypted PDFs, and PDFs with pages without extractable
text are reported as issues. Use `--ocr` to process PDF pages without text locally with `pdftoppm` and `tesseract`. OCR-derived claims always require extraction review; their quotations refer to the OCR transcript. EPUB chapters follow spine reading order. Image-only EPUBs and complex table layouts are not supported.

To inspect extraction before writing to Neo4j:

```bash
python -m knowledge.ingest ./books --collection personal-library \
  --report-only --report data/knowledge/library-review.json
```

Report-only mode calls the model but performs no graph writes. A later normal import
runs extraction again; the report is not an approved payload that gets replayed.
All extraction, comparison, and planning calls have zero permitted tool dispatches.
The document contents are presented as untrusted source material.

Reports contain accepted candidate claims, exact supporting quotations, character/page
locations, proposed connections, skipped passages, and validation/execution issues.
A nonzero exit status indicates issues; successful documents can still have been saved.
Each document's claims and connections are written in a single transaction. Retrying a
complete passage avoids additional model calls. File content hashes identify versions;
changed content at the same path creates a new document linked to its earlier versions.

## Inspect and review

```bash
python -m knowledge.inspect --collection personal-library
python -m knowledge.inspect --claim 'claim:ID_FROM_THE_REPORT'
```

The collection list prioritizes unassessed claims, possible contradictions, then older
assessments. It includes nonempirical and low-confidence claims so they can be reviewed.
The current list defaults to 20 entries; the HTML export supports a larger explicit limit.

An explanation includes the source's position (`endorses`, `quotes`, `rejects`,
`questions`, or `unclear`), exact excerpts, proposed relationships, recorded test plans,
and all evidence-backed assessments. The source's stance is independent of the agent's.
Extraction confidence and attribution confidence are model estimates about reading
accuracy; they are not evidence that the proposition is true.

Low extraction/attribution confidence or unclear stance blocks automatic investigation.
After examining the source, record whether the extraction is accurate:

```bash
python -m knowledge.review 'claim:ID' --accept \
  --note 'The cited passage accurately represents the identified speaker and scope.'
```

`--reject` keeps the claim blocked. This review records a separate `ExtractionReview`
node and never endorses the underlying proposition. Editing inaccurate extracted text
in place is not implemented; preserve the original and import corrected source material
or revisit extraction in a future revision.

## Investigate an empirical claim

```bash
python -m knowledge.investigate 'claim:ID' --report data/knowledge/plan.json
```

By default this generates and records a plan without executing it. The plan states the
question, scope, expected observations under competing explanations, and procedure.

Read the saved plan, then approve and execute its investigation ID:

```bash
python -m knowledge.investigate --approve 'investigation:ID' --note 'Reviewed scope and procedure.'
python -m knowledge.investigate --resume 'investigation:ID'
```

Approval records a hash of the exact plan. Resume does not generate a new plan.
Execution uses the existing Explorer/Analyzer, deny permission policy, and a default
three-dispatch limit. This relies on application tool permission classifications,
not a separate read-only OS sandbox. The Analyzer cannot execute tools.

Durable checkpoints are stored in `<data_dir>/knowledge/journal/investigations.sqlite3`.
Keep this directory across restarts and share it between runners on this installation.
Saved observations can be analyzed without repeating tools; saved assessments can be
persisted without calling the model. Completed results also remain recoverable from Neo4j.
If a crash loses an action receipt, resume reports an uncertain outcome and refuses an
automatic retry. Only after inspecting execution records, explicitly permit repetition
with `--resume 'investigation:ID' --retry-uncertain`. Exactly-once external execution
cannot be guaranteed. The previous CLI `claim:ID --execute` has been replaced; Python
callers retain `investigate(..., execute=True)` for explicit immediate execution.

A successful observation may produce `supported` or `rejected`; absent, failed, or
inconclusive evidence produces `unresolved`. The assessment records its own scope,
justification, timestamp, and heuristic confidence derived from evidence quality.
The original claim and every earlier assessment remain intact. A narrowly scoped
assessment is not automatically a universal judgment of the claim.

Automatic investigation currently accepts only claims classified as empirical.
Definitions, values, fiction, and hypotheses remain available in the graph for
interpretive review. They are not labeled wrong for being unsuitable for an experiment.
The existing EpistemicLoop handles theories. A separate optional knowledge worker handles the claim queue.

## Explore the connections visually

```bash
python -m knowledge.export --collection personal-library \
  --output data/knowledge/library.html --limit 200
```

Open that HTML file in a browser. It is self-contained and makes no network requests.
The default **Whole graph** view includes every exported claim, even isolated nodes
and disconnected components. Drag to pan, scroll or use +/− to zoom, and use Fit all
to reset. Optional layers show concept nodes, source documents, and assessments.
Shared concepts indicate shared topics, not agreement; source links indicate provenance.
Node positions are for navigation and do not measure semantic similarity.
Click a claim to inspect its exact passages and assessment history below the overview.
Click a concept, source, or assessment node to list the claims associated with it.
Search highlights matching nodes without removing disconnected parts of the overview.

Sources groups the claim list by original document. Concepts groups it by extracted
topic. Assessments groups it by the latest recorded judgment (or unassessed).
These views retain the selected claim's local diagram with up to eight neighbors;
the complete connection list for that claim is below it. The whole-graph view has no
per-node eight-neighbor limit. It displays the exported collection, not every node in
Neo4j: source/passage paths are simplified to document-to-claim links, while detailed
passage and evidence records stay in the inspector. The export is a snapshot, and clearly marks when its claim limit truncated
the collection. Source text is rendered as text, never injected as executable HTML.

## Data model and current boundaries

```text
Collection → HAS_DOCUMENT → Document → HAS_PASSAGE → Passage
Passage → EXPRESSES → Claim → ABOUT → Concept
Claim → CLAIM_RELATION {kind, scope_match, status: proposed} → Claim
Investigation → INVESTIGATES → Claim
Investigation → RESULTED_IN → Assessment → TARGETS → Claim
Assessment → BASED_ON → Observation
ExtractionReview → REVIEWS_EXTRACTION → Claim
Concept → ALIAS_OF {status: reviewed} → Concept
Claim / Observation → CITES → EvidenceSource
```

Claim identity includes document, proposition, scope, speaker, stance, and type.
Similarity never merges statements across authors or versions. Candidate retrieval uses
reviewed aliases and lexical overlap, retrieves at most 200 database candidates, and
compares up to twelve. Equivalence requires reported matching scope; contradictions
across reported different scopes are downgraded to `RELATED_TO`. These scope judgments
remain model proposals. Existing belief-context retrieval still reads `Theory` nodes.

Register reviewed concept aliases before importing additional material:

```bash
python -m knowledge.concepts --alias 'electricity consumption' --canonical 'energy use' --note 'Reviewed equivalent terminology.'
```

Aliases are global and affect future comparisons; they do not backfill existing links.
Explicitly quoted DOI/URL citations link claims to shared `EvidenceSource` nodes.
Successful tool receipts also contribute explicit DOI/URL fields to observation
provenance. Explanations and HTML exports show source groups and unknown provenance.
Different identifiers do not prove independence, and repetition adds no automatic
confidence bonus. Indirect citation chains and shared datasets remain unresolved.
Completed imports are skipped rather than re-extracted to populate new citation fields.

## Bounded background investigations

```bash
python -m knowledge.worker --once --collection personal-library
```

The worker prepares plans for eligible empirical claims and executes only approved
plans. It prioritizes unassessed claims, conflicts, objective-word overlap, then older
assessments. Pending plans prevent duplicate planning. To enable it in local settings,
merge these fields into `autonomy`:

```json
{
  "enabled": true,
  "start_with_run_all": true,
  "knowledge_enabled": true,
  "knowledge_worker": {
    "collection": "personal-library",
    "pause_s": 60,
    "max_plans_per_day": 4,
    "max_investigations_per_day": 4,
    "max_tool_calls_per_day": 12,
    "max_actions_per_investigation": 3,
    "reassess_after_days": 30,
    "objective": "Understand household energy consumption"
  }
}
```

The example configuration keeps it disabled; these changes do not enable local autonomy.
Run through the launcher or `python -m autonomy.runtime`. The worker always uses the
deny permission policy independently of other loops. UTC daily budgets are persisted
per collection and reserve the full tool allowance before execution. Failed planning
attempts count. Deleting the journal resets counters. Manual CLI runs and other loops
are outside worker budgets. Dispatch limits count outer tool calls, not operations
inside composite tools. Each pass scans up to 1,000 claims and 10,000 pending plans.
Uncertain actions are never automatically retried; durable receipts can still be
recovered when the execution budget is exhausted.

## Review extraction quality

```bash
python -m knowledge.evaluate sample data/knowledge/ingestion-report.json --count 20 --output data/knowledge/sample.json
```

Compare each sampled claim against its original page. Fill its four `ratings` fields
with JSON `true` or `false` for quote, attribution, scope, and claim-type correctness.
Leave unchecked fields `null`. Then score:

```bash
python -m knowledge.evaluate score data/knowledge/sample.json --output data/knowledge/scores.json
```

Scores report reviewed/unreviewed denominators; unreviewed dimensions have no accuracy
score. This measures sampled extraction correctness, not recall or proposition truth.
Retain the original ingestion report: a reimport containing only skipped passages has
no new claims to sample. No real-library accuracy measurement has been performed yet.

Automatic argument evaluation, citation-chain reconstruction, richer semantic search,
and table/audio adapters remain future work. Model confidence and links are heuristic;
software tests do not establish truth or guaranteed monotonic knowledge improvement.

## Tests

The two fictional Machine X documents under `tests/fixtures/knowledge` intentionally
conflict. Unit tests use scripted extraction and synthetic measurements, never real
claims about a machine:

```bash
python -m unittest discover -s tests -p 'test_knowledge*.py'
```

With a disposable Neo4j instance:

```bash
NEO4J_TEST_URI=bolt://127.0.0.1:7687 \
  python -m unittest discover -s tests -p test_knowledge_graph_integration.py
```

The tests verify versioning, reimport, attribution, contradiction proposals, evidence
assessments, and traceability without changing imported claims into the agent's beliefs.

## Semantic structure and interpretation review

Extraction now requests reusable abstract concepts separately from named entities
(persons, organisms, works, places and organizations). Incidental terms and unresolved
references trigger review flags. These checks are conservative heuristics, not a general
proof that a concept or paraphrase is correct. Existing string concept lists remain
supported. Entity names and kinds are stored separately and linked through `MENTIONS`.

The importer supplies the preceding and following passage as reference context. Quotes
must still match the focal passage; context passage IDs are retained and shown in the
inspector. Unresolvable references stay flagged instead of being silently guessed.

Stronger relationship proposals now require explicit directional checks. Equivalence
requires mutual entailment declarations and is blocked for the known between-species /
within-species scope mismatch. `REFINES` goes from specific to general. `SUPPORTS`
requires an argument warrant; it never becomes independent evidence. Use `CLARIFIES`,
`ELABORATES`, `EXPLAINS`, or `EXEMPLIFIES` for those respective relationships. The model
can still supply incorrect checks, so these are gates, not semantic guarantees.

An explicit review can correct interpretations, retire links, propose replacement links,
and group paraphrases under a shared `Proposition` without deleting attributed claims:

```bash
python -m knowledge.review_semantics path/to/review.json
```

The review must identify its actual reviewer and include reasons. See the local
`data/knowledge/coyne/semantic-review.json` for a concrete example. Claim corrections
are overlays; original claim text, scope, stance, and concepts remain in Neo4j and in
inspector audit history. They cannot edit source quotations or assessments. Reviewed
terms are linked through `REVIEWED_ABOUT` / `REVIEWED_MENTIONS`. Search and investigation
planning use the effective interpretation. Applying a new correction flags extraction
for review again and invalidates unexecuted plan approvals for the affected claims.
Reapplying the identical review is a no-op.

`Claim → EXPRESSES_PROPOSITION → Proposition` groups explicit reviewed paraphrases;
it does not merge source occurrences or create an agent belief. There is no automatic
semantic deduplication of arbitrary paraphrases. In the explorer, enable Entities or
Propositions to inspect these additional layers. Retired connections are excluded from
the active graph but shown in the selected claim's audit history. These are additional
graph layers, not new truth assessments.
