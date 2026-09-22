# Cognition reliability

The four reasoning systems now share a durable execution record. System0 triages,
System1 executes short tasks, System2 maintains deliberative task state, and
System3 reviews strategies before handing execution to System1 or System2.
Strategy scores never count as verification of an executed task.

## Completion contract

Assistant final events include `completion`, `checkpoint_id`, and
`execution_metrics`. Completion has these states:

| Status | Meaning |
| --- | --- |
| `verified` | All caller-supplied deterministic acceptance checks passed; if the final critic is enabled, it also accepted the answer and its evidence. |
| `reviewed` | The final critic accepted the response; execution-dependent answers cite actual successful observations. No deterministic acceptance contract was supplied. |
| `incomplete` | Execution or review could not establish completion within the budgets. The response explicitly says completion could not be verified. |
| `clarify` | A blocking question needs an answer. This is not successful task completion. |

The final critic is enabled by default. Disabling it does not make a model's
answer verified: only passing caller checks can do that. With neither checks nor
review, the result is incomplete. A critic's confidence is not a probability of
correctness. Unknown evidence IDs and superseded successful observations cannot
support acceptance. Failed checks cannot be overridden by a favorable critique.

Provide acceptance checks through `OrchestratorSession.submit_turn`:

```python
await session.submit_turn(
    "Write 42 to result.txt and read it back to verify the contents.",
    acceptance_checks=[{
        "id": "result-file-content",
        "tool_id": "fs.read_file",
        "args": {"path": "result.txt"},
        "path": ["result", "content"],
        "expected": "42",
    }],
)
```

Checks are supplied by the application or user, not extracted from the agent's
claimed success. They match the **latest** tool result with the specified tool
and argument subset, require successful execution, and compare the indicated
result path with the expected value using type-sensitive equality. `path` starts
with `result`. A successful API envelope containing a failed command or an error
does not pass. Missing observations and missing result fields fail closed.

Applications must choose checks that cover the requested outcome. For example,
checking a test command's exit code alone cannot establish that the tests cover
every requirement. Use independent read-back tools to verify mutations; a write
tool saying it accepted a request is weaker evidence than inspecting the result.
Text-only explanations can be reviewed, but are not formally proved by this gate.

## Continuation and recovery

Defaults in `orchestrator` settings:

```json
{
  "cognition_max_segments": 3,
  "cognition_max_model_calls": 60,
  "cognition_timeout_s": 300,
  "final_response_critic_enabled": true,
  "final_response_critic_max_retries": 1
}
```

Each segment retains the existing System1/System2 step limits and per-segment
action limit. The total dispatched-tool limit is `cognition_max_segments` times
`cognition_action_limit`. Reasoning, critic, JSON repair, and semantic-summary
generation requests share the model-call budget; embedding requests and internal
tool-service work are not included in this counter. The time budget is cumulative
across execution, learning, and final review. Continuations and critic retries do
not reset these budgets.

System1 can escalate to System2 when it does not finish. System2 can continue in
another segment if it made observable progress. Repeated segments without new
observations stop. Explicitly forced System1 is respected within its segment;
subsequent continuation uses System2. A forced System3 request fails explicitly
if its proposer/critic pools are unavailable, rather than silently measuring a
System2 run as System3.

The original request, context, checks, query state, todos, strategy, observations,
tool recovery counters, and retry feedback survive continuation. The original
request is included independently of optional model-generated state. Retries
resume this record instead of replacing the task with critique instructions.

Atomic checkpoints live under `data_dir/cognition_reliability/runs`. Snapshots
are written at execution boundaries, including before and after external tool
calls. An OS file lock prevents concurrent executors from owning one checkpoint.
Resume an interrupted run using the same conversation source and original text:

```python
await session.submit_turn(original_request, resume_checkpoint=checkpoint_id)
```

The original checks and remaining budgets are retained. Changing the original
request, checks, or conversation source during resume is rejected. Completed runs
return their stored answer; exhausted budgets are not replenished by resuming.
The session protocol's source ID is a routing identity, not authentication.

Completed identical mutations are reused across continuation segments. Repeating
an action within one segment is allowed (for example repeated scrolling). Tools
explicitly marked read-only, or with read capability and tier0 permissions without
mutation capabilities, obtain fresh observations. Unknown tools are treated as
mutating. Crashes, cancellation, or ambiguous tool failures leave an uncertain
journal entry; the same mutation is blocked on retry. Inspect external state and
resolve the ambiguity before starting a new execution. This prevents blind replay
but cannot promise exactly-once behavior from an external service.

## System3 review

The existing score threshold and distinct-critic quorum still apply. A review
with `blocking: true` excludes that proposal even if its average score is high.
Outstanding reviews and the number of distinct critic model names are retained
in the checkpoint and passed to subsequent reasoning, including after fallback.
Distinct critic identities can still use the same model; the count exposes that
limitation without claiming statistical independence.

## Provisional learning

Per-episode facts, skills, patterns, lessons, and rules are stored under
`data_dir/cognition_reliability/learning` with the original request, execution
evidence, run and segment IDs, and an audit trail. They are not placed in ordinary
memory or callable-pattern retrieval while provisional.

```bash
python scripts/cognition_learning.py list
python scripts/cognition_learning.py show 'RUN_ID:SEGMENT'
python scripts/cognition_learning.py approve 'RUN_ID:SEGMENT' --reviewer alice --reason 'Checked each fact and tested the pattern source'
python scripts/cognition_learning.py reject 'RUN_ID:SEGMENT' --reviewer alice --reason 'Unsupported inference'
python scripts/cognition_learning.py retract 'RUN_ID:SEGMENT' --reviewer alice --reason 'Evidence no longer holds'
```

Promotion requires a deterministically verified run and explicit review of the
candidate. A later retry invalidates eligibility of an earlier segment's
candidate. Task success alone does not automatically validate a general lesson
or generated program: inspect every proposed artifact before approving it.
Promoted artifacts remain in the versioned store; retraction removes them from
future retrieval without erasing provenance. Existing manually maintained
catalogs remain available and are not overwritten by promotion.

The legacy direct cognition persistence method is disabled. Cognition turns also
skip the duplicate conversation distillation path, preventing it from indirectly
promoting the same unreviewed claims. Raw conversation episodes and explicit user
facts remain separate from learned operational knowledge.

## Evaluation

With the local services running:

```bash
python scripts/benchmark_cognition.py tests/fixtures/cognition_benchmark.json --modes SYSTEM2 --repeats 3 --output /tmp/cognition-system2.json
python scripts/benchmark_cognition.py your-complex-cases.json --modes SYSTEM2 SYSTEM3 --repeats 5 --output /tmp/cognition-comparison.json
```

Enable/configure the System3 peer pools before the comparison. The runner uses
the same server budgets for both modes, alternates order, isolates conversation
sources, disables learning, and denies permission requests. Supplied fixtures
are read-only arithmetic smoke tests, **not a complex-task reliability benchmark**.
Custom cases need `id`, `request`, nonempty `acceptance_checks`, and
`answer_contains`. For substantive evaluations, include representative tasks,
failure/recovery cases, and stronger domain-specific answer graders as needed.

Reports include success and false-success rates, latency, orchestrator generation
and tool-call counts, and successes requiring multiple segments. Answer fragments
are checked independently of the completion status. Monetary cost is `null`:
the transport does not expose sufficient provider billing data to calculate it.
An unavailable service or System3 pool is a failed run, not a passing result.
