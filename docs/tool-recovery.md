# Tool selection and recovery

Initial embedding retrieval includes bounded conversation context to resolve
follow-up references. Context is explicitly not evidence of the current screen.
Both category selection and action prompts show descriptions and tool contracts.

Registry tools can declare `source`, `use_for`, `not_for`, `requires`, `produces`,
and `platforms`. These fields are included in retrieval descriptions. For example:

```json
{"tool_id":"image.analyse","source":"image",
 "requires":[{"argument":"image_ref","artifact_type":"image","must_exist":true}]}
```

`computer.screenshot` produces an image; `vision.observe` observes camera frames.
Selecting an artifact consumer also offers available producers. It never executes
them automatically. Existing-image analysis checks the file before calling the
model. Other tools must enforce their own resource availability at runtime.

Actions may declare `intent: {"operation":"capture the screen","source":"desktop"}`.
Preflight rejects known source mismatches and missing required arguments. Source
checking uses declared intent, not another speculative classifier; absent intent
cannot guarantee task suitability.

`retrieve_tools` is available in normal and small cognition modes even with an
empty initial tool selection. It searches the full planner-visible registry,
beyond the initial active-tool cap, using semantic ranking with lexical fallback
if embeddings fail. A turn allows at most two searches. Source/platform/dependency
and unavailable-source failures trigger an alternative search automatically.
Candidates are offered for review, not automatically executed.

Tool failures carry `error_details` with a code and recovery instruction. Known
desktop and image failures distinguish missing inputs, dependencies, permissions,
unsupported platforms, unavailable sources and invalid model output. Unclassified
failures remain `UNKNOWN`. Identical failed calls are blocked; explicitly temporary
failures get one retry. Corrected arguments can be tried within the action budget.

Generation first defers to a registry search, then requires `generation_reason`
explaining the gap and feasibility. One generation action is allowed per turn;
code-generation retries are capped at eight, including legacy `infinity` settings.
Missing-input, permission and model-output failures cannot directly escalate to
generation. Existing approval, static safety, dependency and critic checks remain.
Generated code is compiled before registration, without executing it. This is a
syntax/executability check, **not a behavioral test or a security sandbox**; a
passing critic is also not proof that a generated tool will work.

Search and generation decisions are recorded as `tool_recovery_search` and
`tool_recovery_generation`; execution errors include structured details in tool
output logs. Restart the orchestrator and tool/model services after updating.
