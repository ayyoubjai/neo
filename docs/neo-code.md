# Neo Code

Neo Code implements repository coding sessions with either the configured local model (`--mode direct`) or the existing cognition pipeline (`--mode cognition`). Launchers are available for Bash (`neo-code`), PowerShell (`neo-code.ps1`), and Windows cmd (`neo-code.cmd`).

## Start a task

Add this installation's `scripts` directory to `PATH`, then run from the project you want to change:

```bash
neo-code "Build the requested feature" --mode cognition --apply \
  --test-command "python3 -m pytest -q" --max-steps 24
```

Cognition mode requires the local orchestrator stack. Direct mode uses `--provider ollama|hf|llamacpp` and `--model`, or the existing model settings. Cognition reasons about the next action; Neo Code owns execution. Its calls to cognition use `max_actions=0` so the orchestrator cannot independently run a second set of tools against another workspace.

To authorize model-selected setup commands, package installation, builds, and development servers:

```bash
neo-code "Create the project described in SPEC.md" --apply --allow-exec \
  --allow-network --test-command "npm test" --max-steps 40
```

Choose a verification command appropriate for the project. `--allow-exec` permits local processes with your permissions; this is not an operating-system sandbox. `--allow-network` enables the adapter's web search and HTTP tools; it does not restrict network access by authorized local processes. The agent does not install tools or dependencies merely because a flag is present.

Without `--apply`, the agent can inspect the repository and produce a validated patch for review. Source-changing tools, local process tools, and verification commands are disabled. Mirror refreshes, session checkpoints, and candidate artifacts are still written.

## Continue across milestones

Each response can update a persistent plan, maintain design notes, and request one category of action:

1. Read file ranges or search repository text.
2. Call up to four available tools in sequence, stopping the batch on a failure.
3. Apply a validated patch containing edits and new files.
4. Run verification using `verify: true`.
5. Explicitly finish with `status: complete`, or explain a blocker with `status: blocked`.

A successful patch/check does **not** terminate the task. Its result goes back to the model, which can implement the next milestone. Completion requires all milestones to be marked completed, no remaining background handles, and successful authorized checks for the current repository state. Fingerprints cover discovered source files and explicitly recorded changed paths. A later mutation, resume, or fingerprint change invalidates earlier verification.

The plan is maintained by the model. These checks improve completion discipline; they do not independently prove that every natural-language requirement was implemented.

## Resume a session

The command prints its checkpoint path when it starts:

```text
data/neo_code/sessions/<session-id>.json
```

The checkpoint contains the task, plan, durable notes, full stored transcript, changed paths, step count, execution options, pending action, and verification evidence. Writes use atomic replacement, and an exclusive lock prevents two invocations from using the same session simultaneously.

```bash
neo-code --resume data/neo_code/sessions/<session-id>.json --max-steps 24
```

The step budget is additional to previous invocations. Run from the original target repository, or supply its `--repo-root`. Resume retains the original task, model settings, execution permissions, and verification command unless you override them. Examples include `--no-allow-exec`, `--no-allow-network`, `--no-apply`, or a new `--test-command`.

Earlier actions are not automatically replayed. If interrupted during an action, the next prompt identifies the uncertain action so the model can inspect current files. Verification must run again. Background process handles belong to one invocation; normal exit and interruption handling stop owned processes, and resumed sessions must restart any needed servers. A forcibly killed host process cannot run its cleanup handlers.

The full stored transcript remains in the checkpoint. Prompts use the durable plan/notes plus a bounded recent observation window. This is persistence with bounded context, not unlimited model memory.

## Tools and execution

Neo Code reuses the existing tool-runtime implementations through `src/coding_agent/tools.py`, scoped to the target repository.

| Tools | Availability |
|---|---|
| File-range reads, text/path searches, directory listing, file stat | Default |
| `git.status`, `git.diff`, `git.log` | Default |
| Validated patches; `fs.mkdir`, `fs.move`, `fs.copy`, `fs.delete` | `--apply` |
| `proc.exec`, `proc.start`, `proc.poll`, `proc.stop` | `--apply --allow-exec` |
| `net.search`, `http.request` | `--allow-network` |

Move/copy/delete operate on individual files; move/copy cannot overwrite existing destinations. File and working-directory paths are checked after resolving symlinks. Direct file access to Git metadata and Neo Code session storage is blocked. The adapter exposes selected built-in tools, not arbitrary generated tools or account integrations.

HTTP requests support GET/HEAD and bounded responses. Search uses the configured SearXNG service. Network failures are returned as observations.

Process commands use argv arrays, for example:

```json
{"tools": [{"tool_id": "proc.exec", "args": {
  "command": ["python3", "-m", "pytest", "-q"], "cwd": ".", "timeout_s": 120
}}]}
```

`proc.start` creates a background handle for a server. `proc.poll` returns new output and status; `proc.stop` terminates its process group on POSIX or process tree on Windows. There are at most four owned background handles. The default lifetime is 600 seconds, configurable up to 3600. Stop handles before final verification. Foreground execution has a timeout and bounded captured output; process output is spooled to temporary files.

A successful process tool call alone is not final verification. The agent must run authorized checks. `--test-command` takes precedence. Otherwise, `--allow-model-tests` or `--allow-exec` permits up to three model-proposed shell verification commands per check action.

## Review and saved candidates

Candidate files are saved under `data/neo_code/`. Apply a reviewed candidate without generating it again:

```bash
neo-code --candidate data/neo_code/candidate_<timestamp>_<id>.json \
  --apply --test-command "python3 -m pytest -q"
```

Older candidate files under `data/local_code_agent/` remain readable. Saved-candidate application is a single patch/check operation, not a resumable task. Existing-file edits require matching text, new-file entries cannot overwrite existing files, and the whole patch is validated before writes. Write failures trigger rollback; an incomplete rollback stops automatic repair.

Default initial indexing uses the shared `soul` mirror engine, respects Git ignore rules, and discovers unfamiliar project layouts. A target's `config/soul.json` overrides the default indexing configuration. Root `AGENTS.md` is included in the prompt; nested instruction-file loading is not automatic.

Exit status `0` means a review candidate was produced, a saved candidate passed its checks, or a session explicitly completed after verification. Status `2` means incomplete, blocked, budget exhausted, or a validation error. Saved-candidate check failures return the check's exit code. Applied changes remain in the working tree for review or continuation.
