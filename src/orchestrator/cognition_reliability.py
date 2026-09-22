"""Durable cognition execution and deterministic outcome checks.

Model review is deliberately distinct from verification. Acceptance checks are
provided by the caller, never inferred from a model's claimed success.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import tempfile
import time
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ACTIVE_RUN: ContextVar[dict | None] = ContextVar("cognition_run", default=None)


class CognitionBudgetExceeded(RuntimeError):
    pass


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class RunStore:
    """Atomic snapshots; an in-flight mutation survives crashes as uncertain."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, kind: str, identifier: str) -> Path:
        return self.root / kind / (fingerprint(identifier) + ".json")

    def read(self, kind: str, identifier: str) -> dict:
        with self.path(kind, identifier).open(encoding="utf-8") as handle:
            return json.load(handle)

    def write(self, kind: str, identifier: str, payload: dict) -> None:
        path = self.path(kind, identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @contextmanager
    def lease(self, identifier: str):
        """One executor per checkpoint; OS locks are released after a crash."""
        path = self.path("locks", identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                lock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                unlock = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))
            else:
                import fcntl
                lock = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                unlock = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
            try:
                lock()
            except OSError as exc:
                raise RuntimeError("Checkpoint is already executing") from exc
            try:
                yield
            finally:
                unlock()


def successful(observation: dict) -> bool:
    if not isinstance(observation, dict):
        return False
    result = observation.get("result")
    return (
        observation.get("status") in {"APPROVED", "SUCCESS", "OK"}
        and not observation.get("error")
        and not (isinstance(result, dict) and (
            result.get("success") is False or result.get("ok") is False
            or result.get("exit_code", 0) != 0 or result.get("error")
        ))
    )


def evaluate_checks(checks: list, observations: list) -> list[dict]:
    """Check the latest matching tool result, so stale success cannot hide failure.

    Contract: {id, tool_id, args: {...}, path: ["result", "exit_code"],
    expected: 0}. Equality is deliberately the only operation; no eval/shell.
    """
    results = []
    for check in checks:
        row = {"id": str(check.get("id", "")) if isinstance(check, dict) else "", "passed": False}
        results.append(row)
        if (not isinstance(check, dict) or not row["id"] or not check.get("tool_id")
                or "expected" not in check or not isinstance(check.get("path"), list)
                or not check["path"] or check["path"][0] != "result"
                or not isinstance(check.get("args", {}), dict)):
            row["reason"] = "Invalid acceptance check"
            continue
        matches = [item for item in observations if
                   (item.get("action") or {}).get("tool_id") == check["tool_id"]
                   and all(((item.get("action") or {}).get("args") or {}).get(k) == v
                           for k, v in check.get("args", {}).items())]
        if not matches:
            row["reason"] = "No matching execution evidence"
            continue
        latest = matches[-1]
        row["evidence_id"] = latest.get("evidence_id")
        value = latest.get("observation", {})
        if not successful(value):
            row["reason"] = "Latest matching execution did not succeed"
            continue
        try:
            for key in check["path"]:
                value = value[key]
            row["passed"] = type(value) is type(check["expected"]) and value == check["expected"]
            row["reason"] = "Matched" if row["passed"] else "Observed value differs from expected value"
        except (KeyError, IndexError, TypeError):
            row["reason"] = "Evidence path is absent"
    return results


class CognitionReliabilityMixin:
    def _run_store(self) -> RunStore:
        return RunStore(Path(self._cognition_episodes_path).parent / "cognition_reliability")

    def _checkpoint_cognition(self, run: dict | None = None) -> None:
        run = run if run is not None else ACTIVE_RUN.get()
        if run is not None:
            self._run_store().write("runs", run["run_id"], run)

    async def _run_cognition_loop(
        self, text, context_packet, trace_id, turn_id=None, retrieval_query=None,
        forced_execution_mode=None, skip_distillation=False, *,
        resume_checkpoint=None, acceptance_checks=None,
    ):
        with self._run_store().lease(resume_checkpoint or trace_id):
            if not resume_checkpoint and self._run_store().path("runs", trace_id).exists():
                raise ValueError("Run ID already exists; use a new ID or explicitly resume its checkpoint")
            return await self._run_cognition_owned(
                text, context_packet, trace_id, turn_id, retrieval_query,
                forced_execution_mode, skip_distillation,
                resume_checkpoint=resume_checkpoint, acceptance_checks=acceptance_checks)

    async def _run_cognition_owned(
        self, text, context_packet, trace_id, turn_id=None, retrieval_query=None,
        forced_execution_mode=None, skip_distillation=False, *,
        resume_checkpoint=None, acceptance_checks=None,
    ):
        cfg = self._settings.orchestrator
        max_segments = max(1, int(cfg.get("cognition_max_segments", 3)))
        run = self._run_store().read("runs", resume_checkpoint) if resume_checkpoint else {
            "version": 1, "run_id": trace_id, "original_request": text,
            "created_at": time.time(), "context_packet": copy.deepcopy(context_packet),
            "source_id": str(context_packet.get("cognition_source_id") or ""),
            "query_state": {}, "state_of_mind": {}, "observations": [], "thinking_trace": [],
            "tools": [], "journal": {}, "segments": 0, "actions_used": 0,
            "max_actions": max_segments * self._cognition_action_limit,
            "max_segments": max_segments, "status": "running", "strategy_brief": "",
            "checks": copy.deepcopy(acceptance_checks or []), "skip_distillation": skip_distillation,
            "model_calls": 0, "elapsed_s": 0.0,
            "max_model_calls": max(1, int(cfg.get("cognition_max_model_calls", 60))),
            "timeout_s": max(1, float(cfg.get("cognition_timeout_s", 300))),
        }
        if resume_checkpoint and text != run["original_request"]:
            raise ValueError("A checkpoint must be resumed with its original request")
        if resume_checkpoint and str(context_packet.get("cognition_source_id") or "") != run["source_id"]:
            raise ValueError("A checkpoint belongs to a different conversation source")
        if resume_checkpoint and acceptance_checks is not None and acceptance_checks != run["checks"]:
            raise ValueError("Acceptance checks cannot change during continuation")
        if resume_checkpoint:
            context_packet = copy.deepcopy(run["context_packet"])
        skip_distillation = skip_distillation or run["skip_distillation"]
        run["context_packet"] = context_packet
        self._recovery_state(trace_id)
        if "recovery_state" in run:
            self._tool_recovery_states[trace_id] = run["recovery_state"]
        else:
            run["recovery_state"] = self._tool_recovery_states[trace_id]
        if not hasattr(self, "_cognition_runs"):
            self._cognition_runs = {}
        if turn_id:
            self._cognition_runs[turn_id] = run
        if run.get("status") in {"verified", "reviewed", "clarify"}:
            return run.get("response", "")
        token = ACTIVE_RUN.set(run)
        started = time.monotonic()
        try:
            response = run.get("response", "")
            while run["segments"] < run["max_segments"]:
                prior = len(run["observations"])
                resume = bool(run["segments"])
                run["segments"] += 1
                run["status"] = "running"
                self._checkpoint_cognition(run)
                remaining_s = run["timeout_s"] - run["elapsed_s"] - (time.monotonic() - started)
                if remaining_s <= 0:
                    raise CognitionBudgetExceeded("Total cognition time budget exhausted")
                response = await asyncio.wait_for(self._run_cognition_pass(
                        run["original_request"], context_packet, trace_id, turn_id,
                        retrieval_query=retrieval_query,
                        forced_execution_mode="system2" if resume else forced_execution_mode,
                        skip_distillation=True, resume_state=run if resume else None,
                    ), timeout=remaining_s)
                run["response"] = response
                self._checkpoint_cognition(run)
                if run.get("explicit_result") or len(run["observations"]) == prior:
                    break
                if run["actions_used"] >= run["max_actions"]:
                    break
            run["status"] = "candidate" if run.get("explicit_result") else "incomplete"
            if run.get("result_type") == "clarify":
                run["status"] = "clarify"
            if run["status"] == "incomplete":
                response = self._incomplete_response(run, ["Execution stopped before a complete result was established."])
            run["response"] = response
            if not skip_distillation and run.get("result_type") != "clarify":
                # Learning is advisory and shares the remaining execution budget.
                remaining_s = run["timeout_s"] - run["elapsed_s"] - (time.monotonic() - started)
                if remaining_s > 0 and run["model_calls"] < run["max_model_calls"]:
                    try:
                        await asyncio.wait_for(self._stage_cognition_learning(run, trace_id, turn_id), remaining_s)
                    except (RuntimeError, OSError, asyncio.TimeoutError) as exc:
                        run["learning_error"] = type(exc).__name__
            return response
        except (CognitionBudgetExceeded, asyncio.TimeoutError) as exc:
            run["status"] = "incomplete"
            run["stop_reason"] = str(exc) or "Total cognition time budget exhausted"
            run["response"] = self._incomplete_response(run, [run["stop_reason"]])
            return run["response"]
        except BaseException:
            run["status"] = "interrupted"
            raise
        finally:
            run["elapsed_s"] += time.monotonic() - started
            try:
                self._checkpoint_cognition(run)
            finally:
                ACTIVE_RUN.reset(token)

    def _incomplete_response(self, run: dict, issues: list) -> str:
        detail = "; ".join(str(item) for item in issues[:6])
        return "I couldn't verify completion of this request. " + detail

    async def _call_tool(self, tool_id, args, approval_token, trace_id, required_artifacts):
        run = ACTIVE_RUN.get()
        if run is None or tool_id == "sys.time":
            return await self._call_tool_unjournaled(tool_id, args, approval_token, trace_id, required_artifacts)
        tool = self._tools.get_tool(tool_id) or {}
        # Unknown tools are conservatively treated as mutating. Explicit read
        # capability plus tier0 permits fresh observations on every call.
        read_only = tool.get("read_only") is True or (
            "read" in tool.get("capabilities", [])
            and tool.get("required_permissions") == ["tier0"]
            and not {"write", "delete", "send", "execute"}.intersection(tool.get("capabilities", []))
        )
        key = fingerprint({"tool_id": tool_id, "args": args})
        previous = run["journal"].get(key)
        if not read_only and previous:
            if previous["state"] == "succeeded" and previous.get("segment", 0) < run["segments"]:
                return copy.deepcopy(previous["response"])
            if previous["state"] == "uncertain":
                return {"status": "ERROR", "error": "Previous execution may have taken effect. Inspect external state before repeating this mutation.",
                        "error_details": {"code": "UNCERTAIN_EXECUTION"}}
        if run["actions_used"] >= run["max_actions"]:
            return {"status": "ERROR", "error": "Total cognition action budget exhausted",
                    "error_details": {"code": "BUDGET_EXHAUSTED"}}
        run["actions_used"] += 1
        entry = {"state": "uncertain", "segment": run["segments"], "tool_id": tool_id, "args": copy.deepcopy(args)}
        if not read_only:
            run["journal"][key] = entry
        self._checkpoint_cognition(run)  # Persist intent before an external effect.
        response = await self._call_tool_unjournaled(tool_id, args, approval_token, trace_id, required_artifacts)
        response = {**response, "evidence_id": f"tool-{run['actions_used']}"}
        if not read_only:
            entry["response"] = copy.deepcopy(response)
            # Errors/timeouts may follow a committed write. Only explicit denial
            # proves no effect; uncertain mutations are never automatically replayed.
            entry["state"] = "succeeded" if successful(response) else (
                "denied" if response.get("status") == "DENIED" else "uncertain")
        evidence = {"evidence_id": f"tool-{run['actions_used']}",
                    "action": {"tool_id": tool_id, "args": copy.deepcopy(args)},
                    "observation": copy.deepcopy(response)}
        run.setdefault("tool_evidence", []).append(evidence)
        run["observations"].append(evidence)
        self._checkpoint_cognition(run)
        return response

    async def _stage_cognition_learning(self, run, trace_id, turn_id):
        if not self._cognition_distill_enabled:
            return
        artifacts = await self._distill_cognition_episode(
            {"type": run.get("result_type", "final"), "text": run.get("response", "")},
            run["query_state"], run["state_of_mind"], run["observations"], run["thinking_trace"],
            run["tools"], self._load_cognition_patterns(), trace_id, turn_id,
            budget_exhausted=not run.get("explicit_result", False), user_text=run["original_request"],
        )
        if any(artifacts.get(key) for key in ("memory_facts", "patterns", "reusable_skills", "failure_lessons", "safety_rules")):
            candidate_id = run["run_id"] + ":" + str(run["segments"])
            self._run_store().write("learning", candidate_id, {
                "candidate_id": candidate_id, "status": "provisional", "run_id": run["run_id"],
                "created_at": time.time(), "segment": run["segments"],
                "original_request": run["original_request"], "artifacts": artifacts,
                "evidence": run.get("tool_evidence", run["observations"]), "reviews": [],
            })
            run["learning_candidate_id"] = candidate_id

    async def _apply_final_response_critic(self, initial_response, mode, context_packet,
                                         trace_id, turn_id, query_state=None):
        run = getattr(self, "_cognition_runs", {}).get(turn_id)
        if run is None:
            return initial_response
        with self._run_store().lease(run["run_id"]):
            saved = self._run_store().read("runs", run["run_id"])
            if saved["segments"] != run["segments"]:
                self._cognition_runs[turn_id] = saved
                initial_response = saved["response"]
            return await self._apply_cognition_completion(
                initial_response, mode, context_packet, trace_id, turn_id, query_state)

    async def _apply_cognition_completion(self, initial_response, mode, context_packet,
                                         trace_id, turn_id, query_state=None):
        run = getattr(self, "_cognition_runs", {}).get(turn_id)
        if run is None:
            # Non-cognition callers cannot acquire verified status without a run.
            return initial_response
        if run.get("status") in {"clarify", "verified", "reviewed"}:
            return initial_response
        candidate = initial_response
        for attempt in range(self._final_response_critic_max_retries + 1):
            evidence = run["observations"] or run.get("tool_evidence", [])
            checks = evaluate_checks(run["checks"], evidence)
            review_enabled = (self._final_response_critic_enabled
                              and mode.upper() not in self._final_response_critic_skip_modes)
            review = {"fulfilled": False, "confidence": 0.0, "issues": ["Completion review is disabled"]}
            if review_enabled:
                review_started = time.monotonic()
                try:
                    remaining_s = run["timeout_s"] - run["elapsed_s"]
                    if remaining_s <= 0 or run["model_calls"] >= run["max_model_calls"]:
                        raise CognitionBudgetExceeded("Completion review budget exhausted")
                    # The critic and JSON repair also share the total call budget.
                    token = ACTIVE_RUN.set(run)
                    try:
                        review = await asyncio.wait_for(
                            self._evaluate_final_response_critic(candidate, mode, trace_id, turn_id), remaining_s)
                    finally:
                        ACTIVE_RUN.reset(token)
                except (RuntimeError, OSError, asyncio.TimeoutError) as exc:
                    review = {"fulfilled": False, "confidence": 0.0,
                              "issues": [f"Completion review unavailable: {type(exc).__name__}"]}
                finally:
                    run["elapsed_s"] += time.monotonic() - review_started
            passed_review = (review.get("fulfilled") is True
                             and review.get("confidence", 0) >= self._final_response_critic_min_confidence)
            latest = {}
            for item in evidence:
                action = item.get("action") or {}
                latest[fingerprint({"tool_id": action.get("tool_id"), "args": action.get("args", {})})] = item
            valid_ids = {item.get("evidence_id") for item in latest.values()
                         if item.get("evidence_id") and successful(item.get("observation", {}))}
            refs = review.get("evidence_ids", [])
            if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                refs = []
            if evidence or review.get("requires_execution") is True:
                passed_review = passed_review and bool(refs) and all(ref in valid_ids for ref in refs)
            passed_checks = bool(checks) and all(item["passed"] for item in checks)
            verified = passed_checks and (passed_review or not review_enabled)
            reviewed = not checks and passed_review
            run["completion"] = {"status": "verified" if verified else "reviewed" if reviewed else "incomplete",
                                 "checks": checks, "review": review}
            if verified or reviewed:
                run["status"] = run["completion"]["status"]
                run["response"] = candidate
                self._checkpoint_cognition(run)
                return candidate
            issues = list(review.get("issues", []))
            issues.extend(f"{item['id']}: {item['reason']}" for item in checks if not item["passed"])
            if not issues:
                issues = ["The result lacks sufficient supporting execution evidence."]
            if attempt == self._final_response_critic_max_retries or run["segments"] >= run["max_segments"]:
                run["status"] = "incomplete"
                run["response"] = self._incomplete_response(run, issues)
                self._checkpoint_cognition(run)
                return run["response"]
            run["feedback"] = {"issues": issues, "fix_instructions": review.get("fix_instructions", []),
                               "previous_response": candidate}
            run["status"] = "retrying"
            self._checkpoint_cognition(run)
            candidate = await self._run_cognition_owned(
                run["original_request"], context_packet, trace_id, turn_id,
                resume_checkpoint=run["run_id"], skip_distillation=run["skip_distillation"],
            )
            run = self._cognition_runs[turn_id]
            if run.get("status") == "clarify":
                return candidate

    async def review_cognition_learning(self, candidate_id: str, *, reviewer: str,
                                       reason: str, approve: bool) -> dict:
        """Explicit per-candidate review; never invoked by a model tool.

        Promotion additionally requires verified task checks. A reviewer must
        inspect *all* candidate artifacts, including executable pattern source.
        Rejected/revoked knowledge remains in the audit file, outside retrieval.
        """
        if not reviewer.strip() or not reason.strip():
            raise ValueError("Learning review requires a reviewer and a reason")
        candidate = self._run_store().read("learning", candidate_id)
        run = self._run_store().read("runs", candidate["run_id"])
        if approve and run.get("status") != "verified":
            raise ValueError("Only deterministically verified episodes can be promoted")
        if approve and candidate.get("segment") != run["segments"]:
            raise ValueError("A later retry superseded this candidate; review the final segment's learning")
        if candidate["status"] != "provisional":
            raise ValueError("Candidate has already been reviewed")
        candidate["reviews"].append({"reviewer": reviewer, "reason": reason, "approve": approve, "at": time.time()})
        if approve:
            candidate["status"] = "promoted"
        else:
            candidate["status"] = "rejected"
        self._run_store().write("learning", candidate_id, candidate)
        return candidate

    def _reviewed_learning_items(self, kind, normalizer):
        entries = []
        for candidate in self._promoted_learning():
            for raw in candidate["artifacts"].get(kind, []):
                item = normalizer(raw)
                if item is not None:
                    item["provenance"] = {"candidate_id": candidate["candidate_id"],
                                          "run_id": candidate["run_id"], "reviews": candidate["reviews"]}
                    entries.append(item)
        return entries

    def _promoted_learning(self):
        directory = self._run_store().root / "learning"
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                try:
                    candidate = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if candidate.get("status") == "promoted":
                    yield candidate

    def retract_cognition_learning(self, candidate_id: str, *, reviewer: str, reason: str) -> dict:
        if not reviewer.strip() or not reason.strip():
            raise ValueError("Retraction requires reviewer and reason")
        candidate = self._run_store().read("learning", candidate_id)
        if candidate["status"] != "promoted":
            raise ValueError("Candidate is not promoted")
        # Catalogs and facts both retrieve from this status-filtered store.
        candidate["status"] = "retracted"
        candidate["reviews"].append({"reviewer": reviewer, "reason": reason, "retracted": True, "at": time.time()})
        self._run_store().write("learning", candidate_id, candidate)
        return candidate

    def _reviewed_learning_facts(self, text: str) -> list:
        facts = []
        terms = set(text.lower().split())
        for candidate in self._promoted_learning():
            for fact in candidate["artifacts"].get("memory_facts", []):
                if terms.intersection((fact.get("name", "") + " " + fact.get("value", "")).lower().split()):
                    facts.append({"name": fact["name"], "data": {"value": fact["value"]},
                                  "score": fact.get("confidence", 0), "candidate_id": candidate["candidate_id"]})
        return sorted(facts, key=lambda item: item["score"], reverse=True)[:self._memory_retrieval_k]
