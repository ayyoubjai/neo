import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from common.config import load_settings
from model_server.hf_client import HfError
from model_server.llamacpp_client import LlamacppError
from model_server.local_generation import generate_local_text
from model_server.ollama_client import OllamaError


_LOCAL_MODEL_UNAVAILABLE_MESSAGE = (
    "I hit a model/runtime issue while processing your request, so I cannot give a reliable answer right now. "
    "Please try again or switch to a smaller/local model."
)
_LOCAL_MODEL_UNAVAILABLE_LINES = (
    "I hit a model/runtime issue while processing your request, so I cannot give a reliable answer right now. ",
    "Please try again or switch to a smaller/local model.",
)


def _line_indent(text: str) -> str:
    stripped = text.lstrip(" \t")
    return text[: len(text) - len(stripped)]


def _find_assignment_block(text: str, variable: str) -> str:
    lines = text.splitlines(keepends=True)
    token = f"{variable} ="
    for index, line in enumerate(lines):
        stripped = line.lstrip(" \t")
        if not stripped.startswith(token):
            continue
        indent = _line_indent(line)
        if stripped.rstrip().endswith("("):
            block = [line]
            for follow_index in range(index + 1, len(lines)):
                block.append(lines[follow_index])
                follow_line = lines[follow_index]
                if follow_line.strip() == ")" and _line_indent(follow_line) == indent:
                    return "".join(block)
            return ""
        return line
    return ""


def _find_return_dict_block(text: str, required_fragments: List[str]) -> str:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.lstrip(" \t")
        if not stripped.startswith("return {"):
            continue
        block = [line]
        depth = line.count("{") - line.count("}")
        for follow_index in range(index + 1, len(lines)):
            follow_line = lines[follow_index]
            block.append(follow_line)
            depth += follow_line.count("{") - follow_line.count("}")
            if depth <= 0:
                block_text = "".join(block)
                if all(fragment in block_text for fragment in required_fragments):
                    return block_text
                break
    return ""


def _target_problem_text(payload: Dict[str, Any]) -> str:
    target = payload.get("target", {}) if isinstance(payload.get("target"), dict) else {}
    reflection = payload.get("reflection", {}) if isinstance(payload.get("reflection"), dict) else {}
    parts: List[str] = [
        str(target.get("notes", "") or ""),
        str(reflection.get("problem_statement", "") or ""),
        str(reflection.get("title", "") or ""),
    ]
    for key in ("suggested_actions", "verification_targets", "rationale"):
        values = reflection.get(key, [])
        if not isinstance(values, list):
            continue
        parts.extend(str(item) for item in values if isinstance(item, str))
    return "\n".join(part for part in parts if part.strip()).lower()


def _make_candidate(candidate_id: str, summary: str, path: str, before: str, after: str) -> Dict[str, Any]:
    return {
        "id": candidate_id,
        "summary": summary,
        "edits": [{"path": path, "before": before, "after": after}],
    }


def _build_model_unavailable_candidate(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target = payload.get("target", {}) if isinstance(payload.get("target"), dict) else {}
    path = str(target.get("path", "") or "").strip()
    content = str(target.get("content", "") or "")
    problem_text = _target_problem_text(payload)
    if not path.endswith("src/model_server/text_model.py"):
        return None
    if "model backend was unavailable" not in problem_text and "offline" not in problem_text:
        return None
    before = _find_assignment_block(content, "fallback")
    if not before or _LOCAL_MODEL_UNAVAILABLE_MESSAGE in before:
        return None
    indent = _line_indent(before.splitlines()[0])
    after = (
        f"{indent}fallback = (\n"
        f'{indent}    "{_LOCAL_MODEL_UNAVAILABLE_LINES[0]}"\n'
        f'{indent}    "{_LOCAL_MODEL_UNAVAILABLE_LINES[1]}"\n'
        f"{indent})\n"
    )
    if before == after:
        return None
    return _make_candidate(
        "local_model_unavailable_response",
        "Make offline model fallback explicit instead of optimistic.",
        path,
        before,
        after,
    )


def _build_final_critic_candidate(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target = payload.get("target", {}) if isinstance(payload.get("target"), dict) else {}
    path = str(target.get("path", "") or "").strip()
    content = str(target.get("content", "") or "")
    problem_text = _target_problem_text(payload)
    if not path.endswith("src/orchestrator/main.py"):
        return None
    if "final response critic" not in problem_text and "critic parse" not in problem_text:
        return None
    before = _find_return_dict_block(content, ['"fulfilled"', "True"])
    if not before or "final response critic unavailable or parse failed" in before:
        return None
    indent = _line_indent(before.splitlines()[0])
    inner = indent + "    "
    after = (
        f"{indent}return {{\n"
        f'{inner}"fulfilled": False,\n'
        f'{inner}"confidence": 0.0,\n'
        f'{inner}"issues": ["final response critic unavailable or parse failed"],\n'
        f'{inner}"fix_instructions": [\n'
        f'{inner}    "Return a safe user-visible response that explicitly reflects degraded execution state."\n'
        f"{inner}],\n"
        f"{indent}}}\n"
    )
    if before == after:
        return None
    return _make_candidate(
        "local_final_critic_guard",
        "Mark final-response critic parse failures as degraded instead of fulfilled.",
        path,
        before,
        after,
    )


def _build_model_service_status_candidate(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target = payload.get("target", {}) if isinstance(payload.get("target"), dict) else {}
    path = str(target.get("path", "") or "").strip()
    content = str(target.get("content", "") or "")
    problem_text = _target_problem_text(payload)
    if not path.endswith("src/model_server/main.py"):
        return None
    if "model backend was unavailable" not in problem_text and "offline" not in problem_text:
        return None
    before = '        return {"status": "OK", "text": output}\n'
    if before not in content:
        return None
    after = (
        '        status = "DEGRADED" if not output or output.startswith(\n'
        '            "I hit a model/runtime issue while processing your request"\n'
        '        ) else "OK"\n'
        '        return {"status": status, "text": output}\n'
    )
    return _make_candidate(
        "local_model_service_degraded_status",
        "Surface degraded model generation status instead of always returning OK.",
        path,
        before,
        after,
    )


def _build_local_fallback_result(
    payload: Dict[str, Any],
    *,
    max_candidates: int,
    provider: str,
    model_id: str,
    reason: str,
) -> Dict[str, Any]:
    builders = (
        _build_model_unavailable_candidate,
        _build_model_service_status_candidate,
        _build_final_critic_candidate,
    )
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for builder in builders:
        candidate = builder(payload)
        if not isinstance(candidate, dict):
            continue
        candidate_key = json.dumps(candidate.get("edits", []), ensure_ascii=True, sort_keys=True)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        candidates.append(candidate)
        if max_candidates > 0 and len(candidates) >= max_candidates:
            break
    return {
        "candidates": candidates[:max_candidates] if max_candidates > 0 else candidates,
        "generator": {
            "mode": "local_fallback",
            "provider": provider,
            "model": model_id,
            "reason": reason,
            "strategy": "heuristic_reflection_templates",
            "candidate_count": len(candidates[:max_candidates] if max_candidates > 0 else candidates),
        },
    }


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("No valid JSON object found in output.")


def _trim_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _format_seed_candidates(seed_candidates: List[Dict[str, Any]], max_chars: int = 10000) -> str:
    if not seed_candidates:
        return "None"
    chunks: List[str] = []
    used = 0
    for idx, seed in enumerate(seed_candidates, start=1):
        if not isinstance(seed, dict):
            continue
        seed_id = str(seed.get("id", f"seed_{idx}"))
        summary = str(seed.get("summary", "") or "")
        score = seed.get("score", "")
        generation = seed.get("generation", "")
        lines = [f"[{seed_id}] generation={generation} score={score}", f"summary: {summary}"]
        edits = seed.get("edits", [])
        if isinstance(edits, list):
            for edit in edits[:6]:
                if not isinstance(edit, dict):
                    continue
                path = edit.get("path", "")
                before = _trim_text(str(edit.get("before", "") or ""), 220)
                after = _trim_text(str(edit.get("after", "") or ""), 220)
                occurrence = edit.get("occurrence")
                occ_text = f", occurrence={occurrence}" if occurrence is not None else ""
                lines.append(f"- edit path={path}{occ_text}")
                lines.append(f"  before: {before}")
                lines.append(f"  after: {after}")
        files = seed.get("files", [])
        if isinstance(files, list):
            for file_entry in files[:3]:
                if not isinstance(file_entry, dict):
                    continue
                path = file_entry.get("path", "")
                content = _trim_text(str(file_entry.get("content", "") or ""), 220)
                lines.append(f"- file path={path}")
                lines.append(f"  content: {content}")
        chunk = "\n".join(lines)
        size = len(chunk)
        if max_chars > 0 and used + size > max_chars:
            break
        chunks.append(chunk)
        used += size
    return "\n\n".join(chunks) if chunks else "None"


def _build_prompt(
    payload: Dict[str, Any],
    max_candidates: int,
    enable_thinking: bool,
    run_index: int = 1,
    run_total: int = 1,
    avoid_summaries: Optional[List[str]] = None,
) -> str:
    target = payload.get("target", {})
    metrics = payload.get("metrics", [])
    context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
    generation = payload.get("generation", {}) if isinstance(payload.get("generation"), dict) else {}
    seed_candidates = payload.get("seed_candidates", []) if isinstance(payload.get("seed_candidates"), list) else []
    reflection = payload.get("reflection", {}) if isinstance(payload.get("reflection"), dict) else {}
    path = target.get("path", "")
    start_line = target.get("start_line", 0)
    end_line = target.get("end_line", 0)
    content = target.get("content", "")
    target_notes = str(target.get("notes", "") or "").strip()

    metrics_text = []
    for metric in metrics:
        name = metric.get("name", "")
        weight = metric.get("weight", 0.0)
        required = metric.get("required", False)
        metrics_text.append(f"- {name} (weight={weight}, required={required})")
    metrics_block = "\n".join(metrics_text) if metrics_text else "None"

    context_snippets = context.get("snippets", []) if isinstance(context.get("snippets"), list) else []
    context_notes = context.get("notes", "")
    context_block = ""
    if context_snippets:
        parts = []
        for snippet in context_snippets:
            if not isinstance(snippet, dict):
                continue
            spath = snippet.get("path", "")
            sstart = snippet.get("start_line", 0)
            send = snippet.get("end_line", 0)
            scontent = snippet.get("content", "")
            parts.append(f"[{spath}:{sstart}-{send}]\n{scontent}")
        context_block = "\n\n".join(parts)

    generation_index = int(generation.get("index") or 1)
    generation_total = int(generation.get("total") or 1)
    seed_block = _format_seed_candidates(seed_candidates)
    avoid_block = "\n".join([f"- {s}" for s in avoid_summaries[-8:]]) if avoid_summaries else "None"
    reflection_block = _format_reflection_block(reflection)
    run_note = f"Generation run {run_index} of {run_total}."
    if run_total > 1:
        run_note += " Return a meaningfully different strategy from other runs."
    if seed_candidates:
        mode_text = (
            "You are in evolutionary refinement mode. Build child candidates that combine strengths of seed candidates, "
            "remove their weaknesses, and improve metric outcomes."
        )
    else:
        mode_text = "You are in base mutation mode. Propose strong initial improvements."

    prompt = (
        "You are an expert software engineer. Propose code improvements for the target block. "
        "Return JSON only. Do not include markdown fences or extra commentary.\n\n"
        f"{mode_text}\n\n"
        f"Generation: {generation_index}/{generation_total}\n"
        f"{run_note}\n\n"
        f"Produce up to {max_candidates} candidate(s). Each candidate must include:\n"
        "- id (string)\n"
        "- summary (string)\n"
        "- edits (list of {path, before, after[, occurrence]}). Use exact substrings from repository files.\n\n"
        "Constraints:\n"
        "- Keep changes minimal and focused on the target path unless necessary.\n"
        "- Avoid unrelated refactors.\n"
        "- Do not add new dependencies unless justified.\n"
        "- Preserve behavior unless it improves objectives.\n\n"
        "If seed candidates are provided, do not merely copy one parent unchanged; synthesize and improve.\n"
        "Prefer compatibility with required metrics.\n\n"
        "If you cannot produce valid edits, return an empty candidates list.\n\n"
        f"Metrics:\n{metrics_block}\n\n"
        f"Seed candidates:\n{seed_block}\n\n"
        f"Avoid repeating these recent summaries:\n{avoid_block}\n\n"
        f"Target path: {path}\n"
        f"Target lines: {start_line}-{end_line}\n"
        f"Target notes:\n{target_notes or 'None'}\n\n"
        f"Reflection dossier:\n{reflection_block or 'None'}\n\n"
        "Target content:\n"
        f"{content}\n\n"
        f"Additional context notes:\n{context_notes or 'None'}\n\n"
        f"Additional context snippets:\n{context_block or 'None'}\n\n"
        "Output JSON format:\n"
        "{\"candidates\": ["
        "{\"id\": \"cand1\", \"summary\": \"...\", "
        "\"edits\": [{\"path\": \"src/x.py\", \"before\": \"old\", \"after\": \"new\"}]}"
        "]}"
    )
    if enable_thinking:
        return "/think\n" + prompt
    return prompt


def _format_reflection_block(reflection: Dict[str, Any]) -> str:
    if not reflection:
        return ""
    lines: List[str] = []
    for key in ("title", "problem_statement"):
        value = str(reflection.get(key, "") or "").strip()
        if value:
            label = "Title" if key == "title" else "Problem"
            lines.append(f"{label}: {value}")
    selected_symbol = reflection.get("selected_symbol", {})
    if isinstance(selected_symbol, dict):
        qualname = str(selected_symbol.get("qualname", "") or "").strip()
        if qualname:
            lines.append(f"Selected symbol: {qualname}")
    affected_modules = reflection.get("affected_modules", [])
    if isinstance(affected_modules, list) and affected_modules:
        lines.append("Affected modules: " + ", ".join(str(item) for item in affected_modules[:6]))
    for label, key, limit in (
        ("Suggested action", "suggested_actions", 3),
        ("Verification", "verification_targets", 3),
        ("Rationale", "rationale", 3),
    ):
        values = reflection.get(key, [])
        if not isinstance(values, list):
            continue
        for item in values[:limit]:
            text = str(item).strip()
            if text:
                lines.append(f"{label}: {text}")
    evidence = reflection.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence[:4]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "") or "").strip()
            event_type = str(item.get("event_type", "") or "").strip()
            summary = str(item.get("summary", "") or "").strip()
            line = int(item.get("line", 0) or 0)
            parts = [part for part in (source, f"line={line}" if line > 0 else "", event_type, summary) if part]
            if parts:
                lines.append("Evidence: " + " | ".join(parts))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM candidate generator for evolve_improve.")
    parser.add_argument("--model", default="", help="Override model id for generation.")
    parser.add_argument("--provider", default="", choices=["", "ollama", "hf", "llamacpp"], help="Override provider.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=1200, help="Max new tokens.")
    parser.add_argument("--enable-thinking", action="store_true", help="Prepend /think to the prompt.")
    args = parser.parse_args()

    payload_text = sys.stdin.read()
    if not payload_text.strip():
        raise SystemExit("[error] expected JSON payload on stdin")
    payload = json.loads(payload_text)
    max_candidates = int(payload.get("max_candidates") or 3)
    runs = int(payload.get("runs") or 1)
    candidates_per_run = int(payload.get("candidates_per_run") or 1)
    fallback_cfg = payload.get("generator_fallback", {}) if isinstance(payload.get("generator_fallback"), dict) else {}
    allow_local_fallback = bool(fallback_cfg.get("local_heuristic", True))
    if candidates_per_run <= 0:
        candidates_per_run = 1

    settings = load_settings()
    provider = args.provider or str(settings.models.get("provider", "ollama")).lower()
    model_id = args.model
    if not model_id:
        model_id = settings.evolve.get("evolution_model") or ""
    if not model_id:
        model_id = settings.models.get("text_model", "")
    if not model_id:
        if not allow_local_fallback:
            raise SystemExit("[error] no model id configured (settings.models.text_model)")
        result = _build_local_fallback_result(
            payload,
            max_candidates=max_candidates,
            provider=provider,
            model_id="",
            reason="no model id configured",
        )
        sys.stdout.write(json.dumps(result, ensure_ascii=True))
        return 0

    all_candidates: List[Dict[str, Any]] = []
    avoid_summaries: List[str] = []
    runs = max(1, runs)
    generator_meta: Dict[str, Any] = {
        "mode": "llm",
        "provider": provider,
        "model": model_id,
        "candidate_count": 0,
    }
    for run_index in range(1, runs + 1):
        prompt = _build_prompt(
            payload,
            max_candidates=candidates_per_run if runs > 1 else max_candidates,
            enable_thinking=args.enable_thinking,
            run_index=run_index,
            run_total=runs,
            avoid_summaries=avoid_summaries,
        )
        try:
            output = generate_local_text(
                prompt,
                model_id,
                provider=provider,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                system="Return only valid JSON as instructed.",
                response_format="json",
            )
            data = _extract_json(output)
        except (OllamaError, HfError, LlamacppError, ValueError) as e:
            if not allow_local_fallback:
                raise
            fallback_result = _build_local_fallback_result(
                payload,
                max_candidates=max_candidates - len(all_candidates),
                provider=provider,
                model_id=model_id,
                reason=str(e),
            )
            fallback_candidates = fallback_result.get("candidates", [])
            if isinstance(fallback_candidates, list):
                for cand in fallback_candidates:
                    if isinstance(cand, dict):
                        all_candidates.append(cand)
            generator_meta = dict(fallback_result.get("generator", {}) or {})
            if all_candidates:
                generator_meta["mode"] = "llm_with_local_fallback" if run_index > 1 else generator_meta.get("mode", "local_fallback")
                generator_meta["candidate_count"] = len(all_candidates)
                break
            sys.stdout.write(json.dumps(fallback_result, ensure_ascii=True))
            return 0
        candidates = data.get("candidates", [])
        if not isinstance(candidates, list):
            raise SystemExit("[error] generator output missing candidates list")
        if runs > 1 and candidates_per_run > 0:
            candidates = candidates[:candidates_per_run]
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            if "summary" in cand and isinstance(cand["summary"], str):
                avoid_summaries.append(cand["summary"])
            all_candidates.append(cand)
        if max_candidates > 0 and len(all_candidates) >= max_candidates:
            all_candidates = all_candidates[:max_candidates]
            break

    seen_ids = set()
    for idx, cand in enumerate(all_candidates, start=1):
        base_id = cand.get("id") or f"cand_{idx}"
        new_id = base_id
        if new_id in seen_ids:
            suffix = 1
            while f"{base_id}_{suffix}" in seen_ids:
                suffix += 1
            new_id = f"{base_id}_{suffix}"
        cand["id"] = new_id
        seen_ids.add(new_id)

    generator_meta["candidate_count"] = len(all_candidates)
    sys.stdout.write(json.dumps({"candidates": all_candidates, "generator": generator_meta}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
