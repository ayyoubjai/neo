import ast
import asyncio
import json
import os
import random
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from common.config import load_settings
from common.ids import new_id
from common.jsonlog import append_jsonl
from common.llm_json_log import record_llm_json
from common.jsonl import read_jsonl, write_jsonl
from common.jsonl_rpc import RpcError, send_request
from common.record_log import record_event
from common.tool_metadata import redact_tool_payload
from common.system_entity import load_system_entity
from common.time_utils import utc_now_iso
from common.types import (
    EVENT_ASSISTANT_FINAL,
    EVENT_ASSISTANT_DRAFT,
    EVENT_PERMISSION_REQUEST,
    EVENT_PERMISSION_RESPONSE,
    EVENT_STT_FINAL,
    EVENT_TURN_PATCH,
    make_event,
)
from orchestrator.context_builder import ContextBuilder
from orchestrator.memory_manager import MemoryManager
from orchestrator.policy import PolicyEngine, TIER_MAP
from orchestrator.tool_selector import ToolRegistry


_CURRENT_TURN_ID: ContextVar[Optional[str]] = ContextVar("orchestrator_current_turn_id", default=None)
_CURRENT_TRACE_ID: ContextVar[Optional[str]] = ContextVar("orchestrator_current_trace_id", default=None)
PRIMARY_REASONING_MODE = "COGNITION"


class _AsiSourceReturn(Exception):
    def __init__(self, value: Any):
        super().__init__("ASI source return")
        self.value = value


class _AsiSourceSignal(Exception):
    def __init__(self, payload: Dict[str, Any]):
        super().__init__("ASI source execution signal")
        self.payload = payload


class Orchestrator:
    def __init__(self):
        settings = load_settings()
        self._settings = settings
        self._policy = PolicyEngine()
        self._tools = ToolRegistry()
        self._memory = MemoryManager()
        self._context_builder = ContextBuilder()
        self._system_entity = load_system_entity()
        self._model_host = settings.rpc["orch_host"]
        self._model_port = settings.rpc["model_port"]
        self._tool_host = settings.rpc["orch_host"]
        self._tool_port = settings.rpc["tool_port"]
        self._model_rpc_timeout_s = int(settings.orchestrator.get("model_rpc_timeout_s", 0))
        if self._model_rpc_timeout_s < 0:
            self._model_rpc_timeout_s = 0
        soft_json_setting = settings.orchestrator.get("soft_json_parsing_enabled", True)
        if isinstance(soft_json_setting, str):
            self._soft_json_parsing_enabled = soft_json_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._soft_json_parsing_enabled = bool(soft_json_setting)
        self._auto_approve_all = bool(settings.orchestrator.get("auto_approve_all", False))
        retrieve_tools_setting = settings.orchestrator.get("retrieve_relevant_tools", True)
        if isinstance(retrieve_tools_setting, str):
            self._retrieve_relevant_tools_enabled = retrieve_tools_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._retrieve_relevant_tools_enabled = bool(retrieve_tools_setting)
        select_tools_setting = settings.orchestrator.get("select_relevant_tools", False)
        if isinstance(select_tools_setting, str):
            self._select_relevant_tools_enabled = select_tools_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._select_relevant_tools_enabled = bool(select_tools_setting)
        category_tool_selection_setting = settings.orchestrator.get(
            "tool_selection_category_mode_enabled",
            False,
        )
        if isinstance(category_tool_selection_setting, str):
            self._tool_selection_category_mode_enabled = (
                category_tool_selection_setting.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            )
        else:
            self._tool_selection_category_mode_enabled = bool(category_tool_selection_setting)
        independent_tool_selection_setting = settings.orchestrator.get(
            "tool_selection_independent_from_retrieval",
            False,
        )
        if isinstance(independent_tool_selection_setting, str):
            self._tool_selection_independent_from_retrieval = (
                independent_tool_selection_setting.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            )
        else:
            self._tool_selection_independent_from_retrieval = bool(independent_tool_selection_setting)
        self._tool_retrieval_k = int(settings.orchestrator.get("tool_retrieval_k", 8))
        if self._tool_retrieval_k < 1:
            self._tool_retrieval_k = 1
        self._cognition_system1_max_steps = int(
            settings.orchestrator.get("cognition_system1_max_steps", 3)
        )
        if self._cognition_system1_max_steps < 1:
            self._cognition_system1_max_steps = 1
        self._cognition_system0_max_steps = int(
            settings.orchestrator.get("cognition_system0_max_steps", 1)
        )
        if self._cognition_system0_max_steps < 1:
            self._cognition_system0_max_steps = 1
        self._cognition_system2_max_steps = int(
            settings.orchestrator.get("cognition_system2_max_steps", 4)
        )
        if self._cognition_system2_max_steps < 1:
            self._cognition_system2_max_steps = 1
        self._cognition_action_limit = int(settings.orchestrator.get("cognition_action_limit", 6))
        if self._cognition_action_limit < 1:
            self._cognition_action_limit = 1
        self._cognition_mode = str(settings.orchestrator.get("cognition_mode", "auto")).strip().lower()
        system1_clarify_setting = self._coerce_optional_bool(
            settings.orchestrator.get("cognition_system1_clarify_enabled", True)
        )
        self._cognition_system1_clarify_enabled = (
            True if system1_clarify_setting is None else system1_clarify_setting
        )
        system2_clarify_setting = self._coerce_optional_bool(
            settings.orchestrator.get("cognition_system2_clarify_enabled", True)
        )
        self._cognition_system2_clarify_enabled = (
            True if system2_clarify_setting is None else system2_clarify_setting
        )
        init_controls_tools_setting = self._coerce_optional_bool(
            settings.orchestrator.get("cognition_init_controls_tools_needed", False)
        )
        self._cognition_init_controls_tools_needed = (
            False if init_controls_tools_setting is None else init_controls_tools_setting
        )
        cognition_query_state_enabled_setting = settings.orchestrator.get(
            "cognition_query_state_enabled",
            True,
        )
        if isinstance(cognition_query_state_enabled_setting, str):
            self._cognition_query_state_enabled = cognition_query_state_enabled_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._cognition_query_state_enabled = bool(cognition_query_state_enabled_setting)
        cognition_state_of_mind_enabled_setting = settings.orchestrator.get(
            "cognition_state_of_mind_enabled",
            True,
        )
        if isinstance(cognition_state_of_mind_enabled_setting, str):
            self._cognition_state_of_mind_enabled = cognition_state_of_mind_enabled_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._cognition_state_of_mind_enabled = bool(cognition_state_of_mind_enabled_setting)
        cognition_system3_enabled_setting = settings.orchestrator.get("cognition_system3_enabled", False)
        if isinstance(cognition_system3_enabled_setting, str):
            self._cognition_system3_enabled = cognition_system3_enabled_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._cognition_system3_enabled = bool(cognition_system3_enabled_setting)
        try:
            self._cognition_system3_rounds = int(settings.orchestrator.get("cognition_system3_rounds", 1))
        except (TypeError, ValueError):
            self._cognition_system3_rounds = 1
        if self._cognition_system3_rounds < 1:
            self._cognition_system3_rounds = 1
        cognition_system3_parallel_setting = settings.orchestrator.get("cognition_system3_parallel", False)
        if isinstance(cognition_system3_parallel_setting, str):
            self._cognition_system3_parallel = cognition_system3_parallel_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._cognition_system3_parallel = bool(cognition_system3_parallel_setting)
        trivial_target_raw = str(
            settings.orchestrator.get("cognition_trivial_query_target", "system1") or ""
        ).strip().lower()
        self._cognition_trivial_query_target = (
            trivial_target_raw if trivial_target_raw in {"system0", "system1"} else "system1"
        )
        force_system_raw = str(settings.orchestrator.get("cognition_force_system", "") or "").strip().lower()
        self._cognition_force_system = (
            force_system_raw if force_system_raw in {"system0", "system1", "system2", "system3"} else ""
        )
        self._cognition_peer_pools = self._normalize_cognition_peer_pool_setting(
            settings.orchestrator.get("cognition_peer_pools", {})
        )
        cognition_distill_enabled_setting = settings.orchestrator.get("cognition_distill_enabled", True)
        if isinstance(cognition_distill_enabled_setting, str):
            self._cognition_distill_enabled = cognition_distill_enabled_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._cognition_distill_enabled = bool(cognition_distill_enabled_setting)
        # --- Distillation quality controls ---
        try:
            self._cognition_distill_min_confidence = float(
                settings.orchestrator.get("cognition_distill_min_confidence", 0.55)
            )
        except (TypeError, ValueError):
            self._cognition_distill_min_confidence = 0.55
        self._cognition_distill_min_confidence = max(0.0, min(1.0, self._cognition_distill_min_confidence))
        cognition_distill_skip_trivial_setting = settings.orchestrator.get(
            "cognition_distill_skip_trivial", True
        )
        if isinstance(cognition_distill_skip_trivial_setting, str):
            self._cognition_distill_skip_trivial = (
                cognition_distill_skip_trivial_setting.strip().lower() in {"1", "true", "yes", "on"}
            )
        else:
            self._cognition_distill_skip_trivial = bool(cognition_distill_skip_trivial_setting)
        cognition_routing_skip_trivial_setting = settings.orchestrator.get(
            "cognition_routing_skip_trivial", False
        )
        if isinstance(cognition_routing_skip_trivial_setting, str):
            self._cognition_routing_skip_trivial = (
                cognition_routing_skip_trivial_setting.strip().lower() in {"1", "true", "yes", "on"}
            )
        else:
            self._cognition_routing_skip_trivial = bool(cognition_routing_skip_trivial_setting)
        try:
            self._cognition_distill_trivial_max_chars = int(
                settings.orchestrator.get("cognition_distill_trivial_max_chars", 60)
            )
        except (TypeError, ValueError):
            self._cognition_distill_trivial_max_chars = 60
        if self._cognition_distill_trivial_max_chars < 1:
            self._cognition_distill_trivial_max_chars = 60
        try:
            self._cognition_reflections_limit = int(
                settings.orchestrator.get("cognition_reflections_limit", 6)
            )
        except (TypeError, ValueError):
            self._cognition_reflections_limit = 6
        if self._cognition_reflections_limit < 1:
            self._cognition_reflections_limit = 1
        cognition_budget_exhausted_signal_setting = settings.orchestrator.get(
            "cognition_budget_exhausted_signal", True
        )
        if isinstance(cognition_budget_exhausted_signal_setting, str):
            self._cognition_budget_exhausted_signal = (
                cognition_budget_exhausted_signal_setting.strip().lower() in {"1", "true", "yes", "on"}
            )
        else:
            self._cognition_budget_exhausted_signal = bool(cognition_budget_exhausted_signal_setting)
        self._asi_recursion_limit = int(
            settings.orchestrator.get("asi_recursion_limit", 2)
        )
        if self._asi_recursion_limit < 0:
            self._asi_recursion_limit = 0
        asi_seed_enabled_setting = settings.orchestrator.get("asi_seed_enabled", True)
        if isinstance(asi_seed_enabled_setting, str):
            self._asi_seed_enabled = asi_seed_enabled_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._asi_seed_enabled = bool(asi_seed_enabled_setting)
        self._memory_retrieval_k = int(settings.orchestrator.get("memory_retrieval_k", 3))
        if self._memory_retrieval_k < 1:
            self._memory_retrieval_k = 1
        self._memory_distill_turns = int(settings.orchestrator.get("memory_distill_turns", 12))
        if self._memory_distill_turns < 0:
            self._memory_distill_turns = 0
        critic_enabled_setting = settings.orchestrator.get("final_response_critic_enabled", False)
        if isinstance(critic_enabled_setting, str):
            self._final_response_critic_enabled = critic_enabled_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._final_response_critic_enabled = bool(critic_enabled_setting)
        try:
            self._final_response_critic_max_retries = int(
                settings.orchestrator.get("final_response_critic_max_retries", 1)
            )
        except (TypeError, ValueError):
            self._final_response_critic_max_retries = 1
        if self._final_response_critic_max_retries < 0:
            self._final_response_critic_max_retries = 0
        try:
            self._final_response_critic_min_confidence = float(
                settings.orchestrator.get("final_response_critic_min_confidence", 0.55)
            )
        except (TypeError, ValueError):
            self._final_response_critic_min_confidence = 0.55
        if self._final_response_critic_min_confidence < 0.0:
            self._final_response_critic_min_confidence = 0.0
        if self._final_response_critic_min_confidence > 1.0:
            self._final_response_critic_min_confidence = 1.0
        tool_codegen_critic_enabled_setting = settings.orchestrator.get("tool_codegen_critic_enabled", True)
        if isinstance(tool_codegen_critic_enabled_setting, str):
            self._tool_codegen_critic_enabled = tool_codegen_critic_enabled_setting.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._tool_codegen_critic_enabled = bool(tool_codegen_critic_enabled_setting)
        try:
            self._tool_codegen_critic_min_confidence = float(
                settings.orchestrator.get("tool_codegen_critic_min_confidence", 0.65)
            )
        except (TypeError, ValueError):
            self._tool_codegen_critic_min_confidence = 0.65
        if self._tool_codegen_critic_min_confidence < 0.0:
            self._tool_codegen_critic_min_confidence = 0.0
        if self._tool_codegen_critic_min_confidence > 1.0:
            self._tool_codegen_critic_min_confidence = 1.0
        critic_skip_modes_setting = settings.orchestrator.get("final_response_critic_skip_modes", [])
        skip_modes_values: List[str] = []
        if isinstance(critic_skip_modes_setting, str):
            skip_modes_values = [item.strip() for item in critic_skip_modes_setting.split(",")]
        elif isinstance(critic_skip_modes_setting, (list, tuple, set)):
            skip_modes_values = [str(item).strip() for item in critic_skip_modes_setting]
        self._final_response_critic_skip_modes: Set[str] = {
            item.upper()
            for item in skip_modes_values
            if isinstance(item, str) and item.strip()
        }
        
        thought_feedback_enabled_setting = settings.orchestrator.get("cognition_thought_feedback_enabled", False)
        if isinstance(thought_feedback_enabled_setting, str):
            self._cognition_thought_feedback_enabled = thought_feedback_enabled_setting.strip().lower() in {
                "1", "true", "yes", "on"
            }
        else:
            self._cognition_thought_feedback_enabled = bool(thought_feedback_enabled_setting)

        thought_feedback_steps_setting = settings.orchestrator.get("cognition_thought_feedback_steps", [])
        if isinstance(thought_feedback_steps_setting, str):
            self._cognition_thought_feedback_steps = [s.strip() for s in thought_feedback_steps_setting.split(",") if s.strip()]
        elif isinstance(thought_feedback_steps_setting, (list, tuple, set)):
            self._cognition_thought_feedback_steps = [str(s).strip() for s in thought_feedback_steps_setting if str(s).strip()]
        else:
            self._cognition_thought_feedback_steps = []
        self._generated_tools_path = os.path.join(settings.data_dir, "generated_tools.json")
        self._tool_generation_runs_dir = os.path.join(settings.data_dir, "tool_generation_runs")
        self._cognition_skills_path = os.path.join(settings.data_dir, "cognition_skills.json")
        self._cognition_patterns_path = os.path.join(settings.data_dir, "cognition_patterns.json")
        self._cognition_lessons_path = os.path.join(settings.data_dir, "cognition_lessons.json")
        self._cognition_safety_rules_path = os.path.join(settings.data_dir, "cognition_safety_rules.json")
        self._cognition_episodes_path = os.path.join(settings.data_dir, "cognition_episodes.jsonl")
        self._asi_patterns_path = os.path.join(settings.data_dir, "asi_patterns.json")
        self._generated_tools_module_path = os.path.join(
            settings.workspace_root,
            "src",
            "tool_runtime",
            "generated_tools.py",
        )
        os.makedirs(settings.data_dir, exist_ok=True)
        self._final_response_critic_model = str(settings.orchestrator.get("final_response_critic_model", "")).strip()
        self._tool_codegen_critic_model = str(settings.orchestrator.get("tool_codegen_critic_model", "")).strip()
        self._cognition_init_model = str(settings.orchestrator.get("cognition_init_model", "")).strip()
        self._cognition_route_model = str(settings.orchestrator.get("cognition_route_model", "")).strip()
        self._cognition_system0_model = str(settings.orchestrator.get("cognition_system0_model", "")).strip()
        self._cognition_tool_selection_model = str(settings.orchestrator.get("cognition_tool_selection_model", "")).strip()
        self._cognition_system1_model = str(settings.orchestrator.get("cognition_system1_model", "")).strip()
        self._cognition_system2_model = str(settings.orchestrator.get("cognition_system2_model", "")).strip()
        self._cognition_repair_model = str(settings.orchestrator.get("cognition_repair_model", "")).strip()
        self._cognition_judge_model = str(settings.orchestrator.get("cognition_judge_model", "")).strip()
        self._cognition_distill_model = str(settings.orchestrator.get("cognition_distill_model", "")).strip()
        try:
            self._cognition_repair_timeout_s = int(settings.orchestrator.get("cognition_repair_timeout_s", 60))
        except (TypeError, ValueError):
            self._cognition_repair_timeout_s = 60
        if self._cognition_repair_timeout_s < 1:
            self._cognition_repair_timeout_s = 60
        try:
            self._cognition_distill_timeout_s = int(settings.orchestrator.get("cognition_distill_timeout_s", 90))
        except (TypeError, ValueError):
            self._cognition_distill_timeout_s = 90
        if self._cognition_distill_timeout_s < 1:
            self._cognition_distill_timeout_s = 90
        self._final_response_critic_options = self._normalize_model_call_options(
            settings.orchestrator.get("final_response_critic_options", {})
        )
        self._tool_codegen_critic_options = self._normalize_model_call_options(
            settings.orchestrator.get("tool_codegen_critic_options", {})
        )
        self._cognition_init_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_init_options", {})
        )
        self._cognition_route_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_route_options", {})
        )
        self._cognition_system0_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_system0_options", {})
        )
        self._cognition_tool_selection_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_tool_selection_options", {})
        )
        self._cognition_system1_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_system1_options", {})
        )
        self._cognition_system2_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_system2_options", {})
        )
        self._cognition_system3_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_system3_options", {})
        )
        self._cognition_repair_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_repair_options", {})
        )
        self._cognition_judge_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_judge_options", {})
        )
        self._cognition_distill_options = self._normalize_model_call_options(
            settings.orchestrator.get("cognition_distill_options", {})
        )

        self._mode_logs = {
            PRIMARY_REASONING_MODE: os.path.join(settings.data_dir, "cognition_events.log"),
        }
        self._tool_index: Dict[str, Dict[str, Any]] = {}
        self._embed_cache: Dict[str, List[float]] = {}
        self._embed_cache_limit = int(settings.orchestrator.get("embed_cache_limit", 256))
        if self._embed_cache_limit < 0:
            self._embed_cache_limit = 0

        self._history: List[Dict[str, Any]] = []
        self._summary = ""
        self._pending_turns: Dict[str, str] = {}
        self._pending_turn_modes: Dict[str, str] = {}
        self._pending_turn_skip_distillation: Dict[str, bool] = {}
        self._turn_context_packet: Dict[str, Dict[str, Any]] = {}
        self._inflight_task: Optional[asyncio.Task] = None
        self._inflight_turn_id: Optional[str] = None
        self._out_queue: Optional[asyncio.Queue] = None
        self._permission_waiters: Dict[str, asyncio.Future] = {}
        self._safe_mode = False
        self._tool_failures = 0
        self._asi_patterns_cache: Optional[List[Dict[str, Any]]] = None
        self._asi_patterns_mtime: Optional[float] = None
        self._turn_execution_evidence: Dict[str, Dict[str, Any]] = {}
        self._pending_turn_traces: Dict[str, List[Dict[str, Any]]] = {}

    def _normalize_cognition_peer_profile(
        self,
        value: Any,
        *,
        pool_name: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        peer_id = str(value.get("peer_id") or value.get("id") or "").strip().lower()
        if not peer_id:
            peer_id = f"{pool_name}_{index}"
        model = str(value.get("model") or "").strip()
        if not model:
            return None
        role = str(value.get("role") or "").strip()
        options = value.get("options", {})
        return {
            "peer_id": peer_id,
            "model": model,
            "role": role,
            "options": self._normalize_model_call_options(options),
        }

    def _normalize_cognition_peer_pool_setting(self, value: Any) -> Dict[str, List[Dict[str, Any]]]:
        normalized: Dict[str, List[Dict[str, Any]]] = {
            "proposers": [],
            "critics": [],
        }
        if not isinstance(value, dict):
            return normalized
        for pool_name in ("proposers", "critics"):
            pool_values = value.get(pool_name, [])
            if not isinstance(pool_values, list):
                continue
            peers: List[Dict[str, Any]] = []
            for index, item in enumerate(pool_values, start=1):
                peer = self._normalize_cognition_peer_profile(item, pool_name=pool_name[:-1], index=index)
                if peer is None:
                    continue
                peers.append(peer)
            normalized[pool_name] = peers
        if not normalized["critics"] and normalized["proposers"]:
            normalized["critics"] = [dict(item) for item in normalized["proposers"]]
        return normalized

    def _cognition_system3_available(self) -> bool:
        return bool(
            self._cognition_system3_enabled
            and self._cognition_peer_pools.get("proposers")
            and self._cognition_peer_pools.get("critics")
        )

    def _normalize_model_call_options(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            name = key.strip()
            if not name:
                continue
            if name == "reasoning_effort":
                text = str(item).strip()
                if text:
                    normalized[name] = text
                continue
            if name in {"thinking", "enable_thinking", "think"}:
                parsed = self._coerce_optional_bool(item)
                if parsed is not None:
                    normalized["thinking"] = parsed
                continue
            normalized[name] = item
        return normalized

    def _coerce_optional_bool(self, value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return None

    def _merge_model_call_options(
        self,
        base_options: Optional[Dict[str, Any]],
        override_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        if isinstance(base_options, dict):
            merged.update(self._normalize_model_call_options(base_options))
        if isinstance(override_options, dict):
            merged.update(self._normalize_model_call_options(override_options))
        return merged or None

    def _resolve_cognition_model_override(
        self,
        system_name: str,
        explicit_model: str = "",
    ) -> str:
        explicit_value = str(explicit_model or "").strip()
        if explicit_value:
            return explicit_value
        if system_name == "system0":
            return self._cognition_system0_model
        if system_name == "tool_selection":
            return self._cognition_tool_selection_model
        if system_name == "system1":
            return self._cognition_system1_model
        if system_name == "system2":
            return self._cognition_system2_model
        return ""

    def bind_out_queue(self, queue: asyncio.Queue) -> None:
        self._out_queue = queue

    async def _send_event(self, event: Dict[str, Any]) -> None:
        if self._out_queue is not None:
            await self._out_queue.put(event)

    async def _maybe_emit_thought_feedback(
        self,
        parsed_json: Dict[str, Any],
        step_name: str,
        trace_id: str,
        turn_id: Optional[str]
    ) -> None:
        if not self._cognition_thought_feedback_enabled:
            return
        if self._cognition_thought_feedback_steps and step_name not in self._cognition_thought_feedback_steps:
            return
        thought = parsed_json.get("thought", "")
        if isinstance(thought, str) and thought.strip():
            await self._send_event(
                make_event(
                    EVENT_ASSISTANT_DRAFT,
                    {"text": f"[{step_name} thought]: {thought.strip()}", "trace_id": trace_id, "turn_id": turn_id or ""}
                )
            )

    async def handle_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if event_type == EVENT_STT_FINAL:
            turn_id = payload.get("turn_id", new_id())
            self._pending_turns[turn_id] = payload.get("text", "")
            requested_mode = self._normalize_requested_mode(payload.get("requested_mode"))
            self._pending_turn_modes[turn_id] = requested_mode or PRIMARY_REASONING_MODE
            self._pending_turn_skip_distillation[turn_id] = bool(payload.get("skip_distillation"))
            await self._start_or_restart(turn_id)
        elif event_type == EVENT_TURN_PATCH:
            turn_id = payload.get("turn_id")
            if turn_id:
                existing = self._pending_turns.get(turn_id, "")
                appended = payload.get("appended_text", "")
                self._pending_turns[turn_id] = (existing + " " + appended).strip()
                await self._start_or_restart(turn_id)
        elif event_type == EVENT_PERMISSION_RESPONSE:
            request_id = payload.get("request_id")
            waiter = self._permission_waiters.pop(request_id, None)
            if waiter and not waiter.done():
                waiter.set_result(payload)

    async def _start_or_restart(self, turn_id: str) -> None:
        if self._inflight_task and self._inflight_turn_id == turn_id:
            self._inflight_task.cancel()
        self._inflight_turn_id = turn_id
        self._inflight_task = asyncio.create_task(self._process_turn(turn_id))

    async def _run_mode_once(
        self,
        mode: str,
        text: str,
        context_packet: Dict[str, Any],
        trace_id: str,
        turn_id: str,
        retrieval_query: Optional[str] = None,
        skip_distillation: bool = False,
    ) -> str:
        normalized_mode = self._normalize_requested_mode(mode)
        forced_execution_mode = (
            normalized_mode.lower()
            if normalized_mode in {"SYSTEM0", "SYSTEM1", "SYSTEM2", "SYSTEM3"}
            else None
        )
        kwargs: Dict[str, Any] = {
            "retrieval_query": retrieval_query,
            "forced_execution_mode": forced_execution_mode,
        }
        if skip_distillation:
            kwargs["skip_distillation"] = True
        return await self._run_cognition_loop(
            text,
            context_packet,
            trace_id,
            turn_id,
            **kwargs,
        )

    def _build_critic_retry_text(self, user_text: str, previous_response: str, critique: Dict[str, Any]) -> str:
        issues_raw = critique.get("issues", [])
        fixes_raw = critique.get("fix_instructions", [])
        issues: List[str] = []
        fixes: List[str] = []
        if isinstance(issues_raw, list):
            for item in issues_raw:
                if isinstance(item, str):
                    value = item.strip()
                    if value and value not in issues:
                        issues.append(value)
                if len(issues) >= 6:
                    break
        if isinstance(fixes_raw, list):
            for item in fixes_raw:
                if isinstance(item, str):
                    value = item.strip()
                    if value and value not in fixes:
                        fixes.append(value)
                if len(fixes) >= 6:
                    break
        issues_block = "\n".join(f"- {item}" for item in issues) if issues else "- Not specified."
        fixes_block = "\n".join(f"- {item}" for item in fixes) if fixes else "- Fully satisfy the user request."
        return (
            f"{user_text}\n\n"
            "Self-critique on previous answer:\n"
            f"Previous answer: {self._truncate(previous_response, 1200)}\n"
            "Issues to fix:\n"
            f"{issues_block}\n"
            "Fix instructions:\n"
            f"{fixes_block}\n\n"
            "Produce a corrected final answer that fully satisfies the original user request."
        )

    def _set_turn_execution_evidence(
        self,
        turn_id: Optional[str],
        evidence: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(turn_id, str) or not turn_id:
            return
        if not isinstance(evidence, dict) or not evidence:
            self._turn_execution_evidence.pop(turn_id, None)
            return
        self._turn_execution_evidence[turn_id] = evidence

    def _get_turn_execution_evidence(self, turn_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not isinstance(turn_id, str) or not turn_id:
            return None
        evidence = self._turn_execution_evidence.get(turn_id)
        if not isinstance(evidence, dict):
            return None
        return evidence

    def _clear_turn_execution_evidence(self, turn_id: Optional[str]) -> None:
        if not isinstance(turn_id, str) or not turn_id:
            return
        self._turn_execution_evidence.pop(turn_id, None)

    def _summarize_tool_results_from_observations(
        self,
        observations: List[Dict[str, Any]],
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        if not isinstance(observations, list):
            return []
        summarized: List[Dict[str, Any]] = []
        max_items = max(1, limit)
        for item in observations:
            if not isinstance(item, dict):
                continue
            action = item.get("action")
            observation = item.get("observation")
            if not isinstance(action, dict) or not isinstance(observation, dict):
                continue
            tool_id_raw = action.get("tool_id", "")
            tool_id = tool_id_raw.strip() if isinstance(tool_id_raw, str) else ""
            if not tool_id:
                continue
            status_raw = observation.get("status", "")
            status = status_raw.strip() if isinstance(status_raw, str) else ""
            args_preview = self._truncate(self._safe_json(action.get("args", {})), 220)
            result_preview = ""
            if "result" in observation:
                result_preview = self._truncate(self._safe_json(observation.get("result")), 420)
            error_preview = ""
            error_value = observation.get("error")
            if isinstance(error_value, str) and error_value.strip():
                error_preview = self._truncate(error_value.strip(), 220)
            summarized.append(
                {
                    "tool_id": tool_id,
                    "status": status or "unknown",
                    "args_preview": args_preview,
                    "result_preview": result_preview,
                    "error": error_preview,
                }
            )
            if len(summarized) >= max_items:
                break
        return summarized

    def _summarize_tool_results_from_trace(
        self,
        trace_steps: Any,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        if not isinstance(trace_steps, list):
            return []
        summarized: List[Dict[str, Any]] = []
        max_items = max(1, limit)
        for item in trace_steps:
            if not isinstance(item, dict):
                continue
            tool_id_raw = item.get("tool_id", "")
            if not isinstance(tool_id_raw, str) or not tool_id_raw.strip():
                continue
            status_raw = item.get("status", "")
            status = status_raw.strip() if isinstance(status_raw, str) and status_raw.strip() else "unknown"
            detail_raw = item.get("detail", "")
            detail = detail_raw.strip() if isinstance(detail_raw, str) else ""
            summarized.append(
                {
                    "tool_id": tool_id_raw.strip(),
                    "status": status,
                    "detail": self._truncate(detail, 260) if detail else "",
                }
            )
            if len(summarized) >= max_items:
                break
        return summarized

    def _build_execution_evidence_payload(
        self,
        mode: str,
        source: str,
        entity_kind: str,
        entity: Dict[str, Any],
        result: Dict[str, Any],
        bindings: Optional[Dict[str, Any]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": mode,
            "source": source,
            "entity_kind": entity_kind,
            "entity": entity,
        }
        if isinstance(bindings, dict) and bindings:
            payload["bindings"] = bindings
        if isinstance(tool_results, list) and tool_results:
            payload["tool_results"] = tool_results[:12]
        if isinstance(result, dict):
            result_type = result.get("type", "")
            result_summary: Dict[str, Any] = {"type": result_type}
            if result_type == "final":
                result_summary["text_preview"] = self._truncate(str(result.get("text", "")), 800)
            elif result_type == "clarify":
                result_summary["question_preview"] = self._truncate(str(result.get("question", "")), 800)
            else:
                result_summary["text_preview"] = self._truncate(str(result.get("text", "")), 800)
            payload["result"] = result_summary
            trace_steps = result.get("trace")
            if isinstance(trace_steps, list):
                payload["step_trace"] = trace_steps
        return payload

    def _format_step_trace_lines(
        self,
        trace: List[Any],
        depth: int,
        remaining: List[int],
    ) -> List[str]:
        lines: List[str] = []
        indent = "  " * depth
        for item in trace:
            if remaining[0] <= 0:
                lines.append(f"{indent}- ... truncated ...")
                break
            if not isinstance(item, dict):
                continue
            remaining[0] -= 1
            step_index = item.get("index", "?")
            op_name = item.get("op", "unknown")
            status = item.get("status", "unknown")
            line = f"{indent}- step {step_index}: op={op_name}, status={status}"
            detail = item.get("detail")
            if isinstance(detail, str) and detail.strip():
                line += f", detail={self._truncate(detail.strip(), 160)}"
            missing = item.get("missing")
            if isinstance(missing, list):
                missing_values = [str(entry).strip() for entry in missing if str(entry).strip()]
                if missing_values:
                    line += f", missing={self._truncate(', '.join(missing_values), 160)}"
            lines.append(line)
            sub_trace = item.get("sub_trace")
            if isinstance(sub_trace, list) and sub_trace and depth < 3:
                lines.extend(self._format_step_trace_lines(sub_trace, depth + 1, remaining))
        return lines

    def _format_execution_evidence_for_critic(self, evidence: Optional[Dict[str, Any]]) -> str:
        if not isinstance(evidence, dict) or not evidence:
            return "None"
        mode = str(evidence.get("mode", "")).strip() or "unknown"
        source = str(evidence.get("source", "")).strip() or "unknown"
        entity_kind = str(evidence.get("entity_kind", "")).strip() or "workflow"
        lines: List[str] = [
            f"Mode: {mode}",
            f"Source: {source}",
        ]
        entity = evidence.get("entity")
        if isinstance(entity, (dict, list)):
            lines.append(
                f"Executed {entity_kind} JSON: {self._truncate(self._safe_json(entity), 2800)}"
            )
        elif entity is not None:
            lines.append(f"Executed {entity_kind}: {self._truncate(str(entity), 400)}")
        bindings = evidence.get("bindings")
        if isinstance(bindings, dict) and bindings:
            lines.append(f"Resolved bindings: {self._truncate(self._safe_json(bindings), 1200)}")
        result = evidence.get("result")
        if isinstance(result, dict):
            result_type = str(result.get("type", "")).strip() or "unknown"
            preview = ""
            if isinstance(result.get("text_preview"), str):
                preview = result.get("text_preview", "")
            elif isinstance(result.get("question_preview"), str):
                preview = result.get("question_preview", "")
            if preview:
                lines.append(f"Execution result: type={result_type}, preview={self._truncate(preview, 500)}")
            else:
                lines.append(f"Execution result: type={result_type}")
        tool_results = evidence.get("tool_results")
        if isinstance(tool_results, list) and tool_results:
            lines.append("Tool execution results:")
            for item in tool_results[:12]:
                if not isinstance(item, dict):
                    continue
                lines.append(f"  - {self._truncate(self._safe_json(item), 900)}")
        step_trace = evidence.get("step_trace")
        if isinstance(step_trace, list) and step_trace:
            lines.append("Step execution trace:")
            trace_lines = self._format_step_trace_lines(step_trace, depth=1, remaining=[60])
            if trace_lines:
                lines.extend(trace_lines)
            else:
                lines.append("  - none")
        else:
            lines.append("Step execution trace: none")
        return "\n".join(lines)

    async def _evaluate_final_response_critic(
        self,
        candidate_response: str,
        mode: str,
        trace_id: str,
        turn_id: str,
        query_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        execution_evidence = self._get_turn_execution_evidence(turn_id)
        # If no explicit query_state was passed, try to extract it from the cognition evidence.
        if not query_state and execution_evidence and execution_evidence.get("entity_kind") == "cognition_episode":
            query_state = execution_evidence.get("entity", {}).get("query_state")

        critic_prompt = self._build_final_response_critic_prompt(
            query_state,
            candidate_response,
            execution_evidence=execution_evidence,
        )

        critic_ctx = {"summary": "", "memory": [], "suppress_summary_memory": True}
        judge_model = self._cognition_judge_model if self._cognition_judge_model else self._final_response_critic_model
        raw_critic = await self._generate_response(
            critic_prompt,
            PRIMARY_REASONING_MODE,
            critic_ctx,
            trace_id=trace_id,
            turn_id=turn_id,
            model_override=judge_model,
            options_override=(
                self._cognition_judge_options
                if self._cognition_judge_model
                else self._final_response_critic_options
            ),
        )
        self._log_mode_event(
            mode,
            "final_critic_raw",
            {"trace_id": trace_id, "raw": self._truncate(raw_critic, 2000)},
        )
        parsed_critic = self._parse_final_response_critic(raw_critic)
        if parsed_critic:
            return parsed_critic

        critic_schema = (
            "{\"fulfilled\":true,\"confidence\":0.0,\"issues\":[\"...\"],\"fix_instructions\":[\"...\"]}"
        )
        repair_prompt = self._build_json_repair_prompt(critic_schema, raw_critic)
        repair_model = self._cognition_judge_model if self._cognition_judge_model else self._final_response_critic_model
        repaired = await self._generate_response(
            repair_prompt,
            PRIMARY_REASONING_MODE,
            critic_ctx,
            trace_id=trace_id,
            turn_id=turn_id,
            model_override=repair_model,
            options_override=(
                self._cognition_judge_options
                if self._cognition_judge_model
                else self._final_response_critic_options
            ),
        )
        self._log_mode_event(
            mode,
            "final_critic_repair_raw",
            {"trace_id": trace_id, "raw": self._truncate(repaired, 2000)},
        )
        parsed_critic = self._parse_final_response_critic(repaired)
        if parsed_critic:
            return parsed_critic
        return {
            "fulfilled": False,
            "confidence": 0.0,
            "issues": ["final response critic unavailable or parse failed"],
            "fix_instructions": [
                "Return a safe user-visible response that explicitly reflects degraded execution state."
            ],
        }

    async def _evaluate_tool_codegen_critic(
        self,
        tool_id: str,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        code: str,
        dependencies: List[str],
        trace_id: str,
        attempt: int,
    ) -> Dict[str, Any]:
        if not self._tool_codegen_critic_enabled:
            return {
                "fulfilled": True,
                "confidence": 1.0,
                "issues": [],
                "fix_instructions": [],
            }
        critic_prompt = self._build_tool_codegen_critic_prompt(
            tool_id=tool_id,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            code=code,
            dependencies=dependencies,
        )
        critic_ctx = {"summary": "", "memory": [], "suppress_summary_memory": True}
        raw_critic = await self._generate_response(
            critic_prompt,
            PRIMARY_REASONING_MODE,
            critic_ctx,
            trace_id=trace_id,
            turn_id=None,
            model_override=self._tool_codegen_critic_model,
            options_override=self._tool_codegen_critic_options,
        )
        self._log_mode_event(
            PRIMARY_REASONING_MODE,
            "create_tool_codegen_critic_raw",
            {
                "trace_id": trace_id,
                "tool_id": tool_id,
                "attempt": attempt,
                "raw": self._truncate(raw_critic, 2000),
            },
        )
        parsed_critic = self._parse_final_response_critic(raw_critic)
        if parsed_critic:
            return parsed_critic

        critic_schema = (
            "{\"fulfilled\":true,\"confidence\":0.0,\"issues\":[\"...\"],\"fix_instructions\":[\"...\"]}"
        )
        repair_prompt = self._build_json_repair_prompt(critic_schema, raw_critic)
        repaired = await self._generate_response(
            repair_prompt,
            PRIMARY_REASONING_MODE,
            critic_ctx,
            trace_id=trace_id,
            turn_id=None,
            model_override=self._tool_codegen_critic_model,
            options_override=self._tool_codegen_critic_options,
        )
        self._log_mode_event(
            PRIMARY_REASONING_MODE,
            "create_tool_codegen_critic_repair_raw",
            {
                "trace_id": trace_id,
                "tool_id": tool_id,
                "attempt": attempt,
                "raw": self._truncate(repaired, 2000),
            },
        )
        parsed_critic = self._parse_final_response_critic(repaired)
        if parsed_critic:
            return parsed_critic
        return {
            "fulfilled": False,
            "confidence": 0.0,
            "issues": ["tool codegen critic parse failed"],
            "fix_instructions": ["Return strict JSON with fulfilled/confidence/issues/fix_instructions."],
        }

    async def _apply_final_response_critic(
        self,
        initial_response: str,
        mode: str,
        context_packet: Dict[str, Any],
        trace_id: str,
        turn_id: str,
        query_state: Optional[Dict[str, Any]] = None,
    ) -> str:

        if not self._final_response_critic_enabled:
            return initial_response
        mode_key = mode.strip().upper() if isinstance(mode, str) else ""
        if mode_key and mode_key in self._final_response_critic_skip_modes:
            record_event(
                "final_critic_skipped",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "reason": "mode_config_skip",
                },
            )
            self._log_mode_event(
                mode,
                "final_critic_skipped",
                {"trace_id": trace_id, "mode": mode, "reason": "mode_config_skip"},
            )
            return initial_response

        candidate_response = initial_response
        retry_count = 0
        while True:
            critique = await self._evaluate_final_response_critic(
                candidate_response,
                mode,
                trace_id,
                turn_id,
                query_state=query_state,
            )

            confidence = float(critique.get("confidence", 0.0))
            fulfilled = critique.get("fulfilled") is True and confidence >= self._final_response_critic_min_confidence
            issues = critique.get("issues", [])
            fixes = critique.get("fix_instructions", [])
            record_event(
                "final_critic_result",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": mode,
                    "fulfilled": bool(fulfilled),
                    "confidence": confidence,
                    "issues": issues if isinstance(issues, list) else [],
                    "fix_instructions": fixes if isinstance(fixes, list) else [],
                    "retry_count": retry_count,
                },
            )
            self._log_mode_event(
                mode,
                "final_critic_result",
                {
                    "trace_id": trace_id,
                    "fulfilled": bool(fulfilled),
                    "confidence": confidence,
                    "issues": issues if isinstance(issues, list) else [],
                    "fix_instructions": fixes if isinstance(fixes, list) else [],
                    "retry_count": retry_count,
                },
            )
            if fulfilled:
                return candidate_response
            if retry_count >= self._final_response_critic_max_retries:
                return candidate_response
            retry_count += 1
            # Build retry text based on query_state if available, otherwise fallback to a generic instruction.
            # But the goal is to not use the original text.
            retry_text = (
                f"Your previous response had issues: {', '.join(critique.get('issues', []))}.\n"
                f"Please fix them: {', '.join(critique.get('fix_instructions', []))}.\n"
                "Focus on the task defined in your internal state."
            )
            self._log_mode_event(
                mode,
                "final_critic_retry",
                {
                    "trace_id": trace_id,
                    "retry_count": retry_count,
                    "retry_text_preview": self._truncate(retry_text, 1200),
                },
            )
            candidate_response = await self._run_mode_once(
                mode,
                retry_text,
                context_packet,
                trace_id,
                turn_id,
                retrieval_query=None,
            )


    async def _process_turn(self, turn_id: str) -> None:
        trace_id: Optional[str] = None
        mode: str = PRIMARY_REASONING_MODE
        turn_ctx_token = _CURRENT_TURN_ID.set(turn_id)
        trace_ctx_token = None
        self._clear_turn_execution_evidence(turn_id)
        try:
            text = self._pending_turns.get(turn_id, "")
            mode = self._pending_turn_modes.get(turn_id, PRIMARY_REASONING_MODE)
            skip_distillation = self._pending_turn_skip_distillation.get(turn_id, False)
            trace_id = new_id()
            trace_ctx_token = _CURRENT_TRACE_ID.set(trace_id)
            record_event("user_query", {"turn_id": turn_id, "trace_id": trace_id, "text": text})
            route = {"mode": PRIMARY_REASONING_MODE, "needs_memory": False}
            record_event(
                "router_decision",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "route": route,
                    "mode": mode,
                    "needs_memory": False,
                },
            )
            self._log_mode_event(
                mode,
                "turn_start",
                {"turn_id": turn_id, "trace_id": trace_id, "text": text},
            )

            memory_hits: List[Dict[str, Any]] = []
            context_packet = self._context_builder.build(self._history, self._summary, memory_hits)
            self._turn_context_packet[turn_id] = dict(context_packet)
            record_event(
                "context_built",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "summary": context_packet.get("summary", ""),
                    "memory_count": len(memory_hits),
                    "needs_memory": False,
                    "embodiment_present": bool(context_packet.get("embodiment")),
                },
            )

            response_text = await self._run_mode_once(
                mode,
                text,
                context_packet,
                trace_id,
                turn_id,
                retrieval_query=text,
                skip_distillation=skip_distillation,
            )
            response_text = await self._apply_final_response_critic(
                initial_response=response_text,
                mode=mode,
                context_packet=context_packet,
                trace_id=trace_id,
                turn_id=turn_id,
            )


            final_payload = {"text": response_text, "trace_id": trace_id, "turn_id": turn_id}
            if turn_id in self._pending_turn_modes:
                trace_for_turn = self._pending_turn_traces.get(turn_id, [])
                if trace_for_turn:
                    final_payload["thinking_trace"] = trace_for_turn
            await self._send_event(make_event(EVENT_ASSISTANT_FINAL, final_payload))
            record_event(
                "assistant_final",
                {"turn_id": turn_id, "trace_id": trace_id, "text": response_text},
            )
            self._log_mode_event(
                mode,
                "turn_final",
                {"turn_id": turn_id, "trace_id": trace_id, "text": response_text},
            )

            self._history.append({"role": "user", "text": text, "turn_id": turn_id, "trace_id": trace_id})
            self._history.append({"role": "assistant", "text": response_text, "trace_id": trace_id})

            summary_full = text + " | " + response_text
            self._summary = summary_full

            await self._memory.maybe_store_facts(text, trace_id)
            await self._store_episodic_and_maybe_distill(
                text,
                response_text,
                trace_id,
                turn_id,
                skip_distillation=skip_distillation,
            )

            self._pending_turns.pop(turn_id, None)
            self._pending_turn_modes.pop(turn_id, None)
            self._pending_turn_skip_distillation.pop(turn_id, None)
            self._pending_turn_traces.pop(turn_id, None)
        except asyncio.CancelledError:
            return
        except Exception:
            self._tool_failures += 1
            if self._tool_failures >= 3:
                self._safe_mode = True
            fallback_text = (
                "I ran into a model/runtime issue while processing your request. "
                "Please try again or switch to a smaller model."
            )
            try:
                await self._send_event(
                    make_event(
                        EVENT_ASSISTANT_FINAL,
                        {"text": fallback_text, "trace_id": trace_id or new_id(), "turn_id": turn_id},
                    )
                )
            except Exception:
                pass
            self._log_mode_event(
                mode,
                "turn_error",
                {"turn_id": turn_id, "trace_id": trace_id, "text": fallback_text},
            )
            record_event(
                "turn_error",
                {"turn_id": turn_id, "trace_id": trace_id, "mode": mode},
            )
        finally:
            self._turn_context_packet.pop(turn_id, None)
            if trace_ctx_token is not None:
                _CURRENT_TRACE_ID.reset(trace_ctx_token)
            _CURRENT_TURN_ID.reset(turn_ctx_token)
            self._clear_turn_execution_evidence(turn_id)

    async def _generate_response(
        self,
        text: str,
        mode: str,
        context_packet: Dict[str, Any],
        trace_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        model_override: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        resolved_mode = self._normalize_requested_mode(mode) or PRIMARY_REASONING_MODE
        ctx = self._build_llm_context(resolved_mode, trace_id, turn_id, context_packet=context_packet)
        record_event(
            "llm_request",
            {
                "turn_id": turn_id,
                "trace_id": trace_id,
                "mode": resolved_mode,
                "text": text,
                "summary": ctx.get("summary", ""),
                "memory_count": len(ctx.get("memory", [])),
            },
        )
        try:
            request_payload: Dict[str, Any] = {"text": text, "context": ctx}
            if isinstance(model_override, str) and model_override.strip():
                request_payload["model_override"] = model_override.strip()
            if isinstance(options_override, dict) and options_override:
                request_payload["options_override"] = dict(options_override)
                
            resp = await send_request(
                self._model_host,
                self._model_port,
                "model.Generate",
                request_payload,
                timeout=self._model_rpc_timeout_s,
            )
        except RpcError as e:
            record_event(
                "llm_error",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "mode": resolved_mode,
                    "error": str(e),
                },
            )
            return (
                "I hit a model error while generating a response. "
                "Please try again or switch to a smaller model."
            )
        record_event(
            "llm_response",
            {
                "turn_id": turn_id,
                "trace_id": trace_id,
                "mode": resolved_mode,
                "text": resp.get("text", ""),
            },
        )
        return resp.get("text", "")

    async def _generate_cognition_response(
        self,
        text: str,
        trace_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        model_override: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        ctx = self._build_llm_context("COGNITION", trace_id, turn_id)
        request_payload: Dict[str, Any] = {"text": text, "context": ctx}
        if isinstance(model_override, str) and model_override.strip():
            request_payload["model_override"] = model_override.strip()
        if isinstance(options_override, dict) and options_override:
            request_payload["options_override"] = dict(options_override)
        record_event(
            "llm_request",
            {
                "turn_id": turn_id,
                "trace_id": trace_id,
                "mode": "COGNITION",
                "text": text,
                "model_override": request_payload.get("model_override", ""),
            },
        )
        try:
            resp = await send_request(
                self._model_host,
                self._model_port,
                "model.Generate",
                request_payload,
                timeout=self._model_rpc_timeout_s,
            )
        except RpcError as e:
            record_event(
                "llm_error",
                {"turn_id": turn_id, "trace_id": trace_id, "mode": "COGNITION", "error": str(e)},
            )
            return (
                "{\"type\":\"final\",\"thought\":\"Model unavailable.\","
                "\"text\":\"I hit a model error while running cognition. "
                "Please try again or use a smaller model.\"}"
            )
        record_event(
            "llm_response",
            {"turn_id": turn_id, "trace_id": trace_id, "mode": "COGNITION", "text": resp.get("text", "")},
        )
        return resp.get("text", "")

    async def _call_cognition_json(
        self,
        prompt: str,
        schema: str,
        parser: Callable[[str], Optional[Dict[str, Any]]],
        event_prefix: str,
        trace_id: str,
        turn_id: Optional[str],
        model_override: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        raw = await self._generate_cognition_response(
            prompt,
            trace_id,
            turn_id,
            model_override=model_override,
            options_override=options_override,
        )
        self._log_mode_event(
            "COGNITION",
            f"{event_prefix}_raw",
            {"trace_id": trace_id, "raw": self._truncate(raw, 2000)},
        )
        parsed = parser(raw)
        if parsed is not None:
            await self._maybe_emit_thought_feedback(parsed, event_prefix, trace_id, turn_id)
            return parsed
        repair_prompt = self._build_json_repair_prompt(schema, raw)
        repair_model_override = self._cognition_repair_model if self._cognition_repair_model else model_override
        repair_options_override = (
            self._cognition_repair_options if self._cognition_repair_model else options_override
        )
        repaired = await self._generate_cognition_response_with_timeout(
            repair_prompt,
            trace_id,
            turn_id,
            timeout_s=self._cognition_repair_timeout_s,
            timeout_event=f"{event_prefix}_repair_timeout",
            model_override=repair_model_override,
            options_override=repair_options_override,
        )
        self._log_mode_event(
            "COGNITION",
            f"{event_prefix}_repair_raw",
            {"trace_id": trace_id, "raw": self._truncate(repaired, 2000)},
        )
        repaired_parsed = parser(repaired)
        if repaired_parsed is not None:
            await self._maybe_emit_thought_feedback(repaired_parsed, event_prefix, trace_id, turn_id)
        return repaired_parsed

    def _build_llm_context(
        self,
        mode: str,
        trace_id: Optional[str],
        turn_id: Optional[str],
        context_packet: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        packet = context_packet or {}
        if not packet and isinstance(turn_id, str) and turn_id:
            packet = self._turn_context_packet.get(turn_id, {})
        resolved_mode = self._normalize_requested_mode(mode) or PRIMARY_REASONING_MODE
        ctx = {
            "mode": resolved_mode,
            "summary": packet.get("summary", ""),
            "memory": packet.get("memory", []),
            "trace_id": trace_id,
            "turn_id": turn_id,
            "system_entity": {
                "name": self._system_entity.name,
                "aliases": self._system_entity.aliases,
            }
        }
        if packet.get("suppress_summary_memory"):
            ctx["suppress_summary_memory"] = True
        return ctx

    async def _resolve_embodiment_fallback_tool(
        self,
        tool_id: str,
        trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(tool_id, str) or not tool_id.startswith("embodiment."):
            return None
        if tool_id.startswith("embodiment.generated."):
            return None
        if tool_id in {
            "embodiment.describe_host",
            "embodiment.list_capabilities",
            "embodiment.list_fallbacks",
            "embodiment.get_state",
        }:
            return None

        capability_resp = await self._call_tool(
            "embodiment.list_capabilities",
            {},
            approval_token=None,
            trace_id=trace_id,
            required_artifacts=[],
        )
        capability_payload = capability_resp.get("result")
        if capability_resp.get("status") != "APPROVED" or not isinstance(capability_payload, dict):
            return None
        capability = None
        for item in capability_payload.get("capabilities", []):
            if not isinstance(item, dict):
                continue
            tool_ids = item.get("tool_ids", [])
            if isinstance(tool_ids, list) and tool_id in tool_ids:
                capability = item
                break
        if not isinstance(capability, dict) or capability.get("status") != "configured":
            return None

        fallback_resp = await self._call_tool(
            "embodiment.list_fallbacks",
            {},
            approval_token=None,
            trace_id=trace_id,
            required_artifacts=[],
        )
        fallback_payload = fallback_resp.get("result")
        if fallback_resp.get("status") != "APPROVED" or not isinstance(fallback_payload, dict):
            return None

        capability_id = str(capability.get("capability_id", "")).strip()
        abstract_capability = str(capability.get("abstract_capability", "")).strip()
        candidate = None
        for item in fallback_payload.get("fallbacks", []):
            if not isinstance(item, dict):
                continue
            item_capability_id = str(item.get("capability_id", "")).strip()
            item_abstract = str(item.get("abstract_capability", "")).strip()
            if capability_id and item_capability_id == capability_id:
                candidate = item
                break
            if abstract_capability and item_abstract == abstract_capability:
                candidate = item
                break
        if not isinstance(candidate, dict):
            return None

        tool_spec = candidate.get("tool_spec")
        if not isinstance(tool_spec, dict):
            return None
        generated_tool_id = str(tool_spec.get("tool_id", "")).strip()
        if not generated_tool_id:
            return None
        if self._tools.get_tool(generated_tool_id):
            return {
                "tool_id": generated_tool_id,
                "capability_id": capability_id,
                "created": False,
                "reason": candidate.get("reason", ""),
            }

        create_result = await self._handle_create_tool(tool_spec, trace_id)
        if create_result.get("status") != "APPROVED":
            return {
                "error": str(create_result.get("error") or "Failed to create embodiment fallback tool"),
                "capability_id": capability_id,
                "tool_id": generated_tool_id,
                "reason": candidate.get("reason", ""),
            }
        return {
            "tool_id": generated_tool_id,
            "capability_id": capability_id,
            "created": True,
            "reason": candidate.get("reason", ""),
        }

    async def _execute_tool_action(
        self,
        tool_id: str,
        args: Dict[str, Any],
        trace_id: str,
        thought: str = "",
        allow_embodiment_fallback: bool = False,
    ) -> Dict[str, Any]:
        tool_def = self._tools.get_tool(tool_id)
        action_record = {"tool_id": tool_id, "args": args}
        if not tool_def:
            record_event(
                "tool_error",
                {"trace_id": trace_id, "tool_id": tool_id, "error": "Unknown tool"},
            )
            return {
                "thought": thought,
                "action": action_record,
                "observation": {"status": "ERROR", "error": "Unknown tool"},
            }

        requested_tool_id = tool_id
        if allow_embodiment_fallback:
            fallback_resolution = await self._resolve_embodiment_fallback_tool(tool_id, trace_id)
            if isinstance(fallback_resolution, dict):
                fallback_error = str(fallback_resolution.get("error", "")).strip()
                if fallback_error:
                    action_record["auto_fallback"] = {
                        "tool_id": fallback_resolution.get("tool_id"),
                        "capability_id": fallback_resolution.get("capability_id"),
                    }
                    return {
                        "thought": thought,
                        "action": action_record,
                        "observation": {"status": "ERROR", "error": fallback_error},
                    }
                resolved_tool_id = str(fallback_resolution.get("tool_id", "")).strip()
                if resolved_tool_id:
                    tool_id = resolved_tool_id
                    tool_def = self._tools.get_tool(tool_id)
                    action_record = {
                        "tool_id": tool_id,
                        "args": args,
                        "requested_tool_id": requested_tool_id,
                        "auto_fallback": {
                            "capability_id": fallback_resolution.get("capability_id"),
                            "created": bool(fallback_resolution.get("created")),
                        },
                    }
                    record_event(
                        "embodiment_fallback_resolved",
                        {
                            "trace_id": trace_id,
                            "requested_tool_id": requested_tool_id,
                            "tool_id": tool_id,
                            "capability_id": fallback_resolution.get("capability_id"),
                            "created": bool(fallback_resolution.get("created")),
                        },
                    )
                    if not tool_def:
                        return {
                            "thought": thought,
                            "action": action_record,
                            "observation": {"status": "ERROR", "error": "Generated fallback tool was not registered"},
                        }

        decision = self._policy.evaluate_tool(tool_def, args)
        approval_token = None
        if decision.status == "CONFIRM":
            if self._auto_approve_all:
                approval_token = "auto-all"
            else:
                approval_token = await self._request_permission(tool_id, decision.reason)
                if not approval_token:
                    return {
                        "thought": thought,
                        "action": action_record,
                        "observation": {"status": "DENIED", "error": "Not approved"},
                    }
        elif decision.status == "PASSPHRASE":
            return {
                "thought": thought,
                "action": action_record,
                "observation": {"status": "DENIED", "error": "Passphrase required"},
            }

        required_artifacts = []
        if "write" in tool_def.get("capabilities", []):
            required_artifacts = ["diff", "hash"]

        resp = await self._call_tool(tool_id, args, approval_token, trace_id, required_artifacts)
        status = resp.get("status")
        if status != "APPROVED":
            self._tool_failures += 1
            if self._tool_failures >= 3:
                self._safe_mode = True
        observation = {
            "status": status,
            "result": resp.get("result"),
            "error": resp.get("error"),
        }
        if resp.get("artifacts"):
            observation["artifacts"] = resp.get("artifacts")
        if resp.get("logs_ref"):
            observation["logs_ref"] = resp.get("logs_ref")
        return {"thought": thought, "action": action_record, "observation": observation}

    async def _execute_cognition_action(
        self,
        action: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, Any]:
        action_type = str(action.get("action_type") or "tool").strip().lower()
        if action_type == "generate_tools_from_spec":
            spec = action.get("spec", {})
            label = str(action.get("label") or "").strip()
            action_record: Dict[str, Any] = {"action_type": "generate_tools_from_spec", "spec": spec}
            if label:
                action_record["label"] = label
            if not isinstance(spec, dict):
                return {
                    "action": action_record,
                    "observation": {"status": "ERROR", "error": "spec must be an object"},
                    "new_tools": [],
                }
            result = await self.generate_tools_from_spec(spec, trace_id=trace_id)
            payload = result.get("result", {})
            if not isinstance(payload, dict):
                payload = {}
            observation: Dict[str, Any] = {"status": str(result.get("status") or "ERROR")}
            if payload:
                observation["result"] = payload
            error_text = str(result.get("error") or "").strip()
            if error_text:
                observation["error"] = error_text
            new_tools: List[Dict[str, Any]] = []
            for item in payload.get("tools", []):
                if not isinstance(item, dict):
                    continue
                item_status = str(item.get("status") or "").strip().upper()
                if item_status not in {"APPROVED", "SKIPPED"}:
                    continue
                tool_id = str(item.get("tool_id") or "").strip()
                if not tool_id:
                    continue
                tool_def = self._tools.get_tool(tool_id)
                if isinstance(tool_def, dict):
                    new_tools.append(tool_def)
            return {
                "action": action_record,
                "observation": observation,
                "new_tools": new_tools,
            }

        tool_id = str(action.get("tool_id") or "").strip()
        args = action.get("args", {})
        if not isinstance(args, dict):
            args = {}
        thought = str(action.get("thought") or "").strip()
        result = await self._execute_tool_action(tool_id, args, trace_id, thought=thought)
        result["new_tools"] = []
        return result

    def _safe_json(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return json.dumps(str(obj), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _normalize_requested_mode(value: Any) -> Optional[str]:
        text = str(value or "").strip().upper()
        if text in {PRIMARY_REASONING_MODE, "SYSTEM0", "SYSTEM1", "SYSTEM2", "SYSTEM3"}:
            return text
        return None

    def _log_mode_event(self, mode: str, event_type: str, payload: Dict[str, Any]) -> None:
        path = self._mode_logs.get(mode)
        if not path:
            return
        record = {
            "ts": utc_now_iso(),
            "mode": mode,
            "event_type": event_type,
            "payload": payload,
        }
        try:
            append_jsonl(path, record)
        except Exception:
            pass

    def _format_exchanges_for_summary(self, exchanges: List[Dict[str, str]]) -> str:
        lines = []
        for idx, exchange in enumerate(exchanges, start=1):
            user_text = (exchange.get("user") or "").strip()
            assistant_text = (exchange.get("assistant") or "").strip()
            lines.append(f"{idx}) User: {user_text}")
            lines.append(f"{idx}) Assistant: {assistant_text}")
        return "\n".join(lines).strip()

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(text, validator=lambda item: isinstance(item, dict))
        if isinstance(parsed, dict):
            return parsed
        return None

    async def _distill_exchanges(
        self,
        exchanges: List[Dict[str, str]],
        trace_id: Optional[str],
        turn_id: Optional[str],
    ) -> Dict[str, Any]:
        formatted = self._format_exchanges_for_summary(exchanges)
        prompt = (
            "Extract facts and entities from the conversation excerpts. "
            "Return strict JSON only with keys: "
            "{\"facts\":[{\"name\":string,\"value\":string,\"confidence\":number}],"
            "\"entities\":[{\"name\":string,\"data\":object,\"confidence\":number}]}.\n"
            "- facts: stable preferences, identities, decisions, or constraints.\n"
            "- entities: structured entities with attributes (keep data as a JSON object).\n"
            "If none, return empty arrays.\n\n"
            "Examples:\n"
            "Conversation:\n"
            "1) User: My name is Sam and I prefer short answers.\n"
            "1) Assistant: Got it.\n"
            "JSON:\n"
            "{\"facts\":[{\"name\":\"user.name\",\"value\":\"Sam\",\"confidence\":0.7},"
            "{\"name\":\"user.preference.response_style\",\"value\":\"short\",\"confidence\":0.6}],"
            "\"entities\":[{\"name\":\"user\",\"data\":{\"name\":\"Sam\",\"preference_response_style\":\"short\"},"
            "\"confidence\":0.6}]}\n\n"
            "Conversation:\n"
            "1) User: The project is Atlas, version 1.2.3, owner is team-red.\n"
            "1) Assistant: Noted.\n"
            "JSON:\n"
            "{\"facts\":[{\"name\":\"project.name\",\"value\":\"Atlas\",\"confidence\":0.6},"
            "{\"name\":\"project.owner\",\"value\":\"team-red\",\"confidence\":0.6}],"
            "\"entities\":[{\"name\":\"project\",\"data\":{\"name\":\"Atlas\",\"version\":\"1.2.3\",\"owner\":\"team-red\"},"
            "\"confidence\":0.6}]}\n\n"
            "Conversation:\n"
            "1) User: I use Windows 11 on a ThinkPad X1 with 32GB RAM.\n"
            "1) Assistant: Thanks.\n"
            "JSON:\n"
            "{\"facts\":[{\"name\":\"user.os\",\"value\":\"Windows 11\",\"confidence\":0.6}],"
            "\"entities\":[{\"name\":\"device\",\"data\":{\"model\":\"ThinkPad X1\",\"ram_gb\":32},\"confidence\":0.6}]}\n\n"
            f"{formatted}"
        )
        try:
            resp = await send_request(
                self._model_host,
                self._model_port,
                "model.Generate",
                {
                    "text": prompt,
                    "context": {
                        "mode": "GENERAL",
                        "summary": "",
                        "memory": [],
                        "trace_id": trace_id,
                        "turn_id": turn_id,
                    },
                },
                timeout=self._model_rpc_timeout_s,
            )
            raw = (resp.get("text") or "").strip()
            parsed = self._extract_json_object(raw)
            if parsed:
                return parsed
        except RpcError as e:
            record_event(
                "memory_distill_error",
                {"turn_id": turn_id, "trace_id": trace_id, "error": str(e)},
            )
        return {"facts": [], "entities": []}

    async def _store_episodic_and_maybe_distill(
        self,
        text: str,
        response_text: str,
        trace_id: Optional[str],
        turn_id: Optional[str],
        skip_distillation: bool = False,
    ) -> None:
        summary_full = text + " | " + response_text
        await self._memory.store_episodic(summary_full, trace_id or new_id(), name="conversation.turn")
        if skip_distillation:
            record_event(
                "memory_distill_skipped",
                {
                    "turn_id": turn_id,
                    "trace_id": trace_id,
                    "reason": "skip_distillation",
                },
            )
            return
        if self._memory_distill_turns <= 1:
            return
        batch = await self._memory.fetch_episodic_batch(
            self._memory_distill_turns,
            order="oldest",
            name_filter="conversation.turn",
        )
        if len(batch) < self._memory_distill_turns:
            return
        exchanges = []
        for mem in batch:
            data = mem.get("data") or {}
            summary = data.get("summary") if isinstance(data, dict) else None
            if isinstance(summary, str) and summary.strip():
                if " | " in summary:
                    user_text, assistant_text = summary.split(" | ", 1)
                    exchanges.append({"user": user_text.strip(), "assistant": assistant_text.strip()})
                else:
                    exchanges.append({"user": summary.strip(), "assistant": ""})
        if not exchanges:
            return
        distilled = await self._distill_exchanges(exchanges, trace_id, turn_id)
        facts = distilled.get("facts")
        if isinstance(facts, list):
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                name = fact.get("name")
                value = fact.get("value")
                confidence = fact.get("confidence")
                if isinstance(name, str) and name and isinstance(value, str) and value:
                    conf_value = 0.5
                    if isinstance(confidence, (int, float)):
                        conf_value = float(confidence)
                    await self._memory.store_fact(name, value, trace_id or new_id(), confidence=conf_value)
        entities = distilled.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                name = entity.get("name")
                data = entity.get("data")
                confidence = entity.get("confidence")
                if isinstance(name, str) and name and isinstance(data, dict) and data:
                    conf_value = 0.5
                    if isinstance(confidence, (int, float)):
                        conf_value = float(confidence)
                    await self._memory.store_entity(name, data, trace_id or new_id(), confidence=conf_value)
        record_event(
            "memory_distilled",
            {
                "turn_id": turn_id,
                "trace_id": trace_id,
                "exchange_count": len(batch),
            },
        )
        await self._memory.delete_memory(
            [mem["mem_id"] for mem in batch if mem.get("mem_id")],
            trace_id=trace_id,
        )

    def _truncate(self, text: str, limit: int = 1200) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _tool_description(self, tool: Dict[str, Any]) -> str:
        tool_id = tool.get("tool_id", "")
        name = tool.get("name", "")
        description = tool.get("description", "")
        capabilities = tool.get("capabilities", [])
        input_schema = tool.get("input_schema", {})
        return " | ".join(
            part
            for part in [
                f"{tool_id} {name}".strip(),
                description.strip() if isinstance(description, str) else "",
                f"capabilities={capabilities}",
                f"input_schema={self._safe_json(input_schema)}",
            ]
            if part
        )

    async def _embed_text(self, text: str) -> List[float]:
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached
        resp = await send_request(
            self._model_host,
            self._model_port,
            "model.EmbedText",
            {"text": text},
        )
        embedding = resp.get("embedding", [])
        if self._embed_cache_limit > 0:
            if len(self._embed_cache) >= self._embed_cache_limit:
                try:
                    self._embed_cache.pop(next(iter(self._embed_cache)))
                except StopIteration:
                    pass
            self._embed_cache[text] = embedding
        return embedding

    async def _ensure_tool_index(self) -> None:
        tools = self._tools.list_active()
        current_ids = set()
        for tool in tools:
            tool_id = tool.get("tool_id")
            if not tool_id:
                continue
            current_ids.add(tool_id)
            desc = self._tool_description(tool)
            cached = self._tool_index.get(tool_id)
            if cached and cached.get("desc") == desc:
                continue
            embedding = await self._embed_text(desc)
            self._tool_index[tool_id] = {
                "tool": tool,
                "desc": desc,
                "embedding": embedding,
            }
        stale = [tool_id for tool_id in self._tool_index if tool_id not in current_ids]
        for tool_id in stale:
            self._tool_index.pop(tool_id, None)

    async def _retrieve_relevant_tools(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        all_tools = self._tools.list_active()
        if not self._retrieve_relevant_tools_enabled:
            return all_tools
        if top_k <= 0 or top_k >= len(all_tools):
            return all_tools
        try:
            await self._ensure_tool_index()
            query_embedding = await self._embed_text(query)
            scored = []
            for entry in self._tool_index.values():
                score = self._cosine(query_embedding, entry.get("embedding", []))
                scored.append((score, entry.get("tool")))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [tool for _, tool in scored[:top_k] if tool is not None]
        except Exception:
            return all_tools

    async def _select_relevant_tools(
        self,
        user_text: str,
        context_packet: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
        candidate_tools: List[Dict[str, Any]],
        trace_id: str,
        turn_id: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not self._select_relevant_tools_enabled:
            return candidate_tools, None
        if len(candidate_tools) <= 1:
            selected_ids = [
                str(tool.get("tool_id") or "").strip()
                for tool in candidate_tools
                if str(tool.get("tool_id") or "").strip()
            ]
            return candidate_tools, {
                "phase": "tool_selection",
                "status": "skipped",
                "reason": "selection skipped because candidate tool count is 0 or 1",
                "candidate_count": len(candidate_tools),
                "selected_count": len(candidate_tools),
                "selected_tool_ids": selected_ids,
            }

        allowed_tool_ids = {
            str(tool.get("tool_id") or "").strip()
            for tool in candidate_tools
            if str(tool.get("tool_id") or "").strip()
        }
        if not allowed_tool_ids:
            return candidate_tools, {
                "phase": "tool_selection",
                "status": "skipped",
                "reason": "selection skipped because no candidate tool ids were available",
                "candidate_count": len(candidate_tools),
                "selected_count": len(candidate_tools),
                "selected_tool_ids": [],
            }

        if self._tool_selection_category_mode_enabled:
            return await self._select_relevant_tool_categories(
                user_text,
                context_packet,
                memory_hits,
                candidate_tools,
                allowed_tool_ids,
                trace_id,
                turn_id,
            )

        parsed = await self._call_cognition_json(
            self._build_cognition_tool_selection_prompt(
                user_text,
                context_packet,
                memory_hits,
                candidate_tools,
            ),
            "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"tool.id\"],\"reason\":\"...\"}",
            lambda raw: self._parse_cognition_tool_selection_response(raw, allowed_tool_ids),
            "tool_selection",
            trace_id,
            turn_id,
            model_override=self._resolve_cognition_model_override("tool_selection"),
            options_override=self._cognition_tool_selection_options,
        )
        if parsed is None:
            selected_ids = [
                str(tool.get("tool_id") or "").strip()
                for tool in candidate_tools
                if str(tool.get("tool_id") or "").strip()
            ]
            return candidate_tools, {
                "phase": "tool_selection",
                "status": "fallback",
                "reason": "tool selector returned invalid output; using candidate tool set",
                "candidate_count": len(candidate_tools),
                "selected_count": len(candidate_tools),
                "selected_tool_ids": selected_ids,
            }

        selected_lookup = set(parsed.get("selected_tool_ids", []))
        selected_tools = [
            tool
            for tool in candidate_tools
            if str(tool.get("tool_id") or "").strip() in selected_lookup
        ]
        return selected_tools, {
            "phase": "tool_selection",
            "status": "selected",
            "reason": str(parsed.get("reason") or ""),
            "candidate_count": len(candidate_tools),
            "selected_count": len(selected_tools),
            "selected_tool_ids": [
                str(tool.get("tool_id") or "").strip()
                for tool in selected_tools
                if str(tool.get("tool_id") or "").strip()
            ],
        }

    async def _select_relevant_tool_categories(
        self,
        user_text: str,
        context_packet: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
        candidate_tools: List[Dict[str, Any]],
        allowed_tool_ids: Set[str],
        trace_id: str,
        turn_id: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        category_map = self._group_tools_by_selection_category(candidate_tools)
        allowed_category_ids = set(category_map.keys())
        if not allowed_category_ids:
            selected_ids = [
                str(tool.get("tool_id") or "").strip()
                for tool in candidate_tools
                if str(tool.get("tool_id") or "").strip()
            ]
            return candidate_tools, {
                "phase": "tool_selection",
                "status": "fallback",
                "selection_mode": "category",
                "reason": "category selector could not build categories; using candidate tool set",
                "candidate_count": len(candidate_tools),
                "selected_count": len(candidate_tools),
                "selected_tool_ids": selected_ids,
                "selected_category_ids": [],
            }

        parsed = await self._call_cognition_json(
            self._build_cognition_tool_category_selection_prompt(
                user_text,
                context_packet,
                memory_hits,
                category_map,
            ),
            (
                "{\"type\":\"tool_selection\",\"selected_category_ids\":[\"ui\"],"
                "\"selected_tool_ids\":[\"tool.id\"],\"reason\":\"...\"}"
            ),
            lambda raw: self._parse_cognition_tool_selection_response(
                raw,
                allowed_tool_ids,
                allowed_category_ids,
            ),
            "tool_selection",
            trace_id,
            turn_id,
            model_override=self._resolve_cognition_model_override("tool_selection"),
            options_override=self._cognition_tool_selection_options,
        )
        if parsed is None:
            selected_ids = [
                str(tool.get("tool_id") or "").strip()
                for tool in candidate_tools
                if str(tool.get("tool_id") or "").strip()
            ]
            return candidate_tools, {
                "phase": "tool_selection",
                "status": "fallback",
                "selection_mode": "category",
                "reason": "category selector returned invalid output; using candidate tool set",
                "candidate_count": len(candidate_tools),
                "selected_count": len(candidate_tools),
                "selected_tool_ids": selected_ids,
                "selected_category_ids": [],
            }

        selected_tools = self._expand_selected_tool_categories(
            category_map,
            parsed.get("selected_category_ids", []),
            parsed.get("selected_tool_ids", []),
        )
        selected_tool_ids = [
            str(tool.get("tool_id") or "").strip()
            for tool in selected_tools
            if str(tool.get("tool_id") or "").strip()
        ]
        return selected_tools, {
            "phase": "tool_selection",
            "status": "selected",
            "selection_mode": "category",
            "reason": str(parsed.get("reason") or ""),
            "candidate_count": len(candidate_tools),
            "category_count": len(category_map),
            "selected_count": len(selected_tools),
            "selected_tool_ids": selected_tool_ids,
            "selected_category_ids": list(parsed.get("selected_category_ids", [])),
        }

    def _group_tools_by_selection_category(
        self,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            category_id = self._tool_selection_category_id(tool)
            grouped.setdefault(category_id, []).append(tool)
        return grouped

    def _tool_selection_category_id(self, tool: Dict[str, Any]) -> str:
        for key in ("tool_category", "category"):
            value = str(tool.get(key) or "").strip().lower()
            if value:
                return value
        tool_id = str(tool.get("tool_id") or "").strip().lower()
        bucket = str(tool.get("tool_bucket") or "").strip().lower()
        if bucket:
            if bucket == "base" and "." in tool_id:
                return tool_id.split(".", 1)[0]
            if "." in bucket:
                return bucket.split(".", 1)[0]
            return bucket
        if "." in tool_id:
            return tool_id.split(".", 1)[0]
        return "base"

    def _expand_selected_tool_categories(
        self,
        category_map: Dict[str, List[Dict[str, Any]]],
        selected_category_ids: List[str],
        selected_tool_ids: List[str],
    ) -> List[Dict[str, Any]]:
        selected_lookup = {
            str(tool_id or "").strip()
            for tool_id in selected_tool_ids
            if str(tool_id or "").strip()
        }
        expanded: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for category_id in selected_category_ids:
            for tool in category_map.get(str(category_id or "").strip(), []):
                tool_id = str(tool.get("tool_id") or "").strip()
                if tool_id and tool_id in seen:
                    continue
                if tool_id:
                    seen.add(tool_id)
                expanded.append(tool)
        if selected_lookup:
            for tools in category_map.values():
                for tool in tools:
                    tool_id = str(tool.get("tool_id") or "").strip()
                    if tool_id not in selected_lookup or tool_id in seen:
                        continue
                    seen.add(tool_id)
                    expanded.append(tool)
        return expanded

    def _merge_relevant_tool_lists(self, *tool_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for tools in tool_lists:
            if not isinstance(tools, list):
                continue
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                tool_id = str(tool.get("tool_id") or "").strip()
                if tool_id:
                    if tool_id in seen:
                        continue
                    seen.add(tool_id)
                merged.append(tool)
        return merged

    def _format_tools_for_agent(self, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if tools is None:
            tools = self._tools.list_active()
        if not tools:
            return "None"
        lines = []
        for tool in tools:
            tool_id = tool.get("tool_id")
            name = tool.get("name", "")
            description = tool.get("description") or ""
            input_schema = tool.get("input_schema", {})
            schema_text = self._safe_json(input_schema)
            label = f"{tool_id}: {name}".strip(": ")
            if description:
                lines.append(f"- {label} — {description} input_schema={schema_text}")
            else:
                lines.append(f"- {label} input_schema={schema_text}")
        return "\n".join(lines)

    def _format_cognition_special_actions(self) -> str:
        return (
            "- generate_tools_from_spec: "
            "{\"action_type\":\"generate_tools_from_spec\",\"spec\":{\"request\":\"...\",\"namespace_prefix\":\"...\","
            "\"max_tools\":2},\"label\":\"...\"} "
            "Use this only when the listed tools clearly cannot satisfy a stable capability gap and a reusable tool would help."
        )

    def _format_asi_input_specs(self, inputs_needed: Any) -> str:
        specs = self._normalize_asi_inputs_needed(inputs_needed)
        if not specs:
            return ""
        type_map = {
            "text": "str",
            "string": "str",
            "url": "str",
            "uri": "str",
            "path": "str",
            "file_path": "str",
            "symbol": "str",
            "variable": "str",
            "math_expr": "str",
            "expression": "str",
            "equation": "str",
            "enum": "str",
            "choice": "str",
            "int": "int",
            "integer": "int",
            "float": "float",
            "number": "float",
            "bool": "bool",
            "boolean": "bool",
            "list": "list",
            "array": "list",
            "object": "dict",
            "dict": "dict",
        }
        parts: List[str] = []
        for spec in specs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            value_type = str(spec.get("type", "text")).strip().lower()
            python_type = type_map.get(value_type, "str")
            label = f"{name}: {python_type}"
            if "default" in spec:
                default_value = spec.get("default")
                if isinstance(default_value, str):
                    rendered_default = repr(default_value)
                elif default_value is None:
                    rendered_default = "None"
                elif isinstance(default_value, bool):
                    rendered_default = "True" if default_value else "False"
                else:
                    rendered_default = repr(default_value)
                label += f" = {self._truncate(rendered_default, 40)}"
            elif not spec.get("required", True):
                label += " = None"
            parts.append(label)
        return ", ".join(parts)

    def _format_asi_pattern_signature(self, pattern_id: Any, inputs_needed: Any) -> str:
        normalized_id = self._normalize_asi_pattern_ref(pattern_id)
        if not normalized_id:
            normalized_id = str(pattern_id or "").strip()
        params = self._format_asi_input_specs(inputs_needed)
        return f"{normalized_id}({params})" if params else f"{normalized_id}()"

    def _format_agent_observations(self, observations: List[Dict[str, Any]]) -> str:
        if not observations:
            return "None"
        lines = []
        for idx, obs in enumerate(observations, start=1):
            label = obs.get("label") or ""
            action = obs.get("action")
            observation = obs.get("observation")
            if label:
                lines.append(f"{idx}) {label} -> {self._safe_json(action)}")
            else:
                lines.append(f"{idx}) {self._safe_json(action)}")
            lines.append(f"   Observation: {self._truncate(self._safe_json(observation))}")
        return "\n".join(lines)

    def _normalize_cognition_text_list(self, values: Any, limit: int = 8) -> List[str]:
        if not isinstance(values, list):
            return []
        normalized: List[str] = []
        seen: Set[str] = set()
        max_items = max(1, limit)
        for item in values:
            if not isinstance(item, str):
                continue
            value = item.strip()
            key = value.lower()
            if not value or key in seen:
                continue
            seen.add(key)
            normalized.append(value)
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_cognition_confidence(self, value: Any, default: float = 0.35) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = default
        if confidence < 0.0:
            return 0.0
        if confidence > 1.0:
            return 1.0
        return confidence

    def _cognition_query_state_schema(self) -> str:
        return (
            "\"query_rewrite\":\"...\",\"intent\":\"...\",\"objectives\":[\"...\"],"
            "\"constraints\":[\"...\"],\"unknowns\":[\"...\"],\"success_criteria\":[\"...\"],"
            "\"working_set\":[\"...\"],"
            "\"todo\":[{\"id\":\"...\",\"task\":\"...\",\"status\":\"pending|in_progress|done|blocked\",\"notes\":[\"...\"]}]"
        )

    def _cognition_state_of_mind_schema(self) -> str:
        return (
            "\"stance\":\"...\",\"confidence\":0.0,\"attention\":[\"...\"],"
            "\"risks\":[\"...\"],\"open_questions\":[\"...\"],\"reflections\":[\"...\"]"
        )

    def _build_cognition_init_schema(self) -> str:
        fields = [
            "\"type\":\"init\"",
            "\"thought\":\"...\"",
            "\"tools_needed\":true|false",
        ]
        if self._cognition_query_state_enabled:
            fields.append(f"\"query_state\":{{{self._cognition_query_state_schema()}}}")
        if self._cognition_state_of_mind_enabled:
            fields.append(f"\"state_of_mind\":{{{self._cognition_state_of_mind_schema()}}}")
        return "{" + ",".join(fields) + "}"

    def _cognition_delta_schema_fields(self) -> List[str]:
        fields: List[str] = []
        if self._cognition_query_state_enabled:
            fields.append("\"query_state_delta\":{}")
        if self._cognition_state_of_mind_enabled:
            fields.append("\"state_of_mind_delta\":{}")
        return fields

    def _compose_cognition_schema(self, fields: List[str]) -> str:
        return "{" + ",".join(field for field in fields if field) + "}"

    def _build_cognition_system_schemas(self) -> Dict[str, str]:
        delta_fields = self._cognition_delta_schema_fields()
        action_schema = (
            "["
            "{\"tool_id\":\"...\",\"args\":{},\"label\":\"...\"},"
            "{\"action_type\":\"generate_tools_from_spec\",\"spec\":{\"request\":\"...\",\"namespace_prefix\":\"...\",\"max_tools\":2},\"label\":\"...\"}"
            "]"
        )
        final_fields = ["\"type\":\"final\"", "\"thought\":\"...\"", "\"text\":\"...\""] + delta_fields
        clarify_fields = ["\"type\":\"clarify\"", "\"thought\":\"...\"", "\"question\":\"...\""] + delta_fields
        act_fields = [
            "\"type\":\"act\"",
            "\"thought\":\"...\"",
            f"\"actions\":{action_schema}",
        ] + delta_fields
        step_fields = [
            "\"type\":\"step\"",
            "\"thought\":\"...\"",
            "\"reflection\":\"...\"",
        ] + delta_fields
        if self._cognition_query_state_enabled:
            step_fields.append(
                "\"todo_updates\":[{\"id\":\"...\",\"task\":\"...\",\"status\":\"pending|in_progress|done|blocked\",\"notes\":[\"...\"]}]"
            )
        step_fields.append(f"\"actions\":{action_schema}")
        return {
            "final": self._compose_cognition_schema(final_fields),
            "clarify": self._compose_cognition_schema(clarify_fields),
            "act": self._compose_cognition_schema(act_fields),
            "step": self._compose_cognition_schema(step_fields),
        }

    def _build_cognition_execution_schema_text(
        self,
        *,
        allow_step: bool,
        allow_clarify: bool,
    ) -> str:
        schemas = self._build_cognition_system_schemas()
        schema_parts = [schemas["final"]]
        if allow_clarify:
            schema_parts.append(schemas["clarify"])
        schema_parts.append(schemas["act"])
        if allow_step:
            schema_parts.append(schemas["step"])
        return "\n".join(schema_parts)

    def _format_cognition_optional_state_sections(
        self,
        query_state: Optional[Dict[str, Any]],
        state_of_mind: Optional[Dict[str, Any]],
        *,
        query_limit: int = 2600,
        state_limit: int = 2200,
    ) -> str:
        sections: List[str] = []
        if self._cognition_query_state_enabled:
            sections.append(
                "QUERY STATE:\n"
                f"{self._truncate(self._safe_json(query_state or {}), query_limit)}"
            )
        if self._cognition_state_of_mind_enabled:
            sections.append(
                "STATE OF MIND:\n"
                f"{self._truncate(self._safe_json(state_of_mind or {}), state_limit)}"
            )
        return ("\n\n".join(sections) + "\n\n") if sections else ""

    def _normalize_cognition_catalog_id(self, prefix: str, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        if not slug:
            return ""
        return f"{prefix}.{slug[:64]}"

    def _normalize_cognition_todos(self, values: Any, limit: int = 8) -> List[Dict[str, Any]]:
        if not isinstance(values, list):
            return []
        normalized: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        max_items = max(1, limit)
        for item in values:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task") or "").strip()
            todo_id = str(item.get("id") or "").strip().lower()
            if not todo_id and task:
                todo_id = self._normalize_cognition_catalog_id("todo", task).replace("todo.", "", 1)
            if not todo_id or not task or todo_id in seen:
                continue
            status = str(item.get("status") or "pending").strip().lower()
            if status not in {"pending", "in_progress", "done", "blocked"}:
                status = "pending"
            notes = self._normalize_cognition_text_list(item.get("notes", []), limit=4)
            normalized.append(
                {
                    "id": todo_id,
                    "task": task,
                    "status": status,
                    "notes": notes,
                }
            )
            seen.add(todo_id)
            if len(normalized) >= max_items:
                break
        return normalized

    def _merge_cognition_todos(
        self,
        current: Any,
        updates: Any,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        merged = self._normalize_cognition_todos(current, limit=limit)
        index: Dict[str, int] = {}
        for idx, item in enumerate(merged):
            index[item["id"]] = idx
            index[item["task"].strip().lower()] = idx
        for item in self._normalize_cognition_todos(updates, limit=limit):
            target_idx = index.get(item["id"])
            if target_idx is None:
                target_idx = index.get(item["task"].strip().lower())
            if target_idx is None:
                if len(merged) >= limit:
                    break
                merged.append(item)
                idx = len(merged) - 1
                index[item["id"]] = idx
                index[item["task"].strip().lower()] = idx
                continue
            existing = merged[target_idx]
            existing["task"] = item.get("task", existing.get("task", ""))
            existing["status"] = item.get("status", existing.get("status", "pending"))
            existing["notes"] = self._normalize_cognition_text_list(
                list(existing.get("notes", [])) + list(item.get("notes", [])),
                limit=4,
            )
        return merged[: max(1, limit)]

    def _default_cognition_query_state(self, user_text: str) -> Dict[str, Any]:
        if not self._cognition_query_state_enabled:
            return {}
        return {
            "query_rewrite": str(user_text or "").strip(),
            "intent": "",
            "objectives": [],
            "constraints": [],
            "unknowns": [],
            "success_criteria": [],
            "working_set": [],
            "todo": [],
        }

    def _is_trivial_cognition_request(self, text: str, tools: Optional[List[Dict[str, Any]]] = None) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        if tools:
            return False
        if len(normalized) >= self._cognition_distill_trivial_max_chars:
            return False
        words = normalized.rstrip("?.!").split()
        if len(words) <= 2:
            return True
        trivial_prefixes = (
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "yes",
            "no",
        )
        return normalized.rstrip("?.!") in trivial_prefixes

    def _normalize_cognition_query_state_delta(self, value: Any) -> Dict[str, Any]:
        if not self._cognition_query_state_enabled:
            return {}
        if not isinstance(value, dict):
            return {}
        delta: Dict[str, Any] = {}
        query_rewrite = str(value.get("query_rewrite") or "").strip()
        if query_rewrite:
            delta["query_rewrite"] = query_rewrite
        intent = str(value.get("intent") or "").strip()
        if intent:
            delta["intent"] = intent
        for key in ("objectives", "constraints", "unknowns", "success_criteria", "working_set"):
            items = self._normalize_cognition_text_list(value.get(key, []), limit=10)
            if items:
                delta[key] = items
        todos = self._normalize_cognition_todos(value.get("todo", []), limit=10)
        if todos:
            delta["todo"] = todos
        return delta

    def _merge_cognition_query_state(self, base: Dict[str, Any], update: Any) -> Dict[str, Any]:
        if not self._cognition_query_state_enabled:
            return {}
        if not isinstance(base, dict):
            base = self._default_cognition_query_state("")
        delta = self._normalize_cognition_query_state_delta(update)
        if delta.get("query_rewrite"):
            base["query_rewrite"] = delta["query_rewrite"]
        if delta.get("intent"):
            base["intent"] = delta["intent"]
        for key in ("objectives", "constraints", "unknowns", "success_criteria", "working_set"):
            base[key] = self._normalize_cognition_text_list(
                list(base.get(key, [])) + list(delta.get(key, [])),
                limit=10,
            )
        base["todo"] = self._merge_cognition_todos(base.get("todo", []), delta.get("todo", []), limit=10)
        return base

    def _default_cognition_state_of_mind(self) -> Dict[str, Any]:
        if not self._cognition_state_of_mind_enabled:
            return {}
        return {
            "stance": "careful and evidence-seeking",
            "confidence": 0.35,
            "attention": [],
            "risks": [],
            "open_questions": [],
            "reflections": [],
        }

    def _normalize_cognition_state_of_mind_delta(self, value: Any) -> Dict[str, Any]:
        if not self._cognition_state_of_mind_enabled:
            return {}
        if not isinstance(value, dict):
            return {}
        delta: Dict[str, Any] = {}
        stance = str(value.get("stance") or "").strip()
        if stance:
            delta["stance"] = stance
        if "confidence" in value:
            delta["confidence"] = self._normalize_cognition_confidence(value.get("confidence"))
        for key in ("attention", "risks", "open_questions", "reflections"):
            items = self._normalize_cognition_text_list(value.get(key, []), limit=10)
            if items:
                delta[key] = items
        return delta

    def _merge_cognition_state_of_mind(self, base: Dict[str, Any], update: Any) -> Dict[str, Any]:
        if not self._cognition_state_of_mind_enabled:
            return {}
        if not isinstance(base, dict):
            base = self._default_cognition_state_of_mind()
        delta = self._normalize_cognition_state_of_mind_delta(update)
        if delta.get("stance"):
            base["stance"] = delta["stance"]
        if "confidence" in delta:
            base["confidence"] = delta["confidence"]
        for key in ("attention", "risks", "open_questions"):
            base[key] = self._normalize_cognition_text_list(
                list(base.get(key, [])) + list(delta.get(key, [])),
                limit=10,
            )
        # Reflections: keep only the most recent N entries to prevent unbounded prompt growth.
        # New reflections are appended and old ones are dropped from the front.
        _ref_limit = getattr(self, "_cognition_reflections_limit", 6)
        all_reflections = self._normalize_cognition_text_list(
            list(base.get("reflections", [])) + list(delta.get("reflections", [])),
            limit=100,
        )
        base["reflections"] = all_reflections[-_ref_limit:] if len(all_reflections) > _ref_limit else all_reflections
        return base

    def _normalize_cognition_actions(self, values: Any, limit: int = 2) -> List[Dict[str, Any]]:
        if not isinstance(values, list):
            return []
        normalized: List[Dict[str, Any]] = []
        max_items = max(0, limit)
        for item in values:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            action_type = str(item.get("action_type") or item.get("type") or "").strip().lower()
            tool_id = str(item.get("tool_id") or "").strip()
            is_tool_generation_action = action_type == "generate_tools_from_spec" or tool_id == "generate_tools_from_spec"
            if is_tool_generation_action:
                spec = item.get("spec", item.get("tool_generation_spec", item.get("args", {})))
                if not isinstance(spec, dict):
                    continue
                normalized.append({"action_type": "generate_tools_from_spec", "spec": spec, "label": label})
                if len(normalized) >= max_items:
                    break
                continue
            if not tool_id:
                continue
            args = item.get("args", {})
            if not isinstance(args, dict):
                args = {}
            normalized.append({"action_type": "tool", "tool_id": tool_id, "args": args, "label": label})
            if len(normalized) >= max_items:
                break
        return normalized

    def _normalize_cognition_skill(self, value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        name = str(value.get("name") or value.get("skill") or "").strip()
        description = str(value.get("description") or "").strip()
        when_to_use = self._normalize_cognition_text_list(value.get("when_to_use", []), limit=5)
        if not name or not description:
            return None
        raw_skill_id = str(value.get("skill_id") or "").strip().lower()
        if raw_skill_id.startswith("skill."):
            skill_id = raw_skill_id
        else:
            skill_id = self._normalize_cognition_catalog_id("skill", raw_skill_id or name)
        return {
            "skill_id": skill_id,
            "name": name,
            "description": description,
            "when_to_use": when_to_use,
            "confidence": self._normalize_cognition_confidence(value.get("confidence"), default=0.5),
        }

    def _normalize_cognition_pattern(self, value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        source = str(value.get("source") or "").strip()
        if source:
            func_def, errors = self._extract_asi_source_function(source)
            if func_def is None or errors:
                return None
            raw_pattern_id = str(value.get("pattern_id") or "").strip().lower()
            name = str(value.get("name") or "").strip() or str(func_def.name or "").strip()
            if not name and not raw_pattern_id:
                return None
            if raw_pattern_id.startswith("pattern."):
                pattern_id = raw_pattern_id
            else:
                pattern_id = self._normalize_cognition_catalog_id("pattern", raw_pattern_id or func_def.name or name)
            explicit_inputs = self._normalize_asi_inputs_needed(value.get("inputs_needed", []))
            derived_inputs, input_errors = self._derive_asi_inputs_from_source(source)
            if input_errors:
                return None
            description = str(value.get("description") or "").strip()
            when_to_use = self._normalize_cognition_text_list(value.get("when_to_use", []), limit=5)
            return {
                "pattern_id": pattern_id,
                "name": name or pattern_id,
                "description": description,
                "when_to_use": when_to_use,
                "source": source,
                "inputs_needed": self._merge_asi_source_inputs(derived_inputs, explicit_inputs),
                "confidence": self._normalize_cognition_confidence(value.get("confidence"), default=0.5),
            }
        name = str(value.get("name") or "").strip()
        description = str(value.get("description") or "").strip()
        trigger = str(value.get("trigger") or "").strip()
        if not name or not description:
            return None
        raw_pattern_id = str(value.get("pattern_id") or "").strip().lower()
        if raw_pattern_id.startswith("pattern."):
            pattern_id = raw_pattern_id
        else:
            pattern_id = self._normalize_cognition_catalog_id("pattern", raw_pattern_id or name)
        return {
            "pattern_id": pattern_id,
            "name": name,
            "description": description,
            "trigger": trigger,
            "confidence": self._normalize_cognition_confidence(value.get("confidence"), default=0.5),
        }

    def _normalize_cognition_lesson(self, value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        lesson = str(value.get("lesson") or "").strip()
        when = str(value.get("when") or "").strip()
        if not lesson:
            return None
        raw_lesson_id = str(value.get("lesson_id") or "").strip().lower()
        if raw_lesson_id.startswith("lesson."):
            lesson_id = raw_lesson_id
        else:
            lesson_id = self._normalize_cognition_catalog_id("lesson", raw_lesson_id or lesson)
        return {
            "lesson_id": lesson_id,
            "lesson": lesson,
            "when": when,
            "confidence": self._normalize_cognition_confidence(value.get("confidence"), default=0.5),
        }

    def _normalize_cognition_safety_rule(self, value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None
        rule = str(value.get("rule") or "").strip()
        rationale = str(value.get("rationale") or "").strip()
        if not rule:
            return None
        raw_rule_id = str(value.get("rule_id") or "").strip().lower()
        if raw_rule_id.startswith("safety."):
            rule_id = raw_rule_id
        else:
            rule_id = self._normalize_cognition_catalog_id("safety", raw_rule_id or rule)
        return {
            "rule_id": rule_id,
            "rule": rule,
            "rationale": rationale,
            "confidence": self._normalize_cognition_confidence(value.get("confidence"), default=0.5),
        }

    def _load_cognition_catalog(
        self,
        path: str,
        normalizer: Callable[[Any], Optional[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in data:
            entry = normalizer(item)
            if entry is not None:
                normalized.append(entry)
        return normalized

    def _write_cognition_catalog(self, path: str, entries: List[Dict[str, Any]]) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=True, indent=2)
        except Exception:
            return False
        return True

    def _upsert_cognition_catalog(
        self,
        path: str,
        entries: Any,
        normalizer: Callable[[Any], Optional[Dict[str, Any]]],
        id_key: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        current = list(self._load_cognition_catalog(path, normalizer))
        index = {
            str(item.get(id_key) or "").strip(): idx
            for idx, item in enumerate(current)
            if isinstance(item, dict) and str(item.get(id_key) or "").strip()
        }
        changed: List[Dict[str, Any]] = []
        if not isinstance(entries, list):
            return current, changed
        for raw_item in entries:
            item = normalizer(raw_item)
            if item is None:
                continue
            item_id = str(item.get(id_key) or "").strip()
            if not item_id:
                continue
            existing_idx = index.get(item_id)
            if existing_idx is None:
                current.append(item)
                index[item_id] = len(current) - 1
                changed.append(item)
                continue
            if current[existing_idx] != item:
                current[existing_idx] = item
                changed.append(item)
        if changed:
            self._write_cognition_catalog(path, current)
        return current, changed

    def _load_cognition_skills(self) -> List[Dict[str, Any]]:
        return self._load_cognition_catalog(self._cognition_skills_path, self._normalize_cognition_skill)

    def _load_cognition_patterns(self) -> List[Dict[str, Any]]:
        return self._load_cognition_catalog(self._cognition_patterns_path, self._normalize_cognition_pattern)

    def _find_cognition_pattern(
        self,
        pattern_id: str,
        patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_id = self._normalize_asi_pattern_ref(pattern_id)
        if not normalized_id:
            return None
        search_space = patterns if patterns is not None else self._load_cognition_patterns()
        for pattern in search_space:
            if not isinstance(pattern, dict):
                continue
            existing_id = self._normalize_asi_pattern_ref(pattern.get("pattern_id"))
            if existing_id == normalized_id:
                return pattern
        return None

    def _cognition_source_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        source_patterns: List[Dict[str, Any]] = []
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            source = pattern.get("source", "")
            if isinstance(source, str) and source.strip():
                source_patterns.append(pattern)
        return source_patterns

    def _load_cognition_lessons(self) -> List[Dict[str, Any]]:
        return self._load_cognition_catalog(self._cognition_lessons_path, self._normalize_cognition_lesson)

    def _load_cognition_safety_rules(self) -> List[Dict[str, Any]]:
        return self._load_cognition_catalog(
            self._cognition_safety_rules_path,
            self._normalize_cognition_safety_rule,
        )

    def _format_cognition_memory_hits(self, memory_hits: List[Dict[str, Any]], limit: int = 6) -> str:
        if not memory_hits:
            return "None"
        lines: List[str] = []
        for idx, item in enumerate(memory_hits[: max(1, limit)], start=1):
            name = str(item.get("name") or item.get("type") or "memory").strip()
            score = item.get("score")
            data = item.get("data")
            preview = self._truncate(self._safe_json(data), 260) if isinstance(data, dict) else self._truncate(str(data), 260)
            if isinstance(score, (int, float)):
                lines.append(f"{idx}) {name} score={float(score):.3f} data={preview}")
            else:
                lines.append(f"{idx}) {name} data={preview}")
        return "\n".join(lines)

    def _format_cognition_catalog_brief(self, entries: List[Dict[str, Any]], kind: str, limit: int = 6) -> str:
        if not entries:
            return "None"
        lines: List[str] = []
        for idx, item in enumerate(entries[: max(1, limit)], start=1):
            if kind == "skill":
                lines.append(
                    f"{idx}) {item.get('name', '')}: {item.get('description', '')} "
                    f"when={self._safe_json(item.get('when_to_use', []))}"
                )
            elif kind == "pattern":
                source = item.get("source", "")
                if isinstance(source, str) and source.strip():
                    signature = self._format_asi_pattern_signature(item.get("pattern_id"), item.get("inputs_needed", []))
                    description = str(item.get("description") or "").strip()
                    when_to_use = item.get("when_to_use", [])
                    when_text = (
                        "; ".join(str(part).strip() for part in when_to_use if str(part).strip())
                        if isinstance(when_to_use, list)
                        else ""
                    )
                    line = f"{idx}) {signature}"
                    if description:
                        line += f": {description}"
                    if when_text:
                        line += f" when={when_text}"
                    lines.append(line)
                else:
                    lines.append(
                        f"{idx}) {item.get('name', '')}: {item.get('description', '')} "
                        f"trigger={item.get('trigger', '')}"
                    )
            elif kind == "lesson":
                lines.append(f"{idx}) {item.get('lesson', '')} when={item.get('when', '')}")
            elif kind == "safety":
                lines.append(f"{idx}) {item.get('rule', '')} rationale={item.get('rationale', '')}")
        return "\n".join(lines) if lines else "None"

    def _format_cognition_callable_patterns_brief(self, patterns: List[Dict[str, Any]], limit: int = 6) -> str:
        return self._format_cognition_catalog_brief(self._cognition_source_patterns(patterns), "pattern", limit=limit)

    def _build_cognition_init_prompt(
        self,
        user_text: str,
        context_packet: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> str:
        tools_block = self._format_tools_for_agent(tools)
        memory_block = self._format_cognition_memory_hits(memory_hits)
        context_summary = str(context_packet.get("summary") or "").strip() or "None"

        if self._cognition_mode == "small":
            return (
                "COGNITION INIT\n"
                f"Respond with JSON: {self._build_cognition_init_schema()}\n"
                f"CONTEXT: {context_summary}\n"
                f"MEMORY: {memory_block}\n"
                f"TOOLS: {tools_block}\n"
                f"USER: {user_text}\n"
            )
        state_targets: List[str] = []
        if self._cognition_query_state_enabled:
            state_targets.append("1) query_state: a rigorous representation of the task.")
        if self._cognition_state_of_mind_enabled:
            label = "2" if state_targets else "1"
            state_targets.append(
                f"{label}) state_of_mind: the current internal stance used to reason carefully."
            )
        rules_block = ""
        if self._cognition_query_state_enabled:
            rules_block += "- query_state must stay close to the user's actual request.\n"
        if self._cognition_state_of_mind_enabled:
            rules_block += "- state_of_mind is internal and should capture uncertainty, risks, and attention focus.\n"
        rules_block += "- tools_needed: true if the agent may need to use tools to fulfill the request, false if it can be answered purely from internal knowledge.\n"
        rules_block += "- Keep lists concise.\n- Do not add keys outside the schema.\n\n"
        return (
            "COGNITION INITIALIZER\n\n"
            "Build internal state for the request.\n"
            f"{chr(10).join(state_targets)}\n\n"
            "Respond with STRICT JSON only:\n"
            f"{self._build_cognition_init_schema()}\n\n"
            "Rules:\n"
            f"{rules_block}"
            "CONTEXT SUMMARY:\n"
            f"{context_summary}\n\n"
            "RELEVANT MEMORY:\n"
            f"{memory_block}\n\n"
            "AVAILABLE TOOLS:\n"
            f"{tools_block}\n\n"
            "USER REQUEST:\n"
            f"{user_text}\n"
        )

    def _build_cognition_tool_selection_prompt(
        self,
        user_text: str,
        context_packet: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> str:
        context_summary = str(context_packet.get("summary") or "").strip() or "None"

        if self._cognition_mode == "small":
            return (
                "TOOL SELECTOR\n"
                "JSON: {\"type\":\"tool_selection\",\"selected_tool_ids\":[\"id\"],\"reason\":\"...\"}\n"
                f"CONTEXT: {context_summary}\n"
                f"MEMORY: {self._format_cognition_memory_hits(memory_hits)}\n"
                f"USER: {user_text}\n"
                f"TOOLS: {self._format_tools_for_agent(tools)}\n"
            )

        return (
            "COGNITION TOOL SELECTOR\n\n"
            "Select the minimal set of tools relevant to the request from the candidate list.\n\n"
            "Respond with STRICT JSON only:\n"
            "{\"type\":\"tool_selection\",\"selected_tool_ids\":[\"tool.id\"],\"reason\":\"...\"}\n\n"
            "Rules:\n"
            "- Choose only from the listed candidate tools.\n"
            "- Keep the set minimal but sufficient for the task.\n"
            "- Return an empty selected_tool_ids list when no listed tool is relevant.\n"
            "- Do not add keys outside the schema.\n\n"
            "CONTEXT SUMMARY:\n"
            f"{context_summary}\n\n"
            "RELEVANT MEMORY:\n"
            f"{self._format_cognition_memory_hits(memory_hits)}\n\n"
            "USER REQUEST:\n"
            f"{user_text}\n\n"
            "CANDIDATE TOOLS:\n"
            f"{self._format_tools_for_agent(tools)}\n"
        )

    def _build_cognition_tool_category_selection_prompt(
        self,
        user_text: str,
        context_packet: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
        category_map: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        context_summary = str(context_packet.get("summary") or "").strip() or "None"

        if self._cognition_mode == "small":
            return (
                "CATEGORY SELECTOR\n"
                "JSON: {\"type\":\"tool_selection\",\"selected_category_ids\":[\"id\"],\"selected_tool_ids\":[],\"reason\":\"...\"}\n"
                f"CONTEXT: {context_summary}\n"
                f"MEMORY: {self._format_cognition_memory_hits(memory_hits)}\n"
                f"USER: {user_text}\n"
                f"CATEGORIES: {self._format_tool_categories_for_agent(category_map)}\n"
            )

        return (
            "COGNITION TOOL SELECTOR\n\n"
            "Select the minimal set of tool categories relevant to the request. "
            "Every selected category will forward all active tools under that category.\n\n"
            "Respond with STRICT JSON only:\n"
            "{\"type\":\"tool_selection\",\"selected_category_ids\":[\"ui\"],"
            "\"selected_tool_ids\":[],\"reason\":\"...\"}\n\n"
            "Rules:\n"
            "- Choose only from the listed candidate categories and tools.\n"
            "- Select every category needed to give the agent enough tools for the task.\n"
            "- Use selected_tool_ids only for individual tools that are needed without their whole category.\n"
            "- Return empty selected_category_ids and selected_tool_ids lists when no listed category or tool is relevant.\n"
            "- Do not add keys outside the schema.\n\n"
            "CONTEXT SUMMARY:\n"
            f"{context_summary}\n\n"
            "RELEVANT MEMORY:\n"
            f"{self._format_cognition_memory_hits(memory_hits)}\n\n"
            "USER REQUEST:\n"
            f"{user_text}\n\n"
            "CANDIDATE TOOL CATEGORIES:\n"
            f"{self._format_tool_categories_for_agent(category_map)}\n"
        )

    def _format_tool_categories_for_agent(self, category_map: Dict[str, List[Dict[str, Any]]]) -> str:
        if not category_map:
            return "None"
        lines = []
        for category_id, tools in category_map.items():
            tool_ids = [
                str(tool.get("tool_id") or "").strip()
                for tool in tools
                if str(tool.get("tool_id") or "").strip()
            ]
            names = [
                str(tool.get("name") or "").strip()
                for tool in tools
                if str(tool.get("name") or "").strip()
            ]
            label = ", ".join(tool_ids)
            name_hint = "; names=" + ", ".join(names[:8]) if names else ""
            lines.append(f"- {category_id}: {len(tool_ids)} tools; tool_ids=[{label}]{name_hint}")
        return "\n".join(lines)

    def _build_cognition_pattern_router_prompt(
        self,
        user_or_query_state: Any,
        query_or_state_of_mind: Any,
        state_or_skills: Any,
        skills_or_patterns: Any,
        patterns_or_safety_rules: Any,
        safety_rules: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if safety_rules is None:
            query_state = user_or_query_state if isinstance(user_or_query_state, dict) else {}
            state_of_mind = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            skills = state_or_skills if isinstance(state_or_skills, list) else []
            patterns = skills_or_patterns if isinstance(skills_or_patterns, list) else []
            safety = patterns_or_safety_rules if isinstance(patterns_or_safety_rules, list) else []
        else:
            query_state = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            state_of_mind = state_or_skills if isinstance(state_or_skills, dict) else {}
            skills = skills_or_patterns if isinstance(skills_or_patterns, list) else []
            patterns = patterns_or_safety_rules if isinstance(patterns_or_safety_rules, list) else []
            safety = safety_rules

        if self._cognition_mode == "small":
            return (
                "PATTERN ROUTER\n"
                "JSON: {\"type\":\"pattern_route\",\"decision\":\"use\",\"pattern_id\":\"...\",\"reason\":\"...\"} OR {\"type\":\"pattern_route\",\"decision\":\"route\",\"reason\":\"...\"}\n"
                f"STATE: {query_state} {state_of_mind}\n"
                f"SKILLS: {self._format_cognition_catalog_brief(skills, 'skill')}\n"
                f"PATTERNS: {self._format_cognition_callable_patterns_brief(patterns)}\n"
            )

        return (
            "COGNITION PATTERN ROUTER\n\n"
            "Decide whether one callable cognition pattern should handle the request now,\n"
            "or whether the request should continue to the thinking-systems router.\n\n"
            "Respond with STRICT JSON only using one of:\n"
            "{\"type\":\"pattern_route\",\"decision\":\"use\",\"pattern_id\":\"...\",\"reason\":\"...\"}\n"
            "{\"type\":\"pattern_route\",\"decision\":\"route\",\"reason\":\"...\"}\n\n"
            "Rules:\n"
            "- Use only when one listed callable pattern can likely satisfy the request directly after normal input binding and execution.\n"
            "- Route when no listed callable pattern is a clear fit, or when broader cognition reasoning is still needed.\n"
            "- Choose only from listed callable patterns.\n"
            "- pattern_id may be returned either as the bare id or the call form.\n"
            "- Do not add extra keys.\n\n"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2400, state_limit=2000)}"
            "REUSABLE SKILLS:\n"
            f"{self._format_cognition_catalog_brief(skills, 'skill')}\n\n"
            "PATTERNS:\n"
            f"{self._format_cognition_callable_patterns_brief(patterns)}\n\n"
            "SAFETY RULES:\n"
            f"{self._format_cognition_catalog_brief(safety, 'safety')}\n"
        )

    def _build_cognition_route_prompt(
        self,
        user_or_query_state: Any,
        query_or_state_of_mind: Any,
        state_or_memory_hits: Any,
        memory_or_tools: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if tools is None:
            query_state = user_or_query_state if isinstance(user_or_query_state, dict) else {}
            state_of_mind = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            memory_hits = state_or_memory_hits if isinstance(state_or_memory_hits, list) else []
            active_tools = memory_or_tools if isinstance(memory_or_tools, list) else None
        else:
            query_state = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            state_of_mind = state_or_memory_hits if isinstance(state_or_memory_hits, dict) else {}
            memory_hits = memory_or_tools if isinstance(memory_or_tools, list) else []
            active_tools = tools
        modes_text = "system1|system2|system3" if self._cognition_system3_available() else "system1|system2"
        system3_rules = (
            "Choose system3 when the task would benefit from multiple competing strategies,\n"
            "peer scrutiny, and revision before execution begins.\n\n"
            if self._cognition_system3_available()
            else ""
        )
        system2_rules = (
            "Choose system2 when the task is ambiguous, multi-step, high-risk, or benefits from a reflection loop,\n"
            "todo updates, and iterative state changes.\n\n"
            if self._cognition_query_state_enabled
            else
            "Choose system2 when the task is ambiguous, multi-step, high-risk, or benefits from a reflection loop\n"
            "and iterative state changes.\n\n"
        )
        tools_block = ""
        if active_tools:
            tools_block = (
                "AVAILABLE TOOLS:\n"
                f"{self._format_tools_for_agent(active_tools)}\n\n"
            )

        if self._cognition_mode == "small":
            return (
                "ROUTER\n"
                f"JSON: {{\\\"type\\\":\\\"route\\\",\\\"thinking_mode\\\":\\\"{modes_text}\\\",\\\"reason\\\":\\\"...\\\",\\\"generate_tools_first\\\":false}}\n"
                f"USER/STATE: {query_state} {state_of_mind}\n"
                f"MEMORY: {self._format_cognition_memory_hits(memory_hits)}\n"
                f"{tools_block}"
            )

        return (
            "COGNITION THINKING ROUTER\n\n"
            "Choose which thinking regime should handle the request.\n\n"
            "Respond with STRICT JSON only using one of:\n"
            f"{{\"type\":\"route\",\"thinking_mode\":\"{modes_text}\",\"reason\":\"...\",\"generate_tools_first\":false}}\n"
            f"{{\"type\":\"route\",\"thinking_mode\":\"{modes_text}\",\"reason\":\"...\",\"generate_tools_first\":true,"
            "\"tool_generation_spec\":{\"request\":\"...\",\"namespace_prefix\":\"...\",\"max_tools\":2}}}\n\n"
            "Choose system1 when the task can be handled through an action-oriented execution loop,\n"
            "using iterative tool calls and observations without needing deeper reflective state updates.\n"
            f"{system2_rules}"
            f"{system3_rules}"
            "Set generate_tools_first to true only when the currently listed tools clearly lack a stable reusable capability,\n"
            "and generating that capability before execution would materially improve the episode.\n"
            "When generate_tools_first is true, thinking_mode must name the regime that should continue after generation completes.\n"
            "tool_generation_spec must stay minimal, reusable, and limited to the fewest tools needed.\n"
            "Leave generate_tools_first false when existing tools or normal reasoning loops are sufficient.\n\n"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2500, state_limit=2000)}"
            "RELEVANT MEMORY:\n"
            f"{self._format_cognition_memory_hits(memory_hits)}\n\n"
            f"{tools_block}"
            "Focus only on whether the task needs the faster execution loop, the more reflective loop,\n"
            "or a small capability bootstrap before that loop begins.\n"
        )


    def _build_cognition_system1_prompt(
        self,
        query_state: Dict[str, Any],
        state_of_mind: Dict[str, Any],
        observations: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        cycle_index: int = 1,
        strategy_brief: str = "",
    ) -> str:
        schemas = self._build_cognition_system_schemas()
        allow_clarify = self._cognition_system1_clarify_enabled
        clarify_schema = f"{schemas['clarify']}\n" if allow_clarify else ""
        intro = (
            "Either answer directly, ask one blocking question, or propose a small action set for the acting section.\n\n"
            if allow_clarify
            else "Either answer directly or propose a small action set for the acting section.\n\n"
        )
        clarify_rule = "" if allow_clarify else "- Do not return clarify; proceed with final or act using the available context.\n"
        strategy_block = f"COUNCIL STRATEGY:\n{strategy_brief}\n\n" if strategy_brief else ""
        special_actions_block = self._format_cognition_special_actions()
        rules_block = (
            "- state deltas must use only existing query/state_of_mind keys.\n\n"
            if self._cognition_query_state_enabled or self._cognition_state_of_mind_enabled
            else "\n"
        )
        tools_block = ""
        if tools:
            tools_block = (
                "AVAILABLE TOOLS:\n"
                f"{self._format_tools_for_agent(tools)}\n\n"
            )

        if self._cognition_mode == "small":
            return (
                "SYSTEM1\n"
                f"JSON: {schemas['final']} OR {schemas['act']}\n"
                f"CYCLE: {cycle_index}\n"
                f"STATE: {query_state} {state_of_mind}\n"
                f"OBSERVATIONS: {self._format_agent_observations(observations)}\n"
                f"{tools_block}"
            )

        return (
            "COGNITION SYSTEM1\n\n"
            f"{intro}"
            "Respond with STRICT JSON only using one of:\n"
            f"{schemas['final']}\n"
            f"{clarify_schema}"
            f"{schemas['act']}\n\n"
            "Rules:\n"
            "- Make progress in exactly one cycle.\n"
            "- Use observations from previous actions before proposing more actions.\n"
            "- If actions are used, keep them minimal and concrete.\n"
            "- Use listed tools for normal execution.\n"
            "- Use generate_tools_from_spec only when the current tool set clearly lacks a stable capability and a reusable tool is justified.\n"
            f"{clarify_rule}"
            "- Do not fabricate results.\n"
            f"{rules_block}"
            f"CYCLE INDEX: {cycle_index}\n\n"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2600, state_limit=2200)}"
            f"{strategy_block}"
            "OBSERVATIONS:\n"
            f"{self._format_agent_observations(observations)}\n\n"
            f"{tools_block}"
            "SPECIAL ORCHESTRATION ACTIONS:\n"
            f"{special_actions_block}\n"
        )

    def _build_cognition_system0_prompt(
        self,
        query_state: Dict[str, Any],
        state_of_mind: Dict[str, Any],
        observations: List[Dict[str, Any]],
        cycle_index: int = 1,
    ) -> str:
        schemas = self._build_cognition_system_schemas()
        rules_block = (
            "- state deltas must use only existing query/state_of_mind keys.\n\n"
            if self._cognition_query_state_enabled or self._cognition_state_of_mind_enabled
            else "\n"
        )

        if self._cognition_mode == "small":
            return (
                "SYSTEM0\n"
                f"JSON: {schemas['final']}\n"
                f"STATE: {query_state} {state_of_mind}\n"
                f"OBSERVATIONS: {self._format_agent_observations(observations)}\n"
            )

        return (
            "COGNITION SYSTEM0\n\n"
            "Handle a trivial request with the fastest safe path and return the answer directly.\n\n"
            "Respond with STRICT JSON only using:\n"
            f"{schemas['final']}\n\n"
            "Rules:\n"
            "- Prefer a direct answer when grounded by the request or current observations.\n"
            "- Do not ask clarifying questions or propose actions.\n"
            "- Do not fabricate results.\n"
            f"{rules_block}"
            f"CYCLE INDEX: {cycle_index}\n\n"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=1600, state_limit=1200)}"
            "OBSERVATIONS:\n"
            f"{self._format_agent_observations(observations)}\n\n"
        )


    def _build_cognition_system2_prompt(
        self,
        user_or_query_state: Any,
        query_or_state_of_mind: Any,
        state_or_observations: Any,
        observations_or_trace: Any,
        trace_or_tools: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        cycle_index: int = 1,
        strategy_brief: str = "",
    ) -> str:
        if isinstance(user_or_query_state, str):
            query_state = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            state_of_mind = state_or_observations if isinstance(state_or_observations, dict) else {}
            observations = observations_or_trace if isinstance(observations_or_trace, list) else []
            thinking_trace = trace_or_tools if isinstance(trace_or_tools, list) else []
            active_tools = tools
        else:
            query_state = user_or_query_state if isinstance(user_or_query_state, dict) else {}
            state_of_mind = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            observations = state_or_observations if isinstance(state_or_observations, list) else []
            thinking_trace = observations_or_trace if isinstance(observations_or_trace, list) else []
            active_tools = trace_or_tools if isinstance(trace_or_tools, list) else None
            if tools is not None:
                active_tools = tools
        thinking_trace_block = self._truncate(self._safe_json(thinking_trace), 3000) if thinking_trace else "[]"
        schemas = self._build_cognition_system_schemas()
        allow_clarify = self._cognition_system2_clarify_enabled
        clarify_schema = f"{schemas['clarify']}\n" if allow_clarify else ""
        clarify_rule = "" if allow_clarify else "- Do not return clarify; continue with step or final using the available context.\n"
        strategy_block = f"COUNCIL STRATEGY:\n{strategy_brief}\n\n" if strategy_brief else ""
        special_actions_block = self._format_cognition_special_actions()
        intro_block = (
            "Run one deliberate cognition cycle: reflect, update state, update todo, and optionally plan actions.\n\n"
            if self._cognition_query_state_enabled
            else
            "Run one deliberate cognition cycle: reflect, update state, and optionally plan actions.\n\n"
        )
        rules_block = "- Keep todo updates precise.\n\n" if self._cognition_query_state_enabled else "\n"
        tools_block = ""
        if active_tools:
            tools_block = (
                "AVAILABLE TOOLS:\n"
                f"{self._format_tools_for_agent(active_tools)}\n\n"
            )

        if self._cognition_mode == "small":
            return (
                "SYSTEM2\n"
                f"JSON: {schemas['step']} OR {schemas['final']}\n"
                f"CYCLE: {cycle_index}\n"
                f"STATE: {query_state} {state_of_mind}\n"
                f"OBSERVATIONS: {self._format_agent_observations(observations)}\n"
                f"TRACE: {thinking_trace_block}\n"
                f"{tools_block}"
            )

        return (
            "COGNITION SYSTEM2\n\n"
            f"{intro_block}"
            "Respond with STRICT JSON only using one of:\n"
            f"{schemas['step']}\n"
            f"{schemas['final']}\n"
            f"{clarify_schema}\n"
            "Rules:\n"
            "- Make progress in exactly one cycle.\n"
            "- reflection should capture why the next step is justified.\n"
            "- Use actions only when a tool can reduce uncertainty or complete the task.\n"
            "- Use listed tools for normal execution.\n"
            "- Use generate_tools_from_spec only when the current tool set clearly lacks a stable capability and a reusable tool is justified.\n"
            f"{clarify_rule}"
            f"{rules_block}"
            f"CYCLE INDEX: {cycle_index}\n\n"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2600, state_limit=2200)}"
            f"{strategy_block}"
            "OBSERVATIONS:\n"
            f"{self._format_agent_observations(observations)}\n\n"
            "THINKING TRACE:\n"
            f"{thinking_trace_block}\n\n"
            f"{tools_block}"
            "SPECIAL ORCHESTRATION ACTIONS:\n"
            f"{special_actions_block}\n"
        )


    def _build_cognition_system3_proposal_prompt(
        self,
        query_state: Dict[str, Any],
        state_of_mind: Dict[str, Any],
        peer: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        safety_rules: List[Dict[str, Any]],
    ) -> str:
        fields = [
            "\"type\":\"proposal\"",
            "\"peer_id\":\"...\"",
            "\"recommended_mode\":\"system1|system2\"",
            "\"strategy\":\"...\"",
            "\"argument\":\"...\"",
        ]
        if self._cognition_query_state_enabled:
            fields.append(f"\"query_state\":{{{self._cognition_query_state_schema()}}}")
        if self._cognition_state_of_mind_enabled:
            fields.append(f"\"state_of_mind\":{{{self._cognition_state_of_mind_schema()}}}")
        schema = self._compose_cognition_schema(fields)
        role = str(peer.get("role") or "").strip()
        role_block = f"ROLE:\n{role}\n\n" if role else ""
        return (
            "COGNITION SYSTEM3 PROPOSAL\n\n"
            "Devise a complete execution strategy, defend it, and choose whether system1 or system2 should execute it.\n\n"
            "Respond with STRICT JSON only:\n"
            f"{schema}\n\n"
            "Rules:\n"
            "- Produce one complete strategy, not multiple options.\n"
            "- strategy should be concise but sufficient for execution handoff.\n"
            "- argument should defend why this strategy is better than obvious alternatives.\n"
            "- Use only relevant grounded information from the provided context.\n"
            "- Do not mention hidden chain-of-thought; provide a compact strategic argument.\n\n"
            f"{role_block}"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2600, state_limit=2000)}"
            "RELEVANT MEMORY:\n"
            f"{self._format_cognition_memory_hits(memory_hits)}\n\n"
            "AVAILABLE TOOLS:\n"
            f"{self._format_tools_for_agent(tools)}\n\n"
            "SAFETY RULES:\n"
            f"{self._format_cognition_catalog_brief(safety_rules, 'safety')}\n"
        )

    def _build_cognition_system3_review_prompt(
        self,
        query_state: Dict[str, Any],
        state_of_mind: Dict[str, Any],
        critic: Dict[str, Any],
        proposal: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        safety_rules: List[Dict[str, Any]],
    ) -> str:
        role = str(critic.get("role") or "").strip()
        role_block = f"ROLE:\n{role}\n\n" if role else ""
        return (
            "COGNITION SYSTEM3 SCRUTINY\n\n"
            "Score the proposal and provide one concise critique.\n\n"
            "Respond with STRICT JSON only:\n"
            "{\"type\":\"review\",\"critic_id\":\"...\",\"proposal_peer_id\":\"...\",\"score\":0.0,\"comment\":\"...\"}\n\n"
            "Rules:\n"
            "- score must be on a 0 to 10 scale.\n"
            "- comment should focus on the main strengths and weaknesses.\n"
            "- Evaluate only the proposal shown.\n"
            "- Do not score yourself if the critic and proposer are the same peer.\n\n"
            f"{role_block}"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2600, state_limit=2000)}"
            "PROPOSAL UNDER REVIEW:\n"
            f"{self._truncate(self._safe_json(proposal), 2600)}\n\n"
            "RELEVANT MEMORY:\n"
            f"{self._format_cognition_memory_hits(memory_hits)}\n\n"
            "AVAILABLE TOOLS:\n"
            f"{self._format_tools_for_agent(tools)}\n\n"
            "SAFETY RULES:\n"
            f"{self._format_cognition_catalog_brief(safety_rules, 'safety')}\n"
        )

    def _build_cognition_system3_adjust_prompt(
        self,
        query_state: Dict[str, Any],
        state_of_mind: Dict[str, Any],
        peer: Dict[str, Any],
        current_proposal: Dict[str, Any],
        peer_proposals: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
    ) -> str:
        fields = [
            "\"type\":\"proposal\"",
            "\"peer_id\":\"...\"",
            "\"recommended_mode\":\"system1|system2\"",
            "\"strategy\":\"...\"",
            "\"argument\":\"...\"",
            "\"revision_note\":\"...\"",
        ]
        if self._cognition_query_state_enabled:
            fields.append(f"\"query_state\":{{{self._cognition_query_state_schema()}}}")
        if self._cognition_state_of_mind_enabled:
            fields.append(f"\"state_of_mind\":{{{self._cognition_state_of_mind_schema()}}}")
        schema = self._compose_cognition_schema(fields)
        role = str(peer.get("role") or "").strip()
        role_block = f"ROLE:\n{role}\n\n" if role else ""
        return (
            "COGNITION SYSTEM3 ADJUSTMENT\n\n"
            "Revise your strategy after reading peer proposals and scrutiny.\n\n"
            "Respond with STRICT JSON only:\n"
            f"{schema}\n\n"
            "Rules:\n"
            "- You may keep your original strategy if it still stands.\n"
            "- revision_note should summarize what changed, if anything.\n"
            "- Keep the output focused on the final revised strategy.\n\n"
            f"{role_block}"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2600, state_limit=2000)}"
            "YOUR CURRENT PROPOSAL:\n"
            f"{self._truncate(self._safe_json(current_proposal), 2200)}\n\n"
            "OTHER PEER PROPOSALS:\n"
            f"{self._truncate(self._safe_json(peer_proposals), 2600)}\n\n"
            "SCRUTINY:\n"
            f"{self._truncate(self._safe_json(reviews), 2600)}\n"
        )

    def _build_cognition_final_prompt(
        self,
        user_or_query_state: Any,
        query_or_state_of_mind: Any,
        state_or_observations: Any,
        observations_or_trace: Any,
        thinking_trace_arg: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if isinstance(user_or_query_state, str):
            query_state = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            state_of_mind = state_or_observations if isinstance(state_or_observations, dict) else {}
            observations = observations_or_trace if isinstance(observations_or_trace, list) else []
            thinking_trace = thinking_trace_arg if isinstance(thinking_trace_arg, list) else []
        else:
            query_state = user_or_query_state if isinstance(user_or_query_state, dict) else {}
            state_of_mind = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            observations = state_or_observations if isinstance(state_or_observations, list) else []
            thinking_trace = observations_or_trace if isinstance(observations_or_trace, list) else []
            
        allow_clarify = self._cognition_system1_clarify_enabled or self._cognition_system2_clarify_enabled
        clarify_schema = "{\"type\":\"clarify\",\"thought\":\"...\",\"question\":\"...\"}\n" if allow_clarify else ""
        clarify_rule = (
            "- If key information is still missing, return clarify instead of guessing.\n"
            if allow_clarify
            else "- Do not return clarify; produce a final answer using the available context.\n"
        )

        return (
            "COGNITION FINALIZER\n\n"
            "After thinking and acting, produce the best user-facing result.\n\n"
            "Respond with STRICT JSON only using one of:\n"
            "{\"type\":\"final\",\"thought\":\"...\",\"text\":\"...\"}\n"
            f"{clarify_schema}\n"
            "Rules:\n"
            "- Use only grounded information from the user request, state, and observations.\n"
            f"{clarify_rule}"
            "- Do not reveal internal state_of_mind unless the user explicitly asks for it.\n\n"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2600, state_limit=2200)}"
            "OBSERVATIONS:\n"
            f"{self._format_agent_observations(observations)}\n\n"
            "THINKING TRACE:\n"
            f"{self._truncate(self._safe_json(thinking_trace), 3200)}\n"
        )

    def _build_cognition_distill_prompt(
        self,
        user_or_result_payload: Any,
        result_or_query_state: Any,
        query_or_state_of_mind: Any,
        state_or_observations: Any,
        observations_or_trace: Any,
        trace_or_tools: List[Dict[str, Any]],
        tools_or_patterns: List[Dict[str, Any]],
        patterns_arg: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if isinstance(user_or_result_payload, str):
            result_payload = result_or_query_state if isinstance(result_or_query_state, dict) else {}
            query_state = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            state_of_mind = state_or_observations if isinstance(state_or_observations, dict) else {}
            observations = observations_or_trace if isinstance(observations_or_trace, list) else []
            thinking_trace = trace_or_tools if isinstance(trace_or_tools, list) else []
            tools = tools_or_patterns if isinstance(tools_or_patterns, list) else []
            patterns = patterns_arg if isinstance(patterns_arg, list) else []
        else:
            result_payload = user_or_result_payload if isinstance(user_or_result_payload, dict) else {}
            query_state = result_or_query_state if isinstance(result_or_query_state, dict) else {}
            state_of_mind = query_or_state_of_mind if isinstance(query_or_state_of_mind, dict) else {}
            observations = state_or_observations if isinstance(state_or_observations, list) else []
            thinking_trace = observations_or_trace if isinstance(observations_or_trace, list) else []
            tools = trace_or_tools if isinstance(trace_or_tools, list) else []
            patterns = tools_or_patterns if isinstance(tools_or_patterns, list) else []
        return (
            "COGNITION DISTILLATION\n\n"
            "After the episode, distill reusable knowledge from the run.\n\n"
            "This is not the user-facing answer step. Do not return final/clarify/act/step JSON.\n"
            "Respond with STRICT JSON only:\n"
            "{\"type\":\"distill\","
            "\"reusable_skills\":[{\"name\":\"...\",\"description\":\"...\",\"when_to_use\":[\"...\"],\"confidence\":0.0}],"
            "\"memory_facts\":[{\"name\":\"...\",\"value\":\"...\",\"confidence\":0.0}],"
            "\"patterns\":[{\"pattern_id\":\"...\",\"name\":\"...\",\"description\":\"...\",\"when_to_use\":[\"...\"],"
            "\"source\":\"def pattern_name(arg1, arg2=...):\\n    result = tool.namespace(arg=value)\\n    return result\","
            "\"confidence\":0.0}],"
            "\"failure_lessons\":[{\"lesson\":\"...\",\"when\":\"...\",\"confidence\":0.0}],"
            "\"safety_rules\":[{\"rule\":\"...\",\"rationale\":\"...\",\"confidence\":0.0}]}\n\n"
            "Rules:\n"
            "- The top-level type must be exactly \"distill\".\n"
            "- The only top-level keys allowed are type, reusable_skills, memory_facts, patterns, failure_lessons, and safety_rules.\n"
            "- Include only reusable items.\n"
            "- Keep memory_facts stable and specific.\n"
            "- Put ephemeral run details in neither memory_facts nor reusable_skills.\n"
            "- patterns must be reusable source-backed functions, not prose descriptions.\n"
            "- A distilled pattern may call available tools directly and may call listed callable patterns via pat.<pattern_id>(...).\n"
            "- Do not output a pattern if no reusable source-backed function clearly emerged from the run.\n"
            "- If a category has nothing reusable, return an empty list for it.\n\n"
            "Pattern source shape:\n"
            f"{self._asi_pattern_source_shape()}\n\n"
            "Pattern source rules:\n"
            f"{self._asi_pattern_generation_rules_text()}\n"
            #"AVAILABLE TOOLS:\n"
            #f"{self._format_tools_for_agent(tools)}\n\n"
            "CALLABLE PATTERNS:\n"
            f"{self._format_cognition_callable_patterns_brief(patterns)}\n\n"
            #"RESULT:\n"
            #f"{self._truncate(self._safe_json(result_payload), 1600)}\n\n"
            f"{self._format_cognition_optional_state_sections(query_state, state_of_mind, query_limit=2200, state_limit=2200)}"
            "OBSERVATIONS:\n"
            f"{self._format_agent_observations(observations)}\n\n"
            "THINKING TRACE:\n"
            f"{self._truncate(self._safe_json(thinking_trace), 3200)}\n"
        )

    def _build_final_response_critic_prompt(
        self,
        query_state: Optional[Dict[str, Any]],
        candidate_text: str,
        execution_evidence: Optional[Dict[str, Any]] = None,
    ) -> str:
        candidate_block = self._truncate(candidate_text, 3000)
        evidence_block = self._format_execution_evidence_for_critic(execution_evidence)
        actual_query_state = query_state
        if not actual_query_state and execution_evidence and execution_evidence.get("entity_kind") == "cognition_episode":
            actual_query_state = execution_evidence.get("entity", {}).get("query_state")
        
        if isinstance(actual_query_state, dict) and actual_query_state:
            task_block = self._format_cognition_optional_state_sections(
                actual_query_state, None, query_limit=2600, state_limit=0
            )
        else:
            task_block = "Task description not available.\n\n"
            
        return (
            "Critique whether the candidate response fully satisfies the user request.\n\n"
            "Respond with STRICT JSON only:\n"
            "{\"fulfilled\":true,\"confidence\":0.0,\"issues\":[\"...\"],\"fix_instructions\":[\"...\"]}\n\n"
            "Rules:\n"
            "- fulfilled: true only if the request is fully satisfied as stated.\n"
            "- If key content is missing, unclear, wrong, or unresolved placeholders remain, set fulfilled=false.\n"
            "- Use execution evidence when available to check factual and logical correctness.\n"
            "- When execution evidence includes tool results, prioritize those results over model prior knowledge.\n"
            "- Do not flag a response as wrong if it correctly reflects tool outputs.\n"
            "- confidence: number in [0,1].\n"
            "- issues: 0-6 concrete failures.\n"
            "- fix_instructions: 0-6 actionable instructions for a retry.\n"
            "- Do not add extra keys.\n\n"
            f"{task_block}"
            "Candidate response:\n"
            f"{candidate_block}\n\n"
            "Execution evidence (optional):\n"
            f"{evidence_block}\n"
        )


    def _build_tool_codegen_critic_prompt(
        self,
        tool_id: str,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        code: str,
        dependencies: List[str],
    ) -> str:
        input_schema_text = self._truncate(self._safe_json(input_schema), 3000)
        output_schema_text = self._truncate(self._safe_json(output_schema), 3000)
        deps_text = self._truncate(self._safe_json(dependencies), 1200)
        code_text = self._truncate(code if isinstance(code, str) else "", 9000)
        return (
            "Critique whether this generated tool candidate should be approved for runtime registration.\n\n"
            "Respond with STRICT JSON only:\n"
            "{\"fulfilled\":true,\"confidence\":0.0,\"issues\":[\"...\"],\"fix_instructions\":[\"...\"]}\n\n"
            "Rules:\n"
            "- fulfilled=true only if the candidate is production-ready for this request.\n"
            "- Reject if the code likely fails at runtime (undefined variables, missing imports, wrong return contract).\n"
            "- Reject if input/output schemas are invalid or inconsistent with the requested capability.\n"
            "- Reject if path handling likely breaks workspace:/ usage when path-like args are present.\n"
            "- confidence: number in [0,1].\n"
            "- issues: 0-6 concrete failures.\n"
            "- fix_instructions: 0-6 actionable instructions for a full regeneration.\n"
            "- Do not add extra keys.\n\n"
            "Tool request:\n"
            f"- tool_id: {tool_id}\n"
            f"- description: {description}\n\n"
            "Candidate input_schema:\n"
            f"{input_schema_text}\n\n"
            "Candidate output_schema:\n"
            f"{output_schema_text}\n\n"
            "Candidate dependencies:\n"
            f"{deps_text}\n\n"
            "Candidate code (function body only):\n"
            f"{code_text}\n"
        )


    def _asi_pattern_source_shape(self) -> str:
        return (
            "{"
            "\"pattern_id\":\"...\","
            "\"name\":\"...\","
            "\"when_to_use\":[\"...\"],"
            "\"source\":\"def pattern_name(arg1, arg2=...):\\n    result = tool.namespace(arg=value)\\n    return result\""
            "}"
        )

    def _asi_pattern_generation_rules_text(self) -> str:
        return (
            "- Output source form only.\n"
            "- Source must contain exactly one Python-like function.\n"
            "- pattern_id must be a short reusable identifier in lowercase snake_case.\n"
            "- name should be a short human-readable title.\n"
            "- when_to_use should contain 1-3 short reusable cases, not request-specific literals.\n"
            "- Put required inputs in the function signature.\n"
            "- Put request-specific values in parameters or defaults, not hard-coded prose.\n"
            "- End the function with an explicit return of the final user-facing result.\n"
            "- Prefer the smallest pattern that can solve the task.\n"
            "- Reuse existing callable patterns or direct tool calls when they already fit.\n"
            "- Tool calls must use keyword arguments only.\n"
            "- Use only assignments, if, for-in, return, list/dict comprehensions, list.append, basic expressions, builtin helpers, tool calls, and pat.<pattern_id>(...) calls.\n"
            "- Encode source newlines as \\n exactly once inside JSON.\n"
            "- Do not add steps, save_as, return keys, or other step-form fields.\n"
            "- Do not use import, while, try, with, class, lambda, decorators, or eval/exec.\n"
        )














    def _build_json_repair_prompt(self, required_schema: str, bad_output: str) -> str:
        return (
            "JSON REPAIR PROMPT\n\n"
            "You produced invalid JSON or JSON with the wrong schema. Output STRICTLY valid JSON for the required schema only.\n"
            "- Output ONLY valid JSON, nothing else.\n"
            "- The top-level type/value must match the required schema exactly.\n"
            "- Do not add keys not allowed by the schema.\n"
            "- If the prior output used a different task schema, discard that wrapper and produce the required schema.\n"
            "- Preserve only information that fits the required schema.\n\n"
            "REQUIRED SCHEMA:\n"
            f"{required_schema}\n\n"
            "INVALID OUTPUT:\n"
            "<<<BAD_JSON_START>>>\n"
            f"{bad_output}\n"
            "<<<BAD_JSON_END>>>\n"
        )

    async def _generate_cognition_response_with_timeout(
        self,
        prompt: str,
        trace_id: str,
        turn_id: Optional[str],
        *,
        timeout_s: int,
        timeout_event: str,
        model_override: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            if timeout_s > 0:
                return await asyncio.wait_for(
                    self._generate_cognition_response(
                        prompt,
                        trace_id,
                        turn_id,
                        model_override=model_override,
                        options_override=options_override,
                    ),
                    timeout=timeout_s,
                )
            return await self._generate_cognition_response(
                prompt,
                trace_id,
                turn_id,
                model_override=model_override,
                options_override=options_override,
            )
        except asyncio.TimeoutError:
            self._log_mode_event(
                "COGNITION",
                timeout_event,
                {"trace_id": trace_id, "timeout_s": timeout_s, "model": model_override or "default"},
            )
            return ""







    def _program_signature(self, program: Dict[str, Any]) -> str:
        parts = []
        for key in ("name", "tags", "when_to_use", "inputs_needed"):
            value = program.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.append(" ".join(str(item) for item in value if item))
        source = program.get("source")
        if isinstance(source, str) and source.strip():
            parts.append(source.strip())
        steps = program.get("steps", [])
        if isinstance(steps, list):
            step_ops = [step.get("op", "") for step in steps if isinstance(step, dict)]
            if step_ops:
                parts.append("steps: " + " ".join(step_ops))
        return "; ".join(part for part in parts if part).strip()







    def _annotation_to_asi_input_type(self, annotation: Any) -> str:
        if isinstance(annotation, ast.Name):
            lowered = annotation.id.strip().lower()
            if lowered in {"int", "integer"}:
                return "int"
            if lowered in {"float", "number"}:
                return "float"
            if lowered in {"bool", "boolean"}:
                return "bool"
            if lowered in {"str", "string", "text"}:
                return "string"
            if lowered in {"path"}:
                return "path"
            if lowered in {"dict", "object", "mapping"}:
                return "object"
            if lowered in {"list", "tuple", "set", "sequence"}:
                return "list"
            return "text"
        if isinstance(annotation, ast.Attribute):
            return self._annotation_to_asi_input_type(ast.Name(id=annotation.attr))
        if isinstance(annotation, ast.Subscript):
            root = annotation.value
            if isinstance(root, ast.Name):
                lowered = root.id.strip().lower()
                if lowered in {"optional", "union"}:
                    target = annotation.slice
                    if isinstance(target, ast.Tuple) and target.elts:
                        return self._annotation_to_asi_input_type(target.elts[0])
                    return self._annotation_to_asi_input_type(target)
                if lowered in {"list", "tuple", "set", "sequence"}:
                    return "list"
                if lowered in {"dict", "mapping"}:
                    return "object"
            if isinstance(root, ast.Attribute):
                return self._annotation_to_asi_input_type(root)
        return "text"

    def _json_schema_to_asi_input_type(self, schema: Any, name: str = "") -> str:
        if not isinstance(schema, dict):
            return "text"
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return "enum"
        schema_type = str(schema.get("type", "") or "").strip().lower()
        schema_format = str(schema.get("format", "") or "").strip().lower()
        lowered_name = name.strip().lower()
        if schema_type == "string":
            if schema_format in {"uri", "url"}:
                return "url"
            if schema_format in {"path", "file-path"} or lowered_name in {"path", "file", "file_path", "filename"}:
                return "path"
            return "text"
        if schema_type == "integer":
            return "int"
        if schema_type == "number":
            return "float"
        if schema_type == "boolean":
            return "bool"
        if schema_type == "array":
            return "list"
        if schema_type == "object":
            return "object"
        return "text"

    def _tool_input_specs(self, tool: Dict[str, Any]) -> List[Dict[str, Any]]:
        schema = tool.get("input_schema", {})
        if not isinstance(schema, dict):
            return []
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return []
        required_items = schema.get("required")
        required_names: Set[str] = set()
        if isinstance(required_items, list):
            required_names = {str(item).strip() for item in required_items if str(item).strip()}
        inputs_needed: List[Dict[str, Any]] = []
        for name, prop in properties.items():
            if not isinstance(name, str) or not name.strip():
                continue
            prop_schema = prop if isinstance(prop, dict) else {}
            spec: Dict[str, Any] = {
                "name": name.strip(),
                "type": self._json_schema_to_asi_input_type(prop_schema, name),
                "required": name.strip() in required_names,
            }
            description = prop_schema.get("description", "")
            if isinstance(description, str) and description.strip():
                spec["description"] = description.strip()
            choices = prop_schema.get("enum")
            if isinstance(choices, list):
                cleaned_choices = []
                for choice in choices:
                    if isinstance(choice, (str, int, float, bool)) and choice not in cleaned_choices:
                        cleaned_choices.append(choice)
                if cleaned_choices:
                    spec["choices"] = cleaned_choices
            if "default" in prop_schema:
                spec["default"] = prop_schema.get("default")
            inputs_needed.append(spec)
        return self._normalize_asi_inputs_needed(inputs_needed)

    def _tool_to_asi_pattern(self, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(tool, dict):
            return None
        tool_id = str(tool.get("tool_id", "") or "").strip()
        if not tool_id:
            return None
        name = str(tool.get("name", "") or "").strip() or tool_id
        description = str(tool.get("description", "") or "").strip()
        when_to_use = [description] if description else [f"Direct use of {tool_id}."]
        return {
            "pattern_id": tool_id,
            "name": name,
            "when_to_use": when_to_use,
            "inputs_needed": self._tool_input_specs(tool),
            "tool_id": tool_id,
        }

    def _asi_literal_default_value(self, node: Optional[ast.AST]) -> Tuple[Any, bool]:
        if node is None:
            return None, False
        try:
            return ast.literal_eval(node), True
        except Exception:
            return None, False

    def _extract_asi_source_function(self, source: str) -> Tuple[Optional[ast.FunctionDef], List[str]]:
        if not isinstance(source, str) or not source.strip():
            return None, ["source must be a non-empty string"]
        try:
            module = ast.parse(source)
        except SyntaxError as e:
            return None, [f"source syntax error: {e.msg} (line {e.lineno})"]

        function_defs = [item for item in module.body if isinstance(item, ast.FunctionDef)]
        non_function_defs = [
            item
            for item in module.body
            if not (
                isinstance(item, ast.Expr)
                and isinstance(getattr(item, "value", None), ast.Constant)
                and isinstance(item.value.value, str)
            )
            and not isinstance(item, ast.FunctionDef)
        ]
        errors: List[str] = []
        if non_function_defs:
            errors.append("source must contain only a single top-level function definition")
        if not function_defs:
            errors.append("source must define exactly one function")
            return None, errors
        if len(function_defs) > 1:
            errors.append("source must define exactly one function")
        return function_defs[0], errors

    def _derive_asi_inputs_from_source(self, source: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        func_def, errors = self._extract_asi_source_function(source)
        if func_def is None:
            return [], errors
        args = list(func_def.args.args)
        defaults = list(func_def.args.defaults)
        if func_def.args.posonlyargs:
            errors.append("source functions cannot use positional-only parameters")
        if func_def.args.kwonlyargs:
            errors.append("source functions cannot use keyword-only parameters")
        if func_def.args.vararg is not None or func_def.args.kwarg is not None:
            errors.append("source functions cannot use *args or **kwargs")
        padded_defaults: List[Optional[ast.AST]] = [None] * max(0, len(args) - len(defaults)) + defaults
        inputs_needed: List[Dict[str, Any]] = []
        seen: Dict[str, bool] = {}
        for arg, default_node in zip(args, padded_defaults):
            if not isinstance(arg, ast.arg):
                continue
            name = str(arg.arg or "").strip()
            if not name or seen.get(name):
                continue
            seen[name] = True
            spec: Dict[str, Any] = {
                "name": name,
                "type": self._annotation_to_asi_input_type(arg.annotation),
                "required": default_node is None,
            }
            default_value, ok = self._asi_literal_default_value(default_node)
            if ok:
                spec["default"] = default_value
            elif default_node is not None:
                errors.append(f"parameter {name} default must be a literal value")
            inputs_needed.append(spec)
        return inputs_needed, errors

    def _merge_asi_source_inputs(
        self,
        derived_inputs: List[Dict[str, Any]],
        explicit_inputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        explicit_map = {
            str(item.get("name", "")).strip(): item
            for item in explicit_inputs
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }
        merged: List[Dict[str, Any]] = []
        for spec in derived_inputs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            current = dict(spec)
            explicit = explicit_map.get(name, {})
            if isinstance(explicit.get("description"), str) and explicit.get("description", "").strip():
                current["description"] = explicit.get("description", "").strip()
            if isinstance(explicit.get("choices"), list) and explicit.get("choices"):
                current["choices"] = list(explicit.get("choices", []))
            merged.append(current)
        return merged

    def _normalize_asi_pattern_ref(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        ref = value.strip()
        if not ref:
            return ""
        while True:
            lowered = ref.lower()
            stripped = False
            for prefix in ("pat.", "pattern.", "pattern:", "prog.", "program.", "program:"):
                if lowered.startswith(prefix):
                    ref = ref[len(prefix):].strip()
                    stripped = True
                    break
            if not stripped:
                break
        call_match = re.match(r"^([A-Za-z0-9_.:-]+)\s*\(.*\)\s*$", ref)
        if call_match:
            ref = call_match.group(1).strip()
        return ref




    def _asi_pattern_form(self, pattern: Dict[str, Any]) -> str:
        tool_id = str(pattern.get("tool_id", "") or "").strip()
        source = pattern.get("source", "")
        if tool_id and not (isinstance(source, str) and source.strip()):
            return "tool"
        return "source"

    def _normalize_asi_inputs_needed(self, inputs_needed: Any) -> List[Dict[str, Any]]:
        if not isinstance(inputs_needed, list):
            return []
        normalized: List[Dict[str, Any]] = []
        seen: Dict[str, bool] = {}
        for item in inputs_needed:
            spec: Dict[str, Any] = {}
            if isinstance(item, str):
                name = item.strip()
                if not name:
                    continue
                spec = {"name": name, "type": "text", "required": True}
            elif isinstance(item, dict):
                raw_name = item.get("name")
                if not isinstance(raw_name, str) or not raw_name.strip():
                    continue
                name = raw_name.strip()
                raw_type = item.get("type", "text")
                value_type = raw_type.strip().lower() if isinstance(raw_type, str) and raw_type.strip() else "text"
                required = item.get("required", True)
                if not isinstance(required, bool):
                    required = bool(required)
                spec = {"name": name, "type": value_type, "required": required}
                if "default" in item:
                    spec["default"] = item.get("default")
                description = item.get("description", "")
                if isinstance(description, str) and description.strip():
                    spec["description"] = description.strip()
                choices = item.get("choices")
                if isinstance(choices, list):
                    cleaned_choices = []
                    for choice in choices:
                        if isinstance(choice, (str, int, float, bool)) and choice not in cleaned_choices:
                            cleaned_choices.append(choice)
                    if cleaned_choices:
                        spec["choices"] = cleaned_choices
            else:
                continue
            name_key = str(spec.get("name", "")).strip()
            if not name_key or name_key in seen:
                continue
            seen[name_key] = True
            normalized.append(spec)
        return normalized

    def _normalize_asi_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(pattern)
        pattern_id = self._normalize_asi_pattern_ref(
            normalized.get("pattern_id") if normalized.get("pattern_id") else normalized.get("program_id")
        )
        if pattern_id:
            normalized["pattern_id"] = pattern_id
        if "program_id" in normalized:
            normalized.pop("program_id", None)
        explicit_inputs = self._normalize_asi_inputs_needed(normalized.get("inputs_needed", []))
        source = normalized.get("source")
        tool_ref = self._resolve_asi_tool_id(str(normalized.get("tool_id", "") or ""))
        if tool_ref:
            normalized["tool_id"] = tool_ref
        if isinstance(source, str) and source.strip():
            normalized["source"] = source.strip()
            normalized.pop("tool_id", None)
            derived_inputs, _ = self._derive_asi_inputs_from_source(normalized["source"])
            normalized["inputs_needed"] = self._merge_asi_source_inputs(derived_inputs, explicit_inputs)
        elif tool_ref:
            tool_def = self._tools.get_tool(tool_ref)
            normalized["inputs_needed"] = explicit_inputs or self._tool_input_specs(tool_def or {})
        else:
            normalized["inputs_needed"] = explicit_inputs
        normalized.pop("steps", None)
        return normalized

    def _seed_asi_patterns(self) -> List[Dict[str, Any]]:
        return []

    def _load_asi_patterns(self) -> List[Dict[str, Any]]:
        try:
            current_mtime = os.path.getmtime(self._asi_patterns_path)
        except OSError:
            current_mtime = None
        if self._asi_patterns_cache is not None and current_mtime == self._asi_patterns_mtime:
            return self._asi_patterns_cache
        patterns: List[Dict[str, Any]] = []
        if os.path.exists(self._asi_patterns_path):
            try:
                with open(self._asi_patterns_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    patterns = [
                        normalized
                        for item in data
                        if isinstance(item, dict)
                        for normalized in [self._normalize_asi_pattern(item)]
                        if isinstance(normalized.get("source"), str) and normalized.get("source", "").strip()
                    ]
            except Exception:
                patterns = []
        if not patterns and self._asi_seed_enabled:
            patterns = self._seed_asi_patterns()
            try:
                with open(self._asi_patterns_path, "w", encoding="utf-8") as f:
                    json.dump(patterns, f, ensure_ascii=True, indent=2)
                try:
                    current_mtime = os.path.getmtime(self._asi_patterns_path)
                except OSError:
                    current_mtime = None
            except Exception:
                pass
        self._asi_patterns_cache = patterns
        self._asi_patterns_mtime = current_mtime
        return patterns



    def _find_asi_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        normalized_id = self._normalize_asi_pattern_ref(pattern_id)
        if not normalized_id:
            return None
        for pattern in self._load_asi_patterns():
            existing_id = self._normalize_asi_pattern_ref(pattern.get("pattern_id"))
            if existing_id == normalized_id:
                return pattern
        tool_id = self._resolve_asi_tool_id(normalized_id)
        if tool_id:
            return self._tool_to_asi_pattern(self._tools.get_tool(tool_id) or {"tool_id": tool_id})
        return None








    def _extract_named_value(self, text: str, name: str) -> Optional[str]:
        if not name:
            return None
        pattern = rf"\b{re.escape(name)}\b\s*(?:=|:)\s*([^\n,;]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        value = match.group(1).strip().strip("\"'")
        return value if value else None

    def _extract_math_expression_from_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        candidates: List[str] = []
        for key in ("expr", "expression", "integrand", "equation"):
            named = self._extract_named_value(text, key)
            if named:
                candidates.append(named)
        for pattern in (
            r"\bintegral(?:\s+of)?\s+(.+)$",
            r"\bintegrate\s+(.+)$",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidates.append(match.group(1).strip())
        for candidate in candidates:
            cleaned = candidate.strip().rstrip(".?!;,")
            cleaned = re.sub(r"\s+with respect to\s+[a-zA-Z]\s*$", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*d[a-zA-Z]\s*$", "", cleaned)
            if cleaned:
                return cleaned
        return ""

    def _extract_symbol_from_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        for pattern in (
            r"\bwith respect to\s+([a-zA-Z])\b",
            r"\bw\.r\.t\.\s*([a-zA-Z])\b",
            r"\bvariable\s+([a-zA-Z])\b",
            r"\bd([a-zA-Z])\b",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return ""

    def _extract_symbol_from_expression(self, expr: str) -> str:
        if not isinstance(expr, str):
            return ""
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
        ignore = {"sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "sqrt", "integrate", "diff"}
        for token in tokens:
            if token in ignore:
                continue
            if len(token) == 1 and token.isalpha():
                return token
        for token in tokens:
            if token in ignore:
                continue
            if token.isalpha():
                return token
        return ""

    def _coerce_asi_value(self, value: Any, spec: Dict[str, Any]) -> Any:
        value_type = str(spec.get("type", "text")).strip().lower()
        choices = spec.get("choices", [])
        if value is None:
            return None
        if value_type in {"list", "array", "sequence"}:
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return [part.strip() for part in stripped.split(",") if part.strip()]
                return parsed if isinstance(parsed, list) else None
            return None
        if value_type in {"object", "dict", "mapping", "json"}:
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
            return None
        if value_type in {"math_expr", "expression", "equation"}:
            text = str(value).strip()
            return text if text else None
        if value_type in {"symbol", "variable"}:
            text = str(value).strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", text):
                return text
            return None
        if value_type in {"int", "integer"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if value_type in {"float", "number"}:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if value_type in {"bool", "boolean"}:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"true", "1", "yes", "y"}:
                return True
            if text in {"false", "0", "no", "n"}:
                return False
            return None
        if value_type in {"url", "uri"}:
            text = str(value).strip()
            if re.match(r"^https?://", text, flags=re.IGNORECASE):
                return text
            return None
        if value_type in {"path", "file_path"}:
            text = str(value).strip()
            return text if text else None
        if value_type in {"enum", "choice"} and isinstance(choices, list) and choices:
            candidate = str(value).strip().lower()
            for choice in choices:
                if str(choice).strip().lower() == candidate:
                    return choice
            return None
        text = str(value).strip()
        return text if text else None

    def _deterministic_asi_binding_value(
        self,
        spec: Dict[str, Any],
        user_text: str,
        current_bindings: Dict[str, Any],
    ) -> Any:
        name = str(spec.get("name", "")).strip()
        if not name:
            return None
        value_type = str(spec.get("type", "text")).strip().lower()
        named = self._extract_named_value(user_text, name)
        if named is not None:
            coerced = self._coerce_asi_value(named, spec)
            if coerced is not None:
                return coerced

        if value_type in {"math_expr", "expression", "equation"} or name.lower() in {"expr", "expression", "integrand"}:
            expr = self._extract_math_expression_from_text(user_text)
            if expr:
                return self._coerce_asi_value(expr, spec)
        if value_type in {"symbol", "variable"} or name.lower() in {"var", "variable", "symbol"}:
            symbol = self._extract_symbol_from_text(user_text)
            if not symbol:
                expr = current_bindings.get("expr") or current_bindings.get("expression")
                if isinstance(expr, str):
                    symbol = self._extract_symbol_from_expression(expr)
            if symbol:
                return self._coerce_asi_value(symbol, spec)
        if value_type in {"int", "integer", "float", "number"}:
            number_match = re.search(r"[-+]?\d*\.?\d+", user_text)
            if number_match:
                return self._coerce_asi_value(number_match.group(0), spec)
        if value_type in {"bool", "boolean"}:
            bool_match = re.search(r"\b(true|false|yes|no)\b", user_text, flags=re.IGNORECASE)
            if bool_match:
                return self._coerce_asi_value(bool_match.group(1), spec)
        if value_type in {"url", "uri"}:
            url_match = re.search(r"https?://\S+", user_text, flags=re.IGNORECASE)
            if url_match:
                return self._coerce_asi_value(url_match.group(0).rstrip(".,;!?"), spec)
        if value_type in {"path", "file_path"}:
            path_match = re.search(r"(workspace:/[^\s,;]+|[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)", user_text)
            if path_match:
                return self._coerce_asi_value(path_match.group(1).rstrip(".,;!?"), spec)
        if value_type in {"enum", "choice"}:
            choices = spec.get("choices", [])
            if isinstance(choices, list):
                lowered = user_text.lower()
                for choice in choices:
                    choice_text = str(choice).strip()
                    if choice_text and choice_text.lower() in lowered:
                        return self._coerce_asi_value(choice, spec)
        return None

    async def _bind_asi_inputs(
        self,
        pattern: Dict[str, Any],
        user_text: str,
    ) -> Dict[str, Any]:
        input_specs = self._normalize_asi_inputs_needed(pattern.get("inputs_needed", []))
        bindings: Dict[str, Any] = {}
        for spec in input_specs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            if "default" in spec:
                default_value = self._coerce_asi_value(spec.get("default"), spec)
                if default_value is not None:
                    bindings[name] = default_value

        for spec in input_specs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            current_value = bindings.get(name)
            if current_value is not None and current_value != "":
                continue
            inferred = self._deterministic_asi_binding_value(spec, user_text, bindings)
            if inferred is not None and inferred != "":
                bindings[name] = inferred

        required_names = [str(spec.get("name", "")).strip() for spec in input_specs if spec.get("required", True)]
        required_names = [name for name in required_names if name]
        missing_required = [name for name in required_names if name not in bindings or bindings.get(name) in {"", None}]

        required_total = len(required_names)
        required_hit = required_total - len(missing_required)
        coverage = 1.0
        if required_total > 0:
            coverage = max(0.0, min(1.0, float(required_hit) / float(required_total)))
        return {
            "bindings": bindings,
            "missing_required": missing_required,
            "required_total": required_total,
            "required_hit": required_hit,
            "coverage": coverage,
            "inputs_needed": input_specs,
        }

    def _normalize_asi_callable_name(self, value: Any) -> str:
        normalized = self._normalize_asi_pattern_ref(value)
        if not normalized:
            return ""
        collapsed = re.sub(r"[^A-Za-z0-9_]+", "_", normalized).strip("_")
        return collapsed or normalized.replace(".", "_")

    def _tool_callable_aliases(self, tool_id: str) -> Set[str]:
        aliases: Set[str] = set()
        if not isinstance(tool_id, str) or not tool_id.strip():
            return aliases
        raw = tool_id.strip()
        aliases.add(raw)
        aliases.add(raw.replace(".", "_"))
        return {alias for alias in aliases if alias}

    def _asi_source_builtin_callables(self) -> Dict[str, Callable[..., Any]]:
        return {
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "min": min,
            "max": max,
            "sum": sum,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "reversed": reversed,
            "any": any,
            "all": all,
            "list": list,
            "dict": dict,
            "tuple": tuple,
        }

    def _asi_attr_path(self, node: Any) -> Optional[str]:
        parts: List[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(str(current.attr))
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(str(current.id))
            return ".".join(reversed(parts))
        return None

    def _validate_asi_source_pattern(self, pattern: Dict[str, Any], tool_ids: List[str]) -> List[str]:
        errors: List[str] = []
        source = pattern.get("source", "")
        func_def, source_errors = self._extract_asi_source_function(source)
        errors.extend(source_errors)
        if func_def is None:
            return errors
        derived_inputs, input_errors = self._derive_asi_inputs_from_source(source)
        errors.extend(input_errors)
        explicit_inputs = self._normalize_asi_inputs_needed(pattern.get("inputs_needed", []))
        explicit_names = {
            str(item.get("name", "")).strip()
            for item in explicit_inputs
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }
        derived_names = {str(item.get("name", "")).strip() for item in derived_inputs if str(item.get("name", "")).strip()}
        extras = sorted(name for name in explicit_names if name not in derived_names)
        if extras:
            errors.append("inputs_needed names not present in source signature: " + ", ".join(extras))
        if pattern.get("steps") not in (None, []):
            errors.append("source-backed patterns must not also define steps")
        if func_def.decorator_list:
            errors.append("source functions cannot use decorators")
        if not func_def.body:
            errors.append("source function body must be non-empty")
            return errors
        if not any(isinstance(node, ast.Return) for node in ast.walk(func_def)):
            errors.append("source function must include at least one return statement")

        allowed_nodes = (
            ast.FunctionDef,
            ast.arguments,
            ast.arg,
            ast.Assign,
            ast.AugAssign,
            ast.Return,
            ast.Expr,
            ast.For,
            ast.If,
            ast.Pass,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Constant,
            ast.List,
            ast.Tuple,
            ast.Dict,
            ast.ListComp,
            ast.DictComp,
            ast.comprehension,
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.Slice,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.JoinedStr,
            ast.FormattedValue,
            ast.keyword,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.And,
            ast.Or,
            ast.Not,
            ast.UAdd,
            ast.USub,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.In,
            ast.NotIn,
            ast.Is,
            ast.IsNot,
        )
        disallowed_nodes = (
            ast.Import,
            ast.ImportFrom,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncFunctionDef,
            ast.Await,
            ast.Lambda,
            ast.FunctionDef,
            ast.ClassDef,
            ast.Delete,
            ast.Raise,
            ast.Yield,
            ast.YieldFrom,
            ast.SetComp,
            ast.GeneratorExp,
            ast.NamedExpr,
        )
        builtin_callables = set(self._asi_source_builtin_callables().keys())
        safe_attr_callables = {"append", "get", "items", "keys", "values"}
        pattern_aliases = {
            self._normalize_asi_callable_name(item.get("pattern_id"))
            for item in self._load_asi_patterns()
            if isinstance(item, dict)
        }
        pattern_aliases.add(self._normalize_asi_callable_name(pattern.get("pattern_id")))
        tool_aliases: Set[str] = set()
        for tool_id in tool_ids:
            tool_aliases.update(self._tool_callable_aliases(tool_id))

        for node in ast.walk(func_def):
            if isinstance(node, ast.FunctionDef) and node is not func_def:
                errors.append("source cannot define nested functions")
                continue
            if isinstance(node, disallowed_nodes) and not isinstance(node, ast.FunctionDef):
                errors.append(f"source uses unsupported syntax: {type(node).__name__}")
                continue
            if not isinstance(node, allowed_nodes):
                errors.append(f"source uses unsupported syntax: {type(node).__name__}")
                continue
            if isinstance(node, ast.Call):
                if any(keyword.arg is None for keyword in node.keywords):
                    errors.append("source cannot use **kwargs in calls")
                if isinstance(node.func, ast.Name):
                    func_name = str(node.func.id or "").strip()
                    if func_name and func_name not in builtin_callables and func_name not in pattern_aliases and func_name not in tool_aliases:
                        errors.append(f"unknown callable in source: {func_name}")
                elif isinstance(node.func, ast.Attribute):
                    attr_path = self._asi_attr_path(node.func)
                    if not attr_path:
                        errors.append("call targets must be simple names or dotted names")
                    else:
                        attr_name = attr_path.rsplit(".", 1)[-1]
                        if attr_name in safe_attr_callables:
                            continue
                        if (
                            attr_path in tool_ids
                            or attr_path in tool_aliases
                            or attr_path.startswith("pat.")
                        ):
                            continue
                        root = attr_path.split(".", 1)[0]
                        if root != "pat":
                            errors.append(f"unknown dotted callable in source: {attr_path}")
                else:
                    errors.append("call targets must be simple names or dotted names")
        deduped: List[str] = []
        for error in errors:
            value = str(error).strip()
            if value and value not in deduped:
                deduped.append(value)
        return deduped

    def _validate_asi_pattern(self, pattern: Dict[str, Any], tool_ids: List[str], tools_map: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        pattern_id = self._normalize_asi_pattern_ref(
            pattern.get("pattern_id") if pattern.get("pattern_id") else pattern.get("program_id")
        )
        if not pattern_id:
            errors.append("pattern_id must be a non-empty string")
        inputs_needed = pattern.get("inputs_needed", [])
        if inputs_needed is not None and not isinstance(inputs_needed, list):
            errors.append("inputs_needed must be a list when provided")
        else:
            normalized_inputs = self._normalize_asi_inputs_needed(inputs_needed if isinstance(inputs_needed, list) else [])
            if isinstance(inputs_needed, list) and len(normalized_inputs) < len(
                [item for item in inputs_needed if isinstance(item, (dict, str))]
            ):
                errors.append("inputs_needed contains invalid or duplicate input definitions")
        has_source = isinstance(pattern.get("source"), str) and str(pattern.get("source")).strip()
        raw_tool_id = str(pattern.get("tool_id", "") or "").strip()
        tool_id = self._resolve_asi_tool_id(raw_tool_id)
        has_steps = pattern.get("steps") is not None
        is_tool_pattern = bool(tool_id) and not has_source
        if raw_tool_id and not tool_id:
            errors.append("tool_id not in available tools")
        if has_steps:
            errors.append("step-backed ASI patterns are no longer supported")
        if tool_id and has_source:
            errors.append("tool-backed patterns must not also define source")
        if is_tool_pattern:
            return errors
        if not has_source:
            errors.append("source must be a non-empty string")
            return errors
        errors.extend(self._validate_asi_source_pattern(pattern, tool_ids))
        return errors





    def _resolve_env_path(self, path: str, env: Dict[str, Any]) -> Tuple[Any, bool]:
        if not isinstance(path, str):
            return None, False
        path = path.strip()
        if not path:
            return None, False
        if path in env:
            return env.get(path), True
        parts = [part for part in path.split(".") if part]
        if not parts:
            return None, False
        root = parts[0]
        if root not in env:
            return None, False

        def walk(node: Any, idx: int) -> Tuple[Any, bool]:
            if idx >= len(parts):
                return node, True
            part = parts[idx]

            if isinstance(node, dict):
                if part in node:
                    return walk(node.get(part), idx + 1)
                # Common wrappers in tool observations (status/result/error) and tool payloads.
                for wrapper_key in ("result", "json", "value"):
                    wrapped = node.get(wrapper_key)
                    if wrapped is None:
                        continue
                    resolved, ok = walk(wrapped, idx)
                    if ok:
                        return resolved, True
                return None, False

            if isinstance(node, list):
                if not part.isdigit():
                    return None, False
                list_idx = int(part)
                if list_idx < 0 or list_idx >= len(node):
                    return None, False
                return walk(node[list_idx], idx + 1)

            return None, False

        return walk(env.get(root), 1)

    def _is_missing_env_value(self, value: Any, found: bool) -> bool:
        if not found:
            return True
        return value is None or value == ""

    def _extract_env_reference(self, node: Any) -> Optional[str]:
        if isinstance(node, dict):
            for key in ("image_ref", "path", "file", "uri", "url"):
                ref = node.get(key)
                if isinstance(ref, str) and ref.strip():
                    return ref
            for wrapper in ("result", "json", "value"):
                wrapped = node.get(wrapper)
                if wrapped is None:
                    continue
                nested = self._extract_env_reference(wrapped)
                if nested:
                    return nested
        elif isinstance(node, list):
            for item in node:
                nested = self._extract_env_reference(item)
                if nested:
                    return nested
        return None

    def _normalize_env_direct_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        ref = self._extract_env_reference(value)
        if ref:
            return ref
        return value

    def _stringify_env_for_text(self, value: Any) -> str:
        candidate = value
        if isinstance(candidate, dict) and "result" in candidate and ("status" in candidate or "error" in candidate):
            status = candidate.get("status")
            if status in {"APPROVED", "", None} and candidate.get("result") is not None:
                candidate = candidate.get("result")

        if isinstance(candidate, dict):
            iso = candidate.get("iso")
            if isinstance(iso, (str, int, float, bool)):
                return str(iso)
            ref = self._extract_env_reference(candidate)
            if ref and "image_ref" in candidate:
                return ref

        if isinstance(candidate, (dict, list)):
            try:
                return json.dumps(candidate, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(candidate)
        return str(candidate)



    def _drop_none_values(self, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: Dict[str, Any] = {}
            for key, item in value.items():
                cleaned_item = self._drop_none_values(item)
                if cleaned_item is None:
                    continue
                cleaned[key] = cleaned_item
            return cleaned
        if isinstance(value, list):
            cleaned_list = [self._drop_none_values(item) for item in value]
            return [item for item in cleaned_list if item is not None]
        return value


    def _resolve_asi_pattern_callable(self, target: str) -> Optional[Dict[str, Any]]:
        return self._resolve_asi_pattern_callable_with_candidates(target, None)

    def _resolve_asi_pattern_callable_with_candidates(
        self,
        target: str,
        extra_patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_id = self._normalize_asi_pattern_ref(target)
        if normalized_id:
            if isinstance(extra_patterns, list):
                for pattern in extra_patterns:
                    if not isinstance(pattern, dict):
                        continue
                    existing_id = self._normalize_asi_pattern_ref(pattern.get("pattern_id"))
                    if existing_id == normalized_id:
                        return pattern
            direct = self._find_asi_pattern(normalized_id)
            if direct is not None:
                return direct
        callable_name = self._normalize_asi_callable_name(target)
        if not callable_name:
            return None
        if isinstance(extra_patterns, list):
            for pattern in extra_patterns:
                if not isinstance(pattern, dict):
                    continue
                pattern_id = self._normalize_asi_pattern_ref(pattern.get("pattern_id"))
                if not pattern_id:
                    continue
                if self._normalize_asi_callable_name(pattern_id) == callable_name:
                    return pattern
        for pattern in self._load_asi_patterns():
            pattern_id = self._normalize_asi_pattern_ref(pattern.get("pattern_id"))
            if not pattern_id:
                continue
            if self._normalize_asi_callable_name(pattern_id) == callable_name:
                return pattern
        return None

    def _resolve_asi_tool_id(self, target: str) -> str:
        if not isinstance(target, str) or not target.strip():
            return ""
        clean = target.strip()
        tool = self._tools.get_tool(clean)
        if tool is not None:
            return clean
        for item in self._tools.list_active():
            tool_id = str(item.get("tool_id", "")).strip()
            if not tool_id:
                continue
            if clean in self._tool_callable_aliases(tool_id):
                return tool_id
        return ""

    def _prepare_asi_pattern_call(
        self,
        pattern: Dict[str, Any],
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        input_specs = self._normalize_asi_inputs_needed(pattern.get("inputs_needed", []))
        spec_map = {str(spec.get("name", "")).strip(): spec for spec in input_specs if str(spec.get("name", "")).strip()}
        ordered_names = [str(spec.get("name", "")).strip() for spec in input_specs if str(spec.get("name", "")).strip()]
        if len(args) > len(ordered_names):
            return {"type": "error", "text": "Too many positional arguments for pattern call."}

        bindings: Dict[str, Any] = {}
        for spec in input_specs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            if "default" in spec:
                default_value = self._coerce_asi_value(spec.get("default"), spec)
                if default_value is not None or spec.get("default") is None:
                    bindings[name] = default_value

        for idx, value in enumerate(args):
            name = ordered_names[idx]
            spec = spec_map.get(name, {})
            coerced = self._coerce_asi_value(value, spec) if isinstance(spec, dict) else value
            bindings[name] = value if coerced is None and value not in {None, ""} else coerced

        for key, value in kwargs.items():
            spec = spec_map.get(key)
            if spec is None:
                return {"type": "error", "text": f"Unknown keyword argument for pattern call: {key}."}
            coerced = self._coerce_asi_value(value, spec)
            bindings[key] = value if coerced is None and value not in {None, ""} else coerced

        missing = [
            name
            for name in ordered_names
            if spec_map.get(name, {}).get("required", True) and (name not in bindings or bindings.get(name) in {"", None})
        ]
        if missing:
            return {"type": "clarify", "question": "Please provide: " + ", ".join(missing) + "."}
        return {"type": "ok", "bindings": bindings}

    async def _execute_asi_tool_pattern(
        self,
        pattern: Dict[str, Any],
        bindings: Dict[str, Any],
        trace_id: str,
        tool_budget: Dict[str, int],
    ) -> Dict[str, Any]:
        tool_id = self._resolve_asi_tool_id(str(pattern.get("tool_id") or pattern.get("pattern_id") or ""))
        if not tool_id:
            return {"type": "error", "text": "Unknown tool-backed ASI pattern.", "trace": []}
        if tool_budget.get("remaining", 0) <= 0:
            return {"type": "error", "text": "Tool budget exceeded.", "trace": []}
        args: Dict[str, Any] = {}
        for spec in self._normalize_asi_inputs_needed(pattern.get("inputs_needed", [])):
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            if name in bindings and bindings.get(name) is not None:
                args[name] = bindings.get(name)
        args = self._drop_none_values(args)
        step_entry: Dict[str, Any] = {"index": 1, "op": "tool_call", "tool_id": tool_id, "status": "pending"}
        step_result = await self._execute_tool_action(tool_id, args, trace_id)
        observation = step_result.get("observation")
        if not isinstance(observation, dict):
            step_entry["status"] = "error"
            step_entry["detail"] = "Tool returned invalid observation."
            return {"type": "error", "text": "Tool returned invalid observation.", "trace": [step_entry]}
        status = str(observation.get("status", "") or "").strip()
        if status != "APPROVED":
            error_text = str(observation.get("error", status or "Tool failed") or "Tool failed").strip()
            step_entry["status"] = "error"
            step_entry["detail"] = self._truncate(error_text, 300)
            return {"type": "error", "text": error_text, "trace": [step_entry]}
        result_value = observation.get("result")
        text_value = self._stringify_env_for_text(result_value if result_value is not None else observation)
        step_entry["status"] = "success"
        step_entry["detail"] = f"Executed {tool_id}."
        tool_budget["remaining"] = tool_budget.get("remaining", 0) - 1
        return {"type": "final", "text": text_value, "value": result_value, "trace": [step_entry]}

    async def _execute_asi_pattern_runtime(
        self,
        pattern: Dict[str, Any],
        bindings: Dict[str, Any],
        trace_id: str,
        tool_budget: Dict[str, int],
        *,
        user_request: str,
        depth: int = 0,
        extra_patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        normalized_pattern = self._normalize_asi_pattern(pattern)
        if self._asi_pattern_form(normalized_pattern) == "tool":
            return await self._execute_asi_tool_pattern(normalized_pattern, bindings, trace_id, tool_budget)
        source = normalized_pattern.get("source", "")
        if isinstance(source, str) and source.strip():
            return await self._execute_asi_source_pattern(
                normalized_pattern,
                bindings,
                trace_id,
                tool_budget,
                user_request=user_request,
                depth=depth,
                extra_patterns=extra_patterns,
            )
        return {"type": "error", "text": "Invalid ASI pattern: source is required.", "trace": []}

    async def _execute_asi_source_pattern(
        self,
        pattern: Dict[str, Any],
        bindings: Dict[str, Any],
        trace_id: str,
        tool_budget: Dict[str, int],
        *,
        user_request: str,
        depth: int = 0,
        extra_patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        source = str(pattern.get("source", "") or "").strip()
        func_def, source_errors = self._extract_asi_source_function(source)
        if func_def is None:
            return {
                "type": "error",
                "text": source_errors[0] if source_errors else "Invalid ASI source pattern.",
                "trace": [],
            }

        step_trace: List[Dict[str, Any]] = []
        local_env: Dict[str, Any] = {"user_request": user_request}
        input_specs = self._normalize_asi_inputs_needed(pattern.get("inputs_needed", []))
        for spec in input_specs:
            name = str(spec.get("name", "")).strip()
            if not name:
                continue
            if name in bindings:
                local_env[name] = bindings.get(name)
                continue
            if "default" in spec:
                default_value = self._coerce_asi_value(spec.get("default"), spec)
                local_env[name] = default_value
                continue
            if spec.get("required", True):
                return {
                    "type": "clarify",
                    "question": "Please provide: " + name + ".",
                    "trace": [],
                }

        builtin_callables = self._asi_source_builtin_callables()
        safe_method_names = {"append", "get", "items", "keys", "values"}

        def _wrap_tool_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
            raw = dict(observation) if isinstance(observation, dict) else {"status": "ERROR", "error": "Tool returned invalid observation"}
            status = str(raw.get("status", "") or "").strip()
            ok = status == "APPROVED"
            result_value = raw.get("result")
            error_text = str(raw.get("error", "") or "").strip()
            text_value = self._stringify_env_for_text(
                result_value if ok and result_value is not None else (error_text or raw)
            )
            wrapped = dict(raw)
            wrapped["kind"] = "tool"
            wrapped["ok"] = ok
            wrapped["type"] = "final" if ok else "error"
            wrapped["status"] = status or ("APPROVED" if ok else "ERROR")
            wrapped["result"] = result_value
            wrapped["value"] = result_value
            wrapped["text"] = text_value
            wrapped["error"] = error_text
            wrapped["question"] = ""
            wrapped["trace"] = []
            wrapped["raw"] = dict(raw)
            return wrapped

        def _wrap_pattern_result(
            pattern_obj: Dict[str, Any],
            result: Dict[str, Any],
            sub_trace: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            raw = dict(result) if isinstance(result, dict) else {"type": "error", "text": "Pattern returned invalid result."}
            result_type = str(raw.get("type", "") or "error").strip().lower() or "error"
            ok = result_type == "final"
            value = raw.get("value", raw.get("text", "")) if ok else None
            text_value = str(raw.get("text", "") or "")
            if ok and not text_value:
                text_value = self._stringify_env_for_text(value)
            error_text = ""
            question_text = ""
            status = "ERROR"
            if ok:
                status = "APPROVED"
            elif result_type == "clarify":
                status = "CLARIFY"
                question_text = str(raw.get("question", "") or "").strip()
                text_value = question_text or text_value
            else:
                error_text = str(raw.get("text", raw.get("error", "")) or "").strip()
                text_value = error_text or text_value or "Pattern call failed."
            return {
                "kind": "pattern",
                "pattern_id": self._normalize_asi_pattern_ref(pattern_obj.get("pattern_id")),
                "ok": ok,
                "type": result_type,
                "status": status,
                "result": value,
                "value": value,
                "text": text_value,
                "error": error_text,
                "question": question_text,
                "trace": list(sub_trace),
                "raw": raw,
            }

        def _assign_target(target: ast.AST, value: Any, *, context: str) -> None:
            if isinstance(target, ast.Name):
                local_env[str(target.id)] = value
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                if not isinstance(value, (tuple, list)):
                    raise _AsiSourceSignal(
                        {
                            "type": "error",
                            "text": f"{context} expects an iterable value for tuple/list unpacking.",
                            "trace": step_trace,
                        }
                    )
                if len(target.elts) != len(value):
                    raise _AsiSourceSignal(
                        {
                            "type": "error",
                            "text": f"{context} unpacking arity mismatch in ASI source.",
                            "trace": step_trace,
                        }
                    )
                for elt, item in zip(target.elts, value):
                    _assign_target(elt, item, context=context)
                return
            raise _AsiSourceSignal(
                {
                    "type": "error",
                    "text": f"Unsupported assignment target in ASI source: {type(target).__name__}.",
                    "trace": step_trace,
                }
            )

        def _collect_target_names(target: ast.AST) -> List[str]:
            if isinstance(target, ast.Name):
                return [str(target.id)]
            if isinstance(target, (ast.Tuple, ast.List)):
                names: List[str] = []
                for elt in target.elts:
                    names.extend(_collect_target_names(elt))
                return names
            return []

        async def _eval_comprehension(
            generators: List[ast.comprehension],
            emit: Callable[[], Any],
        ) -> None:
            target_names: List[str] = []
            for generator in generators:
                if generator.is_async:
                    raise _AsiSourceSignal(
                        {"type": "error", "text": "Async comprehensions are not supported in ASI source.", "trace": step_trace}
                    )
                target_names.extend(_collect_target_names(generator.target))
            saved_values = {name: local_env.get(name) for name in target_names if name in local_env}

            async def _walk(index: int) -> None:
                if index >= len(generators):
                    emitted = emit()
                    if asyncio.iscoroutine(emitted):
                        await emitted
                    return
                generator = generators[index]
                iterable = await _eval_expr(generator.iter)
                try:
                    iterator = list(iterable)
                except TypeError:
                    raise _AsiSourceSignal(
                        {"type": "error", "text": "ASI source comprehension target is not iterable.", "trace": step_trace}
                    )
                for item in iterator:
                    _assign_target(generator.target, item, context="Comprehension")
                    passes = True
                    for if_node in generator.ifs:
                        if not await _eval_expr(if_node):
                            passes = False
                            break
                    if passes:
                        await _walk(index + 1)

            try:
                await _walk(0)
            finally:
                for name in target_names:
                    if name in saved_values:
                        local_env[name] = saved_values[name]
                    else:
                        local_env.pop(name, None)

        async def _call_tool(tool_id: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
            if tool_budget.get("remaining", 0) <= 0:
                raise _AsiSourceSignal({"type": "error", "text": "Tool budget exceeded.", "trace": step_trace})
            entry = _trace_entry("tool_call", tool_id=tool_id)
            observation_result = await self._execute_tool_action(tool_id, kwargs, trace_id)
            observation = observation_result.get("observation")
            if not isinstance(observation, dict):
                observation = {"status": "ERROR", "error": "Tool returned invalid observation"}
            tool_budget["remaining"] = tool_budget.get("remaining", 0) - 1
            wrapped = _wrap_tool_observation(observation)
            if wrapped.get("ok"):
                _trace_done(entry, "success", f"Tool {tool_id} completed.")
            else:
                _trace_done(entry, "error", f"Tool {tool_id} failed: {wrapped.get('error') or wrapped.get('status', 'ERROR')}")
            return wrapped

        async def _call_pattern(pattern_obj: Dict[str, Any], args: List[Any], kwargs: Dict[str, Any]) -> Any:
            if depth >= self._asi_recursion_limit:
                return {
                    "kind": "pattern",
                    "pattern_id": self._normalize_asi_pattern_ref(pattern_obj.get("pattern_id")),
                    "ok": False,
                    "type": "error",
                    "status": "ERROR",
                    "result": None,
                    "value": None,
                    "text": "Pattern recursion limit reached.",
                    "error": "Pattern recursion limit reached.",
                    "question": "",
                    "trace": [],
                    "raw": {"type": "error", "text": "Pattern recursion limit reached."},
                }
            prepared = self._prepare_asi_pattern_call(pattern_obj, args, kwargs)
            if prepared.get("type") == "clarify":
                question = str(prepared.get("question", "I need one detail to proceed.") or "I need one detail to proceed.")
                return {
                    "kind": "pattern",
                    "pattern_id": self._normalize_asi_pattern_ref(pattern_obj.get("pattern_id")),
                    "ok": False,
                    "type": "clarify",
                    "status": "CLARIFY",
                    "result": None,
                    "value": None,
                    "text": question,
                    "error": "",
                    "question": question,
                    "trace": [],
                    "raw": {"type": "clarify", "question": question},
                }
            if prepared.get("type") == "error":
                error_text = str(prepared.get("text", "Pattern call failed.") or "Pattern call failed.")
                return {
                    "kind": "pattern",
                    "pattern_id": self._normalize_asi_pattern_ref(pattern_obj.get("pattern_id")),
                    "ok": False,
                    "type": "error",
                    "status": "ERROR",
                    "result": None,
                    "value": None,
                    "text": error_text,
                    "error": error_text,
                    "question": "",
                    "trace": [],
                    "raw": {"type": "error", "text": error_text},
                }
            entry = _trace_entry("pattern_call", program_id=self._normalize_asi_pattern_ref(pattern_obj.get("pattern_id")))
            result = await self._execute_asi_pattern_runtime(
                pattern_obj,
                prepared.get("bindings", {}),
                trace_id,
                tool_budget,
                user_request=user_request,
                depth=depth + 1,
                extra_patterns=extra_patterns,
            )
            sub_trace = result.get("trace") if isinstance(result.get("trace"), list) else []
            wrapped = _wrap_pattern_result(pattern_obj, result, sub_trace)
            if wrapped.get("type") == "final":
                _trace_done(entry, "success", "Pattern call completed.", sub_trace=sub_trace)
            elif wrapped.get("type") == "clarify":
                _trace_done(entry, "clarify", wrapped.get("question", ""), sub_trace=sub_trace)
            else:
                _trace_done(entry, "error", wrapped.get("error", "Pattern call failed."), sub_trace=sub_trace)
            return wrapped

        def _trace_entry(op: str, **fields: Any) -> Dict[str, Any]:
            entry: Dict[str, Any] = {"index": len(step_trace) + 1, "op": op, "status": "pending"}
            for key, value in fields.items():
                if value is not None:
                    entry[key] = value
            return entry

        def _trace_done(
            entry: Dict[str, Any],
            status: str,
            detail: str = "",
            *,
            missing: Optional[List[str]] = None,
            sub_trace: Optional[List[Dict[str, Any]]] = None,
        ) -> None:
            entry["status"] = status
            if detail:
                entry["detail"] = self._truncate(str(detail), 300)
            if isinstance(missing, list) and missing:
                entry["missing"] = [str(item).strip() for item in missing if str(item).strip()][:8]
            if isinstance(sub_trace, list) and sub_trace:
                entry["sub_trace"] = sub_trace
            step_trace.append(entry)

        async def _eval_expr(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                name = str(node.id or "").strip()
                if name in local_env:
                    return local_env.get(name)
                if name in builtin_callables:
                    return builtin_callables[name]
                raise _AsiSourceSignal({"type": "error", "text": f"Unknown variable in ASI source: {name}.", "trace": step_trace})
            if isinstance(node, ast.List):
                return [await _eval_expr(item) for item in node.elts]
            if isinstance(node, ast.Tuple):
                return tuple([await _eval_expr(item) for item in node.elts])
            if isinstance(node, ast.Dict):
                keys = [await _eval_expr(item) for item in node.keys]
                values = [await _eval_expr(item) for item in node.values]
                return {key: value for key, value in zip(keys, values)}
            if isinstance(node, ast.ListComp):
                result: List[Any] = []

                async def _emit() -> None:
                    result.append(await _eval_expr(node.elt))

                await _eval_comprehension(node.generators, _emit)
                return result
            if isinstance(node, ast.DictComp):
                result_dict: Dict[Any, Any] = {}

                async def _emit_dict() -> None:
                    key = await _eval_expr(node.key)
                    value = await _eval_expr(node.value)
                    result_dict[key] = value

                await _eval_comprehension(node.generators, _emit_dict)
                return result_dict
            if isinstance(node, ast.Subscript):
                target = await _eval_expr(node.value)
                if isinstance(node.slice, ast.Slice):
                    lower = await _eval_expr(node.slice.lower) if node.slice.lower is not None else None
                    upper = await _eval_expr(node.slice.upper) if node.slice.upper is not None else None
                    step = await _eval_expr(node.slice.step) if node.slice.step is not None else None
                    return target[slice(lower, upper, step)]
                key = await _eval_expr(node.slice)
                return target[key]
            if isinstance(node, ast.Attribute):
                base = await _eval_expr(node.value)
                if isinstance(base, dict) and node.attr in base:
                    return base.get(node.attr)
                if isinstance(base, list) and node.attr == "append":
                    return base.append
                if isinstance(base, dict) and node.attr in {"get", "items", "keys", "values"}:
                    return getattr(base, node.attr)
                raise _AsiSourceSignal({"type": "error", "text": f"Unsupported attribute access in ASI source: {node.attr}.", "trace": step_trace})
            if isinstance(node, ast.BinOp):
                left = await _eval_expr(node.left)
                right = await _eval_expr(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Mod):
                    return left % right
                raise _AsiSourceSignal({"type": "error", "text": "Unsupported binary operator in ASI source.", "trace": step_trace})
            if isinstance(node, ast.UnaryOp):
                operand = await _eval_expr(node.operand)
                if isinstance(node.op, ast.Not):
                    return not operand
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return +operand
                raise _AsiSourceSignal({"type": "error", "text": "Unsupported unary operator in ASI source.", "trace": step_trace})
            if isinstance(node, ast.BoolOp):
                values = [await _eval_expr(value) for value in node.values]
                if isinstance(node.op, ast.And):
                    result = True
                    for value in values:
                        result = result and value
                    return result
                if isinstance(node.op, ast.Or):
                    result = False
                    for value in values:
                        result = result or value
                    return result
                raise _AsiSourceSignal({"type": "error", "text": "Unsupported boolean operator in ASI source.", "trace": step_trace})
            if isinstance(node, ast.Compare):
                left = await _eval_expr(node.left)
                for operator, comparator_node in zip(node.ops, node.comparators):
                    right = await _eval_expr(comparator_node)
                    ok = False
                    if isinstance(operator, ast.Eq):
                        ok = left == right
                    elif isinstance(operator, ast.NotEq):
                        ok = left != right
                    elif isinstance(operator, ast.Lt):
                        ok = left < right
                    elif isinstance(operator, ast.LtE):
                        ok = left <= right
                    elif isinstance(operator, ast.Gt):
                        ok = left > right
                    elif isinstance(operator, ast.GtE):
                        ok = left >= right
                    elif isinstance(operator, ast.In):
                        ok = left in right
                    elif isinstance(operator, ast.NotIn):
                        ok = left not in right
                    elif isinstance(operator, ast.Is):
                        ok = left is right
                    elif isinstance(operator, ast.IsNot):
                        ok = left is not right
                    else:
                        raise _AsiSourceSignal({"type": "error", "text": "Unsupported comparison in ASI source.", "trace": step_trace})
                    if not ok:
                        return False
                    left = right
                return True
            if isinstance(node, ast.JoinedStr):
                parts: List[str] = []
                for value in node.values:
                    if isinstance(value, ast.Constant):
                        parts.append(str(value.value))
                    elif isinstance(value, ast.FormattedValue):
                        parts.append(self._stringify_env_for_text(await _eval_expr(value.value)))
                return "".join(parts)
            if isinstance(node, ast.Call):
                return await _eval_call(node)
            raise _AsiSourceSignal({"type": "error", "text": f"Unsupported expression in ASI source: {type(node).__name__}.", "trace": step_trace})

        async def _eval_call(node: ast.Call) -> Any:
            args = [await _eval_expr(item) for item in node.args]
            kwargs = {str(item.arg): await _eval_expr(item.value) for item in node.keywords if item.arg}
            if isinstance(node.func, ast.Name):
                func_name = str(node.func.id or "").strip()
                if func_name in builtin_callables:
                    return builtin_callables[func_name](*args, **kwargs)
                pattern = self._resolve_asi_pattern_callable_with_candidates(func_name, extra_patterns)
                if pattern is not None:
                    return await _call_pattern(pattern, args, kwargs)
                tool_id = self._resolve_asi_tool_id(func_name)
                if tool_id:
                    if args:
                        raise _AsiSourceSignal({"type": "error", "text": f"Tool calls in ASI source must use keyword arguments: {tool_id}.", "trace": step_trace})
                    return await _call_tool(tool_id, kwargs)
                raise _AsiSourceSignal({"type": "error", "text": f"Unknown callable in ASI source: {func_name}.", "trace": step_trace})

            attr_path = self._asi_attr_path(node.func)
            if not attr_path:
                raise _AsiSourceSignal({"type": "error", "text": "Call targets must be simple names or dotted names.", "trace": step_trace})
            attr_name = attr_path.rsplit(".", 1)[-1]
            if attr_name in safe_method_names:
                bound_method = await _eval_expr(node.func)
                if not callable(bound_method):
                    raise _AsiSourceSignal({"type": "error", "text": f"{attr_name}() is not callable in ASI source.", "trace": step_trace})
                if attr_name == "append":
                    if len(args) != 1 or kwargs:
                        raise _AsiSourceSignal({"type": "error", "text": "append() accepts exactly one positional argument.", "trace": step_trace})
                    return bound_method(args[0])
                return bound_method(*args, **kwargs)
            if attr_path.startswith("pat."):
                pattern = self._resolve_asi_pattern_callable_with_candidates(attr_path[len("pat."):], extra_patterns)
                if pattern is None:
                    raise _AsiSourceSignal({"type": "error", "text": f"Unknown pattern call in ASI source: {attr_path}.", "trace": step_trace})
                return await _call_pattern(pattern, args, kwargs)

            tool_id = self._resolve_asi_tool_id(attr_path)
            if not tool_id:
                raise _AsiSourceSignal({"type": "error", "text": f"Unknown tool call in ASI source: {attr_path}.", "trace": step_trace})
            if args:
                raise _AsiSourceSignal({"type": "error", "text": f"Tool calls in ASI source must use keyword arguments: {tool_id}.", "trace": step_trace})
            return await _call_tool(tool_id, kwargs)

        async def _exec_block(statements: List[ast.stmt]) -> None:
            for statement in statements:
                if isinstance(statement, ast.Expr) and isinstance(getattr(statement, "value", None), ast.Constant) and isinstance(statement.value.value, str):
                    continue
                await _exec_stmt(statement)

        async def _exec_stmt(statement: ast.stmt) -> None:
            if isinstance(statement, ast.Assign):
                value = await _eval_expr(statement.value)
                for target in statement.targets:
                    _assign_target(target, value, context="Assignment")
                return
            if isinstance(statement, ast.AugAssign):
                if not isinstance(statement.target, ast.Name):
                    raise _AsiSourceSignal({"type": "error", "text": "Unsupported augmented assignment target in ASI source.", "trace": step_trace})
                current = local_env.get(str(statement.target.id))
                value = await _eval_expr(statement.value)
                if isinstance(statement.op, ast.Add):
                    local_env[str(statement.target.id)] = current + value
                    return
                if isinstance(statement.op, ast.Sub):
                    local_env[str(statement.target.id)] = current - value
                    return
                raise _AsiSourceSignal({"type": "error", "text": "Unsupported augmented assignment in ASI source.", "trace": step_trace})
            if isinstance(statement, ast.Expr):
                await _eval_expr(statement.value)
                return
            if isinstance(statement, ast.If):
                condition = await _eval_expr(statement.test)
                await _exec_block(statement.body if condition else statement.orelse)
                return
            if isinstance(statement, ast.For):
                iterable = await _eval_expr(statement.iter)
                try:
                    iterator = list(iterable)
                except TypeError:
                    raise _AsiSourceSignal({"type": "error", "text": "ASI source for-loop target is not iterable.", "trace": step_trace})
                for item in iterator:
                    _assign_target(statement.target, item, context="For loop")
                    await _exec_block(statement.body)
                return
            if isinstance(statement, ast.Return):
                value = await _eval_expr(statement.value) if statement.value is not None else None
                raise _AsiSourceReturn(value)
            if isinstance(statement, ast.Pass):
                return
            raise _AsiSourceSignal({"type": "error", "text": f"Unsupported statement in ASI source: {type(statement).__name__}.", "trace": step_trace})

        try:
            await _exec_block(func_def.body)
        except _AsiSourceReturn as returned:
            return {
                "type": "final",
                "value": returned.value,
                "text": self._stringify_env_for_text(returned.value),
                "trace": step_trace,
            }
        except _AsiSourceSignal as signal:
            payload = dict(signal.payload)
            payload["trace"] = step_trace
            return payload
        return {"type": "error", "text": "ASI source pattern ended without a return statement.", "trace": step_trace}


    def _parse_json_payload(
        self,
        raw: str,
        validator: Optional[Callable[[Any], bool]] = None,
        include_soft_fallback: bool = True,
    ) -> Optional[Any]:
        candidate = self._extract_json_candidate(raw)
        if candidate is not None:
            parsed = self._load_relaxed_json(candidate)
            if parsed is not None and (validator is None or validator(parsed)):
                return parsed
            if not include_soft_fallback or not self._soft_json_parsing_enabled:
                return None
        elif not include_soft_fallback or not self._soft_json_parsing_enabled:
            return None

        for snippet in self._collect_soft_json_candidates(raw):
            if candidate is not None and snippet == candidate:
                continue
            parsed = self._load_relaxed_json(snippet)
            if parsed is None:
                continue
            if validator is None or validator(parsed):
                return parsed
        return None





    def _parse_cognition_init_response(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict) and item.get("type") == "init",
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        tools_needed = bool(parsed.get("tools_needed", True))
        query_state = self._default_cognition_query_state("")
        if self._cognition_query_state_enabled:
            if not isinstance(parsed.get("query_state"), dict):
                return None
            self._merge_cognition_query_state(query_state, parsed.get("query_state", {}))
        state_of_mind = self._default_cognition_state_of_mind()
        if self._cognition_state_of_mind_enabled:
            if not isinstance(parsed.get("state_of_mind"), dict):
                return None
            self._merge_cognition_state_of_mind(state_of_mind, parsed.get("state_of_mind", {}))
        return {
            "type": "init",
            "thought": str(parsed.get("thought") or ""),
            "tools_needed": tools_needed,
            "query_state": query_state,
            "state_of_mind": state_of_mind,
        }


    def _parse_cognition_tool_selection_response(
        self,
        raw: str,
        allowed_tool_ids: Optional[Set[str]] = None,
        allowed_category_ids: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict) and item.get("type") == "tool_selection",
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        selected_raw = parsed.get("selected_tool_ids")
        if selected_raw is None and allowed_category_ids is not None:
            selected_raw = []
        if not isinstance(selected_raw, list):
            return None
        selected_tool_ids: List[str] = []
        seen: Set[str] = set()
        had_invalid = False
        for item in selected_raw:
            tool_id = str(item or "").strip()
            if not tool_id:
                had_invalid = True
                continue
            if allowed_tool_ids is not None and tool_id not in allowed_tool_ids:
                had_invalid = True
                continue
            if tool_id in seen:
                continue
            seen.add(tool_id)
            selected_tool_ids.append(tool_id)
        if had_invalid and not selected_tool_ids and selected_raw:
            return None
        selected_category_ids: List[str] = []
        if allowed_category_ids is not None:
            selected_categories_raw = parsed.get("selected_category_ids")
            if selected_categories_raw is None:
                selected_categories_raw = []
            if not isinstance(selected_categories_raw, list):
                return None
            seen_categories: Set[str] = set()
            had_invalid_category = False
            for item in selected_categories_raw:
                category_id = str(item or "").strip()
                if not category_id:
                    had_invalid_category = True
                    continue
                if category_id not in allowed_category_ids:
                    had_invalid_category = True
                    continue
                if category_id in seen_categories:
                    continue
                seen_categories.add(category_id)
                selected_category_ids.append(category_id)
            if had_invalid_category and not selected_category_ids and selected_categories_raw:
                return None
        result = {
            "type": "tool_selection",
            "selected_tool_ids": selected_tool_ids,
            "reason": str(parsed.get("reason") or "").strip(),
        }
        if allowed_category_ids is not None:
            result["selected_category_ids"] = selected_category_ids
        return result

    def _parse_cognition_route_response(self, raw: str) -> Optional[Dict[str, Any]]:
        allowed_modes = {"system1", "system2"}
        if self._cognition_system3_available():
            allowed_modes.add("system3")
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict)
            and item.get("type") == "route"
            and item.get("thinking_mode") in allowed_modes,
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        thinking_mode = parsed.get("thinking_mode")
        if thinking_mode not in allowed_modes:
            return None
        generate_tools_first_raw = parsed.get("generate_tools_first", False)
        if "generate_tools_first" in parsed and not isinstance(generate_tools_first_raw, bool):
            return None
        generate_tools_first = bool(generate_tools_first_raw)
        tool_generation_spec: Dict[str, Any] = {}
        if generate_tools_first:
            spec = parsed.get("tool_generation_spec", parsed.get("spec"))
            if not isinstance(spec, dict):
                return None
            tool_generation_spec = spec
        return {
            "type": "route",
            "thinking_mode": thinking_mode,
            "reason": str(parsed.get("reason") or ""),
            "generate_tools_first": generate_tools_first,
            "tool_generation_spec": tool_generation_spec,
        }

    def _parse_cognition_system3_proposal_response(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict)
            and item.get("type") == "proposal"
            and item.get("recommended_mode") in {"system1", "system2"},
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        peer_id = str(parsed.get("peer_id") or "").strip().lower()
        strategy = str(parsed.get("strategy") or "").strip()
        argument = str(parsed.get("argument") or "").strip()
        if not peer_id or not strategy or not argument:
            return None
        query_state = self._default_cognition_query_state("")
        if self._cognition_query_state_enabled:
            query_raw = parsed.get("query_state")
            if not isinstance(query_raw, dict):
                return None
            self._merge_cognition_query_state(query_state, query_raw)
        state_of_mind = self._default_cognition_state_of_mind()
        if self._cognition_state_of_mind_enabled:
            state_raw = parsed.get("state_of_mind")
            if not isinstance(state_raw, dict):
                return None
            self._merge_cognition_state_of_mind(state_of_mind, state_raw)
        return {
            "type": "proposal",
            "peer_id": peer_id,
            "recommended_mode": str(parsed.get("recommended_mode") or "").strip(),
            "strategy": strategy,
            "argument": argument,
            "revision_note": str(parsed.get("revision_note") or "").strip(),
            "query_state": query_state,
            "state_of_mind": state_of_mind,
        }

    def _parse_cognition_system3_review_response(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict) and item.get("type") == "review",
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        critic_id = str(parsed.get("critic_id") or "").strip().lower()
        proposal_peer_id = str(parsed.get("proposal_peer_id") or "").strip().lower()
        comment = str(parsed.get("comment") or "").strip()
        if not critic_id or not proposal_peer_id or not comment:
            return None
        try:
            score = float(parsed.get("score"))
        except (TypeError, ValueError):
            return None
        if score < 0.0:
            score = 0.0
        if score > 10.0:
            score = 10.0
        return {
            "type": "review",
            "critic_id": critic_id,
            "proposal_peer_id": proposal_peer_id,
            "score": score,
            "comment": comment,
        }

    def _parse_cognition_pattern_route_response(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict)
            and item.get("type") == "pattern_route"
            and isinstance(item.get("decision"), str),
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        decision_raw = parsed.get("decision", "")
        if not isinstance(decision_raw, str):
            return None
        decision = decision_raw.strip().lower()
        if decision == "route":
            return {"type": "pattern_route", "decision": "route", "reason": str(parsed.get("reason") or "")}
        if decision != "use":
            return None
        pattern_id = parsed.get("pattern_id")
        if not isinstance(pattern_id, str) or not pattern_id.strip():
            return None
        normalized_pattern_id = self._normalize_asi_pattern_ref(pattern_id)
        if not normalized_pattern_id:
            return None
        return {
            "type": "pattern_route",
            "decision": "use",
            "pattern_id": normalized_pattern_id,
            "reason": str(parsed.get("reason") or ""),
        }

    def _parse_cognition_thinking_response(
        self,
        raw: str,
        *,
        allow_step: bool,
        allow_clarify: bool = True,
    ) -> Optional[Dict[str, Any]]:
        allowed_types = {"final", "act"}
        if allow_clarify:
            allowed_types.add("clarify")
        if allow_step:
            allowed_types.add("step")
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict) and item.get("type") in allowed_types,
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        resp_type = parsed.get("type")
        thought = str(parsed.get("thought") or "")
        query_state_delta = self._normalize_cognition_query_state_delta(parsed.get("query_state_delta", {}))
        state_of_mind_delta = self._normalize_cognition_state_of_mind_delta(parsed.get("state_of_mind_delta", {}))
        if resp_type == "final":
            text = parsed.get("text", "")
            if not isinstance(text, str) or not text.strip():
                answer = parsed.get("answer")
                if isinstance(answer, str) and answer.strip():
                    text = answer
            if not isinstance(text, str):
                return None
            return {
                "type": "final",
                "thought": thought,
                "text": text,
                "query_state_delta": query_state_delta,
                "state_of_mind_delta": state_of_mind_delta,
            }
        if resp_type == "clarify":
            question = parsed.get("question", "")
            if not isinstance(question, str) or not question.strip():
                return None
            return {
                "type": "clarify",
                "thought": thought,
                "question": question,
                "query_state_delta": query_state_delta,
                "state_of_mind_delta": state_of_mind_delta,
            }
        if resp_type == "act":
            actions = self._normalize_cognition_actions(parsed.get("actions", []), limit=2)
            return {
                "type": "act",
                "thought": thought,
                "actions": actions,
                "query_state_delta": query_state_delta,
                "state_of_mind_delta": state_of_mind_delta,
            }
        if resp_type != "step":
            return None
        return {
            "type": "step",
            "thought": thought,
            "reflection": str(parsed.get("reflection") or ""),
            "actions": self._normalize_cognition_actions(parsed.get("actions", []), limit=2),
            "todo_updates": self._normalize_cognition_todos(parsed.get("todo_updates", []), limit=6),
            "query_state_delta": query_state_delta,
            "state_of_mind_delta": state_of_mind_delta,
        }

    def _parse_cognition_final_response(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict) and item.get("type") == "final",
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        text = parsed.get("text", "")
        if not isinstance(text, str) or not text.strip():
            answer = parsed.get("answer")
            if isinstance(answer, str) and answer.strip():
                text = answer
        if not isinstance(text, str):
            return None
        return {
            "type": "final",
            "thought": str(parsed.get("thought") or ""),
            "text": text,
            "query_state_delta": self._normalize_cognition_query_state_delta(parsed.get("query_state_delta", {})),
            "state_of_mind_delta": self._normalize_cognition_state_of_mind_delta(
                parsed.get("state_of_mind_delta", {})
            ),
        }

    def _cognition_result_payload_text(self, payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return self._safe_json({"type": "final", "text": ""})
        payload_type = str(payload.get("type") or "").strip()
        if payload_type == "clarify":
            question = payload.get("question")
            if isinstance(question, str) and question.strip():
                return question.strip()
            return self._safe_json({"type": "clarify", "question": ""})
        if payload_type == "final":
            for key in ("text", "answer", "message", "content"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            value = payload.get("value", payload.get("result"))
            if value is not None:
                return self._stringify_env_for_text(value).strip()
            return self._safe_json({"type": "final", "text": ""})
        return self._safe_json(payload)

    def _parse_cognition_distill_response(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict) and item.get("type") == "distill",
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        reusable_skills = []
        for item in parsed.get("reusable_skills", []):
            normalized = self._normalize_cognition_skill(item)
            if normalized is not None:
                reusable_skills.append(normalized)
        memory_facts: List[Dict[str, Any]] = []
        facts_raw = parsed.get("memory_facts", [])
        if isinstance(facts_raw, list):
            for item in facts_raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                value = str(item.get("value") or "").strip()
                if not name or not value:
                    continue
                memory_facts.append(
                    {
                        "name": name,
                        "value": value,
                        "confidence": self._normalize_cognition_confidence(item.get("confidence"), default=0.5),
                    }
                )
        patterns = []
        for item in parsed.get("patterns", []):
            normalized = self._normalize_cognition_pattern(item)
            if normalized is not None:
                patterns.append(normalized)
        failure_lessons = []
        for item in parsed.get("failure_lessons", []):
            normalized = self._normalize_cognition_lesson(item)
            if normalized is not None:
                failure_lessons.append(normalized)
        safety_rules = []
        for item in parsed.get("safety_rules", []):
            normalized = self._normalize_cognition_safety_rule(item)
            if normalized is not None:
                safety_rules.append(normalized)
        return {
            "type": "distill",
            "reusable_skills": reusable_skills[:8],
            "memory_facts": memory_facts[:8],
            "patterns": patterns[:8],
            "failure_lessons": failure_lessons[:8],
            "safety_rules": safety_rules[:8],
        }

    def _parse_final_response_critic(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict) and isinstance(item.get("fulfilled"), bool),
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        fulfilled = parsed.get("fulfilled")
        if not isinstance(fulfilled, bool):
            return None
        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        issues_raw = parsed.get("issues", [])
        fixes_raw = parsed.get("fix_instructions", [])
        issues: List[str] = []
        fixes: List[str] = []
        if isinstance(issues_raw, list):
            for item in issues_raw:
                if not isinstance(item, str):
                    continue
                value = item.strip()
                if not value or value in issues:
                    continue
                issues.append(value)
                if len(issues) >= 6:
                    break
        if isinstance(fixes_raw, list):
            for item in fixes_raw:
                if not isinstance(item, str):
                    continue
                value = item.strip()
                if not value or value in fixes:
                    continue
                fixes.append(value)
                if len(fixes) >= 6:
                    break

        return {
            "fulfilled": fulfilled,
            "confidence": confidence,
            "issues": issues,
            "fix_instructions": fixes,
        }

    def _format_cognition_strategy_brief(
        self,
        proposal: Optional[Dict[str, Any]],
        average_score: Optional[float] = None,
    ) -> str:
        if not isinstance(proposal, dict):
            return ""
        lines = [
            f"peer_id={str(proposal.get('peer_id') or '').strip()}",
            f"recommended_mode={str(proposal.get('recommended_mode') or '').strip()}",
            f"strategy={self._truncate(str(proposal.get('strategy') or '').strip(), 1200)}",
            f"argument={self._truncate(str(proposal.get('argument') or '').strip(), 1200)}",
        ]
        revision_note = str(proposal.get("revision_note") or "").strip()
        if revision_note:
            lines.append(f"revision_note={self._truncate(revision_note, 500)}")
        if average_score is not None:
            lines.append(f"average_score={average_score:.3f}")
        if self._cognition_query_state_enabled:
            lines.append(
                "query_state="
                + self._truncate(self._safe_json(proposal.get("query_state", {})), 1200)
            )
        if self._cognition_state_of_mind_enabled:
            lines.append(
                "state_of_mind="
                + self._truncate(self._safe_json(proposal.get("state_of_mind", {})), 1200)
            )
        return "\n".join(lines)

    def _score_cognition_system3_proposals(
        self,
        proposals: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        proposal_ids = {
            str(item.get("peer_id") or "").strip().lower()
            for item in proposals
            if isinstance(item, dict)
        }
        score_lists: Dict[str, List[float]] = {peer_id: [] for peer_id in proposal_ids if peer_id}
        for review in reviews:
            if not isinstance(review, dict):
                continue
            target = str(review.get("proposal_peer_id") or "").strip().lower()
            if target not in score_lists:
                continue
            try:
                score = float(review.get("score"))
            except (TypeError, ValueError):
                continue
            score_lists[target].append(score)
        averaged: Dict[str, float] = {}
        for peer_id, scores in score_lists.items():
            if not scores:
                averaged[peer_id] = 0.0
                continue
            averaged[peer_id] = sum(scores) / float(len(scores))
        return averaged

    def _select_cognition_system3_winner(
        self,
        proposals: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
        trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not proposals:
            return None
        scores = self._score_cognition_system3_proposals(proposals, reviews)
        top_score = max(scores.get(str(item.get("peer_id") or "").strip().lower(), 0.0) for item in proposals)
        candidates = [
            item
            for item in proposals
            if abs(scores.get(str(item.get("peer_id") or "").strip().lower(), 0.0) - top_score) < 1e-9
        ]
        preferred = [item for item in candidates if str(item.get("recommended_mode") or "").strip() == "system1"]
        if preferred:
            candidates = preferred
        rng = random.Random(trace_id or "cognition_system3")
        winner = dict(rng.choice(candidates))
        winner["average_score"] = scores.get(str(winner.get("peer_id") or "").strip().lower(), 0.0)
        return winner

    async def _run_cognition_system3(
        self,
        query_state: Dict[str, Any],
        state_of_mind: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        safety_rules: List[Dict[str, Any]],
        trace_id: str,
        turn_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        proposers = list(self._cognition_peer_pools.get("proposers", []))
        critics = list(self._cognition_peer_pools.get("critics", []))
        if not proposers or not critics:
            return None

        async def _run_jobs(
            jobs: List[Callable[[], "asyncio.Future[Optional[Dict[str, Any]]]"]],
        ) -> List[Optional[Dict[str, Any]]]:
            if not jobs:
                return []
            if not self._cognition_system3_parallel or len(jobs) == 1:
                results: List[Optional[Dict[str, Any]]] = []
                for job in jobs:
                    results.append(await job())
                return results
            return list(await asyncio.gather(*(job() for job in jobs)))

        rounds_trace: List[Dict[str, Any]] = []
        current_proposals: List[Dict[str, Any]] = []
        latest_reviews: List[Dict[str, Any]] = []

        for round_index in range(1, self._cognition_system3_rounds + 1):
            proposal_jobs: List[Callable[[], "asyncio.Future[Optional[Dict[str, Any]]]"]] = []
            if round_index == 1 or not current_proposals:
                for peer in proposers:
                    peer_id = str(peer.get("peer_id") or "").strip().lower()

                    async def _proposal_job(peer_cfg: Dict[str, Any] = peer, peer_name: str = peer_id):
                        parsed = await self._call_cognition_json(
                            self._build_cognition_system3_proposal_prompt(
                                query_state,
                                state_of_mind,
                                peer_cfg,
                                memory_hits,
                                tools,
                                safety_rules,
                            ),
                            self._compose_cognition_schema(
                                [
                                    "\"type\":\"proposal\"",
                                    "\"peer_id\":\"...\"",
                                    "\"recommended_mode\":\"system1|system2\"",
                                    "\"strategy\":\"...\"",
                                    "\"argument\":\"...\"",
                                ]
                                + (
                                    [f"\"query_state\":{{{self._cognition_query_state_schema()}}}"]
                                    if self._cognition_query_state_enabled
                                    else []
                                )
                                + (
                                    [f"\"state_of_mind\":{{{self._cognition_state_of_mind_schema()}}}"]
                                    if self._cognition_state_of_mind_enabled
                                    else []
                                )
                            ),
                            self._parse_cognition_system3_proposal_response,
                            f"system3_proposal_r{round_index}_{peer_name}",
                            trace_id,
                            turn_id,
                            model_override=self._resolve_cognition_model_override(
                                "system3",
                                explicit_model=str(peer_cfg.get("model") or ""),
                            ),
                            options_override=self._merge_model_call_options(
                                self._cognition_system3_options,
                                peer_cfg.get("options", {}),
                            ),
                        )
                        if parsed is None:
                            return None
                        parsed["peer_id"] = peer_name
                        return parsed

                    proposal_jobs.append(_proposal_job)
            else:
                proposal_index = {
                    str(item.get("peer_id") or "").strip().lower(): item
                    for item in current_proposals
                    if isinstance(item, dict)
                }
                for peer in proposers:
                    peer_id = str(peer.get("peer_id") or "").strip().lower()
                    current = proposal_index.get(peer_id)
                    if current is None:
                        continue
                    peer_proposals = [
                        item
                        for item in current_proposals
                        if str(item.get("peer_id") or "").strip().lower() != peer_id
                    ]

                    async def _adjust_job(
                        peer_cfg: Dict[str, Any] = peer,
                        peer_name: str = peer_id,
                        current_proposal: Dict[str, Any] = current,
                        other_proposals: List[Dict[str, Any]] = peer_proposals,
                    ):
                        parsed = await self._call_cognition_json(
                            self._build_cognition_system3_adjust_prompt(
                                query_state,
                                state_of_mind,
                                peer_cfg,
                                current_proposal,
                                other_proposals,
                                latest_reviews,
                            ),
                            self._compose_cognition_schema(
                                [
                                    "\"type\":\"proposal\"",
                                    "\"peer_id\":\"...\"",
                                    "\"recommended_mode\":\"system1|system2\"",
                                    "\"strategy\":\"...\"",
                                    "\"argument\":\"...\"",
                                    "\"revision_note\":\"...\"",
                                ]
                                + (
                                    [f"\"query_state\":{{{self._cognition_query_state_schema()}}}"]
                                    if self._cognition_query_state_enabled
                                    else []
                                )
                                + (
                                    [f"\"state_of_mind\":{{{self._cognition_state_of_mind_schema()}}}"]
                                    if self._cognition_state_of_mind_enabled
                                    else []
                                )
                            ),
                            self._parse_cognition_system3_proposal_response,
                            f"system3_adjust_r{round_index}_{peer_name}",
                            trace_id,
                            turn_id,
                            model_override=self._resolve_cognition_model_override(
                                "system3",
                                explicit_model=str(peer_cfg.get("model") or ""),
                            ),
                            options_override=self._merge_model_call_options(
                                self._cognition_system3_options,
                                peer_cfg.get("options", {}),
                            ),
                        )
                        if parsed is None:
                            return current_proposal
                        parsed["peer_id"] = peer_name
                        return parsed

                    proposal_jobs.append(_adjust_job)

            proposal_results = await _run_jobs(proposal_jobs)
            current_proposals = [
                item
                for item in proposal_results
                if isinstance(item, dict) and str(item.get("peer_id") or "").strip()
            ]
            if not current_proposals:
                return None

            review_jobs: List[Callable[[], "asyncio.Future[Optional[Dict[str, Any]]]"]] = []
            for critic in critics:
                critic_id = str(critic.get("peer_id") or "").strip().lower()
                for proposal in current_proposals:
                    proposal_peer_id = str(proposal.get("peer_id") or "").strip().lower()
                    if not proposal_peer_id or proposal_peer_id == critic_id:
                        continue

                    async def _review_job(
                        critic_cfg: Dict[str, Any] = critic,
                        critic_name: str = critic_id,
                        proposal_payload: Dict[str, Any] = proposal,
                        target_peer_id: str = proposal_peer_id,
                    ):
                        parsed = await self._call_cognition_json(
                            self._build_cognition_system3_review_prompt(
                                query_state,
                                state_of_mind,
                                critic_cfg,
                                proposal_payload,
                                memory_hits,
                                tools,
                                safety_rules,
                            ),
                            "{\"type\":\"review\",\"critic_id\":\"...\",\"proposal_peer_id\":\"...\",\"score\":0.0,\"comment\":\"...\"}",
                            self._parse_cognition_system3_review_response,
                            f"system3_review_r{round_index}_{critic_name}_{target_peer_id}",
                            trace_id,
                            turn_id,
                            model_override=self._resolve_cognition_model_override(
                                "system3",
                                explicit_model=str(critic_cfg.get("model") or ""),
                            ),
                            options_override=self._merge_model_call_options(
                                self._cognition_system3_options,
                                critic_cfg.get("options", {}),
                            ),
                        )
                        if parsed is None:
                            return None
                        parsed["critic_id"] = critic_name
                        parsed["proposal_peer_id"] = target_peer_id
                        return parsed

                    review_jobs.append(_review_job)

            review_results = await _run_jobs(review_jobs)
            latest_reviews = [
                item
                for item in review_results
                if isinstance(item, dict)
                and str(item.get("critic_id") or "").strip()
                and str(item.get("proposal_peer_id") or "").strip()
            ]
            rounds_trace.append(
                {
                    "round": round_index,
                    "proposals": current_proposals,
                    "reviews": latest_reviews,
                    "scores": self._score_cognition_system3_proposals(current_proposals, latest_reviews),
                }
            )

        winner = self._select_cognition_system3_winner(current_proposals, latest_reviews, trace_id)
        if winner is None:
            return None
        return {
            "selected_mode": str(winner.get("recommended_mode") or "system2"),
            "winner": winner,
            "strategy_brief": self._format_cognition_strategy_brief(
                winner,
                average_score=float(winner.get("average_score", 0.0)),
            ),
            "rounds": rounds_trace,
            "reviews": latest_reviews,
            "scores": self._score_cognition_system3_proposals(current_proposals, latest_reviews),
        }

    def _extract_json_candidate(self, raw: str) -> Optional[str]:
        if not raw:
            return None
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if fence_match:
            fenced = fence_match.group(1).strip()
            if fenced:
                return fenced
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return None
        return raw[start : end + 1]
    """
    def _extract_json_objects(self, raw: str) -> List[str]:
        objects: List[str] = []
        if not raw:
            return objects
        depth = 0
        start = None
        in_str = False
        escape = False
        for idx, ch in enumerate(raw):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
                continue
            if ch == "\"":
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objects.append(raw[start : idx + 1])
                        start = None
        return objects
    """
    def _extract_json_objects(self, raw: str) -> List[str]:
        """Extract all JSON objects, including nested ones, outermost first."""
        objects = []
        depth = 0
        start = None
        starts_at_depth = {}   # depth → start index
        in_str = False
        escape = False
        for idx, ch in enumerate(raw):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
                continue
            if ch == "\"":
                in_str = True
                continue
            if ch == "{":
                starts_at_depth[depth] = idx   # track start at every depth
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    start = starts_at_depth.pop(depth, None)
                    if start is not None:
                        objects.append(raw[start : idx + 1])
        return objects


    def _extract_json_arrays(self, raw: str) -> List[str]:
        arrays: List[str] = []
        if not raw:
            return arrays
        depth = 0
        start = None
        in_str = False
        escape = False
        for idx, ch in enumerate(raw):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
                continue
            if ch == "\"":
                in_str = True
                continue
            if ch == "[":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "]":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        arrays.append(raw[start : idx + 1])
                        start = None
        return arrays

    def _collect_soft_json_candidates(self, raw: str) -> List[str]:
        if not raw:
            return []
        candidates: List[str] = []
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE):
            snippet = match.group(1).strip()
            if snippet:
                candidates.append(snippet)
        stripped = raw.strip()
        if stripped:
            candidates.append(stripped)
        candidates.extend(self._extract_json_arrays(raw))
        candidates.extend(self._extract_json_objects(raw))
        deduped: List[str] = []
        seen: Dict[str, bool] = {}
        for item in candidates:
            candidate = item.strip()
            if not candidate or candidate in seen:
                continue
            seen[candidate] = True
            deduped.append(candidate)
        return deduped

    def _load_relaxed_json(self, candidate: str) -> Optional[Any]:
        if not isinstance(candidate, str):
            return None
        text = candidate.strip()
        if not text:
            return None
        attempts = [text]
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        if cleaned != text:
            attempts.append(cleaned)
        for attempt in attempts:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
        if not self._soft_json_parsing_enabled:
            return None
        literal_attempts = [cleaned]
        normalized_literals = re.sub(r"\bnull\b", "None", cleaned, flags=re.IGNORECASE)
        normalized_literals = re.sub(r"\btrue\b", "True", normalized_literals, flags=re.IGNORECASE)
        normalized_literals = re.sub(r"\bfalse\b", "False", normalized_literals, flags=re.IGNORECASE)
        if normalized_literals != cleaned:
            literal_attempts.append(normalized_literals)
        for attempt in literal_attempts:
            try:
                return ast.literal_eval(attempt)
            except (ValueError, SyntaxError):
                continue
        return None

    def _sanitize_tool_fn_name(self, tool_id: str) -> str:
        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", tool_id)
        if not safe:
            safe = "tool"
        if safe[0].isdigit():
            safe = f"tool_{safe}"
        return safe

    def _default_tool_name(self, tool_id: str) -> str:
        cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", tool_id).strip()
        if not cleaned:
            return "Generated Tool"
        return " ".join(part.capitalize() for part in cleaned.split())

    def _sanitize_tool_id(self, tool_id: str) -> str:
        text = str(tool_id or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^a-z0-9._]+", ".", text)
        text = re.sub(r"\.+", ".", text).strip(".")
        return text

    def _normalize_permission_list(self, value: Any, default: Optional[List[str]] = None) -> List[str]:
        if isinstance(value, str):
            items = [item.strip().lower() for item in value.split(",")]
        elif isinstance(value, (list, tuple, set)):
            items = [str(item).strip().lower() for item in value]
        else:
            items = list(default or [])
        normalized: List[str] = []
        seen: Set[str] = set()
        for item in items:
            if item not in TIER_MAP or item in seen:
                continue
            normalized.append(item)
            seen.add(item)
        if normalized:
            return normalized
        fallback = list(default or ["tier1"])
        result: List[str] = []
        seen_default: Set[str] = set()
        for item in fallback:
            text = str(item).strip().lower()
            if text not in TIER_MAP or text in seen_default:
                continue
            result.append(text)
            seen_default.add(text)
        return result or ["tier1"]

    def _highest_permission_tier(self, perms: List[str]) -> int:
        highest = 0
        for perm in perms:
            highest = max(highest, TIER_MAP.get(str(perm).strip().lower(), 0))
        return highest

    def _permissions_for_tier(self, tier: Any) -> List[str]:
        try:
            parsed = int(tier)
        except (TypeError, ValueError):
            parsed = 1
        parsed = max(0, min(parsed, 3))
        return [f"tier{parsed}"]

    def _default_risk_for_tier(self, tier: int) -> str:
        if tier <= 0:
            return "LOW"
        if tier == 1:
            return "MEDIUM"
        return "HIGH"

    def _normalize_sandbox_profile(self, value: Any, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        fallback = dict(default or {"network": False, "fs": "read"})
        profile = dict(value) if isinstance(value, dict) else fallback
        network = profile.get("network", fallback.get("network", False))
        fs_mode = str(profile.get("fs", fallback.get("fs", "read")) or "read").strip().lower()
        if fs_mode not in {"none", "read", "write"}:
            fs_mode = str(fallback.get("fs", "read") or "read").strip().lower()
        return {"network": bool(network), "fs": fs_mode}

    def _load_registry_specs(self) -> List[Dict[str, Any]]:
        registry_path = os.path.join(self._settings.workspace_root, "config", "tool_registry.json")
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return []
        tools = raw.get("tools", [])
        if not isinstance(tools, list):
            return []
        return [item for item in tools if isinstance(item, dict)]

    def _find_existing_tool_spec(
        self,
        tool_id: str,
        *,
        generated_specs: Optional[List[Dict[str, Any]]] = None,
        registry_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        candidates = generated_specs if isinstance(generated_specs, list) else self._load_generated_specs()
        for item in candidates:
            if str(item.get("tool_id") or "").strip() == tool_id:
                return dict(item)
        registry_items = registry_specs if isinstance(registry_specs, list) else self._load_registry_specs()
        for item in registry_items:
            if str(item.get("tool_id") or "").strip() == tool_id:
                return dict(item)
        existing = self._tools.get_tool(tool_id)
        if isinstance(existing, dict):
            return dict(existing)
        return None

    def _normalize_tool_generation_spec(self, spec: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(spec, dict):
            return None, "tool generation spec must be an object"
        name = str(spec.get("name") or "").strip()
        request = str(spec.get("request") or spec.get("description") or spec.get("goal") or "").strip()
        goals: List[str] = []
        raw_goals = spec.get("goals", [])
        if isinstance(raw_goals, list):
            goals = [str(item).strip() for item in raw_goals if str(item).strip()]
        if not request and goals:
            request = "; ".join(goals)
        if not request and name:
            request = name
        if not request:
            return None, "tool generation spec must include request, description, goal, or goals"

        namespace_prefix = self._sanitize_tool_id(
            str(spec.get("namespace_prefix") or spec.get("tool_namespace") or "").strip()
        )
        max_tools_setting = spec.get("max_tools", self._settings.orchestrator.get("tool_blueprint_max_tools", 4))
        try:
            max_tools = int(max_tools_setting)
        except (TypeError, ValueError):
            max_tools = 4
        max_tools = max(1, min(max_tools, 12))

        default_permissions = self._normalize_permission_list(spec.get("default_permissions"), default=["tier1"])
        default_tier = self._highest_permission_tier(default_permissions)
        if "default_tier" in spec:
            try:
                default_tier = max(default_tier, int(spec.get("default_tier")))
            except (TypeError, ValueError):
                pass
        default_tier = max(0, min(default_tier, 3))
        default_permissions = self._permissions_for_tier(default_tier)

        default_risk = str(
            spec.get("default_risk_level") or spec.get("risk_level_default") or self._default_risk_for_tier(default_tier)
        ).strip().upper()
        if default_risk not in {"LOW", "MEDIUM", "HIGH"}:
            default_risk = self._default_risk_for_tier(default_tier)

        constraints = spec.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}

        default_sandbox_profile = self._normalize_sandbox_profile(spec.get("default_sandbox_profile"))
        tool_hints = spec.get("tool_hints", [])
        if not isinstance(tool_hints, list):
            tool_hints = []
        normalized_hints: List[Any] = []
        for item in tool_hints[:20]:
            if isinstance(item, (dict, str)):
                normalized_hints.append(item)

        default_capabilities = spec.get("default_capabilities", [])
        if not isinstance(default_capabilities, list):
            default_capabilities = []
        capabilities = [str(item).strip() for item in default_capabilities if str(item).strip()]

        try:
            default_timeouts_ms = int(spec.get("default_timeouts_ms", 2000))
        except (TypeError, ValueError):
            default_timeouts_ms = 2000
        default_timeouts_ms = max(500, min(default_timeouts_ms, 600000))

        raw_resource_limits = spec.get("default_resource_limits", {})
        if not isinstance(raw_resource_limits, dict):
            raw_resource_limits = {}
        try:
            cpu_ms = int(raw_resource_limits.get("cpu_ms", 200))
        except (TypeError, ValueError):
            cpu_ms = 200
        try:
            mem_mb = int(raw_resource_limits.get("mem_mb", 64))
        except (TypeError, ValueError):
            mem_mb = 64
        resource_limits = {
            "cpu_ms": max(50, min(cpu_ms, 600000)),
            "mem_mb": max(16, min(mem_mb, 4096)),
        }

        return {
            "name": name,
            "request": request,
            "goals": goals,
            "constraints": constraints,
            "namespace_prefix": namespace_prefix,
            "max_tools": max_tools,
            "default_permissions": default_permissions,
            "default_tier": default_tier,
            "default_risk_level": default_risk,
            "default_sandbox_profile": default_sandbox_profile,
            "default_capabilities": capabilities,
            "tool_hints": normalized_hints,
            "default_timeouts_ms": default_timeouts_ms,
            "default_resource_limits": resource_limits,
        }, None

    def _build_tool_blueprint_prompt(self, spec: Dict[str, Any]) -> str:
        existing_tool_ids = [str(item.get("tool_id") or "").strip() for item in self._load_registry_specs()]
        existing_tool_ids = [item for item in existing_tool_ids if item]
        return (
            "TOOL BLUEPRINT PROMPT\n\n"
            "Design a small set of new runtime tools from a high-level specification.\n"
            "Return STRICT JSON only.\n\n"
            "OUTPUT SCHEMA:\n"
            "{\"type\":\"tool_blueprint_plan\",\"summary\":\"...\",\"tools\":["
            "{\"tool_id\":\"...\",\"name\":\"...\",\"description\":\"...\",\"capabilities\":[\"...\"],"
            "\"input_schema\":{...},\"output_schema\":{...},\"required_permissions\":[\"tier1\"],"
            "\"tier\":1,\"risk_level_default\":\"MEDIUM\",\"timeouts_ms\":2000,"
            "\"resource_limits\":{\"cpu_ms\":200,\"mem_mb\":64},"
            "\"sandbox_profile\":{\"network\":false,\"fs\":\"read\"},"
            "\"dependencies\":[\"...\"],\"examples\":[...]}]}\n\n"
            "RULES:\n"
            "- Do not generate code in this stage.\n"
            f"- Propose at most {int(spec.get('max_tools', 4))} tools.\n"
            "- Prefer the smallest tool set that fully covers the request.\n"
            "- Each tool_id must be lowercase and dot-separated.\n"
            "- Avoid duplicates with already installed tool_ids.\n"
            "- Use conservative permissions and sandbox defaults unless the request clearly needs more power.\n"
            "- input_schema and output_schema may be lightweight hints, but must be valid JSON objects.\n\n"
            "HIGH-LEVEL SPEC:\n"
            f"{self._safe_json(spec)}\n\n"
            "ALREADY INSTALLED TOOL IDS:\n"
            f"{self._truncate(self._safe_json(existing_tool_ids), 5000)}\n"
        )

    def _parse_tool_blueprint_response(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict) and item.get("type") == "tool_blueprint_plan",
        )
        if not isinstance(parsed, dict):
            return None
        tools = parsed.get("tools", [])
        if not isinstance(tools, list):
            return None
        return {
            "summary": str(parsed.get("summary") or "").strip(),
            "tools": [dict(item) for item in tools if isinstance(item, dict)],
        }

    def _normalize_tool_blueprint(
        self,
        item: Dict[str, Any],
        generation_spec: Dict[str, Any],
        index: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        raw_tool_id = self._sanitize_tool_id(str(item.get("tool_id") or item.get("name") or "").strip())
        if not raw_tool_id:
            raw_tool_id = f"tool_{index + 1}"
        namespace_prefix = str(generation_spec.get("namespace_prefix") or "").strip()
        if namespace_prefix and not raw_tool_id.startswith(namespace_prefix + "."):
            raw_tool_id = f"{namespace_prefix}.{raw_tool_id}"
        tool_id = self._sanitize_tool_id(raw_tool_id)
        if not tool_id:
            return None, f"Tool blueprint #{index + 1} is missing a valid tool_id"

        description = str(item.get("description") or "").strip()
        if not description:
            return None, f"Tool blueprint '{tool_id}' is missing description"

        raw_name = str(item.get("name") or "").strip()
        name = raw_name if raw_name else self._default_tool_name(tool_id)

        capabilities = item.get("capabilities", generation_spec.get("default_capabilities", []))
        if not isinstance(capabilities, list):
            capabilities = generation_spec.get("default_capabilities", [])
        normalized_capabilities = [str(value).strip() for value in capabilities if str(value).strip()]

        required_permissions = self._normalize_permission_list(
            item.get("required_permissions"),
            default=generation_spec.get("default_permissions", ["tier1"]),
        )
        tier = self._highest_permission_tier(required_permissions)
        if "tier" in item:
            try:
                tier = max(tier, int(item.get("tier")))
            except (TypeError, ValueError):
                pass
        tier = max(0, min(tier, 3))
        required_permissions = self._permissions_for_tier(tier)

        risk_level = str(item.get("risk_level_default") or generation_spec.get("default_risk_level") or "").strip().upper()
        if risk_level not in {"LOW", "MEDIUM", "HIGH"}:
            risk_level = self._default_risk_for_tier(tier)

        sandbox_profile = self._normalize_sandbox_profile(
            item.get("sandbox_profile"),
            default=generation_spec.get("default_sandbox_profile"),
        )

        input_schema = item.get("input_schema", {})
        if not isinstance(input_schema, dict):
            input_schema = {}
        output_schema = item.get("output_schema", {})
        if not isinstance(output_schema, dict):
            output_schema = {}

        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list):
            dependencies = []
        normalized_dependencies = [str(dep).strip() for dep in dependencies if str(dep).strip()]

        examples = item.get("examples", [])
        if not isinstance(examples, list):
            examples = []

        try:
            timeouts_ms = int(item.get("timeouts_ms", generation_spec.get("default_timeouts_ms", 2000)))
        except (TypeError, ValueError):
            timeouts_ms = int(generation_spec.get("default_timeouts_ms", 2000))
        timeouts_ms = max(500, min(timeouts_ms, 600000))

        resource_limits_value = item.get("resource_limits", generation_spec.get("default_resource_limits", {}))
        if not isinstance(resource_limits_value, dict):
            resource_limits_value = generation_spec.get("default_resource_limits", {})
        try:
            cpu_ms = int(resource_limits_value.get("cpu_ms", generation_spec.get("default_resource_limits", {}).get("cpu_ms", 200)))
        except (TypeError, ValueError):
            cpu_ms = int(generation_spec.get("default_resource_limits", {}).get("cpu_ms", 200))
        try:
            mem_mb = int(resource_limits_value.get("mem_mb", generation_spec.get("default_resource_limits", {}).get("mem_mb", 64)))
        except (TypeError, ValueError):
            mem_mb = int(generation_spec.get("default_resource_limits", {}).get("mem_mb", 64))
        resource_limits = {
            "cpu_ms": max(50, min(cpu_ms, 600000)),
            "mem_mb": max(16, min(mem_mb, 4096)),
        }

        return {
            "tool_id": tool_id,
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "capabilities": normalized_capabilities,
            "required_permissions": required_permissions,
            "tier": tier,
            "risk_level_default": risk_level,
            "sandbox_profile": sandbox_profile,
            "dependencies": normalized_dependencies,
            "examples": examples,
            "timeouts_ms": timeouts_ms,
            "resource_limits": resource_limits,
            "version": str(item.get("version") or "0.1.0").strip() or "0.1.0",
        }, None

    def _persist_tool_generation_run(self, payload: Dict[str, Any], trace_id: str) -> str:
        os.makedirs(self._tool_generation_runs_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_trace = self._sanitize_tool_fn_name(trace_id or new_id())
        run_path = os.path.join(self._tool_generation_runs_dir, f"tool_generation_{ts}_{safe_trace}.json")
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return run_path

    def _build_tool_codegen_prompt(
        self,
        tool_id: str,
        description: str,
        name_hint: str,
        input_schema_hint: Optional[Dict[str, Any]],
        output_schema_hint: Optional[Dict[str, Any]],
        capabilities_hint: Optional[List[str]] = None,
    ) -> str:
        input_hint = self._safe_json(input_schema_hint) if isinstance(input_schema_hint, dict) else "null"
        output_hint = self._safe_json(output_schema_hint) if isinstance(output_schema_hint, dict) else "null"
        allow_local_subprocess = bool(
            tool_id.startswith("embodiment.generated.")
            or (
                isinstance(capabilities_hint, list)
                and any(str(item).strip().lower() == "embodiment" for item in capabilities_hint)
            )
        )
        extra_rule = ""
        if allow_local_subprocess:
            extra_rule = (
                "- For embodiment-generated tools only, local subprocess calls to device CLI utilities are allowed, "
                "but never use shell=True and never use network access.\n"
            )
        return (
            "TOOL CODEGEN PROMPT\n\n"
            "Generate an implementation for a new runtime tool that fulfill the description in TOOL REQUEST.\n"
            "Return STRICT JSON only.\n\n"
            "OUTPUT SCHEMA:\n"
            "{\"type\":\"tool_codegen\",\"name\":\"...\",\"input_schema\":{...},\"output_schema\":{...},"
            "\"code\":\"...\",\"dependencies\":[\"...\"]}\n\n"
            "RULES:\n"
            "- \"code\" is Python function-body code only; do not include a def line.\n"
            "- The generated function receives (args, workspace_root).\n"
            "- The code must return a tuple: (result_dict, io_dict).\n"
            "- Import every module you use inside the generated body, including Python stdlib modules.\n"
            "- Do not rely on module-level imports outside the generated body.\n"
            "- Handle errors in code and return an error result instead of raising when possible.\n"
            "- On invalid input, return a structured error like {\"status\":\"ERROR\",\"error\":\"...\"}.\n"
            "- For file/path args, support workspace:/ refs and normalize to absolute workspace paths before I/O.\n"
            f"{extra_rule}"
            "- \"dependencies\" must include only non-stdlib packages; use [] if none.\n"
            "- \"input_schema\" and \"output_schema\" must be valid JSON Schema objects.\n"
            "- If schema hints are provided, keep them unless clearly invalid.\n\n"
            "- The tool will be executed to fulfill similar tasks so it should be general and rigorous"
            "TOOL REQUEST:\n"
            f"- tool_id: {tool_id}\n"
            f"- description: {description}\n"
            f"- name_hint: {name_hint if name_hint else '(none)'}\n"
            f"- input_schema_hint: {input_hint}\n"
            f"- output_schema_hint: {output_hint}\n"
        )

    def _parse_tool_codegen_response(self, raw: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_json_payload(
            raw,
            validator=lambda item: isinstance(item, dict)
            and item.get("type") == "tool_codegen",
        )
        if parsed is None or not isinstance(parsed, dict):
            return None
        if parsed.get("type") != "tool_codegen":
            return None
        name = parsed.get("name", "")
        input_schema = parsed.get("input_schema", {})
        output_schema = parsed.get("output_schema", {})
        code = parsed.get("code", "")
        dependencies = parsed.get("dependencies", [])
        if not isinstance(name, str):
            name = ""
        if not isinstance(input_schema, dict):
            return None
        if not isinstance(output_schema, dict):
            return None
        if not isinstance(code, str) or not code.strip():
            return None
        if not isinstance(dependencies, list):
            return None
        deps = []
        for dep in dependencies:
            if isinstance(dep, str):
                item = dep.strip()
                if item:
                    deps.append(item)
        return {
            "name": name.strip(),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "code": code,
            "dependencies": deps,
        }

    def _validate_generated_tool_code(self, code: str, allow_local_subprocess: bool = False) -> Optional[str]:
        if not isinstance(code, str) or not code.strip():
            return "Generated code is empty."
        body = code.splitlines()
        wrapper_lines = ["def _tool_impl(args, workspace_root):"]
        wrapper_lines.extend(f"    {line}" for line in body)
        source = "\n".join(wrapper_lines) + "\n"
        try:
            module = ast.parse(source)
        except SyntaxError as exc:
            return f"Generated code has invalid syntax: {exc.msg}"
        if not module.body or not isinstance(module.body[0], ast.FunctionDef):
            return "Generated code wrapper is invalid."
        fn_node = module.body[0]
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(fn_node))
        if not has_return:
            return "Generated code must include at least one return statement."
        safety_enabled = self._coerce_optional_bool(
            self._settings.orchestrator.get("create_tool_code_safety_enabled", True)
        )
        if safety_enabled is False:
            return None
        default_blocked_modules = {
            "socket",
            "http",
            "urllib",
            "ftplib",
            "telnetlib",
            "requests",
            "paramiko",
        }
        configured_blocked_modules = self._settings.orchestrator.get("create_tool_blocked_modules")
        if isinstance(configured_blocked_modules, list):
            blocked_modules = {
                str(item).strip().split(".", 1)[0]
                for item in configured_blocked_modules
                if str(item).strip()
            }
        else:
            blocked_modules = set(default_blocked_modules)
        extra_blocked_modules = self._settings.orchestrator.get("create_tool_extra_blocked_modules", [])
        if isinstance(extra_blocked_modules, list):
            blocked_modules.update(
                str(item).strip().split(".", 1)[0]
                for item in extra_blocked_modules
                if str(item).strip()
            )
        allowed_imports = self._settings.orchestrator.get("create_tool_allowed_imports", [])
        if isinstance(allowed_imports, list):
            blocked_modules.difference_update(
                str(item).strip().split(".", 1)[0]
                for item in allowed_imports
                if str(item).strip()
            )
        if not allow_local_subprocess:
            blocked_modules.add("subprocess")
        blocked_calls = {"eval", "exec", "compile", "__import__"}
        blocked_call_prefixes = (
            "os.system",
            "os.popen",
            "os.spawn",
            "shutil.rmtree",
            "pty.",
        )
        if not allow_local_subprocess:
            blocked_call_prefixes = blocked_call_prefixes + ("subprocess.",)
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in blocked_modules:
                        return f"Blocked import in generated code: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module_name = (node.module or "").split(".", 1)[0]
                if module_name in blocked_modules:
                    return f"Blocked import in generated code: {node.module}"
            elif isinstance(node, ast.Call):
                call_name = self._call_target_name(node.func)
                if call_name in blocked_calls:
                    return f"Blocked call in generated code: {call_name}"
                if allow_local_subprocess and call_name.startswith("subprocess."):
                    for keyword in node.keywords:
                        if keyword.arg == "shell":
                            if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                                return "Embodiment-generated code may not use shell=True."
                for prefix in blocked_call_prefixes:
                    if call_name.startswith(prefix):
                        return f"Blocked call in generated code: {call_name}"
        return None

    def _call_target_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return ""

    def _load_generated_specs(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._generated_tools_path):
            return []
        try:
            with open(self._generated_tools_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _write_generated_specs(self, specs: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self._generated_tools_path), exist_ok=True)
        with open(self._generated_tools_path, "w", encoding="utf-8") as f:
            json.dump(specs, f, ensure_ascii=True, indent=2)

    def _render_generated_tools(self, specs: List[Dict[str, Any]]) -> str:
        lines = [
            "# Auto-generated file. Do not edit by hand.",
            "from typing import Any, Dict, Tuple",
            "",
        ]
        registry_lines = []
        for spec in specs:
            tool_id = spec.get("tool_id", "")
            code = spec.get("code", "")
            tier = int(spec.get("tier", 1))
            fn_name = self._sanitize_tool_fn_name(tool_id)
            lines.append(f"def {fn_name}(args: Dict[str, Any], workspace_root: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:")
            lines.append("    try:")
            if code:
                for line in code.splitlines():
                    lines.append(f"        {line}")
            else:
                lines.append("        return {\"status\":\"ERROR\",\"error\":\"No implementation\"}, {}")
            lines.append("    except Exception as exc:")
            lines.append("        return {\"status\":\"ERROR\",\"error\":f\"Tool exception: {exc}\"}, {}")
            lines.append("    return {\"status\":\"ERROR\",\"error\":\"Tool implementation did not return (result, io).\"}, {}")
            lines.append("")
            registry_lines.append(f"        \"{tool_id}\": {{\"fn\": {fn_name}, \"tier\": {tier}}},")

        lines.append("def generated_registry() -> Dict[str, Dict[str, Any]]:")
        if registry_lines:
            lines.append("    return {")
            lines.extend(registry_lines)
            lines.append("    }")
        else:
            lines.append("    return {}")
        lines.append("")
        return "\n".join(lines)

    def _update_tool_registry(self, spec: Dict[str, Any]) -> Optional[str]:
        registry_path = os.path.join(self._settings.workspace_root, "config", "tool_registry.json")
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return "Unable to load tool registry"
        tools = raw.get("tools", [])
        tool_id = spec.get("tool_id")
        if any(t.get("tool_id") == tool_id for t in tools):
            return None
        tools.append({
            "tool_id": tool_id,
            "name": spec.get("name"),
            "description": spec.get("description", ""),
            "implements_capability_id": spec.get("implements_capability_id"),
            "implements_abstract_capability": spec.get("implements_abstract_capability"),
            "version": spec.get("version", "0.1.0"),
            "input_schema": spec.get("input_schema", {}),
            "output_schema": spec.get("output_schema", {}),
            "dependencies": spec.get("dependencies", []),
            "capabilities": spec.get("capabilities", []),
            "risk_level_default": spec.get("risk_level_default", "MEDIUM"),
            "required_permissions": spec.get("required_permissions", ["tier1"]),
            "examples": spec.get("examples", []),
            "timeouts_ms": spec.get("timeouts_ms", 2000),
            "resource_limits": spec.get("resource_limits", {"cpu_ms": 200, "mem_mb": 64}),
            "sandbox_profile": spec.get("sandbox_profile", {"network": False, "fs": "read"}),
        })
        raw["tools"] = tools
        try:
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=True, indent=2)
        except Exception:
            return "Unable to write tool registry"
        return None

    async def generate_tools_from_spec(
        self,
        spec: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_trace_id = trace_id or new_id()
        normalized_spec, spec_error = self._normalize_tool_generation_spec(spec)
        if spec_error:
            return {"status": "ERROR", "error": spec_error}
        assert normalized_spec is not None

        if not self._auto_approve_all:
            token = await self._request_permission("generate_tools_from_spec", "Generate tools from a high-level spec")
            if not token:
                return {"status": "DENIED", "error": "Not approved"}

        blueprint_schema = (
            "{\"type\":\"tool_blueprint_plan\",\"summary\":\"...\",\"tools\":["
            "{\"tool_id\":\"...\",\"name\":\"...\",\"description\":\"...\"}]}"
        )
        base_prompt = self._build_tool_blueprint_prompt(normalized_spec)
        prompt = base_prompt
        planning_context = {"summary": "", "memory": [], "suppress_summary_memory": True}
        blueprint: Optional[Dict[str, Any]] = None
        validation_error = ""
        attempts_setting = self._settings.orchestrator.get("tool_blueprint_max_attempts", 3)
        try:
            max_attempts = int(attempts_setting)
        except (TypeError, ValueError):
            max_attempts = 3
        max_attempts = max(1, min(max_attempts, 6))

        for attempt in range(1, max_attempts + 1):
            raw = await self._generate_response(prompt, PRIMARY_REASONING_MODE, planning_context, resolved_trace_id, None)
            self._log_mode_event(
                PRIMARY_REASONING_MODE,
                "tool_blueprint_raw",
                {"trace_id": resolved_trace_id, "attempt": attempt, "raw": self._truncate(raw, 2000)},
            )
            parsed = None
            raw_text = raw if isinstance(raw, str) else ""
            empty_raw = not raw_text.strip()
            if not empty_raw:
                parsed = self._parse_tool_blueprint_response(raw_text)
                if parsed is None:
                    repair_prompt = self._build_json_repair_prompt(blueprint_schema, raw_text)
                    repaired = await self._generate_response(
                        repair_prompt,
                        PRIMARY_REASONING_MODE,
                        planning_context,
                        resolved_trace_id,
                        None,
                    )
                    self._log_mode_event(
                        PRIMARY_REASONING_MODE,
                        "tool_blueprint_repair_raw",
                        {"trace_id": resolved_trace_id, "attempt": attempt, "raw": self._truncate(repaired, 2000)},
                    )
                    parsed = self._parse_tool_blueprint_response(repaired)
            if parsed is None:
                validation_error = (
                    "Tool blueprint generation returned empty output."
                    if empty_raw
                    else "Tool blueprint generation returned invalid JSON."
                )
            else:
                normalized_tools: List[Dict[str, Any]] = []
                normalization_errors: List[str] = []
                seen_tool_ids: Set[str] = set()
                for index, item in enumerate(parsed.get("tools", [])[: normalized_spec["max_tools"]]):
                    normalized_item, item_error = self._normalize_tool_blueprint(item, normalized_spec, index)
                    if item_error:
                        normalization_errors.append(item_error)
                        continue
                    assert normalized_item is not None
                    tool_id = str(normalized_item.get("tool_id") or "").strip()
                    if tool_id in seen_tool_ids:
                        normalization_errors.append(f"Duplicate tool_id in blueprint: {tool_id}")
                        continue
                    seen_tool_ids.add(tool_id)
                    normalized_tools.append(normalized_item)
                if not normalized_tools:
                    validation_error = "; ".join(normalization_errors[:4]) or "Tool blueprint contained no valid tools."
                else:
                    blueprint = {
                        "summary": str(parsed.get("summary") or "").strip(),
                        "tools": normalized_tools,
                        "normalization_errors": normalization_errors,
                    }
                    break
            self._log_mode_event(
                PRIMARY_REASONING_MODE,
                "tool_blueprint_validation_error",
                {
                    "trace_id": resolved_trace_id,
                    "attempt": attempt,
                    "error": self._truncate(validation_error, 1200),
                },
            )
            prompt = (
                f"{base_prompt}\n\n"
                "Previous blueprint failed validation. Regenerate a full replacement and fix every issue.\n"
                f"Validation error: {validation_error}\n"
            )

        if blueprint is None:
            run_path = self._persist_tool_generation_run(
                {
                    "ts": utc_now_iso(),
                    "trace_id": resolved_trace_id,
                    "status": "ERROR",
                    "error": validation_error,
                    "spec": spec,
                    "normalized_spec": normalized_spec,
                },
                resolved_trace_id,
            )
            return {
                "status": "ERROR",
                "error": f"Failed to generate a valid tool blueprint: {validation_error}",
                "run_path": run_path,
            }

        generated_specs = self._load_generated_specs()
        registry_specs = self._load_registry_specs()
        results: List[Dict[str, Any]] = []
        created_count = 0
        skipped_count = 0
        failed_count = 0

        for tool_spec in blueprint["tools"]:
            tool_id = str(tool_spec.get("tool_id") or "").strip()
            existing = self._find_existing_tool_spec(
                tool_id,
                generated_specs=generated_specs,
                registry_specs=registry_specs,
            )
            if existing is not None:
                skipped_count += 1
                results.append(
                    {
                        "tool_id": tool_id,
                        "name": str(existing.get("name") or self._default_tool_name(tool_id)),
                        "status": "SKIPPED",
                        "reason": "Tool already exists",
                    }
                )
                continue

            create_result = await self._handle_create_tool(tool_spec, resolved_trace_id, skip_permission_request=True)
            result_payload = create_result.get("result", {})
            if create_result.get("status") == "APPROVED":
                if isinstance(result_payload, dict) and result_payload.get("already_exists") is True:
                    skipped_count += 1
                    results.append(
                        {
                            "tool_id": tool_id,
                            "name": str(result_payload.get("name") or self._default_tool_name(tool_id)),
                            "status": "SKIPPED",
                            "reason": "Tool already exists",
                        }
                    )
                else:
                    created_count += 1
                    results.append(
                        {
                            "tool_id": tool_id,
                            "name": str(result_payload.get("name") or tool_spec.get("name") or self._default_tool_name(tool_id)),
                            "status": "APPROVED",
                        }
                    )
                    generated_specs.append(dict(tool_spec))
                    registry_specs.append(dict(tool_spec))
            else:
                failed_count += 1
                results.append(
                    {
                        "tool_id": tool_id,
                        "name": str(tool_spec.get("name") or self._default_tool_name(tool_id)),
                        "status": str(create_result.get("status") or "ERROR"),
                        "error": str(create_result.get("error") or "Unknown tool generation failure"),
                    }
                )

        run_payload = {
            "ts": utc_now_iso(),
            "trace_id": resolved_trace_id,
            "status": "APPROVED" if (created_count or skipped_count) else "ERROR",
            "spec": spec,
            "normalized_spec": normalized_spec,
            "blueprint": blueprint,
            "results": results,
            "summary": {
                "planned_count": len(blueprint["tools"]),
                "created_count": created_count,
                "skipped_count": skipped_count,
                "failed_count": failed_count,
            },
        }
        run_path = self._persist_tool_generation_run(run_payload, resolved_trace_id)
        final_status = "APPROVED" if (created_count or skipped_count) else "ERROR"
        return {
            "status": final_status,
            "result": {
                "summary": str(blueprint.get("summary") or "").strip(),
                "planned_count": len(blueprint["tools"]),
                "created_count": created_count,
                "skipped_count": skipped_count,
                "failed_count": failed_count,
                "tools": results,
                "run_path": run_path,
            },
        }

    async def bootstrap_embodiment(self, manifest_path: Optional[str] = None) -> Dict[str, Any]:
        from embodiment.bootstrap import EmbodimentBootstrap

        bootstrap = EmbodimentBootstrap(settings=self._settings)
        trace_id = new_id()

        async def _create_tool(spec: Dict[str, Any]) -> Dict[str, Any]:
            return await self._handle_create_tool(spec, trace_id)

        return await bootstrap.apply(_create_tool, manifest_path=manifest_path)

    async def _handle_create_tool(
        self,
        args: Dict[str, Any],
        trace_id: str,
        skip_permission_request: bool = False,
    ) -> Dict[str, Any]:
        if not self._auto_approve_all and not skip_permission_request:
            token = await self._request_permission("create_tool", "Create a new tool")
            if not token:
                return {"status": "DENIED", "error": "Not approved"}

        required_fields = ["tool_id", "description"]
        missing = [field for field in required_fields if field not in args]
        if missing:
            return {"status": "ERROR", "error": f"Missing fields: {', '.join(missing)}"}

        tool_id = args.get("tool_id", "")
        if not isinstance(tool_id, str) or not tool_id:
            return {"status": "ERROR", "error": "Invalid tool_id"}
        description = args.get("description", "")
        if not isinstance(description, str) or not description.strip():
            return {"status": "ERROR", "error": "Invalid description"}
        description = description.strip()
        name_hint = args.get("name", "")
        if not isinstance(name_hint, str):
            name_hint = ""
        input_schema_hint = args.get("input_schema")
        if input_schema_hint is not None and not isinstance(input_schema_hint, dict):
            return {"status": "ERROR", "error": "input_schema must be an object when provided"}
        output_schema_hint = args.get("output_schema")
        if output_schema_hint is not None and not isinstance(output_schema_hint, dict):
            return {"status": "ERROR", "error": "output_schema must be an object when provided"}
        capabilities_hint = args.get("capabilities", [])
        if capabilities_hint is not None and not isinstance(capabilities_hint, list):
            return {"status": "ERROR", "error": "capabilities must be an array when provided"}

        specs = self._load_generated_specs()
        existing_tool = self._find_existing_tool_spec(tool_id, generated_specs=specs)
        if existing_tool is not None:
            return {
                "status": "APPROVED",
                "result": {
                    "tool_id": tool_id,
                    "name": str(existing_tool.get("name") or name_hint or self._default_tool_name(tool_id)),
                    "dependencies": list(existing_tool.get("dependencies", []))
                    if isinstance(existing_tool.get("dependencies", []), list)
                    else [],
                    "already_exists": True,
                },
            }

        codegen_schema = (
            "{\"type\":\"tool_codegen\",\"name\":\"...\",\"input_schema\":{...},\"output_schema\":{...},"
            "\"code\":\"...\",\"dependencies\":[\"...\"]}"
        )
        base_prompt = self._build_tool_codegen_prompt(
            tool_id=tool_id,
            description=description,
            name_hint=name_hint.strip(),
            input_schema_hint=input_schema_hint if isinstance(input_schema_hint, dict) else None,
            output_schema_hint=output_schema_hint if isinstance(output_schema_hint, dict) else None,
            capabilities_hint=capabilities_hint if isinstance(capabilities_hint, list) else None,
        )
        prompt = base_prompt
        codegen_context = {"summary": "", "memory": [], "suppress_summary_memory": True}
        generated: Optional[Dict[str, Any]] = None
        validation_error = ""
        max_attempts_setting = self._settings.orchestrator.get("create_tool_max_attempts", 4)
        max_attempts: Optional[int]
        if isinstance(max_attempts_setting, str):
            normalized_attempts = max_attempts_setting.strip().lower()
            if normalized_attempts in {"infinite", "infinity"}:
                max_attempts = None
            else:
                try:
                    max_attempts = int(normalized_attempts)
                except (TypeError, ValueError):
                    max_attempts = 4
        else:
            try:
                max_attempts = int(max_attempts_setting)
            except (TypeError, ValueError):
                max_attempts = 4
        if max_attempts is not None:
            if max_attempts < 1:
                max_attempts = 1
            if max_attempts > 8:
                max_attempts = 8

        attempt = 0
        allow_local_subprocess = bool(
            tool_id.startswith("embodiment.generated.")
            or (
                isinstance(capabilities_hint, list)
                and any(str(item).strip().lower() == "embodiment" for item in capabilities_hint)
            )
        )
        while True:
            if max_attempts is not None and attempt >= max_attempts:
                break
            attempt += 1
            raw = await self._generate_response(prompt, PRIMARY_REASONING_MODE, codegen_context, trace_id, None)
            self._log_mode_event(
                PRIMARY_REASONING_MODE,
                "create_tool_codegen_raw",
                {"trace_id": trace_id, "tool_id": tool_id, "attempt": attempt, "raw": self._truncate(raw, 2000)},
            )
            parsed = None
            raw_text = raw if isinstance(raw, str) else ""
            empty_raw = not raw_text.strip()
            if not empty_raw:
                parsed = self._parse_tool_codegen_response(raw_text)
                if not parsed:
                    repair_prompt = self._build_json_repair_prompt(codegen_schema, raw_text)
                    repaired = await self._generate_response(
                        repair_prompt,
                        PRIMARY_REASONING_MODE,
                        codegen_context,
                        trace_id,
                        None,
                    )
                    self._log_mode_event(
                        PRIMARY_REASONING_MODE,
                        "create_tool_codegen_repair_raw",
                        {"trace_id": trace_id, "tool_id": tool_id, "attempt": attempt, "raw": self._truncate(repaired, 2000)},
                    )
                    parsed = self._parse_tool_codegen_response(repaired)
            if not parsed:
                validation_error = (
                    "Code generation returned empty output."
                    if empty_raw
                    else "Code generation returned invalid JSON."
                )
            else:
                candidate_input_schema = (
                    input_schema_hint if isinstance(input_schema_hint, dict) else parsed.get("input_schema", {})
                )
                candidate_output_schema = (
                    output_schema_hint if isinstance(output_schema_hint, dict) else parsed.get("output_schema", {})
                )
                if not isinstance(candidate_input_schema, dict) or not isinstance(candidate_output_schema, dict):
                    validation_error = "Generated schema is invalid."
                else:
                    static_validation_error = self._validate_generated_tool_code(
                        parsed.get("code", ""),
                        allow_local_subprocess=allow_local_subprocess,
                    ) or ""
                    if static_validation_error:
                        validation_error = static_validation_error
                    else:
                        critic_result = await self._evaluate_tool_codegen_critic(
                            tool_id=tool_id,
                            description=description,
                            input_schema=candidate_input_schema,
                            output_schema=candidate_output_schema,
                            code=parsed.get("code", ""),
                            dependencies=parsed.get("dependencies", []),
                            trace_id=trace_id,
                            attempt=attempt,
                        )
                        critic_confidence = float(critic_result.get("confidence", 0.0))
                        critic_approved = (
                            critic_result.get("fulfilled") is True
                            and critic_confidence >= self._tool_codegen_critic_min_confidence
                        )
                        issues = critic_result.get("issues", [])
                        if not isinstance(issues, list):
                            issues = []
                        fixes = critic_result.get("fix_instructions", [])
                        if not isinstance(fixes, list):
                            fixes = []
                        self._log_mode_event(
                            PRIMARY_REASONING_MODE,
                            "create_tool_codegen_critic_result",
                            {
                                "trace_id": trace_id,
                                "tool_id": tool_id,
                                "attempt": attempt,
                                "fulfilled": bool(critic_result.get("fulfilled") is True),
                                "confidence": critic_confidence,
                                "approved": bool(critic_approved),
                                "issues": [str(item) for item in issues][:6],
                                "fix_instructions": [str(item) for item in fixes][:6],
                            },
                        )
                        if critic_approved:
                            generated = dict(parsed)
                            generated["input_schema"] = candidate_input_schema
                            generated["output_schema"] = candidate_output_schema
                            break
                        error_parts: List[str] = []
                        if issues:
                            error_parts.extend(str(item).strip() for item in issues if str(item).strip())
                        if fixes:
                            fix_text = "; ".join(str(item).strip() for item in fixes if str(item).strip())
                            if fix_text:
                                error_parts.append(f"Fix: {fix_text}")
                        threshold_text = f"{self._tool_codegen_critic_min_confidence:.2f}"
                        validation_error = (
                            "Codegen critic rejected candidate "
                            f"(fulfilled={bool(critic_result.get('fulfilled') is True)}, confidence={critic_confidence:.2f}, threshold={threshold_text})"
                        )
                        if error_parts:
                            validation_error += ": " + "; ".join(error_parts[:6])
            self._log_mode_event(
                PRIMARY_REASONING_MODE,
                "create_tool_codegen_validation_error",
                {
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "attempt": attempt,
                    "error": self._truncate(validation_error, 1200),
                },
            )
            prompt = (
                f"{base_prompt}\n\n"
                "Previous generation failed validation. Regenerate a full replacement and fix every issue.\n"
                f"Validation error: {validation_error}\n"
            )

        if not generated:
            attempts_text = "infinity" if max_attempts is None else str(max_attempts)
            return {
                "status": "ERROR",
                "error": f"Failed to generate valid tool code after {attempts_text} attempts: {validation_error}",
            }

        resolved_name = (name_hint.strip() if name_hint.strip() else generated.get("name", "").strip())
        if not resolved_name:
            resolved_name = self._default_tool_name(tool_id)
        resolved_input_schema = input_schema_hint if isinstance(input_schema_hint, dict) else generated.get("input_schema", {})
        resolved_output_schema = output_schema_hint if isinstance(output_schema_hint, dict) else generated.get("output_schema", {})
        if not isinstance(resolved_input_schema, dict) or not isinstance(resolved_output_schema, dict):
            return {"status": "ERROR", "error": "Generated schema is invalid"}
        dependencies = generated.get("dependencies", [])
        if not isinstance(dependencies, list):
            dependencies = []

        spec = {
            "tool_id": tool_id,
            "name": resolved_name,
            "description": description,
            "implements_capability_id": args.get("implements_capability_id"),
            "implements_abstract_capability": args.get("implements_abstract_capability"),
            "input_schema": resolved_input_schema,
            "output_schema": resolved_output_schema,
            "code": generated.get("code", ""),
            "dependencies": [d for d in dependencies if isinstance(d, str) and d.strip()],
            "capabilities": args.get("capabilities", []),
            "tier": int(args.get("tier", 1)),
            "risk_level_default": args.get("risk_level_default", "MEDIUM"),
            "required_permissions": args.get("required_permissions", ["tier1"]),
            "version": args.get("version", "0.1.0"),
            "examples": args.get("examples", []),
            "timeouts_ms": args.get("timeouts_ms", 2000),
            "resource_limits": args.get("resource_limits", {"cpu_ms": 200, "mem_mb": 64}),
            "sandbox_profile": args.get("sandbox_profile", {"network": False, "fs": "read"}),
        }

        specs.append(spec)
        self._write_generated_specs(specs)

        os.makedirs(os.path.dirname(self._generated_tools_module_path), exist_ok=True)
        module_text = self._render_generated_tools(specs)
        with open(self._generated_tools_module_path, "w", encoding="utf-8") as f:
            f.write(module_text)

        registry_error = self._update_tool_registry(spec)
        if registry_error:
            return {"status": "ERROR", "error": registry_error}

        self._tools = ToolRegistry()
        return {
            "status": "APPROVED",
            "result": {
                "tool_id": tool_id,
                "name": resolved_name,
                "dependencies": spec.get("dependencies", []),
            },
        }



    async def _run_cognition_actions(
        self,
        actions: List[Dict[str, Any]],
        trace_id: str,
        action_budget: int,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        if action_budget <= 0:
            return observations
        for action in actions[:action_budget]:
            label = action.get("label", "")
            step_result = await self._execute_cognition_action(action, trace_id)
            new_tools = step_result.get("new_tools", [])
            if isinstance(tools, list) and isinstance(new_tools, list) and new_tools:
                merged_tools = self._merge_relevant_tool_lists(tools, new_tools)
                tools[:] = merged_tools
            observations.append(
                {
                    "label": label,
                    "action": step_result.get("action"),
                    "observation": step_result.get("observation"),
                }
            )
        return observations

    async def _finalize_cognition_result(
        self,
        query_state: Dict[str, Any],
        state_of_mind: Dict[str, Any],
        observations: List[Dict[str, Any]],
        thinking_trace: List[Dict[str, Any]],
        trace_id: str,
        turn_id: Optional[str],
    ) -> Dict[str, Any]:
        final_schema = (
            "{\"type\":\"final\",\"thought\":\"...\",\"text\":\"...\"}\n"
            "{\"type\":\"clarify\",\"thought\":\"...\",\"question\":\"...\"}"
        )
        prompt = self._build_cognition_final_prompt(
            query_state,
            state_of_mind,
            observations,
            thinking_trace,
        )
        raw = await self._generate_cognition_response(prompt, trace_id, turn_id)
        self._log_mode_event(
            "COGNITION",
            "final_raw",
            {"trace_id": trace_id, "raw": self._truncate(raw, 2000)},
        )
        parsed = self._parse_cognition_thinking_response(raw, allow_step=False)
        if parsed is None:
            repair_prompt = self._build_json_repair_prompt(final_schema, raw)
            repaired = await self._generate_cognition_response_with_timeout(
                repair_prompt,
                trace_id,
                turn_id,
                timeout_s=self._cognition_repair_timeout_s,
                timeout_event="final_repair_timeout",
                model_override=self._cognition_repair_model or None,
                options_override=self._cognition_repair_options or None,
            )
            self._log_mode_event(
                "COGNITION",
                "final_repair_raw",
                {"trace_id": trace_id, "raw": self._truncate(repaired, 2000)},
            )
            parsed = self._parse_cognition_thinking_response(repaired, allow_step=False)
        if parsed is None:
            return {
                "type": "final",
                "text": "I couldn't produce a reliable cognition result. Please try again.",
                "thought": "Fallback after parse failure.",
            }
        if parsed.get("type") == "clarify":
            return parsed
        return {"type": "final", "text": parsed.get("text", ""), "thought": parsed.get("thought", "")}

    async def _distill_cognition_episode(
        self,
        result_payload: Dict[str, Any],
        query_state: Dict[str, Any],
        state_of_mind: Dict[str, Any],
        observations: List[Dict[str, Any]],
        thinking_trace: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        trace_id: str,
        turn_id: Optional[str],
        *,
        budget_exhausted: bool = False,
    ) -> Dict[str, Any]:
        if not self._cognition_distill_enabled:
            return {
                "reusable_skills": [],
                "memory_facts": [],
                "patterns": [],
                "failure_lessons": [],
                "safety_rules": [],
            }
        distill_schema = (
            "{\"type\":\"distill\","
            "\"reusable_skills\":[{\"name\":\"...\",\"description\":\"...\",\"when_to_use\":[\"...\"],\"confidence\":0.0}],"
            "\"memory_facts\":[{\"name\":\"...\",\"value\":\"...\",\"confidence\":0.0}],"
            "\"patterns\":[{\"pattern_id\":\"...\",\"name\":\"...\",\"description\":\"...\",\"when_to_use\":[\"...\"],"
            "\"source\":\"def pattern_name(arg1, arg2=...):\\n    result = tool.namespace(arg=value)\\n    return result\","
            "\"confidence\":0.0}],"
            "\"failure_lessons\":[{\"lesson\":\"...\",\"when\":\"...\",\"confidence\":0.0}],"
            "\"safety_rules\":[{\"rule\":\"...\",\"rationale\":\"...\",\"confidence\":0.0}]}"
        )
        prompt = self._build_cognition_distill_prompt(
            result_payload,
            query_state,
            state_of_mind,
            observations,
            thinking_trace,
            tools,
            patterns,
        )
        # Prepend a conservative distillation note when reasoning was budget-exhausted
        if budget_exhausted:
            prompt = (
                "NOTE: The cognition loop was budget-exhausted before producing a complete answer. "
                "Distill conservatively — prefer failure_lessons over skills or patterns. "
                "Do not create skills from incomplete execution traces.\n\n"
            ) + prompt
        _distill_model = self._cognition_distill_model or None
        raw = await self._generate_cognition_response_with_timeout(
            prompt,
            trace_id,
            turn_id,
            timeout_s=self._cognition_distill_timeout_s,
            timeout_event="distill_timeout",
            model_override=_distill_model,
            options_override=self._cognition_distill_options or None,
        )
        self._log_mode_event(
            "COGNITION",
            "distill_raw",
            {"trace_id": trace_id, "raw": self._truncate(raw, 2000), "model": _distill_model or "default"},
        )
        parsed = self._parse_cognition_distill_response(raw)
        if parsed is None:
            repair_prompt = self._build_json_repair_prompt(distill_schema, raw)
            repair_model = self._cognition_repair_model or _distill_model
            repair_options = self._cognition_repair_options if self._cognition_repair_model else self._cognition_distill_options
            repaired = await self._generate_cognition_response_with_timeout(
                repair_prompt,
                trace_id,
                turn_id,
                timeout_s=self._cognition_repair_timeout_s,
                timeout_event="distill_repair_timeout",
                model_override=repair_model,
                options_override=repair_options or None,
            )
            self._log_mode_event(
                "COGNITION",
                "distill_repair_raw",
                {"trace_id": trace_id, "raw": self._truncate(repaired, 2000), "model": repair_model or "default"},
            )
            parsed = self._parse_cognition_distill_response(repaired)
        if parsed is None:
            return {
                "reusable_skills": [],
                "memory_facts": [],
                "patterns": [],
                "failure_lessons": [],
                "safety_rules": [],
            }
        return parsed

    async def _persist_cognition_distillation(
        self,
        distillation: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, int]:
        reusable_skills = distillation.get("reusable_skills", [])
        patterns = distillation.get("patterns", [])
        failure_lessons = distillation.get("failure_lessons", [])
        safety_rules = distillation.get("safety_rules", [])
        memory_facts = distillation.get("memory_facts", [])
        # Filter out low-confidence distillation entries to avoid polluting catalogs
        # with hallucinated or weakly-supported skills/patterns/lessons.
        _min_conf = getattr(self, "_cognition_distill_min_confidence", 0.55)
        def _meets_confidence(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            try:
                return float(item.get("confidence", 0)) >= _min_conf
            except (TypeError, ValueError):
                return False
        reusable_skills = [s for s in reusable_skills if _meets_confidence(s)]
        patterns = [p for p in patterns if _meets_confidence(p)]
        failure_lessons = [l for l in failure_lessons if _meets_confidence(l)]
        safety_rules = [r for r in safety_rules if _meets_confidence(r)]
        memory_facts = [f for f in memory_facts if _meets_confidence(f)]

        _, changed_skills = self._upsert_cognition_catalog(
            self._cognition_skills_path,
            reusable_skills,
            self._normalize_cognition_skill,
            "skill_id",
        )
        _, changed_patterns = self._upsert_cognition_catalog(
            self._cognition_patterns_path,
            patterns,
            self._normalize_cognition_pattern,
            "pattern_id",
        )
        _, changed_lessons = self._upsert_cognition_catalog(
            self._cognition_lessons_path,
            failure_lessons,
            self._normalize_cognition_lesson,
            "lesson_id",
        )
        _, changed_safety_rules = self._upsert_cognition_catalog(
            self._cognition_safety_rules_path,
            safety_rules,
            self._normalize_cognition_safety_rule,
            "rule_id",
        )

        for skill in changed_skills:
            await self._memory.store_entity(
                str(skill.get("skill_id") or ""),
                skill,
                trace_id,
                confidence=float(skill.get("confidence", 0.5)),
            )
        for pattern in changed_patterns:
            await self._memory.store_entity(
                str(pattern.get("pattern_id") or ""),
                pattern,
                trace_id,
                confidence=float(pattern.get("confidence", 0.5)),
            )
        for lesson in changed_lessons:
            await self._memory.store_entity(
                str(lesson.get("lesson_id") or ""),
                lesson,
                trace_id,
                confidence=float(lesson.get("confidence", 0.5)),
            )
        for rule in changed_safety_rules:
            await self._memory.store_entity(
                str(rule.get("rule_id") or ""),
                rule,
                trace_id,
                confidence=float(rule.get("confidence", 0.5)),
            )

        fact_count = 0
        if isinstance(memory_facts, list):
            for item in memory_facts:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                value = str(item.get("value") or "").strip()
                if not name or not value:
                    continue
                existing = await self._memory.get_facts(name)
                duplicate = False
                for fact in existing:
                    if not isinstance(fact, dict):
                        continue
                    fact_name = str(fact.get("name") or "").strip()
                    data = fact.get("data", {})
                    fact_value = ""
                    if isinstance(data, dict):
                        fact_value = str(data.get("value") or "").strip()
                    if fact_name == name and fact_value == value:
                        duplicate = True
                        break
                if duplicate:
                    continue
                await self._memory.store_fact(
                    name,
                    value,
                    trace_id,
                    confidence=self._normalize_cognition_confidence(item.get("confidence"), default=0.5),
                )
                fact_count += 1

        return {
            "skills": len(changed_skills),
            "facts": fact_count,
            "patterns": len(changed_patterns),
            "lessons": len(changed_lessons),
            "safety_rules": len(changed_safety_rules),
        }

    async def _run_cognition_loop(
        self,
        text: str,
        context_packet: Dict[str, Any],
        trace_id: str,
        turn_id: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        forced_execution_mode: Optional[str] = None,
        skip_distillation: bool = False,
    ) -> str:
        retrieve_text = retrieval_query.strip() if isinstance(retrieval_query, str) and retrieval_query.strip() else text
        memory_hits = await self._memory.retrieve(retrieve_text, top_k=self._memory_retrieval_k)
        retrieved_tools = await self._retrieve_relevant_tools(retrieve_text, self._tool_retrieval_k)
        selection_candidates = (
            self._tools.list_active()
            if self._select_relevant_tools_enabled and self._tool_selection_independent_from_retrieval
            else retrieved_tools
        )
        selected_tools, tool_selection_trace = await self._select_relevant_tools(
            text,
            context_packet,
            memory_hits,
            selection_candidates,
            trace_id,
            turn_id,
        )
        tools = retrieved_tools
        if self._select_relevant_tools_enabled:
            if self._retrieve_relevant_tools_enabled and self._tool_selection_independent_from_retrieval:
                if tool_selection_trace is not None and tool_selection_trace.get("status") == "fallback":
                    tools = retrieved_tools
                else:
                    tools = self._merge_relevant_tool_lists(retrieved_tools, selected_tools)
            else:
                tools = selected_tools
        tools_from_selector = (
            self._select_relevant_tools_enabled
            and tool_selection_trace is not None
            and tool_selection_trace.get("status") == "selected"
            and bool(tools)
        )
        skills = self._load_cognition_skills()
        patterns = self._load_cognition_patterns()
        callable_patterns = self._cognition_source_patterns(patterns)
        safety_rules = self._load_cognition_safety_rules()

        query_state = self._default_cognition_query_state(text)
        state_of_mind = self._default_cognition_state_of_mind()
        observations: List[Dict[str, Any]] = []
        thinking_trace: List[Dict[str, Any]] = []
        if tool_selection_trace is not None:
            selection_trace = dict(tool_selection_trace)
            selection_trace["candidate_source"] = (
                "all_tools" if self._tool_selection_independent_from_retrieval else "retrieved_tools"
            )
            selection_trace["retrieval_enabled"] = self._retrieve_relevant_tools_enabled
            selection_trace["selection_enabled"] = self._select_relevant_tools_enabled
            selection_trace["retrieved_count"] = len(retrieved_tools)
            selection_trace["retrieved_tool_ids"] = [
                str(tool.get("tool_id") or "").strip()
                for tool in retrieved_tools
                if str(tool.get("tool_id") or "").strip()
            ]
            selection_trace["forwarded_count"] = len(tools)
            selection_trace["forwarded_tool_ids"] = [
                str(tool.get("tool_id") or "").strip()
                for tool in tools
                if str(tool.get("tool_id") or "").strip()
            ]
            thinking_trace.append(selection_trace)
            self._log_mode_event(
                "COGNITION",
                "tool_selection",
                {
                    "trace_id": trace_id,
                    "status": selection_trace.get("status", ""),
                    "candidate_source": selection_trace.get("candidate_source", ""),
                    "candidate_count": selection_trace.get("candidate_count", 0),
                    "selected_count": selection_trace.get("selected_count", 0),
                    "forwarded_count": selection_trace.get("forwarded_count", 0),
                    "selected_tool_ids": selection_trace.get("selected_tool_ids", []),
                    "forwarded_tool_ids": selection_trace.get("forwarded_tool_ids", []),
                },
            )
        action_budget = self._cognition_action_limit
        route_mode = "system2"
        execution_mode = "system2"
        strategy_brief = ""
        result_payload: Optional[Dict[str, Any]] = None

        if forced_execution_mode in {"system0", "system1", "system2", "system3"}:
            route_mode = forced_execution_mode
            execution_mode = forced_execution_mode
            thinking_trace.append(
                {
                    "phase": "route",
                    "thinking_mode": route_mode,
                    "reason": "requested mode override",
                }
            )
            self._log_mode_event(
                "COGNITION",
                "thinking_route",
                {"trace_id": trace_id, "thinking_mode": route_mode, "forced": True, "reason": "requested_mode"},
            )

        _trivial_routing_skip = (
            forced_execution_mode is None
            and not self._cognition_force_system
            and self._cognition_routing_skip_trivial
            and self._is_trivial_cognition_request(text, tools)
        )

        if _trivial_routing_skip:
            execution_mode = self._cognition_trivial_query_target
            route_mode = execution_mode
            thinking_trace.append({
                "phase": "route",
                "thinking_mode": route_mode,
                "reason": "trivial turn skipped init and router",
            })
            self._log_mode_event(
                "COGNITION",
                "thinking_route",
                {"trace_id": trace_id, "thinking_mode": route_mode, "forced": True, "reason": "trivial_turn"},
            )

        if forced_execution_mode is None and not _trivial_routing_skip and (
            self._cognition_query_state_enabled or self._cognition_state_of_mind_enabled
        ):
            parsed_init = await self._call_cognition_json(
                self._build_cognition_init_prompt(text, context_packet, memory_hits, tools),
                self._build_cognition_init_schema(),
                self._parse_cognition_init_response,
                "init",
                trace_id,
                turn_id,
                model_override=self._cognition_init_model,
                options_override=self._cognition_init_options or None,
            )
            if parsed_init is not None:
                init_qs = parsed_init.get("query_state", {})
                # Validate that the init call populated an intent before merging.
                # An empty intent is a sign the init call hallucinated or parsed incorrectly.
                proposed_intent = str(init_qs.get("intent") or "").strip() if isinstance(init_qs, dict) else ""
                if proposed_intent:
                    self._merge_cognition_query_state(query_state, init_qs)
                else:
                    self._log_mode_event(
                        "COGNITION",
                        "init_intent_empty",
                        {"trace_id": trace_id, "text_preview": text[:80]},
                    )
                self._merge_cognition_state_of_mind(state_of_mind, parsed_init.get("state_of_mind", {}))
                tools_needed = bool(parsed_init.get("tools_needed", True))
                if (
                    not tools_needed
                    and (
                        self._cognition_init_controls_tools_needed
                        or not tools_from_selector
                    )
                ):
                    tools = []
                thinking_trace.append(
                    {
                        "phase": "init",
                        "tools_needed": tools_needed,
                        "thought": parsed_init.get("thought", ""),
                        "query_state": dict(query_state),
                        "state_of_mind": dict(state_of_mind),
                        "intent_validated": bool(proposed_intent),
                    }
                )

        if forced_execution_mode is None and not _trivial_routing_skip and callable_patterns and not self._cognition_force_system:
            pattern_route_schema = (
                "{\"type\":\"pattern_route\",\"decision\":\"use\",\"pattern_id\":\"...\",\"reason\":\"...\"}\n"
                "{\"type\":\"pattern_route\",\"decision\":\"route\",\"reason\":\"...\"}"
            )
            parsed_pattern_route = await self._call_cognition_json(
                self._build_cognition_pattern_router_prompt(
                    query_state,
                    state_of_mind,
                    skills,
                    callable_patterns,
                    safety_rules,
                ),
                pattern_route_schema,
                self._parse_cognition_pattern_route_response,
                "pattern_route",
                trace_id,
                turn_id,
            )
            if parsed_pattern_route is not None:
                thinking_trace.append(
                    {
                        "phase": "pattern_route",
                        "decision": parsed_pattern_route.get("decision", "route"),
                        "pattern_id": parsed_pattern_route.get("pattern_id", ""),
                        "reason": parsed_pattern_route.get("reason", ""),
                    }
                )
                if parsed_pattern_route.get("decision") == "use":
                    selected_pattern = self._find_cognition_pattern(
                        str(parsed_pattern_route.get("pattern_id") or ""),
                        callable_patterns,
                    )
                    if selected_pattern is not None:
                        # Use query_rewrite or intent for pattern binding instead of raw text
                        digested_text = str(query_state.get("query_rewrite") or query_state.get("intent") or text)
                        binding_result = await self._bind_asi_inputs(selected_pattern, digested_text)
                        # Save budget snapshot so it can be restored if the pattern fails.
                        # This prevents a failed pattern from starving System1/2 of actions.
                        _pre_pattern_budget = action_budget
                        tool_budget = {"remaining": action_budget}
                        pattern_result = await self._execute_asi_pattern_runtime(
                            selected_pattern,
                            binding_result.get("bindings", {}),
                            trace_id,
                            tool_budget,
                            user_request=digested_text,
                            depth=0,
                            extra_patterns=callable_patterns,
                        )
                        action_budget = max(0, int(tool_budget.get("remaining", 0)))
                        pattern_result_type = str(pattern_result.get("type") or "error").strip().lower()
                        observation_payload: Dict[str, Any] = {
                            "status": "APPROVED" if pattern_result_type == "final" else "CLARIFY" if pattern_result_type == "clarify" else "ERROR",
                            "pattern_id": self._normalize_asi_pattern_ref(selected_pattern.get("pattern_id")),
                            "bindings": binding_result.get("bindings", {}),
                            "trace": pattern_result.get("trace", []),
                        }
                        if pattern_result_type == "final":
                            observation_payload["result"] = pattern_result.get("value", pattern_result.get("text", ""))
                        elif pattern_result_type == "clarify":
                            observation_payload["question"] = pattern_result.get("question", "")
                        else:
                            observation_payload["error"] = pattern_result.get("text", pattern_result.get("error", ""))
                        observations.append(
                            {
                                "label": f"pattern {observation_payload.get('pattern_id', '')}".strip(),
                                "action": {
                                    "pattern_id": observation_payload.get("pattern_id", ""),
                                    "bindings": binding_result.get("bindings", {}),
                                },
                                "observation": observation_payload,
                            }
                        )
                        thinking_trace.append(
                            {
                                "phase": "pattern_execution",
                                "pattern_id": observation_payload.get("pattern_id", ""),
                                "bindings": binding_result.get("bindings", {}),
                                "missing_required": binding_result.get("missing_required", []),
                                "type": pattern_result_type,
                                "trace": pattern_result.get("trace", []),
                            }
                        )
                        if pattern_result_type == "final":
                            route_mode = "pattern"
                            execution_mode = "pattern"
                            result_payload = {"type": "final", "text": pattern_result.get("text", "")}
                        elif pattern_result_type == "clarify":
                            route_mode = "pattern"
                            execution_mode = "pattern"
                            result_payload = {"type": "clarify", "question": pattern_result.get("question", "")}
                        else:
                            # Pattern failed or errored — restore pre-pattern budget so
                            # System1/2 is not penalised for the pattern's failed attempt.
                            action_budget = _pre_pattern_budget

        if result_payload is None:
            if _trivial_routing_skip:
                pass
            elif forced_execution_mode is not None:
                pass
            elif self._cognition_force_system:
                route_mode = self._cognition_force_system
                execution_mode = route_mode
                thinking_trace.append(
                    {
                        "phase": "route",
                        "thinking_mode": route_mode,
                        "reason": "forced by cognition_force_system setting",
                    }
                )
                self._log_mode_event(
                    "COGNITION",
                    "thinking_route",
                    {"trace_id": trace_id, "thinking_mode": route_mode, "forced": True},
                )
            else:
                route_schema = (
                    "{\"type\":\"route\",\"thinking_mode\":\"system1|system2|system3\",\"reason\":\"...\","
                    "\"generate_tools_first\":false}\n"
                    "{\"type\":\"route\",\"thinking_mode\":\"system1|system2|system3\",\"reason\":\"...\","
                    "\"generate_tools_first\":true,\"tool_generation_spec\":{\"request\":\"...\","
                    "\"namespace_prefix\":\"...\",\"max_tools\":2}}"
                    if self._cognition_system3_available()
                    else
                    "{\"type\":\"route\",\"thinking_mode\":\"system1|system2\",\"reason\":\"...\","
                    "\"generate_tools_first\":false}\n"
                    "{\"type\":\"route\",\"thinking_mode\":\"system1|system2\",\"reason\":\"...\","
                    "\"generate_tools_first\":true,\"tool_generation_spec\":{\"request\":\"...\","
                    "\"namespace_prefix\":\"...\",\"max_tools\":2}}"
                )
                parsed_route = await self._call_cognition_json(
                    self._build_cognition_route_prompt(
                        query_state,
                        state_of_mind,
                        memory_hits,
                        tools,
                    ),
                    route_schema,
                    self._parse_cognition_route_response,
                    "route",
                    trace_id,
                    turn_id,
                    model_override=self._cognition_route_model,
                    options_override=self._cognition_route_options or None,
                )
                route_generation_requested = False
                route_generation_status = ""
                if parsed_route is not None:
                    route_mode = str(parsed_route.get("thinking_mode") or "system2")
                    execution_mode = route_mode
                    route_generation_requested = bool(parsed_route.get("generate_tools_first"))
                    thinking_trace.append(
                        {
                            "phase": "route",
                            "thinking_mode": route_mode,
                            "reason": parsed_route.get("reason", ""),
                            "generate_tools_first": route_generation_requested,
                        }
                    )
                    if route_generation_requested:
                        route_generation_spec = parsed_route.get("tool_generation_spec", {})
                        route_generation_action = {
                            "action_type": "generate_tools_from_spec",
                            "spec": route_generation_spec,
                            "label": "route-time tool generation",
                        }
                        if action_budget <= 0:
                            route_generation_step = {
                                "action": route_generation_action,
                                "observation": {
                                    "status": "SKIPPED",
                                    "error": "No remaining action budget for route-time tool generation",
                                },
                                "new_tools": [],
                            }
                        else:
                            route_generation_step = await self._execute_cognition_action(
                                route_generation_action,
                                trace_id,
                            )
                            action_budget = max(0, action_budget - 1)
                        route_generation_observation = route_generation_step.get("observation", {})
                        route_generation_status = str(route_generation_observation.get("status") or "")
                        route_new_tools = route_generation_step.get("new_tools", [])
                        if isinstance(route_new_tools, list) and route_new_tools:
                            tools = self._merge_relevant_tool_lists(tools, route_new_tools)
                        observations.append(
                            {
                                "label": "route-time tool generation",
                                "action": route_generation_step.get("action"),
                                "observation": route_generation_observation,
                            }
                        )
                        thinking_trace.append(
                            {
                                "phase": "route_tool_generation",
                                "thinking_mode": route_mode,
                                "status": route_generation_status,
                                "spec": route_generation_spec,
                                "generated_tool_ids": [
                                    str(tool.get("tool_id") or "").strip()
                                    for tool in route_new_tools
                                    if isinstance(tool, dict) and str(tool.get("tool_id") or "").strip()
                                ],
                                "error": str(route_generation_observation.get("error") or ""),
                            }
                        )
                self._log_mode_event(
                    "COGNITION",
                    "thinking_route",
                    {
                        "trace_id": trace_id,
                        "thinking_mode": route_mode,
                        "generate_tools_first": route_generation_requested,
                        "route_generation_status": route_generation_status,
                    },
                )
        else:
            thinking_trace.append(
                {
                    "phase": "route",
                    "thinking_mode": route_mode,
                    "reason": "pattern execution completed before the thinking router",
                }
            )
            self._log_mode_event(
                "COGNITION",
                "thinking_route",
                {"trace_id": trace_id, "thinking_mode": route_mode},
            )

        if route_mode == "system3" and result_payload is None:
            if not self._cognition_system3_available():
                thinking_trace.append(
                    {
                        "phase": "system3",
                        "selected_mode": "system3",
                        "reason": "system3 forced or selected, but proposer/critic pools are unavailable",
                    }
                )
                if self._cognition_force_system == "system3":
                    result_payload = {
                        "type": "final",
                        "text": (
                            "System3 is forced but unavailable. "
                            "Configure cognition_peer_pools and enable cognition_system3."
                        ),
                    }
                else:
                    execution_mode = "system2"
            system3_result = (
                await self._run_cognition_system3(
                    query_state,
                    state_of_mind,
                    memory_hits,
                    tools,
                    safety_rules,
                    trace_id,
                    turn_id,
                )
                if result_payload is None and self._cognition_system3_available()
                else None
            )
            if system3_result is not None:
                winner = dict(system3_result.get("winner", {}))
                execution_mode = str(system3_result.get("selected_mode") or "system2")
                strategy_brief = str(system3_result.get("strategy_brief") or "")
                if self._cognition_query_state_enabled and isinstance(winner.get("query_state"), dict):
                    query_state = dict(winner.get("query_state", {}))
                if self._cognition_state_of_mind_enabled and isinstance(winner.get("state_of_mind"), dict):
                    state_of_mind = dict(winner.get("state_of_mind", {}))
                thinking_trace.append(
                    {
                        "phase": "system3",
                        "selected_mode": execution_mode,
                        "winner_peer": winner.get("peer_id", ""),
                        "average_score": winner.get("average_score", 0.0),
                        "strategy": winner.get("strategy", ""),
                        "argument": winner.get("argument", ""),
                        "rounds": system3_result.get("rounds", []),
                        "scores": system3_result.get("scores", {}),
                    }
                )
            elif result_payload is None and self._cognition_force_system == "system3":
                result_payload = {
                    "type": "final",
                    "text": "System3 is forced but produced no valid winning strategy.",
                }
                thinking_trace.append(
                    {
                        "phase": "system3",
                        "selected_mode": "system3",
                        "reason": "forced system3 produced no valid winner",
                    }
                )
            elif result_payload is None:
                execution_mode = "system2"
                thinking_trace.append(
                    {
                        "phase": "system3",
                        "selected_mode": execution_mode,
                        "reason": "system3 produced no valid winner; falling back to system2",
                    }
                )

        system_schemas = self._build_cognition_system_schemas()

        if execution_mode == "system0":
            for cycle_index in range(1, self._cognition_system0_max_steps + 1):
                parsed_system0 = await self._call_cognition_json(
                    self._build_cognition_system0_prompt(
                        query_state,
                        state_of_mind,
                        observations,
                        cycle_index,
                    ),
                    system_schemas["final"],
                    self._parse_cognition_final_response,
                    f"system0_{cycle_index}",
                    trace_id,
                    turn_id,
                    model_override=self._resolve_cognition_model_override("system0"),
                    options_override=self._cognition_system0_options or None,
                )
                if parsed_system0 is None:
                    break
                self._merge_cognition_query_state(query_state, parsed_system0.get("query_state_delta", {}))
                self._merge_cognition_state_of_mind(state_of_mind, parsed_system0.get("state_of_mind_delta", {}))
                thinking_trace.append(
                    {
                        "phase": f"system0_{cycle_index}",
                        "thought": parsed_system0.get("thought", ""),
                        "type": "final",
                        "actions": [],
                    }
                )
                result_payload = {"type": "final", "text": parsed_system0.get("text", "")}
                break

        if execution_mode == "system1" and result_payload is None:
            for cycle_index in range(1, self._cognition_system1_max_steps + 1):
                parsed_system1 = await self._call_cognition_json(
                    self._build_cognition_system1_prompt(
                        query_state,
                        state_of_mind,
                        observations,
                        tools,
                        cycle_index,
                        strategy_brief,
                    ),
                    self._build_cognition_execution_schema_text(
                        allow_step=False,
                        allow_clarify=self._cognition_system1_clarify_enabled,
                    ),
                    lambda raw: self._parse_cognition_thinking_response(
                        raw,
                        allow_step=False,
                        allow_clarify=self._cognition_system1_clarify_enabled,
                    ),
                    f"system1_{cycle_index}",
                    trace_id,
                    turn_id,
                    model_override=self._resolve_cognition_model_override("system1"),
                    options_override=self._cognition_system1_options or None,
                )
                if parsed_system1 is None:
                    break
                self._merge_cognition_query_state(query_state, parsed_system1.get("query_state_delta", {}))
                self._merge_cognition_state_of_mind(state_of_mind, parsed_system1.get("state_of_mind_delta", {}))
                thinking_trace.append(
                    {
                        "phase": f"system1_{cycle_index}",
                        "thought": parsed_system1.get("thought", ""),
                        "type": parsed_system1.get("type", ""),
                        "actions": parsed_system1.get("actions", []),
                    }
                )
                if parsed_system1.get("type") == "final":
                    result_payload = {"type": "final", "text": parsed_system1.get("text", "")}
                    break
                if parsed_system1.get("type") == "clarify":
                    result_payload = {"type": "clarify", "question": parsed_system1.get("question", "")}
                    break
                planned_actions = parsed_system1.get("actions", [])
                if action_budget <= 0 or not planned_actions:
                    break
                executed = await self._run_cognition_actions(planned_actions, trace_id, action_budget, tools=tools)
                observations.extend(executed)
                action_budget = max(0, action_budget - len(executed))
                if not executed:
                    break

        if execution_mode == "system2" and result_payload is None:
            for cycle_index in range(1, self._cognition_system2_max_steps + 1):
                parsed_step = await self._call_cognition_json(
                    self._build_cognition_system2_prompt(
                        query_state,
                        state_of_mind,
                        observations,
                        thinking_trace,
                        tools,
                        cycle_index,
                        strategy_brief,
                    ),
                    self._build_cognition_execution_schema_text(
                        allow_step=True,
                        allow_clarify=self._cognition_system2_clarify_enabled,
                    ),
                    lambda raw: self._parse_cognition_thinking_response(
                        raw,
                        allow_step=True,
                        allow_clarify=self._cognition_system2_clarify_enabled,
                    ),
                    f"system2_{cycle_index}",
                    trace_id,
                    turn_id,
                    model_override=self._resolve_cognition_model_override("system2"),
                    options_override=self._cognition_system2_options or None,
                )
                if parsed_step is None:
                    break
                self._merge_cognition_query_state(query_state, parsed_step.get("query_state_delta", {}))
                self._merge_cognition_state_of_mind(state_of_mind, parsed_step.get("state_of_mind_delta", {}))
                if self._cognition_query_state_enabled and parsed_step.get("todo_updates"):
                    query_state["todo"] = self._merge_cognition_todos(
                        query_state.get("todo", []),
                        parsed_step.get("todo_updates", []),
                        limit=10,
                    )
                thinking_trace.append(
                    {
                        "phase": f"system2_{cycle_index}",
                        "thought": parsed_step.get("thought", ""),
                        "reflection": parsed_step.get("reflection", ""),
                        "type": parsed_step.get("type", ""),
                        "actions": parsed_step.get("actions", []),
                        "todo_updates": parsed_step.get("todo_updates", []) if self._cognition_query_state_enabled else [],
                    }
                )
                if parsed_step.get("reflection"):
                    self._merge_cognition_state_of_mind(
                        state_of_mind,
                        {"reflections": [parsed_step.get("reflection", "")]},
                    )
                if parsed_step.get("type") == "final":
                    result_payload = {"type": "final", "text": parsed_step.get("text", "")}
                    break
                if parsed_step.get("type") == "clarify":
                    result_payload = {"type": "clarify", "question": parsed_step.get("question", "")}
                    break
                if action_budget > 0:
                    executed = await self._run_cognition_actions(
                        parsed_step.get("actions", []),
                        trace_id,
                        action_budget,
                        tools=tools,
                    )
                    observations.extend(executed)
                    action_budget = max(0, action_budget - len(executed))

        # Detect budget exhaustion before finalize so we can signal it downstream.
        _budget_was_exhausted = result_payload is None and action_budget == 0
        if result_payload is None:
            if self._cognition_budget_exhausted_signal and _budget_was_exhausted:
                observations.append({
                    "label": "budget_exhausted",
                    "action": None,
                    "observation": {
                        "status": "BUDGET_EXHAUSTED",
                        "error": "Action budget fully consumed before a final answer was reached.",
                    },
                })
            result_payload = await self._finalize_cognition_result(
                query_state,
                state_of_mind,
                observations,
                thinking_trace,
                trace_id,
                turn_id,
            )

        distillation: Dict[str, Any] = {}
        persisted_counts = {
            "skills": 0,
            "facts": 0,
            "patterns": 0,
            "lessons": 0,
            "safety_rules": 0,
        }
        # Skip distillation for:
        # 1. Successful pattern executions — the pattern already encodes the knowledge.
        # 2. Trivial turns (too short and no observations) — nothing meaningful to learn.
        _pattern_success = route_mode == "pattern" and result_payload.get("type") == "final"
        _trivial_turn = (
            self._cognition_distill_skip_trivial
            and self._is_trivial_cognition_request(text)
            and not observations
        )
        if skip_distillation or _pattern_success or _trivial_turn:
            self._log_mode_event(
                "COGNITION",
                "distill_skipped",
                {
                    "trace_id": trace_id,
                    "reason": (
                        "skip_distillation"
                        if skip_distillation
                        else "pattern_success"
                        if _pattern_success
                        else "trivial_turn"
                    ),
                },
            )
        else:
            distillation = await self._distill_cognition_episode(
                result_payload,
                query_state,
                state_of_mind,
                observations,
                thinking_trace,
                tools,
                patterns,
                trace_id,
                turn_id,
                budget_exhausted=_budget_was_exhausted,
            )
            persisted_counts = await self._persist_cognition_distillation(distillation, trace_id)
        try:
            append_jsonl(
                self._cognition_episodes_path,
                {
                    "ts": utc_now_iso(),
                    "task": text,
                    "thinking_mode": route_mode,
                    "execution_mode": execution_mode,
                    "query_state": query_state,
                    "state_of_mind": state_of_mind,
                    "observations": observations,
                    "thinking_trace": thinking_trace,
                    "result": result_payload,
                    "distillation": distillation,
                    "persisted_counts": persisted_counts,
                },
            )
        except Exception as _episode_log_exc:
            self._log_mode_event(
                "COGNITION",
                "episode_log_error",
                {"trace_id": trace_id, "error": str(_episode_log_exc)},
            )

        evidence_source = (
            f"cognition_{route_mode}"
            if route_mode in {"pattern", "system3"}
            else f"cognition_{execution_mode}"
        )
        self._set_turn_execution_evidence(
            turn_id,
            self._build_execution_evidence_payload(
                mode="COGNITION",
                source=evidence_source,
                entity_kind="cognition_episode",
                entity={
                    "thinking_mode": route_mode,
                    "execution_mode": execution_mode,
                    "query_state": query_state,
                    "state_of_mind": state_of_mind,
                    "thinking_trace": thinking_trace,
                    "persisted_counts": persisted_counts,
                },
                result=result_payload,
                tool_results=self._summarize_tool_results_from_observations(observations),
            ),
        )
        if turn_id:
            self._pending_turn_traces[turn_id] = list(thinking_trace)
        if result_payload.get("type") == "clarify":
            return self._cognition_result_payload_text(result_payload)
        return self._cognition_result_payload_text(result_payload)



    async def _execute_plan(self, plan, trace_id: str) -> List[Dict[str, Any]]:
        results = []
        for step in plan.steps:
            tool_id = step.get("tool_id")
            args = step.get("args", {})
            tool_def = self._tools.get_tool(tool_id)
            if not tool_def:
                results.append({"tool_id": tool_id, "status": "DENIED", "error": "Unknown tool"})
                continue

            decision = self._policy.evaluate_tool(tool_def, args)
            approval_token = None
            if decision.status == "CONFIRM":
                approval_token = await self._request_permission(tool_id, decision.reason)
                if not approval_token:
                    results.append({"tool_id": tool_id, "status": "DENIED", "error": "Not approved"})
                    continue
            elif decision.status == "PASSPHRASE":
                results.append({"tool_id": tool_id, "status": "DENIED", "error": "Passphrase required"})
                continue

            required_artifacts = []
            if "write" in tool_def.get("capabilities", []):
                required_artifacts = ["diff", "hash"]

            resp = await self._call_tool(tool_id, args, approval_token, trace_id, required_artifacts)
            results.append({
                "tool_id": tool_id,
                "status": resp.get("status"),
                "result": resp.get("result"),
                "artifacts": resp.get("artifacts"),
                "logs_ref": resp.get("logs_ref"),
                "error": resp.get("error"),
            })
            if resp.get("status") != "APPROVED":
                self._tool_failures += 1
                if self._tool_failures >= 3:
                    self._safe_mode = True
        return results

    async def _request_permission(self, tool_id: str, justification: str) -> Optional[str]:
        request_id = new_id()
        fut = asyncio.get_running_loop().create_future()
        self._permission_waiters[request_id] = fut
        turn_id = _CURRENT_TURN_ID.get()
        trace_id = _CURRENT_TRACE_ID.get()
        payload: Dict[str, Any] = {
            "request_id": request_id,
            "tool_id": tool_id,
            "justification": justification,
            "tier": 1,
        }
        if turn_id:
            payload["turn_id"] = turn_id
        if trace_id:
            payload["trace_id"] = trace_id
        await self._send_event(
            make_event(
                EVENT_PERMISSION_REQUEST,
                payload,
            )
        )
        try:
            result = await asyncio.wait_for(fut, timeout=30)
        except asyncio.TimeoutError:
            return None
        if not result.get("approved"):
            return None
        return result.get("token")

    async def _call_tool(
        self,
        tool_id: str,
        args: Dict[str, Any],
        approval_token: Optional[str],
        trace_id: str,
        required_artifacts: List[str],
    ) -> Dict[str, Any]:
        tool_def = self._tools.get_tool(tool_id) or {"tool_id": tool_id}
        req = {
            "request_id": new_id(),
            "tool_id": tool_id,
            "args": args,
            "proposed_risk": "MEDIUM",
            "justification": "User request",
            "required_artifacts": required_artifacts,
            "trace_id": trace_id,
        }
        record_event(
            "tool_request",
            {
                "request_id": req.get("request_id"),
                "trace_id": trace_id,
                "tool_id": tool_id,
                "args": redact_tool_payload(tool_def, "args", args),
                "required_artifacts": required_artifacts,
                "approval_present": bool(approval_token),
            },
        )
        if approval_token:
            req["approval_token"] = approval_token
        try:
            resp = await send_request(
                self._tool_host,
                self._tool_port,
                "tool.Execute",
                req,
                timeout=self._model_rpc_timeout_s,
            )
        except RpcError as e:
            record_event(
                "tool_response",
                {
                    "request_id": req.get("request_id"),
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": "ERROR",
                    "result": redact_tool_payload(tool_def, "result", None),
                    "error": str(e),
                    "logs_ref": None,
                },
            )
            return {"status": "ERROR", "error": str(e)}
        record_event(
            "tool_response",
            {
                "request_id": req.get("request_id"),
                "trace_id": trace_id,
                "tool_id": tool_id,
                "status": resp.get("status"),
                "result": redact_tool_payload(tool_def, "result", resp.get("result")),
                "error": resp.get("error"),
                "logs_ref": resp.get("logs_ref"),
            },
        )
        if resp.get("status") == "NEEDS_CONFIRMATION" and not approval_token:
            approval_token = await self._request_permission(tool_id, "Tool runtime requested confirmation")
            if not approval_token:
                return {"status": "DENIED", "error": "Not approved"}
            req["approval_token"] = approval_token
            try:
                resp = await send_request(
                    self._tool_host,
                    self._tool_port,
                    "tool.Execute",
                    req,
                    timeout=self._model_rpc_timeout_s,
                )
            except RpcError as e:
                record_event(
                    "tool_response",
                    {
                        "request_id": req.get("request_id"),
                        "trace_id": trace_id,
                        "tool_id": tool_id,
                        "status": "ERROR",
                        "result": redact_tool_payload(tool_def, "result", None),
                        "error": str(e),
                        "logs_ref": None,
                    },
                )
                return {"status": "ERROR", "error": str(e)}
            record_event(
                "tool_response",
                {
                    "request_id": req.get("request_id"),
                    "trace_id": trace_id,
                    "tool_id": tool_id,
                    "status": resp.get("status"),
                    "result": redact_tool_payload(tool_def, "result", resp.get("result")),
                    "error": resp.get("error"),
                    "logs_ref": resp.get("logs_ref"),
                },
            )
        return resp


async def serve() -> None:
    settings = load_settings()
    interface_cfg = dict(settings.interface or {})
    conversation_mode = str(interface_cfg.get("conversation_mode", "separate") or "separate").strip().lower()
    if conversation_mode not in {"shared", "separate"}:
        conversation_mode = "separate"

    conversations: Dict[str, Dict[str, Any]] = {}
    connection_counter = 0

    def _event_payload(event: Dict[str, Any]) -> Dict[str, Any]:
        payload = event.get("payload", {})
        return payload if isinstance(payload, dict) else {}

    def _event_turn_id(event: Dict[str, Any]) -> Optional[str]:
        turn_id = _event_payload(event).get("turn_id")
        return turn_id if isinstance(turn_id, str) and turn_id else None

    def _event_source_id(event: Dict[str, Any], fallback: str) -> str:
        source_id = _event_payload(event).get("source_id")
        if isinstance(source_id, str) and source_id.strip():
            return source_id.strip().lower()
        return fallback

    def _conversation_key(source_id: str) -> str:
        if conversation_mode == "shared":
            return "shared"
        return source_id or "default"

    def _drop_client_routes(client_queue: asyncio.Queue) -> None:
        for context in conversations.values():
            turn_targets = context["turn_targets"]
            stale_turns = [turn_id for turn_id, target in turn_targets.items() if target is client_queue]
            for turn_id in stale_turns:
                turn_targets.pop(turn_id, None)

    def _get_conversation(key: str) -> Dict[str, Any]:
        context = conversations.get(key)
        if context is not None:
            return context

        orchestrator = Orchestrator()
        out_queue: asyncio.Queue = asyncio.Queue()
        turn_targets: Dict[str, asyncio.Queue] = {}
        input_lock = asyncio.Lock()
        orchestrator.bind_out_queue(out_queue)

        async def dispatch() -> None:
            while True:
                msg = await out_queue.get()
                if msg is None:
                    break
                turn_id = _event_turn_id(msg)
                if not turn_id:
                    continue
                target_queue = turn_targets.get(turn_id)
                if target_queue is None:
                    continue
                await target_queue.put(msg)
                if msg.get("event_type") == EVENT_ASSISTANT_FINAL:
                    turn_targets.pop(turn_id, None)

        context = {
            "orchestrator": orchestrator,
            "out_queue": out_queue,
            "turn_targets": turn_targets,
            "input_lock": input_lock,
            "dispatch_task": asyncio.create_task(dispatch()),
        }
        conversations[key] = context
        return context

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connection_counter
        connection_counter += 1
        connection_id = f"connection:{connection_counter}"
        client_queue: asyncio.Queue = asyncio.Queue()

        async def consume() -> None:
            while True:
                event = await read_jsonl(reader)
                if event is None:
                    break
                if not isinstance(event, dict):
                    continue
                source_id = _event_source_id(event, connection_id)
                context = _get_conversation(_conversation_key(source_id))
                turn_id = _event_turn_id(event)
                if turn_id and event.get("event_type") in {EVENT_STT_FINAL, EVENT_TURN_PATCH}:
                    context["turn_targets"][turn_id] = client_queue
                async with context["input_lock"]:
                    await context["orchestrator"].handle_event(event)

        async def produce() -> None:
            while True:
                msg = await client_queue.get()
                if msg is None:
                    break
                await write_jsonl(writer, msg)

        consumer = asyncio.create_task(consume())
        producer = asyncio.create_task(produce())
        try:
            done, pending = await asyncio.wait(
                [consumer, producer],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc:
                    raise exc
        finally:
            _drop_client_routes(client_queue)
            try:
                client_queue.put_nowait(None)
            except Exception:
                pass
            consumer.cancel()
            producer.cancel()
            await asyncio.gather(consumer, producer, return_exceptions=True)
            writer.close()
            await writer.wait_closed()

    listen_host = settings.rpc["orch_host"]
    listen_port = settings.rpc["orch_port"]
    server = await asyncio.start_server(handle_client, listen_host, listen_port)
    print(f"[orchestrator] listening on {listen_host}:{listen_port}")
    print(f"[orchestrator] interface conversation mode: {conversation_mode}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve())
