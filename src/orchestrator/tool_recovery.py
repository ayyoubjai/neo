"""Bounded recovery actions shared by the cognition execution regimes."""
import math

from common.record_log import record_event
from common.tool_recovery import failure_details, lexical_score, source_mismatch


class ToolRecoveryMixin:
    def _recovery_state(self, trace_id):
        if not hasattr(self, '_tool_recovery_states'):
            self._tool_recovery_states = {}
        return self._tool_recovery_states.setdefault(trace_id, {
            'searches': 0, 'generations': 0, 'failures': {}, 'last_error': None,
        })

    def _recovery_error(self, action, code, message):
        return {'action': action, 'observation': {
            'status': 'ERROR', 'error': message, 'error_details': failure_details(code),
        }, 'new_tools': []}

    async def _retrieve_tools_action(self, action, trace_id):
        query = str(action.get('query') or '').strip()[:2000]
        if not query:
            return self._recovery_error(action, 'MISSING_INPUT', 'retrieve_tools requires a capability query.')
        state = self._recovery_state(trace_id)
        if state['searches'] >= 2:
            return self._recovery_error(action, 'RECOVERY_LIMIT', 'At most two full-registry searches per turn.')
        state['searches'] += 1
        # list_available respects the planner profile but not the initial cap.
        available = self._tools.list_available()
        intent = action.get('intent', {})
        excluded = action.get('exclude_tool_ids', [])
        candidates = [tool for tool in available if not source_mismatch(tool, intent)
                      and tool.get('tool_id') not in excluded]
        status = 'semantic'
        try:
            query_embedding = await self._embed_text(query)
            scored = []
            for tool in candidates:
                vector = await self._embed_text(self._tool_description(tool), task='search_document')
                score = self._cosine(query_embedding, vector)
                if math.isfinite(score) and score >= self._tool_retrieval_min_similarity:
                    scored.append((score, tool))
        except Exception:
            status = 'lexical_fallback'
            scored = [(lexical_score(query, tool), tool) for tool in candidates]
            scored = [(score, tool) for score, tool in scored if score > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        found = [tool for _, tool in scored[:6]]
        result = {'query': query, 'search_status': status, 'searched_count': len(candidates),
                  'tools': found, 'guidance': 'Review these candidates. No matches does not prove that a new tool can solve the problem.'}
        record_event('tool_recovery_search', {'trace_id': trace_id, 'query': query, 'status': status,
                     'searched_count': len(candidates), 'tool_ids': [tool['tool_id'] for tool in found]})
        state['last_search'] = {'query': query, 'tool_ids': [tool['tool_id'] for tool in found]}
        return {'action': action, 'observation': {'status': 'APPROVED', 'result': result}, 'new_tools': found}

    async def _generation_recovery_gate(self, action, trace_id):
        state = self._recovery_state(trace_id)
        if state['last_error'] in {'PERMISSION_DENIED', 'MISSING_INPUT', 'INVALID_MODEL_OUTPUT'}:
            return self._recovery_error(action, state['last_error'],
                'Resolve the input, permission, or model-output failure before generating a replacement tool.')
        spec = action.get('spec', {})
        if not isinstance(spec, dict):
            return self._recovery_error(action, 'MISSING_INPUT', 'spec must be an object')
        if not state['searches']:
            result = await self._retrieve_tools_action({
                'action_type': 'retrieve_tools', 'query': spec.get('request', ''),
                'intent': action.get('intent', {}),
            }, trace_id)
            result['observation']['recovery_note'] = (
                'Tool generation deferred: review the registry search first. If no candidate fits, '
                'resubmit generation with generation_reason explaining the implementation gap and '
                'why it is feasible with the available dependencies and permissions.')
            return result
        if not str(action.get('generation_reason') or '').strip():
            return self._recovery_error(action, 'MISSING_INPUT',
                'generation_reason must explain why searched tools cannot implement the operation '
                'and why a new implementation is feasible in this environment.')
        if state['generations'] >= 1:
            return self._recovery_error(action, 'RECOVERY_LIMIT', 'At most one tool-generation attempt per turn.')
        state['generations'] += 1
        record_event('tool_recovery_generation', {'trace_id': trace_id,
                     'reason': str(action['generation_reason'])[:2000], 'search': state.get('last_search')})
        return None

    def _preflight_tool_action(self, action, trace_id):
        tool = self._tools.get_tool(str(action.get('tool_id') or ''))
        if not tool:
            return self._recovery_error(action, 'MISSING_CAPABILITY', 'Unknown tool. Retrieve alternatives from the registry.')
        if source_mismatch(tool, action.get('intent', {})):
            return self._recovery_error(action, 'SOURCE_MISMATCH',
                f"Declared intent source differs from tool source {tool.get('source')!r}.")
        args = action.get('args') or {}
        if not isinstance(args, dict):
            return self._recovery_error(action, 'MISSING_INPUT', 'Tool args must be an object.')
        schema = tool.get('input_schema') or {}
        for argument in schema.get('required', []):
            if argument not in args:
                return self._recovery_error(action, 'MISSING_INPUT', f'Required argument is missing: {argument}')
        for requirement in tool.get('requires', []):
            if not isinstance(requirement, dict):
                continue
            argument = requirement.get('argument')
            if argument and (not args.get(argument) or str(args[argument]).startswith('<')):
                return self._recovery_error(action, 'MISSING_INPUT',
                    f"Supply {argument} from an existing {requirement.get('artifact_type', 'input')}. "
                    'Retrieve a producer if needed; never invent a result reference.')
        key = self._safe_json({'tool_id': action.get('tool_id'), 'args': args})
        previous = self._recovery_state(trace_id)['failures'].get(key)
        if previous and (previous['code'] != 'TEMPORARY_FAILURE' or previous['count'] >= 2):
            return self._recovery_error(action, previous['code'],
                'This unchanged action already failed. Correct its input or retrieve an alternative instead of repeating it.')
        return None

    def _remember_tool_outcome(self, action, observation, trace_id):
        state = self._recovery_state(trace_id)
        if observation.get('status') == 'APPROVED':
            state['last_error'] = None
            state['last_error_message'] = None
            return
        details = observation.get('error_details') or failure_details(
            'PERMISSION_DENIED' if observation.get('status') == 'DENIED' else 'UNKNOWN')
        observation['error_details'] = details
        state['last_error'] = details.get('code', 'UNKNOWN')
        state['last_error_message'] = str(observation.get('error') or details.get('recovery') or '')[:1500]
        key = self._safe_json({'tool_id': action.get('tool_id'), 'args': action.get('args') or {}})
        previous = state['failures'].get(key, {'count': 0})
        state['failures'][key] = {'count': previous['count'] + 1, 'code': state['last_error']}
