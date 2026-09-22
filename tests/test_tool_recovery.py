import unittest
from unittest.mock import AsyncMock, Mock, patch

from common.tool_recovery import contextual_query, expand_prerequisites, failure_details
from orchestrator.main import Orchestrator
from orchestrator.tool_selector import ToolRegistry
from tool_runtime.tools import ToolError, image_analyse
from tool_runtime.main import ToolRuntime


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = Orchestrator.__new__(Orchestrator)
        self.camera = {'tool_id': 'vision.observe', 'source': 'camera', 'description': 'Observe camera frames'}
        self.screen = {'tool_id': 'computer.screenshot', 'source': 'desktop', 'description': 'Capture desktop screen',
                       'produces': [{'artifact_type': 'image'}]}
        self.analysis = {'tool_id': 'image.analyse', 'source': 'image',
                         'requires': [{'argument': 'image_ref', 'artifact_type': 'image'}]}
        catalog = [self.camera, self.screen, self.analysis]
        self.agent._tools = Mock()
        self.agent._tools.list_available.return_value = catalog
        self.agent._tools.list_active.return_value = catalog
        self.agent._tools.get_tool.side_effect = lambda tool_id: next((t for t in catalog if t['tool_id'] == tool_id), None)
        self.agent._embed_text = AsyncMock(return_value=[1, 0])
        self.agent._tool_retrieval_min_similarity = .7
        self.agent._execute_tool_action = AsyncMock(return_value={'observation': {'status': 'APPROVED'}})
        self.agent.generate_tools_from_spec = AsyncMock(return_value={'status': 'APPROVED', 'result': {'tools': []}})
        p = patch('orchestrator.tool_recovery.record_event')
        p.start()
        self.addCleanup(p.stop)

    def test_context_is_bounded_and_not_live_evidence(self):
        self.assertEqual(contextual_query('hello', {}), 'hello')
        query = contextual_query('what is it?', {'summary': 'x' * 2000})
        self.assertIn('not current observations', query)
        self.assertNotIn('x' * 801, query)

    def test_prerequisites_are_generic_and_only_offer_tools(self):
        producer = {'tool_id': 'custom.capture', 'produces': [{'artifact_type': 'sample'}]}
        consumer = {'tool_id': 'custom.read', 'requires': [{'artifact_type': 'sample'}]}
        self.assertEqual(expand_prerequisites([consumer], [producer]), [consumer, producer])
        self.assertEqual(expand_prerequisites([consumer], []), [consumer])

    async def test_source_mismatch_never_executes_camera(self):
        result = await self.agent._execute_cognition_action({'tool_id': 'vision.observe', 'args': {},
                    'intent': {'source': 'desktop'}}, 'trace')
        self.assertEqual(result['observation']['error_details']['code'], 'SOURCE_MISMATCH')
        self.agent._execute_tool_action.assert_not_awaited()

    async def test_missing_artifact_blocks_execution_and_generation(self):
        result = await self.agent._execute_cognition_action({'tool_id': 'image.analyse', 'args': {}}, 'trace')
        self.assertEqual(result['observation']['error_details']['code'], 'MISSING_INPUT')
        generation = await self.agent._generation_recovery_gate({'spec': {'request': 'analyze'}}, 'trace')
        self.assertEqual(generation['observation']['error_details']['code'], 'MISSING_INPUT')
        self.agent._execute_tool_action.assert_not_awaited()

    def test_nonexistent_image_is_checked_before_model_call(self):
        with self.assertRaises(ToolError) as caught:
            image_analyse({'image_ref': 'workspace:/does-not-exist-neo-test.png'}, '/tmp')
        self.assertEqual(caught.exception.details['code'], 'MISSING_INPUT')

    async def test_full_search_filters_source_and_is_bounded(self):
        action = {'query': 'desktop screenshot', 'intent': {'source': 'desktop'}}
        result = await self.agent._retrieve_tools_action(action, 'trace')
        self.assertNotIn(self.camera, result['new_tools'])
        self.assertIn(self.screen, result['new_tools'])
        await self.agent._retrieve_tools_action(action, 'trace')
        limited = await self.agent._retrieve_tools_action(action, 'trace')
        self.assertEqual(limited['observation']['error_details']['code'], 'RECOVERY_LIMIT')

    async def test_search_recovers_without_embedder(self):
        self.agent._embed_text.side_effect = RuntimeError('offline')
        result = await self.agent._retrieve_tools_action({'query': 'desktop'}, 'trace')
        self.assertEqual(result['observation']['result']['search_status'], 'lexical_fallback')
        self.assertEqual(result['new_tools'], [self.screen])

    async def test_generation_requires_search_reason_and_only_one_attempt(self):
        action = {'action_type': 'generate_tools_from_spec', 'spec': {'request': 'new capability'}}
        result = await self.agent._execute_cognition_action(action, 'trace')
        self.assertIn('recovery_note', result['observation'])
        self.agent.generate_tools_from_spec.assert_not_awaited()
        result = await self.agent._execute_cognition_action(action, 'trace')
        self.assertEqual(result['observation']['error_details']['code'], 'MISSING_INPUT')
        action['generation_reason'] = 'Existing tools do not support this transformation; use installed stdlib.'
        await self.agent._execute_cognition_action(action, 'trace')
        self.agent.generate_tools_from_spec.assert_awaited_once()
        result = await self.agent._execute_cognition_action(action, 'trace')
        self.assertEqual(result['observation']['error_details']['code'], 'RECOVERY_LIMIT')

    async def test_permission_denial_does_not_search_or_generate(self):
        self.agent._execute_tool_action.return_value = {'observation': {'status': 'DENIED'}}
        await self.agent._run_cognition_actions([{'tool_id': 'computer.screenshot', 'args': {}}], 'trace', 1)
        self.agent._tools.list_available.assert_not_called()
        result = await self.agent._generation_recovery_gate({'spec': {'request': 'capture'}}, 'trace')
        self.assertEqual(result['observation']['error_details']['code'], 'PERMISSION_DENIED')

    async def test_failure_automatically_offers_alternative(self):
        tools = [self.camera]
        result = await self.agent._run_cognition_actions([{'tool_id': 'vision.observe', 'args': {},
                  'intent': {'source': 'desktop', 'operation': 'capture desktop screen'}}], 'trace', 1, tools)
        self.assertIn(self.screen, tools)
        self.assertIn('recovery_search', result[0]['observation'])
        self.agent._execute_tool_action.assert_not_awaited()

    def test_identical_failures_are_not_retried_except_once_for_transient(self):
        action = {'tool_id': 'computer.screenshot', 'args': {}}
        for code, trace in [('UNKNOWN', 'unknown'), ('TEMPORARY_FAILURE', 'temporary')]:
            error = {'status': 'ERROR', 'error_details': failure_details(code)}
            self.agent._remember_tool_outcome(action, error, trace)
            result = self.agent._preflight_tool_action(action, trace)
            self.assertEqual(result is None, code == 'TEMPORARY_FAILURE')
            self.agent._remember_tool_outcome(action, error, trace)
            self.assertIsNotNone(self.agent._preflight_tool_action(action, trace))

    def test_available_registry_is_not_limited_to_initial_cap(self):
        registry = ToolRegistry.__new__(ToolRegistry)
        registry._planner_profile = 'default'
        registry._tool_order = [dict(self.camera, tool_bucket='base'), dict(self.screen, tool_bucket='base'),
                                {'tool_id': 'hidden', 'tool_bucket': 'restricted'}]
        registry._active_tool_limit = 1
        self.assertEqual([t['tool_id'] for t in registry.list_available()], ['vision.observe', 'computer.screenshot'])

    def test_compile_validation_rejects_parseable_invalid_code(self):
        self.agent._settings = Mock(orchestrator={})
        self.assertIn('invalid syntax', self.agent._validate_generated_tool_code('break\nreturn {}, {}'))

    def test_tool_prompt_includes_descriptions_and_contracts(self):
        prompt = self.agent._format_tools_for_agent([self.camera, self.analysis])
        self.assertIn('Observe camera frames', prompt)
        self.assertIn('source=camera', prompt)
        self.assertIn('image_ref', prompt)
        category = self.agent._format_tool_categories_for_agent({'vision': [self.camera]})
        self.assertIn('Observe camera frames', category)
        self.assertIn('source=camera', category)

    def test_small_mode_exposes_recovery_with_empty_tool_selection(self):
        self.agent._cognition_mode = 'small'
        self.agent._cognition_query_state_enabled = False
        self.agent._cognition_state_of_mind_enabled = False
        self.agent._cognition_system1_clarify_enabled = False
        self.agent._cognition_system2_clarify_enabled = False
        prompts = [self.agent._build_cognition_system1_prompt({}, {}, [], tools=[]),
                   self.agent._build_cognition_system2_prompt({}, {}, [], [], tools=[])]
        for prompt in prompts:
            self.assertIn('retrieve_tools', prompt)
            self.assertIn('generation_reason', prompt)
            self.assertIn('PERMISSION_DENIED', prompt)

    def test_normalization_keeps_recovery_fields_and_honors_zero_budget(self):
        action = {'action_type': 'generate_tools_from_spec', 'spec': {'request': 'test'},
                  'generation_reason': 'gap', 'intent': {'source': 'desktop'}}
        normalized = self.agent._normalize_cognition_actions([action])[0]
        self.assertEqual(normalized['generation_reason'], 'gap')
        self.assertEqual(normalized['intent'], action['intent'])
        self.assertEqual(self.agent._normalize_cognition_actions([action], limit=0), [])

    async def test_runtime_preserves_structured_error_in_response_and_logs(self):
        runtime = ToolRuntime.__new__(ToolRuntime)
        runtime._workspace_root = '/tmp'
        runtime._refresh_generated_tools = Mock()
        runtime._append_tool_io = Mock()
        runtime._tools = {'test.fail': {'fn': Mock(side_effect=ToolError('missing package', code='MISSING_DEPENDENCY')), 'tier': 0}}
        with patch('tool_runtime.main.record_event') as log:
            result = await runtime.Execute({'tool_id': 'test.fail', 'args': {}})
        self.assertEqual(result['error_details']['code'], 'MISSING_DEPENDENCY')
        self.assertEqual(runtime._append_tool_io.call_args.args[0]['error_details'], result['error_details'])
        self.assertEqual(log.call_args.args[1]['error_details'], result['error_details'])

    async def test_model_connection_failure_preserves_desktop_blocker(self):
        from common.jsonl_rpc import RpcError
        self.agent._build_llm_context = Mock(return_value={})
        self.agent._model_host = 'localhost'
        self.agent._model_port = 1
        self.agent._model_rpc_timeout_s = 1
        self.agent._recovery_state('trace')['last_error_message'] = 'Desktop input requires an X11 session.'
        with patch('orchestrator.main.send_request', AsyncMock(side_effect=RpcError('Empty RPC response'))), \
                patch('orchestrator.main.record_event'):
            response = await self.agent._generate_cognition_response('prompt', 'trace')
        self.assertIn('Desktop input requires an X11 session', response)
        self.assertIn('Empty RPC response', response)
        self.assertNotIn('smaller model', response)

    async def test_invalid_final_and_repair_preserve_latest_unresolved_tool_failure(self):
        self.agent._build_cognition_final_prompt = Mock(return_value='final prompt')
        self.agent._generate_cognition_response = AsyncMock(return_value='')
        self.agent._generate_cognition_response_with_timeout = AsyncMock(return_value='coordinates unavailable')
        self.agent._parse_cognition_thinking_response = Mock(return_value=None)
        self.agent._log_mode_event = Mock()
        self.agent._cognition_repair_timeout_s = 1
        self.agent._cognition_repair_model = ''
        self.agent._cognition_repair_options = {}
        observations = [
            {'action': {'tool_id': 'ui.predict_coords'}, 'observation': {'status': 'APPROVED'}},
            {'action': {'tool_id': 'computer.click'}, 'observation': {
                'status': 'ERROR', 'error': 'PyGObject missing in Neo venv',
                'error_details': {'code': 'MISSING_DEPENDENCY'}}},
            {'action': {'action_type': 'retrieve_tools'}, 'observation': {'status': 'APPROVED'}},
        ]
        result = await self.agent._finalize_cognition_result({}, {}, observations, [], 'trace', None)
        self.assertIn('computer.click failed (MISSING_DEPENDENCY)', result['text'])
        self.assertIn('PyGObject missing in Neo venv', result['text'])
        self.assertNotIn('coordinates unavailable', result['text'])
        observations.append({'action': {'tool_id': 'computer.click'}, 'observation': {'status': 'APPROVED'}})
        result = await self.agent._finalize_cognition_result({}, {}, observations, [], 'trace', None)
        self.assertNotIn('PyGObject', result['text'])
