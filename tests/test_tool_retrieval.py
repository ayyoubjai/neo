from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.record_log import _format_human_event
from orchestrator.main import Orchestrator


class ToolRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def test_visual_followup_keeps_capture_and_analysis_available(self):
        camera = {'tool_id': 'vision.observe'}
        capture = {'tool_id': 'computer.screenshot', 'produces': [{'artifact_type': 'image', 'field': 'image_ref'}]}
        analysis = {'tool_id': 'image.analyse', 'requires': [{'artifact_type': 'image', 'argument': 'image_ref'}]}
        self.agent._tools.list_active.return_value = [camera, capture, analysis]
        result = self.agent._merge_relevant_tool_lists([camera, analysis])
        self.assertEqual([tool['tool_id'] for tool in result], ['vision.observe', 'image.analyse', 'computer.screenshot'])
        self.assertEqual(self.agent._merge_relevant_tool_lists(result), result)
        prompt = self.agent._format_tools_for_agent(result)
        self.assertIn('NEXT action cycle', prompt)
        self.assertIn('Past conversation/memory is not evidence', prompt)
        self.assertIn('CAMERA frames', prompt)

    def test_visual_pair_does_not_enable_inactive_tools(self):
        analysis = {'tool_id': 'image.analyse'}
        self.agent._tools.list_active.return_value = [analysis]
        self.assertEqual(self.agent._merge_relevant_tool_lists([analysis]), [analysis])

    def test_click_adds_transitive_grounding_and_screenshot_prerequisites(self):
        click = {'tool_id': 'computer.click', 'requires': [{'artifact_type': 'screen_coordinates'}]}
        ground = {'tool_id': 'ui.predict_coords',
                  'requires': [{'artifact_type': 'image', 'argument': 'image_ref'}],
                  'produces': [{'artifact_type': 'screen_coordinates'}]}
        capture = {'tool_id': 'computer.screenshot', 'produces': [{'artifact_type': 'image'}]}
        self.agent._tools.list_active.return_value = [click, ground, capture]
        result = self.agent._merge_relevant_tool_lists([click])
        self.assertEqual([tool['tool_id'] for tool in result],
                         ['computer.click', 'ui.predict_coords', 'computer.screenshot'])
        prompt = self.agent._format_tools_for_agent(result)
        self.assertIn('ui.predict_coords', prompt)
        self.assertIn('must not be used for click coordinates', prompt)

    def setUp(self) -> None:
        self.agent = Orchestrator.__new__(Orchestrator)
        self.agent._retrieve_relevant_tools_enabled = True
        self.agent._tool_retrieval_min_similarity = 0.7
        self.tools = [{"tool_id": name} for name in ("weak", "best", "good", "opposite")]
        vectors = [[0, 1], [1, 0], [1, 1], [-1, 0]]
        self.agent._tools = Mock()
        self.agent._tools.list_active.return_value = self.tools
        self.agent._tool_index = {
            tool["tool_id"]: {"tool": tool, "embedding": vector}
            for tool, vector in zip(self.tools, vectors)
        }
        self.agent._ensure_tool_index = AsyncMock()
        self.agent._embed_text = AsyncMock(return_value=[1, 0])
        self.log_patch = patch("orchestrator.main.record_event")
        self.log = self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    async def test_returns_only_relevant_tools_even_when_catalog_is_smaller_than_cap(self) -> None:
        result = await self.agent._retrieve_relevant_tools("query", 8)
        self.assertEqual([tool["tool_id"] for tool in result], ["best", "good"])
        self.assertEqual(self.log.call_args.args[1]["selected_tool_ids"], ["best", "good"])

    async def test_cap_keeps_highest_scoring_match(self) -> None:
        result = await self.agent._retrieve_relevant_tools("query", 1)
        self.assertEqual(result, [self.tools[1]])

    async def test_zero_matches_is_valid_and_not_replaced_with_catalog(self) -> None:
        self.agent._embed_text.return_value = [0, 0, 1]
        result = await self.agent._retrieve_relevant_tools("query", 8)
        self.assertEqual(result, [])
        self.assertEqual(self.log.call_args.args[1]["status"], "selected")

    async def test_threshold_is_inclusive(self) -> None:
        self.agent._tool_retrieval_min_similarity = 1.0
        self.assertEqual(await self.agent._retrieve_relevant_tools("query", 8), [self.tools[1]])

    async def test_minus_one_restores_unfiltered_ranking(self) -> None:
        self.agent._tool_retrieval_min_similarity = -1.0
        result = await self.agent._retrieve_relevant_tools("query", 8)
        self.assertEqual([tool["tool_id"] for tool in result], ["best", "good", "weak", "opposite"])

    async def test_embedding_failure_fallback_is_bounded_and_logged(self) -> None:
        self.agent._embed_text.side_effect = RuntimeError("embedding offline")
        result = await self.agent._retrieve_relevant_tools("query", 2)
        self.assertEqual(result, self.tools[:2])
        payload = self.log.call_args.args[1]
        self.assertEqual(payload["status"], "fallback")
        self.assertEqual(payload["error"], "embedding offline")

    async def test_disabled_retrieval_preserves_full_catalog(self) -> None:
        self.agent._retrieve_relevant_tools_enabled = False
        self.assertEqual(await self.agent._retrieve_relevant_tools("query", 1), self.tools)
        self.agent._embed_text.assert_not_awaited()

    async def test_zero_cap_skips_embedding(self) -> None:
        self.assertEqual(await self.agent._retrieve_relevant_tools("query", 0), [])
        self.agent._embed_text.assert_not_awaited()

    async def test_log_shows_scores_and_explicit_empty_selection_despite_truncation(self) -> None:
        self.agent._tool_retrieval_min_similarity = 1.0
        self.agent._embed_text.return_value = [1, 2]
        await self.agent._retrieve_relevant_tools("query", 8)
        payload = self.log.call_args.args[1]
        line = _format_human_event("tool_retrieval", payload, 20)
        self.assertIn("selected=0/8", line)
        self.assertIn("RETRIEVED_TOOLS: none", line)
        self.assertIn("TOP_TOOL_SCORES:", line)
        self.assertIn("good=", line)


class ToolEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.agent = Orchestrator.__new__(Orchestrator)
        self.agent._tool_embedding_model = "nomic-embed-text-v1.5.Q4_K_M.gguf"
        self.agent._embed_cache = {}
        self.agent._embed_cache_limit = 10
        self.agent._model_host = "localhost"
        self.agent._model_port = 1234

    async def test_nomic_task_prefixes_and_cache_are_distinct(self) -> None:
        with patch("orchestrator.main.send_request", new_callable=AsyncMock) as rpc:
            rpc.return_value = {"embedding": [1.0, 0.0], "model": "llamacpp:nomic"}
            await self.agent._embed_text("same text")
            await self.agent._embed_text("same text", task="search_document")
            await self.agent._embed_text("same text")
            self.assertEqual(rpc.await_count, 2)
            self.assertEqual(rpc.await_args_list[0].args[3]["text"], "search_query: same text")
            self.assertEqual(rpc.await_args_list[1].args[3]["text"], "search_document: same text")

    async def test_other_models_do_not_receive_nomic_prefix(self) -> None:
        self.agent._tool_embedding_model = "other-model"
        with patch("orchestrator.main.send_request", new_callable=AsyncMock) as rpc:
            rpc.return_value = {"embedding": [1.0]}
            await self.agent._embed_text("query")
            self.assertEqual(rpc.call_args.args[3]["text"], "query")

    async def test_placeholder_and_invalid_vectors_are_not_cached(self) -> None:
        responses = [
            {"embedding": [1.0], "model": "sha256-text-v1"},
            {"embedding": [], "status": "ERROR"},
            {"embedding": []}, {"embedding": [0, 0]},
            {"embedding": [float("nan")]}, {"embedding": ["invalid"]},
        ]
        for response in responses:
            with self.subTest(response=response), patch(
                "orchestrator.main.send_request", new_callable=AsyncMock, return_value=response
            ):
                with self.assertRaises(ValueError):
                    await self.agent._embed_text("query")
                self.assertEqual(self.agent._embed_cache, {})


if __name__ == "__main__":
    unittest.main()
