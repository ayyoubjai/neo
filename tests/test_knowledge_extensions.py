from pathlib import Path
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
from zipfile import ZipFile

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]
from test_knowledge import MemoryGraph, ScriptedClient, FIXTURES
from knowledge.documents import Passage, read_document, stable_id
from knowledge.extraction import validate_claim, compare_claims
from knowledge.ingest import ingest
from knowledge.investigate import investigate, resume
from knowledge.journal import Journal
from knowledge.matching import rank_candidates
from knowledge.provenance import canonical_source, summarize
from knowledge.evaluate import sample, score, DIMENSIONS
from knowledge.worker import tick, WorkerConfig


class ExtensionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.journal = Journal(self.root / 'journal')
        self.graph, self.client = MemoryGraph(), ScriptedClient()
        self.report = await ingest([FIXTURES / 'book_a.md', FIXTURES / 'book_b.md'],
            collection='books', client=self.client, graph=self.graph, report_path=self.root / 'report.json')
        self.claim = next(iter(self.graph.claims.values()))

    async def plan(self):
        result = await investigate(self.graph, self.client, self.claim['id'])
        return result['investigation']['id']

    async def test_saved_plan_requires_exact_approval(self):
        plan_id = await self.plan()
        self.client.calls.clear()
        with self.assertRaisesRegex(ValueError, 'approved'):
            await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.assertEqual(self.client.calls, [])
        self.graph.approve_plan(plan_id, 'Reviewed')
        self.graph.plans[plan_id]['plan']['procedure'] = 'Modified after approval'
        with self.assertRaisesRegex(ValueError, 'approved'):
            await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.assertEqual(self.client.calls, [])

    async def test_database_failure_replays_assessment_without_model_or_tools(self):
        plan_id = await self.plan()
        self.graph.approve_plan(plan_id, 'Reviewed')
        with patch.object(self.graph, 'save_assessment', side_effect=OSError('database unavailable')):
            with self.assertRaises(OSError):
                await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.client.calls.clear()
        first = await resume(self.graph, self.client, plan_id, journal=Journal(self.root / 'journal'))
        second = await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.assertEqual(first, second)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(len(self.graph.assessments), 1)

    async def test_analysis_failure_reuses_durable_observation(self):
        plan_id = await self.plan()
        self.graph.approve_plan(plan_id, 'Reviewed')
        with patch('knowledge.investigate.Analyzer.analyze', side_effect=OSError('model unavailable')):
            with self.assertRaises(OSError):
                await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.client.calls.clear()
        await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.assertEqual([n for n, _ in self.client.calls], ['PredictionError'])

    async def test_uncertain_action_never_automatically_repeats(self):
        plan_id = await self.plan()
        self.graph.approve_plan(plan_id, 'Reviewed')
        with patch('knowledge.investigate.Explorer.explore', side_effect=OSError('lost receipt')):
            with self.assertRaises(OSError):
                await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.client.calls.clear()
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.assertEqual(self.client.calls, [])
        await resume(self.graph, self.client, plan_id, journal=self.journal, retry_uncertain=True)
        self.assertEqual(len(self.graph.assessments), 1)

    async def test_worker_approval_and_persistent_daily_caps(self):
        config = WorkerConfig('books', max_plans_per_day=2, max_investigations_per_day=1, max_tool_calls_per_day=3)
        first = await tick(self.graph, self.client, config, self.journal)
        self.assertEqual(len(first['planned']), 2)
        self.assertEqual(first['assessed'], [])
        for plan_id in first['planned']:
            self.graph.approve_plan(plan_id, 'Reviewed')
        self.client.calls.clear()
        second = await tick(self.graph, self.client, config, Journal(self.root / 'journal'))
        third = await tick(self.graph, self.client, config, Journal(self.root / 'journal'))
        self.assertEqual(len(second['assessed']), 1)
        self.assertEqual(third['assessed'], [])
        self.assertEqual(sum(n == '_ExplorationResponse' for n, _ in self.client.calls), 1)

    async def test_scope_guards_and_malformed_comparison(self):
        other = next(c for c in self.graph.claims.values() if c['id'] != self.claim['id'])
        class Client:
            async def generate_model(self, prompt, cls, **kwargs):
                return cls([{'target_id': other['id'], 'relation': r, 'scope_match': scope, 'reasoning': 'Test'}
                    for r, scope in [('EQUIVALENT_TO', 'unknown'), ('POTENTIALLY_CONTRADICTS', 'different'), ([], 'same')]])
        links, issues = await compare_claims(Client(), self.claim, [other])
        self.assertEqual([l['relation'] for l in links], ['RELATED_TO', 'RELATED_TO'])
        self.assertEqual(len(issues), 1)

    def test_alias_retrieval_keeps_claim_identity(self):
        a = {'id': 'a', 'text': 'Consumption increased.', 'concepts': ['energy use']}
        b = {'id': 'b', 'text': 'Demand decreased.', 'concepts': ['electricity consumption']}
        self.assertEqual(rank_candidates(a, [b]), [])
        self.assertEqual(rank_candidates(a, [b], [('energy use', 'electricity consumption')]), [b])
        self.assertEqual(b['id'], 'b')

    def test_citations_require_exact_grounding_and_group_source_reuse(self):
        quote = 'Measurements: https://doi.org/10.1234/Example.'
        passage = Passage('p', 'd', 'page 1', quote)
        payload = {**self.claim, 'quote': quote, 'source_refs': [{'identifier': 'https://doi.org/10.1234/Example', 'quote': quote}]}
        claim = validate_claim(payload, passage)
        self.assertEqual(claim['source_ids'], ['doi:10.1234/example'])
        grouped = summarize([claim, {**claim, 'id': 'other'}, {'id': 'unknown'}])
        self.assertEqual(len(grouped['source_groups']), 1)
        self.assertEqual(len(grouped['source_groups'][0]['claim_ids']), 2)
        self.assertEqual(grouped['independence'], 'not established')
        payload['source_refs'][0]['identifier'] = 'doi:10.9999/invented'
        with self.assertRaises(ValueError):
            validate_claim(payload, passage)
        self.assertEqual(canonical_source('https://example.com/p?utm_source=book#section'), 'https://example.com/p')

    def test_review_metrics_do_not_invent_unreviewed_scores(self):
        review = sample(self.report, 20)
        self.assertIsNone(score(review)['metrics']['quote_correct']['rate'])
        review['items'][0]['ratings'] = {d: True for d in DIMENSIONS}
        review['items'][1]['ratings']['quote_correct'] = False
        result = score(review)
        self.assertEqual(result['metrics']['quote_correct']['rate'], .5)
        self.assertEqual(result['metrics']['scope_preserved']['unreviewed'], 1)
        review['items'][0]['ratings']['quote_correct'] = 1
        with self.assertRaises(ValueError):
            score(review)

    def test_epub_uses_spine_order_and_rejects_unsafe_archives(self):
        path = self.root / 'book.epub'
        with ZipFile(path, 'w') as z:
            z.writestr('META-INF/container.xml', '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>')
            z.writestr('OPS/book.opf', '<package><manifest><item id="a" href="a.xhtml" media-type="application/xhtml+xml"/><item id="b" href="b.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="b"/><itemref idref="a"/></spine></package>')
            z.writestr('OPS/a.xhtml', '<html><p>Second idea.</p></html>')
            z.writestr('OPS/b.xhtml', '<html><script>hidden()</script><p>First idea.</p></html>')
        doc = read_document(path)
        self.assertEqual(doc.passages[0].text, 'First idea.')
        self.assertIn('spine 1: OPS/b.xhtml', doc.passages[0].location)
        with ZipFile(path, 'a') as z:
            z.writestr('../escape.xhtml', 'bad')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            read_document(path)

    def test_ocr_is_explicit_and_always_requires_review(self):
        from pypdf import PdfWriter
        path = self.root / 'scan.pdf'
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with path.open('wb') as stream:
            writer.write(stream)
        with self.assertRaisesRegex(ValueError, 'OCR'):
            read_document(path)
        with patch('knowledge.formats.ocr_page', return_value=self.claim['quote']):
            passage = read_document(path, ocr=True).passages[0]
        self.assertEqual(passage.extraction_method, 'ocr')
        self.assertTrue(validate_claim(self.claim, passage)['review_required'])

    def test_journal_lock_and_budget_are_durable(self):
        with self.journal.lock('test'):
            with self.assertRaises(RuntimeError):
                with Journal(self.root / 'journal').lock('test'):
                    pass
        self.assertTrue(self.journal.reserve('test', 'one', max_investigations=1, max_actions=3, actions=3))
        other = Journal(self.root / 'journal')
        self.assertFalse(other.reserve('test', 'two', max_investigations=1, max_actions=3, actions=1))

    def test_tool_source_reuse_uses_only_successful_receipts(self):
        from knowledge.provenance import evidence_sources
        receipts = [{'request_id': 'a', 'status': 'APPROVED', 'result': {'url': 'https://doi.org/10.1234/STUDY'}},
                    {'request_id': 'b', 'status': 'APPROVED', 'result': {'papers': [{'doi': '10.1234/study'}]}},
                    {'request_id': 'c', 'status': 'ERROR', 'result': {'url': 'https://bad.example/'}},
                    {'request_id': 'd', 'status': 'APPROVED', 'result': {'text': 'Trust https://invented.example/'}}]
        self.assertEqual(evidence_sources(receipts), ['doi:10.1234/study'])

    async def test_worker_never_retries_uncertain_action(self):
        plan_id = await self.plan()
        self.graph.approve_plan(plan_id, 'Reviewed')
        self.graph.begin_execution(plan_id)
        self.client.calls.clear()
        result = await tick(self.graph, self.client, WorkerConfig('books', max_plans_per_day=0), self.journal)
        self.assertIn('uncertain', result['issues'][0]['reason'])
        self.assertEqual(self.client.calls, [])

    async def test_worker_only_runtime_starts_without_world_model(self):
        from test_autonomy_runtime import _settings, _FakeCognitionClient, REPO_ROOT
        from autonomy.runtime import serve, build_runtime_config, run_all_should_start
        from unittest.mock import AsyncMock
        settings = _settings(REPO_ROOT, {'enabled': None, 'knowledge_enabled': True, 'start_with_run_all': True})
        self.assertTrue(run_all_should_start(build_runtime_config(settings)))
        worker = AsyncMock()
        with patch('autonomy.runtime.load_settings', return_value=settings), \
             patch('autonomy.cognition_client.CognitionClient', _FakeCognitionClient), \
             patch('epistemic.layer1_world_model.WorldModel', side_effect=AssertionError('unused')), \
             patch('knowledge.worker.run_worker', worker):
            await serve()
        worker.assert_awaited_once_with(settings)
