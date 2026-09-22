"""Requires a disposable Neo4j instance via NEO4J_TEST_URI."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

from knowledge.graph import KnowledgeGraph
from knowledge.ingest import ingest
from knowledge.investigate import investigate, resume
from knowledge.journal import Journal
from test_knowledge import ScriptedClient, FIXTURES


@unittest.skipUnless(os.environ.get("NEO4J_TEST_URI"), "requires disposable Neo4j")
class KnowledgeGraphTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.graph = KnowledgeGraph(uri=os.environ["NEO4J_TEST_URI"])
        self.collection = "test-knowledge-" + uuid.uuid4().hex
        self.directory = tempfile.TemporaryDirectory()
        self.report = Path(self.directory.name) / "report.json"
        self.journal = Journal(Path(self.directory.name) / "journal")
        self.client = ScriptedClient()
        self.books = []
        for name in ("book_a.md", "book_b.md"):
            path = Path(self.directory.name) / name
            path.write_text(self.collection + "\n" + (FIXTURES / name).read_text())
            self.books.append(path)

    async def asyncTearDown(self):
        # Remove only this test collection's uniquely prefixed documents and their records.
        with self.graph.driver.session() as session:
            row = session.run("""
                MATCH (k:Collection {name:$name})
                OPTIONAL MATCH (k)-[:HAS_DOCUMENT]->(d:Document)-[:HAS_PASSAGE]->(p:Passage)
                OPTIONAL MATCH (p)-[:EXPRESSES]->(c:Claim)
                OPTIONAL MATCH (i:Investigation)-[:INVESTIGATES]->(c)
                OPTIONAL MATCH (a:Assessment)-[:TARGETS]->(c)
                OPTIONAL MATCH (a)-[:BASED_ON]->(o:Observation)
                OPTIONAL MATCH (review:ExtractionReview)-[:REVIEWS_EXTRACTION]->(c)
                OPTIONAL MATCH (semantic:SemanticReview)-[:INTERPRETS]->(c)
                OPTIONAL MATCH (c)-[:EXPRESSES_PROPOSITION]->(proposition:Proposition)
                RETURN collect(DISTINCT elementId(k))+collect(DISTINCT elementId(d))+
                    collect(DISTINCT elementId(p))+collect(DISTINCT elementId(c))+
                    collect(DISTINCT elementId(i))+collect(DISTINCT elementId(a))+collect(DISTINCT elementId(o))+collect(DISTINCT elementId(review))+collect(DISTINCT elementId(semantic))+collect(DISTINCT elementId(proposition)) AS ids
                """, name=self.collection).single()
            if row:
                session.run("MATCH (n) WHERE elementId(n) IN $ids DETACH DELETE n", ids=row["ids"]).consume()
        self.graph.close()
        self.directory.cleanup()

    async def test_ingestion_conflicts_assessment_and_source_explanation(self):
        report = await ingest(self.books, collection=self.collection,
            client=self.client, graph=self.graph, report_path=self.report)
        self.assertEqual(report["issues"], [])
        queue = self.graph.queue(self.collection)
        self.assertEqual(len(queue), 2)
        self.assertTrue(all(item["conflicts"] == 1 for item in queue))
        target = next(c for c in queue if "always" in c["text"])
        explanation = self.graph.explain(target["id"])
        self.assertEqual(explanation["sources"][0]["quote"], target["text"])
        self.assertEqual(explanation["relationships"][0]["relationship"]["status"], "proposed")
        self.assertEqual(explanation["assessments"], [])
        self.graph.review_extraction(target["id"], accepted=False, note="Check attribution")
        with self.assertRaisesRegex(ValueError, "Review"):
            await investigate(self.graph, self.client, target["id"], execute=True, journal=self.journal)
        self.graph.review_extraction(target["id"], accepted=True, note="Confirmed against original passage")
        self.assertEqual(len(self.graph.explain(target["id"])["extraction_reviews"]), 2)
        result = await investigate(self.graph, self.client, target["id"], execute=True, journal=self.journal)
        self.assertEqual(result["assessment"]["position"], "rejected")
        result2 = await investigate(self.graph, self.client, target["id"], execute=True, journal=self.journal)
        explanation = self.graph.explain(target["id"])
        self.assertEqual(len(explanation["assessments"]), 2)
        self.assertEqual(len(explanation["investigations"]), 2)
        self.assertEqual(explanation["claim"]["text"], target["text"])
        self.assertTrue(explanation["assessments"][0]["evidence"])
        # Imported claims were never promoted to world-model theories.
        self.assertEqual(self.graph.theory_count(), 0)
        self.client.calls.clear()
        await ingest(self.books, collection=self.collection,
            client=self.client, graph=self.graph, report_path=self.report)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(len(self.graph.queue(self.collection)), 2)

    async def test_changed_file_preserves_previous_version(self):
        path = Path(self.directory.name) / "version.md"
        path.write_text(self.collection + "\n" + (FIXTURES / "book_a.md").read_text())
        first = await ingest([path], collection=self.collection, client=self.client, graph=self.graph, report_path=self.report)
        path.write_text(self.collection + "\n" + (FIXTURES / "book_b.md").read_text())
        second = await ingest([path], collection=self.collection, client=self.client, graph=self.graph, report_path=self.report)
        self.assertNotEqual(first["documents"][0]["document_id"], second["documents"][0]["document_id"])
        with self.graph.driver.session() as session:
            row = session.run("MATCH (:Document)-[r:REVISION_OF]->(:Document) RETURN count(r) AS count").single()
        self.assertEqual(row["count"], 1)

    async def test_document_transaction_rolls_back_when_a_link_write_fails(self):
        from knowledge.documents import read_document
        from knowledge.extraction import extract_passage
        document = read_document(self.books[0])
        passage = document.passages[0]
        claims, issues = await extract_passage(self.client, passage)
        bad_link = {"source_id": claims[0]["id"], "target_id": claims[0]["id"], "id": "bad-link",
            "relation": {"not": "a Neo4j property value"}, "scope_match": "unknown", "reasoning": "injected failure", "passage_ids": [passage.id]}
        with self.assertRaises(Exception):
            self.graph.save_document(document, self.collection,
                [{"passage": passage, "claims": claims, "issues": issues}], [bad_link])
        self.assertIsNone(self.graph.get_claim(claims[0]["id"]))
        with self.graph.driver.session() as session:
            row = session.run("MATCH (d:Document {id:$id}) RETURN count(d) AS count", id=document.id).single()
        self.assertEqual(row["count"], 0)

    async def test_alias_and_shared_citation_queries(self):
        from knowledge.documents import read_document
        from knowledge.extraction import validate_claim
        from knowledge.provenance import canonical_source
        import uuid
        marker = uuid.uuid4().hex
        aliases = ('consumption ' + marker, 'energy use ' + marker)
        self.graph.add_alias(*aliases, note='Reviewed as equivalent concepts in this collection')
        claims = []
        for index, concept in enumerate(aliases):
            path = Path(self.directory.name) / f'citation-{index}.md'
            quote = f'Source {index}: https://doi.org/10.1234/{marker}'
            path.write_text(quote)
            doc = read_document(path)
            passage = doc.passages[0]
            claim = validate_claim({'text': f'Unique proposition {index}', 'quote': quote, 'speaker': 'narrator',
                'stance': 'endorses', 'scope': 'test', 'claim_type': 'empirical', 'concepts': [concept],
                'extraction_confidence': .9, 'attribution_confidence': .9,
                'source_refs': [{'identifier': f'https://doi.org/10.1234/{marker}', 'quote': quote}]}, passage)
            self.graph.save_document(doc, self.collection, [{'passage': passage, 'claims': [claim], 'issues': []}], [])
            claims.append(claim)
        self.assertEqual(self.graph.candidates(claims[1], self.collection)[0]['id'], claims[0]['id'])
        groups = self.graph.explain(claims[0]['id'])['provenance']['source_groups']
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['claim_ids']), 2)
        with self.graph.driver.session() as session:
            session.run('MATCH (a:Concept)-[r:ALIAS_OF]->(b:Concept) WHERE a.name=$a AND b.name=$b DELETE r',
                        a=aliases[0], b=aliases[1]).consume()

    async def test_saved_plan_completion_is_idempotent_across_journals(self):
        await ingest(self.books, collection=self.collection, client=self.client, graph=self.graph, report_path=self.report)
        target = self.graph.queue(self.collection)[0]
        planned = await investigate(self.graph, self.client, target['id'])
        plan_id = planned['investigation']['id']
        self.assertEqual(self.graph.pending_plans(self.collection)[0]['id'], plan_id)
        self.graph.approve_plan(plan_id, 'Reviewed procedure and scope')
        first = await resume(self.graph, self.client, plan_id, journal=self.journal)
        self.client.calls.clear()
        second = await resume(self.graph, self.client, plan_id, journal=Journal(Path(self.directory.name) / 'other-journal'))
        self.assertEqual(first, second)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(len(self.graph.explain(target['id'])['assessments']), 1)
        self.assertEqual(self.graph.pending_plans(self.collection), [])

    async def test_semantic_review_preserves_originals_and_groups_propositions(self):
        from knowledge.review_semantics import apply_review
        import json
        other=Path(self.directory.name)/'paraphrase.md'
        other.write_text('Another copy '+self.collection+'\n'+self.books[0].read_text())
        await ingest([self.books[0],other],collection=self.collection,client=self.client,graph=self.graph,report_path=self.report)
        a,b=self.graph.queue(self.collection)
        original=self.graph.get_claim(a['id'])
        link=self.graph.explain(a['id'])['relationships'][0]['relationship']['id']
        payload={'reviewer':'assistant:test','claims':[{'claim_id':a['id'],'correction':{'concepts':['energy use']},'note':'Reviewed terminology'}],
                 'relationships':[{'id':link,'status':'rejected','note':'Identical statements do not contradict'}],
                 'propositions':[{'text':a['text'],'scope':'total electricity per complete cycle','claim_ids':[a['id'],b['id']],'note':'Same assertion in two source documents'}]}
        review=apply_review(self.graph,payload)
        self.assertEqual(apply_review(self.graph,payload),review)
        explained=self.graph.explain(a['id'])
        self.assertEqual(explained['claim']['concepts'],['energy use'])
        self.assertEqual(explained['semantic_reviews'][0]['original_claim']['concepts'],original['concepts'])
        self.assertEqual(explained['relationships'],[])
        self.assertEqual(len(explained['archived_relationships']),1)
        self.assertEqual(len(explained['propositions']),1)
        self.assertEqual(explained['propositions'],self.graph.explain(b['id'])['propositions'])
        # A missing target must roll back the entire review.
        bad={**payload,'claims':[{'claim_id':'missing','correction':{'text':'bad'},'note':'invalid'}]}
        with self.assertRaises(ValueError):apply_review(self.graph,bad)
        self.assertEqual(self.graph.get_claim(a['id'])['text'],original['text'])
