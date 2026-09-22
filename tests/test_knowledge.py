from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knowledge.documents import read_document
from knowledge.extraction import validate_claim
from knowledge.ingest import ingest
from knowledge.investigate import investigate
from knowledge.journal import Journal
from knowledge.documents import stable_id

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


class ScriptedClient:
    """Deterministic model responses; never calls a model or external tools."""
    def __init__(self):
        self.calls = []
        self.execution_ok = True
        self.graph = None

    async def generate_model(self, prompt, cls, **kwargs):
        self.calls.append((cls.__name__, kwargs))
        if cls.__name__ == "Extraction":
            data = json.loads(prompt.split("PASSAGE:\n", 1)[1])
            text = data["text"]
            claim = "Machine X uses more electricity in Eco mode during short cycles." if "second account" in text else "Machine X always uses less electricity in Eco mode."
            return cls([{"text": claim, "quote": claim, "speaker": "narrator", "stance": "endorses",
                         "scope": "total electricity per complete cycle", "claim_type": "empirical",
                         "extraction_confidence": .95, "attribution_confidence": .9, "concepts": ["Machine X", "electricity"]}])
        if cls.__name__ == "Comparison":
            data = json.loads(prompt[prompt.index('{"focal"'):])
            return cls([{"target_id": data["candidates"][0]["id"], "relation": "POTENTIALLY_CONTRADICTS",
                         "scope_match": "unknown", "reasoning": "Check cycle length and machine identity"}])
        if cls.__name__ == "TestPlan":
            return cls("Does Eco always save electricity?", "Machine X short cycles",
                "Eco consumes less than normal mode", "An Eco cycle consumes more",
                "Read controlled measurements of equal short cycles")
        if cls.__name__ == "_ExplorationResponse":
            if self.graph is not None:
                assert self.graph.plans, "Plan must be saved before action"
            return cls(action_taken="Read controlled short-cycle measurements", sensor_data="Eco: 12 Wh; normal: 10 Wh",
                ok=self.execution_ok, evidence=[{"request_id": "measurement-1", "status": "APPROVED" if self.execution_ok else "ERROR",
                "result": {"eco_wh": 12, "normal_wh": 10}}])
        if cls.__name__ == "PredictionError":
            return cls(match=False, reasoning="A measured counterexample contradicts the universal claim", confidence_in_data=.9,
                suggested_confidence_adjustment=-.2, verdict="contradicts")
        raise AssertionError(cls)


class MemoryGraph:
    def __init__(self):
        self.claims, self.passages, self.documents, self.plans = {}, {}, {}, {}
        self.links, self.assessments = [], []
        self.states = {}

    def passage_complete(self, passage_id):
        return passage_id in self.passages and not self.passages[passage_id]["issues"]

    def candidates(self, claim, collection):
        return [c for c in self.claims.values() if c["id"] != claim["id"] and set(c["concepts"]) & set(claim["concepts"])]

    def save_document(self, document, collection, results, links):
        self.documents[document.id] = document
        for result in results:
            self.passages[result["passage"].id] = result
            for claim in result["claims"]:
                self.claims[claim["id"]] = claim
        self.links.extend(links)

    def explain(self, claim_id):
        c = self.claims[claim_id]
        return {"claim": c, "sources": [{"quote": c["quote"], "location": c["location"]}],
                "relationships": self.links, "assessments": self.assessments}

    def save_plan(self, investigation):
        self.plans[investigation["id"]] = copy.deepcopy(investigation)
        self.states[investigation["id"]] = {"status": "planned"}

    def get_plan(self, investigation_id):
        return {"investigation": copy.deepcopy(self.plans[investigation_id]), "state": dict(self.states[investigation_id])}

    def approve_plan(self, investigation_id, note):
        state = self.states[investigation_id]
        if state["status"] != "planned" or not note:
            raise ValueError("Only planned investigations can be approved with a note")
        state.update(status="approved", approval_hash=stable_id("plan", json.dumps(self.plans[investigation_id], sort_keys=True)))

    def begin_execution(self, investigation_id, retry_uncertain=False):
        state = self.states[investigation_id]
        if state["status"] != "approved" and not (retry_uncertain and state["status"] == "executing"):
            raise RuntimeError("Execution already started")
        state["status"] = "executing"

    def pending_plans(self, collection, limit=20):
        return [{"id": i, "status": s["status"], "claim_id": self.plans[i]["claim_id"]}
                for i, s in self.states.items() if s["status"] != "assessed"][:limit]

    def queue(self, collection, limit=20):
        return [{**c, "conflicts": 0, "last_assessed": max((a["assessment"]["created_at"] for a in self.assessments
                if a["assessment"]["claim_id"] == c["id"]), default=None)} for c in self.claims.values()][:limit]

    def save_assessment(self, assessment, observation):
        assert assessment["investigation_id"] in self.plans
        if not any(a["assessment"]["id"] == assessment["id"] for a in self.assessments):
            self.assessments.append({"assessment": assessment, "observation": observation})
        self.states[assessment["investigation_id"]]["status"] = "assessed"


class DocumentTests(unittest.TestCase):
    def test_stable_locations_and_changed_document_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.md"
            path.write_text("A claim. " * 20)
            doc = read_document(path, chunk_size=50, overlap=10)
            self.assertEqual(doc.id, read_document(path, chunk_size=50, overlap=10).id)
            self.assertEqual(doc.passages[0].location, "text, characters 0:50")
            path.write_text("A revised claim.")
            self.assertNotEqual(doc.id, read_document(path).id)

    def test_quote_must_be_in_source_and_attribution_is_separate(self):
        passage = read_document(FIXTURES / "book_b.md").passages[0]
        candidate = {"text": "Eco mode always saves electricity.", "quote": "Eco mode always saves electricity.",
            "speaker": "opponent", "stance": "quotes", "scope": "unspecified", "claim_type": "empirical",
            "extraction_confidence": .9, "attribution_confidence": .8, "concepts": ["Eco"]}
        first = validate_claim(candidate, passage)
        second = validate_claim({**candidate, "speaker": "narrator", "stance": "endorses"}, passage)
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotIn("confidence_score", first)
        with self.assertRaisesRegex(ValueError, "substring"):
            validate_claim({**candidate, "quote": "Invented passage"}, passage)
        with self.assertRaises(ValueError):
            validate_claim({**candidate, "extraction_confidence": float("nan")}, passage)

    @unittest.skipUnless(importlib.util.find_spec("pypdf"), "requires pypdf")
    def test_pdf_pages_keep_locations_and_scans_require_ocr(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=612, height=792)
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
            content = DecodedStreamObject()
            content.set_data(b"BT /F1 12 Tf 72 720 Td (A testable claim.) Tj ET")
            page[NameObject("/Contents")] = writer._add_object(content)
            with path.open("wb") as stream:
                writer.write(stream)
            document = read_document(path)
            self.assertIn("page 1", document.passages[0].location)
            self.assertIn("A testable claim.", document.passages[0].text)
            writer.add_blank_page(width=612, height=792)
            with path.open("wb") as stream:
                writer.write(stream)
            with self.assertRaisesRegex(ValueError, "OCR"):
                read_document(path)

    def test_export_does_not_execute_source_markup(self):
        from knowledge.export import render_explorer
        html = render_explorer([], "</script><script>alert(1)</script>")
        self.assertNotIn("</script><script>alert(1)</script>", html)
        self.assertIn("\\u003c/script>", html)
        self.assertNotIn("__KNOWLEDGE_DATA__", html)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.report = Path(self.directory.name) / "report.json"
        self.journal = Journal(Path(self.directory.name) / "journal")
        self.graph = MemoryGraph()
        self.client = ScriptedClient()
        self.client.graph = self.graph

    async def import_books(self):
        return await ingest([FIXTURES / "book_a.md", FIXTURES / "book_b.md"], collection="books",
            client=self.client, graph=self.graph, report_path=self.report)

    async def test_conflicting_books_assessment_preserves_sources_and_history(self):
        report = await self.import_books()
        self.assertFalse(report["issues"])
        self.assertEqual(len(self.graph.claims), 2)
        self.assertEqual(self.graph.links[0]["relation"], "POTENTIALLY_CONTRADICTS")
        claim = next(c for c in self.graph.claims.values() if "always" in c["text"])
        original = copy.deepcopy(claim)
        result = await investigate(self.graph, self.client, claim["id"], execute=True, journal=self.journal)
        self.assertEqual(result["assessment"]["position"], "rejected")
        self.assertEqual(self.graph.claims[claim["id"]], original)
        self.assertEqual(self.graph.explain(claim["id"])["sources"][0]["quote"], claim["quote"])
        self.client.execution_ok = False
        result = await investigate(self.graph, self.client, claim["id"], execute=True, journal=self.journal)
        self.assertEqual(result["assessment"]["position"], "unresolved")
        self.assertEqual(len(self.graph.assessments), 2)
        self.assertEqual(self.graph.assessments[0]["assessment"]["position"], "rejected")

    async def test_reimport_does_not_reextract_or_duplicate(self):
        await self.import_books()
        self.client.calls.clear()
        report = await self.import_books()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(len(self.graph.claims), 2)
        self.assertTrue(all(d["skipped_passages"] == 1 for d in report["documents"]))

    async def test_plan_only_does_not_execute_and_value_claims_are_preserved(self):
        await self.import_books()
        claim = next(iter(self.graph.claims.values()))
        self.client.calls.clear()
        result = await investigate(self.graph, self.client, claim["id"])
        self.assertEqual(result["status"], "planned")
        self.assertEqual([name for name, _ in self.client.calls], ["TestPlan"])
        claim["claim_type"] = "value"
        with self.assertRaisesRegex(ValueError, "empirical"):
            await investigate(self.graph, self.client, claim["id"], execute=True, journal=self.journal)
        self.assertIn(claim["id"], self.graph.claims)

    async def test_bad_extraction_is_a_review_item_not_a_claim(self):
        class BadClient:
            async def generate_model(self, prompt, cls, **kwargs):
                return cls([{"text": "fabricated", "quote": "invented"}])
        report = await ingest([FIXTURES / "book_a.md"], collection="books", client=BadClient(),
            graph=self.graph, report_path=self.report)
        self.assertEqual(self.graph.claims, {})
        self.assertTrue(report["issues"])
        self.assertEqual(report["documents"][0]["status"], "review")
        self.assertTrue(json.loads(self.report.read_text())["issues"])

    async def test_report_only_needs_no_graph(self):
        report = await ingest([FIXTURES / "book_a.md"], collection="books", client=self.client,
            report_path=self.report)
        self.assertFalse(report["persisted"])
        self.assertEqual(len(report["documents"][0]["claims"]), 1)


if __name__ == "__main__":
    unittest.main()
