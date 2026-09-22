from __future__ import annotations

import json
import time

from epistemic.layer1_world_model import WorldModel
from knowledge.documents import Document
from knowledge.matching import expanded, normalize, rank_candidates
from knowledge.provenance import summarize


class KnowledgeGraph(WorldModel):
    def _ensure_schema(self):
        super()._ensure_schema()
        with self.driver.session() as session:
            for label in ("Document", "Passage", "Claim", "Assessment", "Investigation", "ExtractionReview", "EvidenceSource", "SemanticReview", "Proposition", "Entity"):
                session.run(f"CREATE CONSTRAINT knowledge_{label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT knowledge_collection_name IF NOT EXISTS FOR (n:Collection) REQUIRE n.name IS UNIQUE").consume()

    def passage_complete(self, passage_id: str) -> bool:
        with self.driver.session() as session:
            return session.run("MATCH (p:Passage {id:$id, extraction_status:'complete'}) RETURN p.id", id=passage_id).single() is not None

    def aliases(self):
        with self.driver.session() as session:
            return [(r["a"], r["b"]) for r in session.run("""
                MATCH (a:Concept)-[r:ALIAS_OF]->(b:Concept) WHERE r.status='reviewed'
                RETURN a.name AS a, b.name AS b
                """)]

    def add_alias(self, alias, canonical, note):
        alias, canonical = normalize(alias), normalize(canonical)
        if not alias or not canonical or alias == canonical or not note.strip():
            raise ValueError("Two different concept names and a review note are required")
        with self.driver.session() as session:
            session.run("""
                MERGE (a:Concept {name:$a}) MERGE (b:Concept {name:$b})
                MERGE (a)-[r:ALIAS_OF]->(b)
                SET r.status='reviewed', r.note=$note, r.reviewer='local_user', r.updated_at=$now
                """, a=alias, b=canonical, note=note, now=time.time()).consume()

    def candidates(self, claim: dict, collection: str, *, limit: int = 12) -> list[dict]:
        aliases = self.aliases()
        concepts = sorted(expanded(claim["concepts"], aliases))
        import re
        words = re.findall(r"\w{4,}", normalize(claim["text"]))
        with self.driver.session() as session:
            rows = session.run("""
                MATCH (:Collection {name:$collection})-[:HAS_DOCUMENT]->(:Document)-[:HAS_PASSAGE]->(p:Passage)-[:EXPRESSES]->(c:Claim)
                WHERE c.id <> $id AND (any(name IN $concepts WHERE name IN coalesce(c.search_concepts,c.concepts))
                    OR any(word IN $words WHERE toLower(coalesce(c.search_text,c.text)) CONTAINS word))
                WITH c, min(p.id) AS passage_id
                WITH c, passage_id, size([name IN coalesce(c.search_concepts,c.concepts) WHERE name IN $concepts]) AS shared
                WITH c, passage_id, shared ORDER BY shared DESC, c.id LIMIT 200
                OPTIONAL MATCH (r:SemanticReview)-[e:INTERPRETS]->(c)
                WITH c,passage_id,shared,r,e ORDER BY r.created_at,r.id
                RETURN properties(c) AS claim, passage_id,shared,collect(e.correction_json) AS corrections
                ORDER BY shared DESC,claim.id
                """, collection=collection, id=claim["id"], concepts=concepts, words=words)
            candidates = []
            for row in rows:
                candidate = {**dict(row["claim"]), "passage_id": row["passage_id"]}
                for correction in row["corrections"]:candidate.update(json.loads(correction))
                candidates.append(candidate)
        return rank_candidates(claim, candidates, aliases, limit)

    def save_document(self, document: Document, collection: str, results: list[dict], links: list[dict]):
        def persist(model):
            with model.driver.session() as tx:
                tx.run("""
                    MERGE (d:Document {id:$id})
                    ON CREATE SET d.title=$title, d.sha256=$sha256, d.format=$format, d.created_at=$now, d.sources=[]
                    SET d.sources=CASE WHEN $source IN d.sources THEN d.sources ELSE d.sources+[$source] END
                    MERGE (collection:Collection {name:$collection})
                    MERGE (collection)-[:HAS_DOCUMENT]->(d)
                    WITH d
                    MATCH (old:Document) WHERE old.id <> d.id AND $source IN old.sources
                      AND old.created_at < d.created_at
                    MERGE (d)-[:REVISION_OF]->(old)
                    """, id=document.id, title=document.title, sha256=document.sha256,
                    format=document.format, source=document.source, collection=collection, now=time.time()).consume()
                for result in results:
                    passage = result["passage"]
                    tx.run("""
                        MATCH (d:Document {id:$document_id})
                        MERGE (p:Passage {id:$id})
                        ON CREATE SET p.location=$location, p.text=$text, p.extraction_method=$method
                        SET p.extraction_status=$status
                        MERGE (d)-[:HAS_PASSAGE]->(p)
                        """, document_id=document.id, id=passage.id, location=passage.location,
                        text=passage.text, method=passage.extraction_method, status="review" if result["issues"] else "complete").consume()
                    for claim in result["claims"]:
                        properties = {k: v for k, v in claim.items() if k not in {"quote", "passage_id"}}
                        tx.run("""
                            MATCH (p:Passage {id:$passage_id})
                            MERGE (c:Claim {id:$id})
                            ON CREATE SET c += $properties, c.created_at=$now
                            MERGE (p)-[r:EXPRESSES]->(c)
                            SET r.quote=$quote, r.extraction_confidence=$extraction_confidence,
                                r.attribution_confidence=$attribution_confidence
                            WITH c UNWIND $concepts AS name
                            MERGE (concept:Concept {name:name})
                            MERGE (c)-[:ABOUT]->(concept)
                            """, passage_id=passage.id, id=claim["id"], properties=properties,
                            quote=claim["quote"], extraction_confidence=claim["extraction_confidence"],
                            attribution_confidence=claim["attribution_confidence"], concepts=claim["concepts"], now=time.time()).consume()
                        for name, kind in zip(claim.get("entities", []), claim.get("entity_kinds", [])):
                            from knowledge.documents import stable_id
                            tx.run("MATCH (c:Claim {id:$id}) MERGE (e:Entity {id:$entity_id}) ON CREATE SET e.name=$name,e.kind=$kind MERGE (c)-[:MENTIONS]->(e)",id=claim["id"],entity_id=stable_id("entity",kind,name),name=name,kind=kind).consume()
                        for source_id in claim.get("source_ids", []):
                            tx.run("""
                                MATCH (c:Claim {id:$id})
                                MERGE (s:EvidenceSource {id:$source})
                                MERGE (c)-[:CITES]->(s)
                                """, id=claim["id"], source=source_id).consume()
                for link in links:
                    tx.run("""
                        MATCH (a:Claim {id:$source_id}), (b:Claim {id:$target_id})
                        MERGE (a)-[r:CLAIM_RELATION {id:$id}]->(b)
                        ON CREATE SET r.kind=$relation, r.scope_match=$scope_match, r.reasoning=$reasoning,
                            r.passage_ids=$passage_ids, r.status='proposed', r.origin='model',r.validation_json=$validation_json
                        """, **{**link, "validation_json": json.dumps(link)}).consume()
        self.run_transaction(persist)

    def review_extraction(self, claim_id: str, *, accepted: bool, note: str):
        import uuid
        if not note.strip():
            raise ValueError("An extraction review needs a justification")
        with self.driver.session() as session:
            row = session.run("""
                MATCH (c:Claim {id:$claim_id})
                CREATE (r:ExtractionReview {id:$id, reviewer:'local_user', accepted:$accepted, note:$note, created_at:$now})
                CREATE (r)-[:REVIEWS_EXTRACTION]->(c)
                SET c.review_required=NOT $accepted
                RETURN r.id AS id
                """, claim_id=claim_id, id="review:"+uuid.uuid4().hex, accepted=accepted, note=note, now=time.time()).single()
            if row is None:
                raise ValueError("Unknown claim")
            return row["id"]

    def get_claim(self, claim_id: str) -> dict | None:
        with self.driver.session() as session:
            row = session.run("MATCH (c:Claim {id:$id}) RETURN properties(c) AS claim", id=claim_id).single()
            if row is None:return None
            claim=dict(row["claim"])
            # Reviews are incremental overlays; preserve earlier fields not changed later.
            reviews=session.run("""MATCH (r:SemanticReview)-[e:INTERPRETS]->(:Claim {id:$id})
                RETURN e.correction_json AS correction ORDER BY r.created_at ASC,r.id ASC""",id=claim_id)
            for interpretation in reviews:claim.update(json.loads(interpretation["correction"]))
            return claim

    def explain(self, claim_id: str) -> dict:
        claim = self.get_claim(claim_id)
        if claim is None:
            raise ValueError(f"Unknown claim: {claim_id}")
        with self.driver.session() as session:
            sources = [dict(row) for row in session.run("""
                MATCH (d:Document)-[:HAS_PASSAGE]->(p:Passage)-[r:EXPRESSES]->(:Claim {id:$id})
                RETURN d.title AS document, d.sha256 AS document_hash, d.sources AS paths,
                    p.id AS passage_id, p.location AS location, p.extraction_method AS extraction_method, r.quote AS quote,
                    r.extraction_confidence AS extraction_confidence, r.attribution_confidence AS attribution_confidence
                """, id=claim_id)]
            relationships = [dict(row) for row in session.run("""
                MATCH (c:Claim {id:$id})-[r:CLAIM_RELATION]-(other:Claim)
                RETURN other.id AS other_id, other.text AS text, properties(r) AS relationship,
                    startNode(r).id AS source_id, endNode(r).id AS target_id
                """, id=claim_id)]
            assessments = [dict(row) for row in session.run("""
                MATCH (a:Assessment)-[:TARGETS]->(:Claim {id:$id})
                OPTIONAL MATCH (a)-[:BASED_ON]->(o:Observation)
                RETURN properties(a) AS assessment, collect(properties(o)) AS evidence
                ORDER BY assessment.created_at DESC
                """, id=claim_id)]
            reviews = [dict(row["review"]) for row in session.run("""
                MATCH (r:ExtractionReview)-[:REVIEWS_EXTRACTION]->(:Claim {id:$id})
                RETURN properties(r) AS review ORDER BY r.created_at DESC
                """, id=claim_id)]
            investigations = [dict(row["investigation"]) for row in session.run("""
                MATCH (i:Investigation)-[:INVESTIGATES]->(:Claim {id:$id})
                RETURN properties(i) AS investigation ORDER BY i.created_at DESC
                """, id=claim_id)]
            context_passages=[dict(r) for r in session.run("MATCH (c:Claim {id:$id}),(p:Passage) WHERE p.id IN coalesce(c.context_passage_ids,[]) RETURN p.id AS id,p.location AS location,p.text AS text",id=claim_id)]
            semantic_reviews=[dict(r) for r in session.run("""MATCH (r:SemanticReview)-[e:INTERPRETS]->(c:Claim {id:$id}) RETURN r.id AS id,r.reviewer AS reviewer,e.note AS note,e.correction_json AS correction_json,e.context_json AS context_json,properties(c) AS original_claim ORDER BY r.created_at DESC""",id=claim_id)]
            propositions=[dict(r["proposition"]) for r in session.run("MATCH (:Claim {id:$id})-[:EXPRESSES_PROPOSITION]->(p:Proposition) RETURN properties(p) AS proposition",id=claim_id)]
        return {"claim": claim, "context_passages": context_passages, "semantic_reviews": semantic_reviews, "propositions": propositions,
                "archived_relationships": [r for r in relationships if r["relationship"].get("review_status")=="rejected"], "sources": sources, "relationships": [r for r in relationships if r["relationship"].get("review_status")!="rejected"],
                "assessments": assessments, "investigations": investigations, "extraction_reviews": reviews,
                "provenance": self.provenance(claim_id)}

    def provenance(self, claim_id):
        with self.driver.session() as session:
            rows = session.run("""
                MATCH (c:Claim {id:$id})
                OPTIONAL MATCH (c)-[:CITES]->(:EvidenceSource)<-[:CITES]-(other:Claim)
                RETURN properties(c) AS claim, collect(DISTINCT properties(other)) AS related
                """, id=claim_id).single()
            observations = [dict(r["observation"]) for r in session.run("""
                MATCH (:Assessment)-[:TARGETS]->(:Claim {id:$id})
                WITH DISTINCT $id AS id
                MATCH (a:Assessment)-[:TARGETS]->(:Claim {id:id})
                MATCH (a)-[:BASED_ON]->(o:Observation)
                RETURN DISTINCT properties(o) AS observation
                """, id=claim_id)]
        result = summarize([dict(rows["claim"])] + [dict(c) for c in rows["related"]]) if rows else summarize([])
        result["observation_sources"] = summarize(observations)
        return result

    def queue(self, collection: str, *, limit: int = 20) -> list[dict]:
        with self.driver.session() as session:
            rows = session.run("""
                MATCH (:Collection {name:$collection})-[:HAS_DOCUMENT]->(:Document)-[:HAS_PASSAGE]->(:Passage)-[:EXPRESSES]->(c:Claim)
                WITH DISTINCT c
                OPTIONAL MATCH (c)-[r:CLAIM_RELATION]-(:Claim)
                WITH c, count(CASE WHEN r.kind='POTENTIALLY_CONTRADICTS' AND coalesce(r.review_status,'') <> 'rejected' THEN 1 END) AS conflicts
                OPTIONAL MATCH (a:Assessment)-[:TARGETS]->(c)
                WITH c, conflicts, max(a.created_at) AS last_assessed
                WITH c,conflicts,last_assessed ORDER BY last_assessed IS NOT NULL,conflicts DESC,last_assessed ASC LIMIT $limit
                OPTIONAL MATCH (r:SemanticReview)-[e:INTERPRETS]->(c)
                WITH c,conflicts,last_assessed,r,e ORDER BY r.created_at,r.id
                RETURN c.id AS id, c.text AS text, c.claim_type AS claim_type, c.review_required AS review_required,
                    conflicts, last_assessed, collect(e.correction_json) AS corrections
                ORDER BY last_assessed IS NOT NULL, conflicts DESC, last_assessed ASC LIMIT $limit
                """, collection=collection, limit=limit)
            result=[]
            for row in rows:
                item=dict(row)
                for correction in item.pop("corrections"):
                    fields=json.loads(correction)
                    item.update({k:fields[k] for k in ("text","claim_type") if k in fields})
                result.append(item)
            return result

    def get_plan(self, investigation_id: str) -> dict:
        with self.driver.session() as session:
            row = session.run("MATCH (i:Investigation {id:$id}) RETURN properties(i) AS state", id=investigation_id).single()
        if row is None:
            raise ValueError("Unknown investigation")
        state = dict(row["state"])
        return {"investigation": json.loads(state["plan_json"]), "state": state}

    def investigation_result(self, investigation_id):
        investigation = self.get_plan(investigation_id)["investigation"]
        with self.driver.session() as session:
            row = session.run("""
                MATCH (:Investigation {id:$id})-[:RESULTED_IN]->(a:Assessment)-[:BASED_ON]->(o:Observation)
                RETURN properties(a) AS assessment, properties(o) AS observation
                """, id=investigation_id).single()
        if row is None:
            raise ValueError("Completed investigation has no persisted result")
        return {"investigation": investigation, **dict(row)}

    def approve_plan(self, investigation_id: str, note: str):
        from knowledge.documents import stable_id
        if not note.strip():
            raise ValueError("Approval needs a review note")
        plan = self.get_plan(investigation_id)["investigation"]
        digest = stable_id("plan", json.dumps(plan, sort_keys=True))
        with self.driver.session() as session:
            row = session.run("""
                MATCH (i:Investigation {id:$id}) WHERE i.status='planned'
                SET i.status='approved', i.approval_hash=$hash, i.approval_note=$note, i.approved_at=$now
                RETURN i.id
                """, id=investigation_id, hash=digest, note=note, now=time.time()).single()
            if row is None:
                raise ValueError("Only a planned investigation can be approved")
        return digest

    def begin_execution(self, investigation_id: str, *, retry_uncertain: bool = False):
        # A write lock on i serializes runners even when they use different journals.
        with self.driver.session() as session:
            row = session.run("""
                MATCH (i:Investigation {id:$id}) SET i.execution_lock=coalesce(i.execution_lock,0)+1
                WITH i WHERE i.status='approved' OR ($retry AND i.status='executing')
                SET i.status='executing', i.started_at=$now
                RETURN i.id
                """, id=investigation_id, retry=retry_uncertain, now=time.time()).single()
            if row is None:
                raise RuntimeError("Execution is already started or finished; inspect its journal before retrying")

    def pending_plans(self, collection: str, *, limit: int = 20):
        with self.driver.session() as session:
            return [dict(row) for row in session.run("""
                MATCH (:Collection {name:$collection})-[:HAS_DOCUMENT]->(:Document)-[:HAS_PASSAGE]->(:Passage)-[:EXPRESSES]->(c:Claim)
                MATCH (i:Investigation)-[:INVESTIGATES]->(c)
                WHERE i.status IN ['planned','approved','executing']
                RETURN DISTINCT i.id AS id, i.status AS status, c.id AS claim_id
                ORDER BY status, id LIMIT $limit
                """, collection=collection, limit=limit)]

    def save_plan(self, investigation: dict):
        with self.driver.session() as session:
            row = session.run("""
                MATCH (c:Claim {id:$claim_id})
                MERGE (i:Investigation {id:$id})
                ON CREATE SET i.plan_json=$plan_json, i.created_at=$now, i.status='planned'
                MERGE (i)-[:INVESTIGATES]->(c)
                RETURN i.id
                """, id=investigation["id"], claim_id=investigation["claim_id"],
                plan_json=json.dumps(investigation), now=time.time()).single()
            if row is None:
                raise ValueError("Investigation target no longer exists")

    def save_assessment(self, assessment: dict, observation: dict):
        def persist(model):
            with model.driver.session() as tx:
                row = tx.run("""
                    MATCH (i:Investigation {id:$investigation_id})-[:INVESTIGATES]->(c:Claim {id:$claim_id})
                    MERGE (o:Observation {id:$observation_id})
                    ON CREATE SET o += $observation
                    MERGE (a:Assessment {id:$id})
                    ON CREATE SET a += $assessment
                    MERGE (a)-[:TARGETS]->(c)
                    MERGE (a)-[:BASED_ON]->(o)
                    MERGE (i)-[:RESULTED_IN]->(a)
                    SET i.status='assessed'
                    RETURN a.id
                    """, investigation_id=assessment["investigation_id"], claim_id=assessment["claim_id"],
                    observation_id=observation["id"], id=assessment["id"], observation=observation,
                    assessment=assessment).single()
                if row is None:
                    raise ValueError("Assessment requires an existing investigation and claim")
                for source_id in observation.get("source_ids", []):
                    tx.run("""
                        MATCH (o:Observation {id:$id})
                        MERGE (s:EvidenceSource {id:$source})
                        MERGE (o)-[:CITES]->(s)
                        """, id=observation["id"], source=source_id).consume()
        self.run_transaction(persist)
