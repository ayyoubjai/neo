from __future__ import annotations

import os
import copy
import json
from contextlib import nullcontext
import time
import uuid
from typing import Iterable, Optional

from epistemic.types import CURIOSITY_PLACEHOLDER_ID, Observation, PredictionError, Theory
from power_process.types import ExecutionOutcome, Goal, GoalExecution


DEFAULT_AXIOMS = (
    "Autonomous agents that verify assumptions before acting make fewer catastrophic errors.",
    "Information retrieved from the internet can be outdated, biased, or fabricated; cross-referencing multiple sources improves reliability.",
    "Local filesystem state changes persisted by one action are observable by subsequent actions in the same session.",
    "Memory retrieval quality degrades when stored facts are redundant or contradictory; deduplication improves recall precision.",
    "Tool failures carry diagnostic information that, when analyzed, often reveal the correct next action.",
)


class WorldModel:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        *,
        ensure_schema: bool = True,
    ) -> None:
        from neo4j import GraphDatabase

        resolved_uri = uri or os.getenv("EPISTEMIC_NEO4J_URI", "bolt://localhost:7687")
        resolved_user = user or os.getenv("EPISTEMIC_NEO4J_USER", "neo4j")
        resolved_password = password or os.getenv("EPISTEMIC_NEO4J_PASSWORD", "epistemic123")
        self.driver = GraphDatabase.driver(resolved_uri, auth=(resolved_user, resolved_password),
            connection_timeout=2.0, connection_acquisition_timeout=3.0, max_transaction_retry_time=3.0)
        if ensure_schema:
            self._ensure_schema()

    def run_transaction(self, operation):
        """Run a whole belief/goal transition atomically, including its provenance."""
        with self.driver.session() as session:
            def apply(tx):
                bound = copy.copy(self)
                class TransactionDriver:
                    def session(self):
                        return nullcontext(tx)
                bound.driver = TransactionDriver()
                return operation(bound)
            return session.execute_write(apply)

    def close(self) -> None:
        self.driver.close()

    def _ensure_schema(self) -> None:
        queries = (
            "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT theory_id_unique IF NOT EXISTS FOR (t:Theory) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT theory_text_unique IF NOT EXISTS FOR (t:Theory) REQUIRE t.text IS UNIQUE",
            "CREATE CONSTRAINT goal_id_unique IF NOT EXISTS FOR (g:Goal) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT observation_id_unique IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE",
        )
        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _theory_from_node(node) -> Theory:
        return Theory(
            id=node["id"],
            text=node["text"],
            confidence_score=float(node.get("confidence_score", 0.0)),
            predictive_success_rate=float(node.get("predictive_success_rate", 0.0)),
            attempts=int(node.get("attempts", 0)),
            status=str(node.get("status", "active")),
            kind=str(node.get("kind", "hypothesis")),
            updated_at=float(node.get("updated_at", 0)),
        )

    @staticmethod
    def _goal_from_node(node) -> Goal:
        return Goal(
            id=node["id"],
            text=node["text"],
            reasoning=node["reasoning"],
            status=node["status"],
            attempts=int(node.get("attempts", 0)),
            success_criteria=str(node.get("success_criteria", node["text"])),
        )

    def has_observation(self, observation_id: str) -> bool:
        with self.driver.session() as session:
            return session.run("MATCH (o:Observation {id: $id}) RETURN o.id AS id", id=observation_id).single() is not None

    def theory_count(self) -> int:
        query = "MATCH (t:Theory) RETURN count(t) AS count"
        with self.driver.session() as session:
            return int(session.run(query).single()["count"])

    def goal_count(self) -> int:
        query = "MATCH (g:Goal) RETURN count(g) AS count"
        with self.driver.session() as session:
            return int(session.run(query).single()["count"])

    def bootstrap_axioms(self, axioms: Optional[Iterable[str]] = None) -> int:
        if self.theory_count() > 0:
            return 0
        seeded = 0
        for axiom in axioms or DEFAULT_AXIOMS:
            theory = self.add_theory(axiom)
            if theory:
                seeded += 1
        if seeded:
            print(f"[WorldModel] Bootstrapped {seeded} initial axioms into Neo4j.")
        return seeded

    def get_theory(self, theory_id: str) -> Optional[Theory]:
        query = "MATCH (t:Theory {id: $id}) RETURN t LIMIT 1"
        with self.driver.session() as session:
            record = session.run(query, id=theory_id).single()
            if record:
                return self._theory_from_node(record["t"])
        return None

    def get_untested_theory(self, *, max_attempts: Optional[int] = None) -> Optional[Theory]:
        where_clause = "WHERE coalesce(t.status, 'active') = 'active' AND coalesce(t.next_test_at, 0) <= $now"
        if max_attempts is not None:
            where_clause += " AND t.attempts < $max_attempts"
        query = f"""
        MATCH (t:Theory)
        {where_clause}
        RETURN t
        ORDER BY t.attempts ASC, t.confidence_score ASC, t.predictive_success_rate ASC
        LIMIT 1
        """
        with self.driver.session() as session:
            kwargs = {"max_attempts": int(max_attempts)} if max_attempts is not None else {}
            kwargs["now"] = self._now()
            record = session.run(query, **kwargs).single()
            if record:
                return self._theory_from_node(record["t"])
        return None

    def get_revalidation_theory(self, *, stale_after_days: float = 30) -> Optional[Theory]:
        # Retired beliefs remain in the graph, but do not compete with active beliefs.
        query = """
        MATCH (t:Theory)
        WHERE coalesce(t.status, 'active') = 'active'
          AND t.attempts > 0 AND coalesce(t.updated_at, 0) < $cutoff
          AND coalesce(t.next_test_at, 0) <= $now
        RETURN t ORDER BY t.updated_at ASC LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(query, cutoff=self._now() - stale_after_days * 86400, now=self._now()).single()
            return self._theory_from_node(record["t"]) if record else None

    def update_theory(self, theory_id: str, new_confidence: float, success: bool) -> None:
        query = """
        MATCH (t:Theory {id: $id})
        WITH t, coalesce(t.attempts, 0) AS previous_attempts,
             coalesce(t.predictive_success_rate, 0.0) AS previous_rate
        SET t.confidence_score = $new_confidence,
            t.attempts = previous_attempts + 1,
            t.predictive_success_rate = (previous_rate * previous_attempts + $success_value) / (previous_attempts + 1),
            t.status = CASE WHEN $new_confidence <= 0 THEN 'retired' ELSE 'active' END,
            t.updated_at = $updated_at
        """
        with self.driver.session() as session:
            session.run(query, id=theory_id, new_confidence=float(max(0.0, min(1.0, new_confidence))),
                        success_value=1.0 if success else 0.0, updated_at=self._now())

    def delete_theory(self, theory_id: str) -> None:
        query = "MATCH (t:Theory {id: $id}) DETACH DELETE t"
        with self.driver.session() as session:
            session.run(query, id=theory_id)

    def get_all_theories(self) -> list[Theory]:
        query = """
        MATCH (t:Theory)
        RETURN t
        ORDER BY t.confidence_score DESC, t.attempts ASC
        """
        theories: list[Theory] = []
        with self.driver.session() as session:
            for record in session.run(query):
                theories.append(self._theory_from_node(record["t"]))
        return theories

    def get_grounded_theories(self, *, limit: int = 5, min_confidence: float = 0.3) -> list[Theory]:
        # Staleness reduces eligibility without silently rewriting historical confidence
        # or increasing confidence in weak beliefs merely because time passed.
        query = """
        MATCH (t:Theory)
        WHERE coalesce(t.status, 'active') = 'active' AND t.attempts > 0
        WITH t, CASE WHEN $now - coalesce(t.updated_at, 0) <= 2592000 THEN 1.0
                     ELSE 1.0 / (1.0 + ($now - coalesce(t.updated_at, 0) - 2592000) / 2592000) END AS freshness
        WHERE t.confidence_score * freshness >= $min_confidence
        RETURN t ORDER BY t.predictive_success_rate DESC, t.confidence_score DESC LIMIT $limit
        """
        with self.driver.session() as session:
            return [self._theory_from_node(record["t"]) for record in session.run(
                query, now=self._now(), limit=int(limit), min_confidence=float(min_confidence))]

    def enrich_theory(self, theory_id: str, *, kind: str, concepts: Iterable[str],
                      parent_id: Optional[str], relation: str, observation_id: str) -> None:
        kind = kind if kind in {"idea", "hypothesis", "belief"} else "hypothesis"
        relation = relation if relation in {"REFINES", "EXTENDS", "CONTRADICTS", "RELATED_TO"} else "RELATED_TO"
        names = list(dict.fromkeys(" ".join(c.lower().split())[:120] for c in concepts if isinstance(c, str) and c.strip()))[:8]
        with self.driver.session() as session:
            session.run("MATCH (t:Theory {id: $id}) SET t.kind = coalesce(t.kind, $kind)", id=theory_id, kind=kind)
            session.run("""
                MATCH (t:Theory {id: $id})
                UNWIND $names AS name
                MERGE (c:Concept {name: name})
                MERGE (t)-[r:ABOUT {source_observation_id: $observation_id}]->(c)
                SET r.status = 'proposed', r.origin = 'model'
                """, id=theory_id, names=names, observation_id=observation_id)
            if parent_id and parent_id != theory_id:
                # Only an allowlisted relationship name is interpolated, never model text.
                session.run(f"""
                    MATCH (child:Theory {{id: $id}}), (parent:Theory {{id: $parent_id}})
                    MERGE (child)-[r:{relation} {{source_observation_id: $observation_id}}]->(parent)
                    SET r.status = 'proposed', r.origin = 'model'
                    """, id=theory_id, parent_id=parent_id, observation_id=observation_id)

    def search_beliefs(self, terms: list[str], *, limit: int = 5) -> list[dict]:
        query = """
        MATCH (t:Theory)
        WITH t, size([term IN $terms WHERE toLower(t.text) CONTAINS term]) AS relevance
        WHERE relevance > 0
        WITH t, relevance ORDER BY relevance DESC, t.updated_at DESC LIMIT $limit
        OPTIONAL MATCH (t)-[:TESTED_BY]->(o:Observation)
        RETURN t.id AS id, t.text AS text, coalesce(t.kind, 'hypothesis') AS kind,
               coalesce(t.status, 'active') AS status, t.confidence_score AS confidence,
               t.updated_at AS last_tested_at, collect(o.id)[0..3] AS observation_ids
        """
        with self.driver.session() as session:
            records = [dict(record) for record in session.run(query, terms=terms, limit=limit)]
        for record in records:
            record["relationships"] = self.get_theory_context(record["id"])
        return records

    def get_theory_context(self, theory_id: str) -> list[dict]:
        query = """
        MATCH (t:Theory {id: $id})-[r]-(other:Theory)
        WHERE type(r) IN ['REFINES', 'EXTENDS', 'CONTRADICTS', 'RELATED_TO']
        RETURN other.id AS id, other.text AS text, other.status AS status,
               type(r) AS relation, r.status AS relation_status,
               startNode(r).id AS source_id, endNode(r).id AS target_id
        LIMIT 6
        UNION
        MATCH (t:Theory {id: $id})-[:ABOUT]->(c:Concept)<-[:ABOUT]-(other:Theory)
        WHERE other.id <> t.id
        RETURN DISTINCT other.id AS id, other.text AS text, other.status AS status,
               'SHARES_CONCEPT' AS relation, 'proposed' AS relation_status,
               t.id AS source_id, other.id AS target_id
        LIMIT 6
        """
        with self.driver.session() as session:
            return [dict(record) for record in session.run(query, id=theory_id)]

    def get_goal_execution_outcome(self, observation_id: str) -> Optional[ExecutionOutcome]:
        with self.driver.session() as session:
            record = session.run(
                "MATCH (o:Observation {id: $id, kind: 'goal_execution'}) RETURN o",
                id=observation_id,
            ).single()
            return ExecutionOutcome.from_dict(dict(record["o"])) if record else None

    def get_goal_history(self, goal_id: str) -> list[dict]:
        query = """
        MATCH (:Goal {id: $id})-[:ATTEMPTED_BY]->(o:Observation)
        RETURN o.action_taken AS action_taken, o.sensor_data AS sensor_data,
               o.ok AS ok, o.reasoning AS reasoning, o.completion_score AS completion_score
        ORDER BY o.created_at DESC LIMIT 5
        """
        with self.driver.session() as session:
            return [dict(record) for record in session.run(query, id=goal_id)]

    def add_goal(
        self,
        text: str,
        reasoning: str,
        *,
        deduplicate_pending: bool = True,
        success_criteria: str = "",
        grounding_theory_ids: Optional[Iterable[str]] = None,
    ) -> Goal:
        normalized_text = " ".join(str(text).split()).strip()
        normalized_reasoning = " ".join(str(reasoning).split()).strip()
        if not normalized_text:
            raise ValueError("Goal text cannot be empty.")
        goal: Goal
        if deduplicate_pending and self.goal_count() > 0:
            query = "MATCH (g:Goal {text: $text, status: 'pending'}) RETURN g LIMIT 1"
            with self.driver.session() as session:
                existing = session.run(query, text=normalized_text).single()
                if existing:
                    goal = self._goal_from_node(existing["g"])
                    if grounding_theory_ids:
                        self.link_goal_to_theories(goal.id, grounding_theory_ids)
                    return goal
        goal = Goal(
            id=str(uuid.uuid4()),
            text=normalized_text,
            reasoning=normalized_reasoning or "No explicit reasoning provided.",
            success_criteria=success_criteria or normalized_text,
        )
        timestamp = self._now()
        query = """
        CREATE (
            g:Goal {
                id: $id,
                text: $text,
                reasoning: $reasoning,
                success_criteria: $success_criteria,
                status: $status,
                attempts: $attempts,
                created_at: $created_at,
                updated_at: $updated_at
            }
        )
        """
        with self.driver.session() as session:
            session.run(
                query,
                id=goal.id,
                text=goal.text,
                reasoning=goal.reasoning,
                success_criteria=goal.success_criteria,
                status=goal.status,
                attempts=goal.attempts,
                created_at=timestamp,
                updated_at=timestamp,
            )
        if grounding_theory_ids:
            self.link_goal_to_theories(goal.id, grounding_theory_ids)
        return goal

    def get_pending_goal(self) -> Optional[Goal]:
        if self.goal_count() == 0:
            return None
        query = """
        MATCH (g:Goal)
        WHERE g.status = 'pending'
        RETURN g
        ORDER BY g.attempts ASC
        LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(query).single()
            if record:
                return self._goal_from_node(record["g"])
        return None

    def update_goal_status(self, goal_id: str, new_status: str) -> None:
        query = """
        MATCH (g:Goal {id: $id})
        SET g.status = $status,
            g.attempts = g.attempts + 1,
            g.updated_at = $updated_at
        """
        with self.driver.session() as session:
            session.run(query, id=goal_id, status=new_status, updated_at=self._now())

    def link_goal_to_theories(self, goal_id: str, theory_ids: Iterable[str]) -> None:
        unique_ids = [str(theory_id).strip() for theory_id in theory_ids if str(theory_id).strip()]
        if not unique_ids:
            return
        query = """
        MATCH (g:Goal {id: $goal_id})
        UNWIND $theory_ids AS theory_id
        MATCH (t:Theory {id: theory_id})
        MERGE (g)-[:MOTIVATED_BY]->(t)
        """
        with self.driver.session() as session:
            session.run(query, goal_id=goal_id, theory_ids=unique_ids)

    def _link_observation_to_theory(self, observation_id: str, theory_id: str) -> None:
        query = """
        MATCH (o:Observation {id: $observation_id})
        MATCH (t:Theory {id: $theory_id})
        MERGE (o)-[:LEARNED_THEORY]->(t)
        """
        with self.driver.session() as session:
            session.run(query, observation_id=observation_id, theory_id=theory_id)

    def add_theory(
        self,
        text: str,
        *,
        deduplicate: bool = True,
        source_observation_id: Optional[str] = None,
    ) -> Theory:
        normalized = " ".join(str(text).split()).strip()
        if not normalized:
            raise ValueError("Theory text cannot be empty.")
        theory: Theory
        if deduplicate:
            query = "MATCH (t:Theory {text: $text}) RETURN t LIMIT 1"
            with self.driver.session() as session:
                existing = session.run(query, text=normalized).single()
                if existing:
                    theory = self._theory_from_node(existing["t"])
                    if source_observation_id:
                        self._link_observation_to_theory(source_observation_id, theory.id)
                    return theory
        theory = Theory(id=str(uuid.uuid4()), text=normalized)
        timestamp = self._now()
        query = """
        MERGE (t:Theory {text: $text})
        ON CREATE SET t.id = $id, t.confidence_score = $confidence_score,
            t.predictive_success_rate = $predictive_success_rate, t.attempts = $attempts,
            t.created_at = $created_at, t.updated_at = $updated_at, t.status = 'active'
        RETURN t
        """
        with self.driver.session() as session:
            record = session.run(
                query,
                id=theory.id,
                text=theory.text,
                confidence_score=theory.confidence_score,
                predictive_success_rate=theory.predictive_success_rate,
                attempts=theory.attempts,
                created_at=timestamp,
                updated_at=timestamp,
            ).single()
            theory = self._theory_from_node(record["t"])
        if source_observation_id:
            self._link_observation_to_theory(source_observation_id, theory.id)
        return theory

    def record_epistemic_observation(
        self,
        theory: Theory,
        observation: Observation,
        error: PredictionError,
        *,
        mode: str,
        focus_text: str,
    ) -> str:
        timestamp = self._now()
        query = """
        MERGE (o:Observation {id: $id})
        SET o.kind = 'epistemic',
            o.mode = $mode,
            o.focus_text = $focus_text,
            o.theory_id = $theory_id,
            o.action_taken = $action_taken,
            o.sensor_data = $sensor_data,
            o.ok = $ok,
            o.match = $match,
            o.verdict = $verdict,
            o.evidence_json = $evidence_json,
            o.reasoning = $reasoning,
            o.confidence_in_data = $confidence_in_data,
            o.suggested_confidence_adjustment = $suggested_confidence_adjustment,
            o.created_at = coalesce(o.created_at, $created_at),
            o.updated_at = $updated_at
        """
        with self.driver.session() as session:
            session.run(
                query,
                id=observation.id,
                mode=str(mode or "theory"),
                focus_text=str(focus_text or theory.text),
                theory_id=observation.theory_id,
                action_taken=observation.action_taken,
                sensor_data=observation.sensor_data,
                ok=bool(observation.ok),
                match=error.verdict == "supports",
                verdict=error.verdict,
                evidence_json=json.dumps(observation.evidence),
                reasoning=error.reasoning,
                confidence_in_data=float(error.confidence_in_data),
                suggested_confidence_adjustment=float(error.suggested_confidence_adjustment),
                created_at=timestamp,
                updated_at=timestamp,
            )
        if theory.id != CURIOSITY_PLACEHOLDER_ID:
            rel_query = """
            MATCH (t:Theory {id: $theory_id})
            MATCH (o:Observation {id: $observation_id})
            MERGE (t)-[r:TESTED_BY]->(o)
            SET r.verdict = $verdict,
                t.next_test_at = $next_test_at
            """
            with self.driver.session() as session:
                session.run(rel_query, theory_id=theory.id, observation_id=observation.id,
                            verdict=error.verdict, next_test_at=timestamp + (300 if error.verdict == "inconclusive" else 0))
        return observation.id

    def record_goal_execution(
        self,
        goal: Goal,
        execution: GoalExecution,
        outcome: ExecutionOutcome,
    ) -> str:
        observation_id = execution.id
        timestamp = self._now()
        completion_score = float(getattr(outcome, "completion_score", 1.0 if outcome.match else 0.0))
        query = """
        CREATE (
            o:Observation {
                id: $id,
                kind: 'goal_execution',
                evidence_json: $evidence_json,
                goal_id: $goal_id,
                action_taken: $action_taken,
                sensor_data: $sensor_data,
                ok: $ok,
                match: $match,
                completion_score: $completion_score,
                reasoning: $reasoning,
                goal_status: $goal_status,
                created_at: $created_at,
                updated_at: $updated_at
            }
        )
        """
        rel_query = """
        MATCH (g:Goal {id: $goal_id})
        MATCH (o:Observation {id: $observation_id})
        MERGE (g)-[:ATTEMPTED_BY]->(o)
        """
        with self.driver.session() as session:
            session.run(
                query,
                id=observation_id,
                goal_id=goal.id,
                evidence_json=json.dumps(execution.evidence),
                action_taken=execution.action_taken,
                sensor_data=execution.sensor_data,
                ok=bool(execution.ok),
                match=bool(outcome.match),
                completion_score=completion_score,
                reasoning=outcome.reasoning,
                goal_status=outcome.goal_status,
                created_at=timestamp,
                updated_at=timestamp,
            )
            session.run(rel_query, goal_id=goal.id, observation_id=observation_id)
        return observation_id
