"""Run against a disposable Neo4j instance with NEO4J_TEST_URI set."""
from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epistemic.layer1_world_model import WorldModel
from epistemic.layer4_optimizer import Optimizer
from epistemic.types import Observation, PredictionError


@unittest.skipUnless(os.environ.get("NEO4J_TEST_URI"), "requires a disposable Neo4j instance")
class GraphIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model = WorldModel(uri=os.environ["NEO4J_TEST_URI"])
        self.prefix = "integration" + uuid.uuid4().hex
        self.ids = []

    def tearDown(self):
        # Delete only nodes created by this test, including linked observations.
        with self.model.driver.session() as session:
            session.run("MATCH (o:Observation) WHERE o.id STARTS WITH $prefix DETACH DELETE o", prefix=self.prefix).consume()
            session.run("MATCH (t:Theory) WHERE t.text STARTS WITH $prefix DETACH DELETE t", prefix=self.prefix).consume()
            session.run("MATCH (c:Concept) WHERE c.name STARTS WITH $prefix DETACH DELETE c", prefix=self.prefix).consume()
            session.run("MATCH (g:Goal) WHERE g.text STARTS WITH $prefix DETACH DELETE g", prefix=self.prefix).consume()
        self.model.close()

    def observation(self, theory, suffix="obs", ok=True):
        return Observation(self.prefix + suffix, theory.id, "read", "observed result", ok)

    def test_atomic_belief_history_relationships_and_revalidation(self):
        theory = self.model.add_theory(self.prefix + " old claim")
        optimizer = Optimizer(self.model)
        supported = PredictionError(True, "support", 1.0, .25, verdict="supports")
        observation = self.observation(theory)
        optimizer.optimize(theory, observation, supported)
        updated = self.model.get_theory(theory.id)
        self.assertEqual(updated.attempts, 1)
        self.assertEqual(updated.predictive_success_rate, 1.0)
        self.assertAlmostEqual(updated.confidence_score, .35)
        # Replaying an observation must not count it twice.
        optimizer.optimize(theory, observation, supported)
        self.assertEqual(self.model.get_theory(theory.id).attempts, 1)
        optimizer.optimize(updated, self.observation(theory, "failed", False), supported)
        self.assertEqual(self.model.get_theory(theory.id).attempts, 1)
        self.assertEqual(self.model.get_theory(theory.id).updated_at, updated.updated_at)
        # Confidence can reach zero without erasing the belief or its evidence.
        self.model.update_theory(theory.id, .1, False)
        result = optimizer.optimize(self.model.get_theory(theory.id), self.observation(theory, "contradiction"),
            PredictionError(False, "contradiction", 1.0, -.25,
                self.prefix + " alternative", verdict="contradicts",
                hypothesis_relation="CONTRADICTS", concepts=[self.prefix + " concept"]))
        self.assertEqual(self.model.get_theory(theory.id).status, "retired")
        context = self.model.get_theory_context(theory.id)
        self.assertEqual(context[0]["relation"], "CONTRADICTS")
        self.assertEqual(context[0]["relation_status"], "proposed")
        self.assertEqual(context[0]["source_id"], result.injected_theory_id)
        grounded = self.model.get_grounded_theories(limit=100, min_confidence=0)
        self.assertNotIn(theory.id, [t.id for t in grounded])
        self.assertTrue(self.model.search_beliefs([self.prefix]))
        # Active, exhausted beliefs become eligible after their validation ages.
        child = self.model.get_theory(result.injected_theory_id)
        self.model.update_theory(child.id, .9, True)
        with self.model.driver.session() as session:
            session.run("MATCH (t:Theory {id:$id}) SET t.attempts=3, t.updated_at=$old", id=child.id, old=self.model._now()-86400*365).consume()
        self.assertEqual(self.model.get_revalidation_theory().id, child.id)
        self.assertNotIn(child.id, [t.id for t in self.model.get_grounded_theories(min_confidence=.3)])

    def test_transaction_rolls_back_observation_and_belief_together(self):
        theory = self.model.add_theory(self.prefix + " rollback")
        observation = self.observation(theory)
        def failing(model):
            model.record_epistemic_observation(theory, observation, PredictionError(False,"unknown"), mode="theory", focus_text=theory.text)
            model.update_theory(theory.id, .8, True)
            raise RuntimeError("injected persistence failure")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.model.run_transaction(failing)
        self.assertFalse(self.model.has_observation(observation.id))
        self.assertEqual(self.model.get_theory(theory.id).attempts, 0)

    def test_shared_concepts_join_separate_ideas_and_exact_duplicates_merge(self):
        first = self.model.add_theory(self.prefix + " first")
        second = self.model.add_theory(self.prefix + " second")
        self.assertEqual(first.id, self.model.add_theory(first.text).id)
        for theory in (first, second):
            self.model.enrich_theory(theory.id, kind="idea", concepts=[self.prefix + " shared"],
                parent_id=None, relation="RELATED_TO", observation_id=self.prefix + "obs")
        related = self.model.get_theory_context(first.id)
        self.assertEqual(related[0]["id"], second.id)
        self.assertEqual(related[0]["relation"], "SHARES_CONCEPT")
        self.assertEqual(self.model.get_theory(first.id).kind, "idea")

    def test_goal_history_includes_prior_result_and_success_criteria(self):
        from power_process.types import GoalExecution, ExecutionOutcome
        goal = self.model.add_goal(self.prefix + " goal", "learn", success_criteria="Observe result")
        execution = GoalExecution("inspect", "partial result", True, id=self.prefix + "goalobs")
        def persist(model):
            model.record_goal_execution(goal, execution, ExecutionOutcome(False, "partial", completion_score=.7))
            model.update_goal_status(goal.id, "pending")
        self.model.run_transaction(persist)
        self.assertEqual(self.model.get_goal_history(goal.id)[0]["completion_score"], .7)
        with self.model.driver.session() as session:
            node = session.run("MATCH (g:Goal {id:$id}) RETURN g", id=goal.id).single()["g"]
        self.assertEqual(self.model._goal_from_node(node).success_criteria, "Observe result")


if __name__ == "__main__":
    unittest.main()
