import unittest
from knowledge.retrieval import hybrid_candidates, unit_vector, embedding_key
from knowledge.pilot_stages import project


class RetrievalTests(unittest.TestCase):
    def test_semantics_can_retrieve_without_shared_words(self):
        a = {'id': 'a', 'text': 'Rain falls', 'concepts': ['weather']}
        b = {'id': 'b', 'text': 'Precipitation occurs', 'concepts': ['hydrology']}
        c = {'id': 'c', 'text': 'A novel', 'concepts': ['literature']}
        result = hybrid_candidates(a, [a,b,c], {'a': [1,0], 'b': [1,0], 'c': [-1,0]})
        self.assertEqual([r['id'] for r in result], ['b'])
        self.assertNotEqual(embedding_key(a, 'model-a'), embedding_key(a, 'model-b'))
        with self.assertRaises(ValueError):
            unit_vector([float('nan')])
        with self.assertRaises(ValueError):
            unit_vector([0,0])

    def test_dedup_keeps_occurrences_and_distinct_scope_ids(self):
        claim = {'id': 'same', 'text': 'Species change.', 'quote': 'Species change.',
                 'extraction_confidence': .9, 'attribution_confidence': .9}
        state = {'extractions': {
            'p1': {'claims': [claim, claim], 'source': {'passage_id': 'p1'}, 'context': []},
            'p2': {'claims': [claim, {**claim, 'id': 'different-scope', 'scope': 'sometimes'}],
                   'source': {'passage_id': 'p2'}, 'context': []}}, 'comparisons': {}}
        records, _ = project(state)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[0]['sources']), 2)
