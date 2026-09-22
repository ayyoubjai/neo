import asyncio
import json
import unittest

from knowledge.comparison_modes import comparison_jobs, compare_group, GroupComparison
from knowledge.documents import Passage
from knowledge.extraction import extract_passage, Extraction


class ComparisonModeTests(unittest.TestCase):
    def setUp(self):
        self.claims = [{'id': str(i), 'passage_id': 'p'+str(i), 'text': 'Species change '+str(i),
                        'scope': 'all species', 'concepts': ['evolution']} for i in range(12)]
        self.vectors = {c['id']: [1, 0] for c in self.claims}

    def test_job_counts_and_cross_batch_candidates(self):
        def jobs(mode, budget=12000):
            return comparison_jobs(self.claims, self.vectors, mode, 4, 12, budget)
        self.assertEqual(len(jobs('per-claim')), 12)
        batches = jobs('batched')
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0][1]), 12)
        self.assertEqual(len(jobs('whole-set')), 1)
        self.assertEqual({c['id'] for focals,_ in batches for c in focals}, set(self.vectors))
        with self.assertRaises(ValueError):
            jobs('whole-set', 10)

    def test_group_validates_ids_direction_and_keeps_more_than_twelve(self):
        class Client:
            async def generate_model(inner, prompt, cls, **kwargs):
                links = [{'source_id': 'C1', 'target_id': 'C2', 'relation': 'REFINES',
                          'scope_match': 'same', 'reasoning': 'More specific',
                          'direction': 'focal_to_target', 'more_specific_id': 'C1'},
                         {'source_id': 'C99', 'target_id': 'C2'},
                         {'source_id': 'C1', 'target_id': 'C1'},
                         {'source_id': 'C1', 'target_id': 'C2', 'relation': 'SUPPORTS',
                          'scope_match': 'same', 'reasoning': 'Same words'}]
                return cls.from_dict({'relationships': links})
        links, issues = asyncio.run(compare_group(Client(), self.claims[:1], self.claims))
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['source_id'], '0')
        self.assertEqual(links[0]['more_specific_id'], '0')
        self.assertEqual(len(issues), 3)
        self.assertEqual(len(GroupComparison.from_dict({'relationships': [{}]*20}).relationships), 20)

    def test_required_entities_rejects_omission_but_accepts_empty(self):
        passage = Passage('p', 'd', 'page 1', 'Species change.', 'text')
        candidate = {'text': passage.text, 'quote': passage.text, 'speaker': 'narrator',
                     'stance': 'endorses', 'claim_type': 'empirical', 'scope': 'unspecified',
                     'concepts': ['evolution'], 'extraction_confidence': .9, 'attribution_confidence': .9}
        class Client:
            async def generate_model(inner, prompt, cls, **kwargs):
                return Extraction([candidate])
        self.assertEqual(len(asyncio.run(extract_passage(Client(), passage))[0]), 1)
        claims, issues = asyncio.run(extract_passage(Client(), passage, entities='required'))
        self.assertFalse(claims)
        self.assertIn('Required entities', issues[0]['reason'])
        candidate['entities'] = []
        self.assertEqual(len(asyncio.run(extract_passage(Client(), passage, entities='required'))[0]), 1)
