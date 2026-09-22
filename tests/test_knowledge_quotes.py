import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from knowledge.quotes import recover_quote

class QuoteTests(unittest.TestCase):
    def test_layout_recovery_returns_original_substring(self):
        source='Darwin used evi-\ndence about the ﬁrst species. More text.'
        self.assertEqual(recover_quote('Darwin used evidence about the first species.',source),source[:source.index(' More')])

    def test_meaning_and_ambiguous_quotes_are_not_repaired(self):
        for quote,source in [('Evolution is false','Evolution is true'),('first','ﬁrst and ﬁrst'),('1859','')]:
            with self.assertRaises(ValueError):recover_quote(quote,source)

    def test_smart_apostrophes_recover_the_original_typography(self):
        self.assertEqual(recover_quote("species don't evolve at the same rate", "species don’t evolve at the same rate"),
                         "species don’t evolve at the same rate")
        with self.assertRaises(ValueError):recover_quote('f', 'ﬁ')

    def test_ordinary_hyphens_preserved(self):
        with self.assertRaises(ValueError):recover_quote('selfreplicating','self-replicating')
        self.assertEqual(recover_quote('self-replicating molecule','self-replicating\nmolecule'),'self-replicating\nmolecule')

    def test_singular_stance_is_canonicalized_without_changing_position(self):
        from knowledge.documents import Passage
        from knowledge.extraction import validate_claim
        p=Passage('p','d','1','A source claim.')
        candidate={'text':'A source claim.','quote':'A source claim.','speaker':'narrator','stance':'endorse',
                   'claim_type':'empirical','scope':'unspecified','concepts':[],
                   'extraction_confidence':.9,'attribution_confidence':.9}
        result=validate_claim(candidate,p)
        self.assertEqual(result['stance'],'endorses')
        self.assertEqual(result['proposed_stance'],'endorse')
        with self.assertRaises(ValueError):validate_claim({**candidate,'stance':'invented'},p)

class StructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_json_is_not_rewritten_by_a_second_turn(self):
        import json
        from unittest.mock import AsyncMock
        from autonomy.cognition_client import CognitionClient
        from knowledge.extraction import Extraction
        client=CognitionClient()
        payload={'claims':[{'quote':'An exact source\nquotation with ﬁ ligature.'}]}
        client._run_turn_with_trace=AsyncMock(return_value=(json.dumps(payload),[]))
        result=await client.generate_model('Extract',Extraction,max_actions=0)
        self.assertEqual(result.claims,payload['claims'])
        client._run_turn_with_trace.assert_awaited_once()
