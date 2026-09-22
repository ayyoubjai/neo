import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from knowledge.semantics import relation_guard, semantic_fields
from knowledge.review_semantics import validate_review

class SemanticTests(unittest.TestCase):
    def test_false_equivalence_despite_model_scope_agreement(self):
        a={'id':'a','text':'Different species vary in how fast they evolve.'}
        b={'id':'b','text':'A single species evolves faster or slower as pressures wax and wane.'}
        link={'relation':'EQUIVALENT_TO','direction':'focal_to_target','focal_entails_target':True,'target_entails_focal':True}
        self.assertIn('Between-species',relation_guard(link,a,b))

    def test_refinement_direction_and_clarification_are_not_support(self):
        a={'id':'summary','text':'Six components.'};b={'id':'detail','text':'One component defined.'}
        self.assertIsNotNone(relation_guard({'relation':'REFINES','direction':'focal_to_target','more_specific_id':'detail'},a,b))
        self.assertIsNone(relation_guard({'relation':'REFINES','direction':'focal_to_target','more_specific_id':'detail'},b,a))
        self.assertIsNotNone(relation_guard({'relation':'SUPPORTS','direction':'focal_to_target','support_basis':'shared_wording','warrant':'Same two ideas'},a,b))
        self.assertIsNone(relation_guard({'relation':'CLARIFIES'},a,b))

    def test_entities_and_incidental_terms_are_distinct(self):
        fields=semantic_fields({'concepts':['many','evolution'],'entities':[{'name':'Charles Darwin','kind':'person'}]},'Those two ideas were important.')
        self.assertEqual(fields['entities'],['Charles Darwin'])
        self.assertEqual(len(fields['semantic_flags']),2)
        with self.assertRaises(ValueError):semantic_fields({'entities':[{'name':'bad','kind':{}}]},'Text')

    def test_review_cannot_change_provenance_or_silently_merge_claims(self):
        base={'reviewer':'assistant:test','claims':[{'claim_id':'a','note':'Review','correction':{'quote':'invented'}}]}
        with self.assertRaises(ValueError):validate_review(base)
        base['claims'][0]['correction']={'concepts':['evolutionary rate']}
        self.assertTrue(validate_review(base).startswith('semantic-review:'))
        base['propositions']=[{'text':'Same','scope':'same','claim_ids':['a','a'],'note':'duplicate'}]
        with self.assertRaises(ValueError):validate_review(base)

class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_resolves_names_but_is_not_an_extra_quote_source(self):
        from knowledge.extraction import extract_passage
        from knowledge.documents import Passage
        context=Passage('context','doc','page 1','Two ideas were evolution and natural selection.')
        focal=Passage('focal','doc','page 2','Those ideas explain diversity.')
        class Client:
            async def generate_model(self,prompt,cls,**kwargs):
                self_prompt=prompt
                assert 'REFERENCE CONTEXT' in self_prompt
                return cls([{'text':'Evolution and natural selection explain diversity.','quote':q,
                    'speaker':'narrator','stance':'endorses','claim_type':'empirical','scope':'diversity',
                    'concepts':['evolution'],'extraction_confidence':.9,'attribution_confidence':.9}
                    for q in (focal.text,context.text)])
        claims,issues=await extract_passage(Client(),focal,context=[context])
        self.assertEqual(len(claims),1)
        self.assertEqual(claims[0]['context_passage_ids'],['context'])
        self.assertEqual(len(issues),1)
