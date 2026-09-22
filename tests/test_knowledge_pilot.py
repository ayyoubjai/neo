import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge.pilot import run, parse_model_json


class FreshPilotTests(unittest.TestCase):
    def test_json_fences_are_transport_formatting_only(self):
        self.assertEqual(parse_model_json('```json\n{"claims": []}\n```'), {'claims': []})
        self.assertEqual(parse_model_json('{"claims": []}'), {'claims': []})
        for invalid in ['Here is JSON: {"claims": []}', '```json\n{"claims": [}\n```']:
            with self.assertRaises(json.JSONDecodeError):
                parse_model_json(invalid)

    def test_fresh_export_preserves_context_and_proposed_links(self):
        class Client:
            model = 'test-only'
            extraction_calls = 0
            comparison_calls = 0
            cancel_next_comparison = False

            def __init__(self, output, timeout):
                assert timeout == 600

            async def generate_model(self, prompt, cls, **kwargs):
                if cls.__name__ == 'Extraction':
                    Client.extraction_calls += 1
                    focal = json.loads(prompt.split('PASSAGE:\n')[1])
                    return cls([dict(text=focal['text'], quote=focal['text'], speaker='unidentified narrator',
                                     stance='endorses', claim_type='empirical', scope='unspecified',
                                     extraction_confidence=.9, attribution_confidence=.9,
                                     concepts=['change'], entities=[])])
                payload = json.loads(prompt.split('An empty list is valid.\n')[1])
                if Client.cancel_next_comparison:
                    Client.cancel_next_comparison = False
                    raise asyncio.CancelledError()
                Client.comparison_calls += 1
                return cls([dict(target_id=payload['candidates'][0]['id'], relation='RELATED_TO',
                                 scope_match='unknown', reasoning='Shared topic')])

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'book.pdf'
            source.write_bytes(b'test fixture')
            output = Path(tmp) / 'fresh'
            args = argparse.Namespace(pdf=source, first_page=24, last_page=25, output=output, timeout=600, stage='extract')
            transcript = argparse.Namespace(stdout='Earlier context\fSpecies change.\fRates change.\fLater context\f')
            with patch('knowledge.pilot.RecordedClient', Client), patch('knowledge.pilot_stages.subprocess.run', return_value=transcript), patch('knowledge.pilot_stages.embed', return_value={'embedding': [1, 0]}):
                self.assertEqual(asyncio.run(run(args)), 0)
            self.assertEqual(Client.comparison_calls, 0)
            self.assertFalse((output / 'embeddings.json').exists())
            args.resume = output
            args.pdf = None
            args.stage = 'all'
            Client.cancel_next_comparison = True
            with patch('knowledge.pilot.RecordedClient', Client), patch('knowledge.pilot_stages.embed', return_value={'embedding': [1, 0]}):
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(run(args))
            self.assertEqual(len(json.loads((output / 'graph.json').read_text())), 2)
            with patch('knowledge.pilot.RecordedClient', Client), patch('knowledge.pilot_stages.embed', side_effect=AssertionError('Should reuse vectors after interruption')):
                self.assertEqual(asyncio.run(run(args)), 0)
            self.assertEqual(Client.extraction_calls, 2)
            self.assertEqual(Client.comparison_calls, 2)
            records = json.loads((output / 'graph.json').read_text())
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]['context_passages'][0]['text'], 'Earlier context')
            self.assertEqual(records[1]['context_passages'][-1]['text'], 'Later context')
            self.assertEqual(records[0]['relationships'][0]['relationship']['kind'], 'RELATED_TO')
            self.assertTrue(all(not r['semantic_reviews'] and not r['assessments'] for r in records))
            self.assertIn('graph-connections', (output / 'explorer.html').read_text())
            args.resume = output
            args.pdf = None
            with patch('knowledge.pilot.RecordedClient', Client), patch('knowledge.pilot_stages.embed', side_effect=AssertionError('Should reuse embeddings')):
                self.assertEqual(asyncio.run(run(args)), 0)
            self.assertEqual(Client.extraction_calls, 2)
            self.assertEqual(Client.comparison_calls, 2)
            args.resume = None
            args.pdf = source
            with self.assertRaises(FileExistsError):
                asyncio.run(run(args))
