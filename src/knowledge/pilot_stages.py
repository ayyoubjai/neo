"""Checkpointed extraction, embedding, and comparison for the offline pilot."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common.config import load_settings
from knowledge.documents import Passage, stable_id
from knowledge.extraction import extract_passage, compare_claims
from knowledge.export import render_explorer
from knowledge.ingest import write_report
from knowledge.retrieval import embedding_key, embedding_text, unit_vector, hybrid_candidates
from knowledge.comparison_modes import comparison_jobs, compare_group
from model_server.llamacpp_client import embed


def project(state):
    """Collapse identical attributed claims, retaining every supporting occurrence."""
    records = {}
    for page in state['extractions'].values():
        for claim in page['claims']:
            source = {**page['source'], 'quote': claim['quote'],
                      'extraction_confidence': claim['extraction_confidence'],
                      'attribution_confidence': claim['attribution_confidence']}
            if claim['id'] not in records:
                records[claim['id']] = {'claim': {**claim, 'review_required': True}, 'sources': [],
                    'context_passages': [], 'relationships': [], 'assessments': [], 'investigations': [],
                    'semantic_reviews': [], 'propositions': [], 'archived_relationships': []}
            record = records[claim['id']]
            if source not in record['sources']:
                record['sources'].append(source)
            for context in page['context']:
                if context not in record['context_passages']:
                    record['context_passages'].append(context)
    links = {}
    for comparison in state['comparisons'].values():
        for link in comparison['links']:
            links[link['id']] = link
    for link in links.values():
        if link['source_id'] not in records or link['target_id'] not in records:
            continue
        for focal, other in [(link['source_id'], link['target_id']), (link['target_id'], link['source_id'])]:
            records[focal]['relationships'].append({'other_id': other, 'text': records[other]['claim']['text'],
                'source_id': link['source_id'], 'target_id': link['target_id'],
                'relationship': {**link, 'kind': link['relation'], 'status': 'proposed'}})
    return list(records.values()), list(links.values())


def checkpoint(output, state):
    write_report(output / 'checkpoint.json', state)
    records, links = project(state)
    issues = [issue for page in state['extractions'].values() for issue in page['issues']]
    issues += [issue for comparison in state['comparisons'].values() for issue in comparison['issues']]
    issues += list(state['failures'].values())
    report = {k: state[k] for k in ('source', 'sha256', 'pages', 'model', 'phase')}
    report.update(entities=state.get('entities', 'auto'), comparison_mode=state.get('comparison_mode', 'per-claim'))
    report.update(persisted=False, claims=[r['claim'] for r in records], relationships=links, issues=issues,
                  duplicate_occurrences=sum(len(p['claims']) for p in state['extractions'].values())-len(records))
    write_report(output / 'report.json', report)
    write_report(output / 'graph.json', records)
    (output / 'explorer.html').write_text(render_explorer(records, f'Fresh model-only PDF pilot — {state["phase"]}'), encoding='utf-8')
    return records, issues


async def run_stages(args):
    from knowledge.pilot import RecordedClient, timed_call
    resume = getattr(args, 'resume', None)
    stage = getattr(args, 'stage', 'all')
    output = resume or args.output or Path('data/knowledge') / datetime.now(timezone.utc).strftime('fresh-pilot-%Y%m%d-%H%M%S-%f')
    if resume:
        saved = json.loads((output / 'report.json').read_text())
        source = (args.pdf or Path(saved['source'])).resolve()
        first_page, last_page = saved['pages']
    else:
        source = args.pdf.resolve()
        first_page, last_page = args.first_page, args.last_page
        output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if resume and digest != saved['sha256']:
        raise ValueError('PDF content changed; start a new run')
    client = RecordedClient(output, timeout=args.timeout)
    path = output / 'checkpoint.json'
    if resume and path.exists():
        state = json.loads(path.read_text())
        if state['model'] != client.model:
            raise ValueError('Text model changed; start a new run')
    else:
        if resume and saved['model'] != client.model:
            raise ValueError('Text model changed; start a new run')
        state = {'version': 1, 'source': str(source), 'sha256': digest, 'pages': [first_page, last_page],
                 'model': client.model, 'extractions': {}, 'comparisons': {}, 'failures': {}, 'phase': 'extracting'}
        if resume:
            # Retain the previous partial report; exact request caches reconstruct full pages.
            backup = output / 'before-staged-resume-report.json'
            if not backup.exists():
                write_report(backup, saved)
    entities = getattr(args, 'entities', None) or state.get('entities', 'auto')
    if state['extractions'] and entities != state.get('entities', 'auto'):
        raise ValueError('Entity extraction policy changed; start a fresh run with --entities required to re-extract claims')
    state['entities'] = entities
    mode = getattr(args, 'comparison_mode', None) or state.get('comparison_mode', 'per-claim')
    options = state.get('comparison_options', {})
    limit = getattr(args, 'candidate_limit', None) or options.get('candidate_limit', 12)
    batch_size = getattr(args, 'batch_size', None) or options.get('batch_size', 6)
    max_chars = getattr(args, 'comparison_max_chars', None) or options.get('max_chars', 12000)
    print(f'Output: {output}; through stage: {stage}; comparison: {mode}; entities: {entities}', flush=True)
    first = max(1, first_page-1)
    transcript = output / 'source.txt'
    if not resume or not transcript.exists():
        result = subprocess.run(['pdftotext', '-layout', '-f', str(first), '-l', str(last_page+1), str(source), '-'],
                                check=True, capture_output=True, text=True)
        transcript.write_text(result.stdout, encoding='utf-8')
    texts = transcript.read_text().split('\f')
    if texts and not texts[-1].strip():
        texts.pop()
    if first+len(texts)-1 < last_page:
        raise ValueError('Requested pages exceed extracted pages')
    docid = stable_id('document', digest)
    passages = [Passage(stable_id('passage', docid, 'poppler-layout', str(first+i), text), docid,
                        f'PDF page {first+i}', text, 'poppler-layout') for i, text in enumerate(texts)]
    try:
        state['phase'] = 'extracting'
        checkpoint(output, state)
        for i, passage in enumerate(passages):
            if not first_page <= first+i <= last_page or passage.id in state['extractions']:
                continue
            print(f'Extraction page {first+i} ({first+i-first_page+1}/{last_page-first_page+1})', flush=True)
            context = passages[max(0, i-1):i] + passages[i+1:i+2]
            try:
                claims, issues = await extract_passage(client, passage, context=context, entities=entities)
            except Exception as exc:
                state['failures'][passage.id] = {'passage_id': passage.id, 'reason': str(exc)}
                print(f'Extraction failed: {exc}; resume will retry', flush=True)
            else:
                state['failures'].pop(passage.id, None)
                state['extractions'][passage.id] = {'claims': claims, 'issues': issues,
                    'source': {'document': source.stem, 'document_hash': digest, 'paths': [str(source)],
                               'passage_id': passage.id, 'location': passage.location},
                    'context': [{'location': p.location, 'text': p.text} for p in context]}
                print(f'Accepted {len(claims)} claims; {len(issues)} validation issues', flush=True)
            checkpoint(output, state)
        state['phase'] = 'extracted' if not state['failures'] else 'extraction incomplete'
        records, _ = checkpoint(output, state)
        claims = [r['claim'] for r in records]
        print(f'{len(claims)} unique attributed claims ready', flush=True)
        # Do not compare an incomplete population: resume failed pages first.
        if stage == 'extract' or any(p.id not in state['extractions'] for i,p in enumerate(passages) if first_page <= first+i <= last_page):
            return 1 if checkpoint(output, state)[1] else 0

        settings = load_settings().models
        model = settings.get('embedding_model')
        if not model:
            raise ValueError('Configure models.embedding_model before the embedding stage')
        identity = json.dumps([model, settings.get('embedding_host') or settings.get('llamacpp_host')])
        cache_path = output / 'embeddings.json'
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        vectors = {}
        state['phase'] = 'embedding'
        for i, claim in enumerate(claims, 1):
            key = embedding_key(claim, identity)
            print(f'Embedding {i}/{len(claims)}: {"cached" if key in cache else "generating"}', flush=True)
            if key not in cache:
                response = await timed_call(f'Embedding {i}/{len(claims)}', embed, embedding_text(claim), model, timeout=args.timeout)
                cache[key] = {'model': model, 'endpoint': identity, 'claim_id': claim['id'],
                              'text': embedding_text(claim), 'vector': unit_vector(response['embedding'])}
                write_report(cache_path, cache)
            vectors[claim['id']] = unit_vector(cache[key]['vector'])
        state['phase'] = 'embedded'
        checkpoint(output, state)
        if stage == 'embed':
            return 1 if checkpoint(output, state)[1] else 0

        # Invalidate comparison checkpoints if retrieval model, population or limit changes.
        signature = hashlib.sha256(json.dumps([claims, identity, limit, mode, batch_size, max_chars, 'comparison-modes-v1'], sort_keys=True).encode()).hexdigest()
        if state.get('comparison_signature') != signature:
            if state['comparisons']:
                write_report(output / f'comparisons-{state.get("comparison_signature", "legacy")}.json', state['comparisons'])
            state['comparisons'] = {}
            state['failures'] = {}
            state['comparison_signature'] = signature
        state['comparison_mode'] = mode
        state['comparison_options'] = {'candidate_limit': limit, 'batch_size': batch_size, 'max_chars': max_chars}
        state['phase'] = 'comparing'
        jobs = comparison_jobs(claims, vectors, mode, batch_size, limit, max_chars)
        print(f'{mode}: {len(jobs)} comparison jobs planned (oversized groups split to fit the character budget)', flush=True)
        for i, (focals, candidates) in enumerate(jobs, 1):
            job_id = hashlib.sha256(json.dumps([[c['id'] for c in focals], [c['id'] for c in candidates]]).encode()).hexdigest()
            if job_id in state['comparisons']:
                print(f'Comparison {i}/{len(jobs)}: cached', flush=True)
                continue
            print(f'Comparison {i}/{len(jobs)}: {len(focals)} focal claims, {len(candidates)} candidates', flush=True)
            try:
                if mode == 'per-claim':
                    links, issues = await compare_claims(client, focals[0], candidates)
                else:
                    links, issues = await compare_group(client, focals, candidates)
            except Exception as exc:
                state['failures'][job_id] = {'job_id': job_id, 'reason': str(exc)}
                print(f'Comparison failed: {exc}; resume will retry', flush=True)
            else:
                state['failures'].pop(job_id, None)
                state['comparisons'][job_id] = {'links': links, 'issues': issues,
                    'focal_ids': [c['id'] for c in focals],
                    'candidate_ids': [c['id'] for c in candidates]}
                print(f'{len(links)} connections; {len(issues)} validation issues', flush=True)
            checkpoint(output, state)
        state['phase'] = 'complete' if not state['failures'] else 'comparison incomplete'
        _, issues = checkpoint(output, state)
        print(f'Finished. Open {output / "explorer.html"}', flush=True)
        return 1 if issues else 0
    finally:
        # Even interruption leaves an inspectable partial graph and durable progress.
        checkpoint(output, state)
