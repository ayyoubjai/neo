"""Create and score human-reviewed extraction samples; no model self-grading."""
import argparse
import json
from pathlib import Path
import random
from knowledge.ingest import write_report

DIMENSIONS = ('quote_correct', 'attribution_correct', 'scope_preserved', 'claim_type_correct')


def sample(report, count=20, seed=0):
    if count < 1:
        raise ValueError('Sample size must be positive')
    claims = {(c['id'], c['passage_id']): {**c, 'source': d['source']}
              for d in report['documents'] for c in d['claims']}
    selected = random.Random(seed).sample(list(claims.values()), min(count, len(claims)))
    return {'schema': 'extraction-review-v1', 'population': len(claims), 'seed': seed,
            'items': [{'claim': c, 'ratings': {key: None for key in DIMENSIONS}, 'note': ''} for c in selected]}


def score(review):
    if review.get('schema') != 'extraction-review-v1' or not isinstance(review.get('items'), list):
        raise ValueError('Expected an extraction-review-v1 sample')
    metrics = {}
    seen = set()
    for item in review['items']:
        key = (item['claim']['id'], item['claim']['passage_id'])
        if key in seen:
            raise ValueError('Duplicate claim/passage in review sample')
        seen.add(key)
    for dimension in DIMENSIONS:
        values = [item['ratings'].get(dimension) for item in review['items']]
        if any(v is not None and type(v) is not bool for v in values):
            raise ValueError('Ratings must be JSON true, false, or null (not reviewed)')
        reviewed = [v for v in values if v is not None]
        metrics[dimension] = {'reviewed': len(reviewed), 'unreviewed': len(values) - len(reviewed),
                              'correct': sum(reviewed), 'rate': sum(reviewed) / len(reviewed) if reviewed else None}
    return {'sample_size': len(review['items']), 'metrics': metrics,
            'limitation': 'Measures correctness of sampled extractions, not recall, truth, or overall corpus coverage.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['sample', 'score'])
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    result = sample(payload, args.count, args.seed) if args.operation == 'sample' else score(payload)
    write_report(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
