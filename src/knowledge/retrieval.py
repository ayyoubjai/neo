"""Cached embeddings improve retrieval; similarity never establishes equivalence."""
import hashlib
import json
import math

from knowledge.matching import rank_candidates


def embedding_text(claim):
    return json.dumps({k: claim.get(k, '') for k in ('text', 'scope', 'speaker', 'stance', 'claim_type')}, ensure_ascii=False, sort_keys=True)


def embedding_key(claim, model):
    return hashlib.sha256((model + '\n' + embedding_text(claim)).encode()).hexdigest()


def unit_vector(values):
    if not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('Embedding must contain finite numbers')
    norm = math.sqrt(sum(v*v for v in values))
    if not norm or not math.isfinite(norm):
        raise ValueError('Embedding must have a finite nonzero norm')
    return [v/norm for v in values]


def hybrid_candidates(claim, claims, vectors, limit=12):
    """Reciprocal rank fusion of lexical/concept and positive cosine rankings."""
    lexical = rank_candidates(claim, claims, limit=limit)
    focal = vectors[claim['id']]
    semantic = []
    for other in claims:
        if other['id'] == claim['id']:
            continue
        vector = vectors[other['id']]
        if len(vector) != len(focal):
            raise ValueError('Embedding dimensions differ; regenerate embeddings with one model')
        similarity = sum(a*b for a, b in zip(focal, vector))
        if similarity > 0:
            semantic.append((similarity, other))
    semantic.sort(key=lambda item: (-item[0], item[1]['id']))
    scores = {}
    for ranking in (lexical, [other for _, other in semantic[:limit]]):
        for rank, other in enumerate(ranking, 1):
            scores[other['id']] = scores.get(other['id'], 0) + 1/(60+rank)
    by_id = {c['id']: c for c in claims}
    return [by_id[id] for id in sorted(scores, key=lambda id: (-scores[id], id))[:limit]]
