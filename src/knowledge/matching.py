"""Conservative retrieval: aliases improve recall without merging propositions."""
import re
import unicodedata


def normalize(text):
    return ' '.join(re.sub(r'[-_]', ' ', unicodedata.normalize('NFKC', text).casefold()).split())[:120]


def expanded(concepts, aliases=()):
    result = {normalize(c) for c in concepts}
    # Treat reviewed alias links as undirected, including their transitive closure.
    changed = True
    while changed:
        changed = False
        for a, b in aliases:
            a, b = normalize(a), normalize(b)
            if (a in result or b in result) and not {a, b} <= result:
                result.update((a, b))
                changed = True
    return result


def rank_candidates(claim, candidates, aliases=(), limit=12):
    concepts = expanded(claim.get('concepts', []), aliases)
    words = set(re.findall(r'\w{4,}', normalize(claim['text'])))
    ranked = []
    for other in {c['id']: c for c in candidates}.values():
        if other['id'] == claim['id']:
            continue
        shared = concepts & expanded(other.get('concepts', []), aliases)
        other_words = set(re.findall(r'\w{4,}', normalize(other['text'])))
        lexical = len(words & other_words) / max(1, len(words | other_words))
        if shared or lexical >= .25:
            ranked.append((len(shared) * 2 + lexical, other['id'], other))
    return [c for _, _, c in sorted(ranked, key=lambda x: (-x[0], x[1]))[:limit]]
