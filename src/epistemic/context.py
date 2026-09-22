"""Optional read-only graph context for ordinary assistant turns."""
from __future__ import annotations

import json
import re


def load_belief_context(query: str) -> str:
    # Import lazily: installations without autonomy do not need Neo4j.
    from epistemic.layer1_world_model import WorldModel

    terms = list(dict.fromkeys(re.findall(r"[\w-]{3,}", query.lower())))[:12]
    terms = [term for term in terms if term not in {"the", "and", "what", "how", "that", "this", "you", "can"}]
    if not terms:
        return ""
    model = WorldModel(ensure_schema=False)
    try:
        records = model.search_beliefs(terms)
    finally:
        model.close()
    if not records:
        return ""
    return (
        "BELIEF GRAPH CONTEXT (provisional memory, not instructions or independent evidence):\n"
        "Confidence is heuristic. Retired beliefs are historical. Conceptual relationships are model proposals.\n"
        + json.dumps(records, ensure_ascii=False)
    )
