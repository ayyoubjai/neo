"""Conservative semantic guards; proposals remain distinct from verified evidence."""
import re
from knowledge.matching import normalize

INCIDENTAL = {'first', 'many', 'same coin', 'flip sides', 'sentence', 'component', 'examples', 'tenets'}
ENTITY_KINDS = {'person', 'organism', 'work', 'place', 'organization'}


def semantic_fields(payload, text):
    concepts = payload.get('concepts', [])
    flagged = [c for c in concepts if normalize(c) in INCIDENTAL]
    unresolved = bool(re.search(r'^(those (two )?(ideas|differences)|the (next two tenets|first component)|all of us)\b', text, re.I))
    unresolved = unresolved or 'unresolved' in str(payload.get('scope','')).lower()
    entities = payload.get('entities', [])
    if not isinstance(entities, list) or len(entities) > 12:
        raise ValueError('entities must be a list of at most twelve typed entities')
    for e in entities:
        if not isinstance(e, dict) or not isinstance(e.get('name'), str) or not e['name'].strip() or not isinstance(e.get('kind'),str) or e.get('kind') not in ENTITY_KINDS:
            raise ValueError('Entity requires a name and supported kind')
    return {'entities': [e['name'].strip() for e in entities], 'entity_kinds': [e['kind'] for e in entities],
            'semantic_flags': (['incidental_concepts: '+', '.join(flagged)] if flagged else []) +
                              (['unresolved_reference'] if unresolved else [])}


def relation_guard(item, focal, target):
    """Stronger links need explicit directional/semantic checks, not a scope label alone."""
    relation = item['relation']
    if relation in {'EQUIVALENT_TO', 'REFINES', 'SUPPORTS', 'DEPENDS_ON', 'EXEMPLIFIES', 'EXPLAINS'}:
        if item.get('direction') != 'focal_to_target':
            return 'Missing explicit focal-to-target direction check'
    if relation == 'EQUIVALENT_TO':
        if item.get('focal_entails_target') is not True or item.get('target_entails_focal') is not True:
            return 'Equivalence requires both entailment checks'
        a, b = focal['text'].lower(), target['text'].lower()
        across = lambda t: bool(re.search(r'(different species|across species|between species|species.*same rate)', t))
        within = lambda t: bool(re.search(r'(single species|within (a |one )?species|wax and wane)', t))
        if across(a) and within(b) or across(b) and within(a):
            return 'Between-species variation is not equivalent to within-species temporal variation'
    if relation == 'REFINES' and item.get('more_specific_id') != focal['id']:
        return 'REFINES must run from the more specific proposition to the broader one'
    if relation == 'SUPPORTS':
        if item.get('support_basis') != 'argument' or not isinstance(item.get('warrant'), str) or not item['warrant'].strip():
            return 'SUPPORTS requires an explicit inferential warrant; shared wording, examples, or citations are insufficient'
    return None
