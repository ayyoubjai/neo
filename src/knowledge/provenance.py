"""Identifiable source reuse; absence of reuse never proves independence."""
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


def canonical_source(identifier):
    identifier = identifier.strip()
    doi = re.fullmatch(r'(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/\S+)', identifier, re.I)
    if doi:
        return 'doi:' + doi.group(1).lower().rstrip('.,;')
    url = urlsplit(identifier)
    if url.scheme.lower() in {'http', 'https'} and url.hostname and not url.username:
        query = [(k, v) for k, v in parse_qsl(url.query) if not k.lower().startswith('utm_') and k.lower() not in {'fbclid', 'gclid'}]
        return urlunsplit((url.scheme.lower(), url.netloc.lower(), url.path or '/', urlencode(sorted(query)), ''))
    raise ValueError('Source identifiers must be explicit DOI or HTTP(S) URLs')


def validate_sources(payload, passage):
    refs = payload.get('source_refs', [])
    if not isinstance(refs, list) or len(refs) > 20:
        raise ValueError('source_refs must be a list of at most 20 citations')
    sources = []
    for ref in refs:
        if not isinstance(ref, dict) or not isinstance(ref.get('identifier'), str) or not isinstance(ref.get('quote'), str):
            raise ValueError('Citation requires identifier and exact quote')
        if not ref['quote'] or ref['quote'] not in passage.text or ref['identifier'] not in ref['quote']:
            raise ValueError('Citation identifier must occur literally in its exact passage quote')
        sources.append(canonical_source(ref['identifier']))
    return list(dict.fromkeys(sources))


def summarize(claims):
    groups, unknown = {}, []
    for claim in claims:
        refs = claim.get('source_ids', [])
        if not refs:
            unknown.append(claim['id'])
        for ref in refs:
            groups.setdefault(ref, set()).add(claim['id'])
    return {'source_groups': [{'source': k, 'claim_ids': sorted(v)} for k, v in sorted(groups.items())],
            'unknown_provenance_claim_ids': sorted(set(unknown)),
            'independence': 'not established',
            'note': 'Shared citations reveal source reuse. Distinct citations do not establish independent evidence.'}


def evidence_sources(evidence):
    """Read explicit citation fields from successful tool receipts, not model prose."""
    sources = set()
    def visit(value, depth=0):
        if depth > 12:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {'url', 'source_url', 'doi', 'source_id'} and isinstance(item, str):
                    try:
                        sources.add(canonical_source(item))
                    except ValueError:
                        pass
                elif isinstance(item, (dict, list)):
                    visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                visit(item, depth + 1)
    for receipt in evidence:
        if isinstance(receipt, dict) and receipt.get('status') == 'APPROVED' and receipt.get('request_id'):
            visit(receipt.get('result'))
    return sorted(sources)
