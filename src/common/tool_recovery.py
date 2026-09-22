"""Shared, declarative tool contracts and failure recovery guidance."""
import re


RECOVERY = {
    'MISSING_INPUT': 'Obtain or correct the required input before retrying.',
    'MISSING_DEPENDENCY': 'Use a supported alternative or resolve the dependency; do not repeat unchanged.',
    'PERMISSION_DENIED': 'Follow the approval flow or report the denial. Do not use another tool to bypass it.',
    'UNSUPPORTED_PLATFORM': 'Retrieve an implementation compatible with the current platform.',
    'SOURCE_MISMATCH': 'Retrieve a tool whose declared source matches the intended operation.',
    'MISSING_CAPABILITY': 'Search the full permitted registry before proposing tool generation.',
    'TEMPORARY_FAILURE': 'Retry once if appropriate, then reassess.',
    'INVALID_MODEL_OUTPUT': 'Adjust or repair the model request within the action budget.',
    'UNAVAILABLE_SOURCE': 'Obtain the requested source or choose the correct source; do not retry unchanged.',
    'RECOVERY_LIMIT': 'Recovery budget exhausted. Report the blocker.',
    'UNKNOWN': 'Inspect the failure and inputs; do not assume missing capability or generate a tool blindly.',
}


def failure_details(code='UNKNOWN', *, missing_argument=None):
    if code not in RECOVERY:
        code = 'UNKNOWN'
    result = {'code': code, 'retryable': code == 'TEMPORARY_FAILURE', 'recovery': RECOVERY[code]}
    if missing_argument:
        result['missing_argument'] = missing_argument
    return result


def contract_text(tool):
    parts = []
    for key in ('source', 'use_for', 'not_for', 'requires', 'produces', 'platforms'):
        if tool.get(key):
            parts.append(f'{key}={tool[key]}')
    return '; '.join(parts)


def contextual_query(query, context):
    """Bound context and keep the current request prominent; no memory facts."""
    query = str(query or '').strip()[:2000]
    summary = str(context.get('summary') or '').strip()[:800]
    if not summary:
        return query
    return (f'Current request: {query}\nConversation context for resolving references only '
            f'(not current observations): {summary}\nFind tools for: {query}')


def source_mismatch(tool, intent):
    source = intent.get('source') if isinstance(intent, dict) else None
    actual = tool.get('source')
    # "image" is an artifact which can originate from any source.
    return bool(source and actual and actual not in ('image', 'any') and source != actual)


def lexical_score(query, tool):
    words = set(re.findall(r'\w+', query.lower()))
    document = f"{tool.get('tool_id', '')} {tool.get('description', '')} {contract_text(tool)}"
    return len(words.intersection(re.findall(r'\w+', document.lower())))


def expand_prerequisites(selected, available):
    """Offer transitive producers for required artifacts; never run them automatically."""
    result = list(selected)
    seen = {tool.get('tool_id') for tool in result}
    while True:
        required = {req.get('artifact_type') for tool in result for req in tool.get('requires', [])
                    if isinstance(req, dict) and req.get('artifact_type')}
        added = False
        for tool in available:
            if tool.get('tool_id') in seen:
                continue
            if any(isinstance(output, dict) and output.get('artifact_type') in required
                   for output in tool.get('produces', [])):
                result.append(tool)
                seen.add(tool.get('tool_id'))
                added = True
        if not added:
            break
    return result
