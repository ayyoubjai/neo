"""Recover exact PDF substrings after strictly typographic normalization."""
import re
import unicodedata


def normalized_offsets(text):
    # Ignore only line-wrap hyphens between letters; retain ordinary lexical hyphens.
    wrapped = {m.start() for m in re.finditer(r'(?<=[^\W\d_])-\r?\n\s*(?=[^\W\d_])', text)}
    chars, offsets = [], []
    skip_space = False
    for index, char in enumerate(text):
        if index in wrapped:
            skip_space = True
            continue
        if skip_space and char.isspace():
            continue
        skip_space = False
        for normalized in unicodedata.normalize('NFKC', char).translate(str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"'})):
            if normalized.isspace():
                if chars and chars[-1] != ' ':
                    chars.append(' '); offsets.append(index)
            else:
                chars.append(normalized); offsets.append(index)
    return ''.join(chars), offsets


def recover_quote(proposed, source):
    if proposed in source:
        return proposed
    normalized, offsets = normalized_offsets(source)
    needle, _ = normalized_offsets(proposed)
    needle = needle.strip()
    if not needle:
        raise ValueError('Empty quote')
    start = normalized.find(needle)
    if start < 0 or normalized.find(needle, start + 1) >= 0:
        raise ValueError('Quote does not uniquely match the source after PDF layout normalization')
    end = start + len(needle)
    if (start > 0 and offsets[start - 1] == offsets[start]
            or end < len(offsets) and offsets[end - 1] == offsets[end]):
        raise ValueError('Quote matches only part of a normalized source character')
    return source[offsets[start]:offsets[end - 1] + 1]
