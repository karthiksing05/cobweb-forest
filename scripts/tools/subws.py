"""Whitespace-tolerant exact text replacement for editing the LaTeX source."""
import re


def subws(s, old, new):
    pat = r'\s+'.join(re.escape(w) for w in old.split())
    m = list(re.finditer(pat, s))
    if len(m) != 1:
        raise SystemExit(f"NOMATCH({len(m)}): {old[:90]!r}")
    return s[:m[0].start()] + new + s[m[0].end():]


def apply(path, edits):
    s = open(path, encoding='utf-8').read()
    for old, new in edits:
        s = subws(s, old, new)
    open(path, 'w', encoding='utf-8').write(s)
    print("applied", len(edits))
