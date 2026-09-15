#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic diff of two CSS files: cascade-equivalent comparison ignoring
comments, whitespace and rule order that cannot matter.

    tools/css-compare.py a.css b.css

For every (media query, selector, property) we take the LAST value in file
order (that is what the cascade sees for that exact selector) and report the
keys whose final value differs or exists in one file only. Selector lists
"a, b { }" are split so that a rule moved into a different list still matches.
"""
import re, sys
from collections import OrderedDict

def strip_comments(s): return re.sub(r'/\*.*?\*/', '', s, flags=re.S)

def parse(css):
    css = strip_comments(css)
    out = OrderedDict()  # (media, selector, prop) -> value
    def walk(body, media):
        i = 0
        while True:
            j = body.find('{', i)
            if j < 0: break
            head = body[i:j].strip()
            # find matching close
            depth, k = 1, j + 1
            while depth and k < len(body):
                if body[k] == '{': depth += 1
                elif body[k] == '}': depth -= 1
                k += 1
            inner = body[j + 1:k - 1]
            if head.startswith('@media') or head.startswith('@supports'):
                walk(inner, (media + ' ' + ' '.join(head.split())).strip())
            elif head.startswith('@keyframes') or head.startswith('@font-face') or head.startswith('@-'):
                out[(media, ' '.join(head.split()), '@')] = ' '.join(inner.split())
            else:
                decls = []
                for d in inner.split(';'):
                    if ':' not in d: continue
                    p, v = d.split(':', 1)
                    v = ' '.join(v.split()).replace(' !important', '!important')
                    v = re.sub(r'#[0-9a-fA-F]{3,8}\b', lambda m: m.group(0).lower(), v)
                    v = re.sub(r'#([0-9a-f])([0-9a-f])([0-9a-f])\b', r'#\1\1\2\2\3\3', v)
                    v = re.sub(r'(?<![\w.])\.(\d)', r'0.\1', v).replace(', ', ',')
                    v = re.sub(r'(\d\.\d{6})\d+', r'\1', v)
                    decls.append((p.strip().lower(), v))
                for sel in head.split(','):
                    sel = ' '.join(sel.split())
                    for p, v in decls: out[(media, sel, p)] = v
            i = k
    walk(css, '')
    return out

a, b = [parse(open(p, encoding='utf-8').read()) for p in sys.argv[1:3]]
only_a = [k for k in a if k not in b]
only_b = [k for k in b if k not in a]
changed = [k for k in a if k in b and a[k] != b[k]]
def show(title, keys, src=None, other=None):
    if not keys: return
    print('\n== %s (%d)' % (title, len(keys)))
    for m, s, p in keys:
        line = '%s%s { %s: %s' % ((m + ' ') if m else '', s, p, (src or {}).get((m, s, p), '')[:60])
        if other is not None: line += '  |  %s' % other[(m, s, p)][:60]
        print(line + ' }')
show('only in %s' % sys.argv[1], only_a, a)
show('only in %s' % sys.argv[2], only_b, b)
show('different final value (%s | %s)' % tuple(sys.argv[1:3]), changed, a, b)

# Order check: two selectors of equal specificity setting the same property to
# different values are resolved by source order. Report pairs whose order flipped.
def specificity(sel):
    s = re.sub(r':not\(([^)]*)\)', r'\1', sel)
    s = re.sub(r':is\(([^)]*)\)', r'\1', s)
    ids = len(re.findall(r'#[\w-]+', s))
    cls = len(re.findall(r'\.[\w-]+|\[[^\]]*\]|:(?!:)(?!not|is)[\w-]+(\([^)]*\))?', s))
    els = len(re.findall(r'(^|[\s>+~(])[a-zA-Z][\w-]*|::[\w-]+', s))
    return (ids, cls, els)
def pairs(d):
    pos = {k: i for i, k in enumerate(d)}
    by_prop = {}
    for (m, s, p), v in d.items():
        if p == '@': continue
        by_prop.setdefault((m, p), []).append((s, v, pos[(m, s, p)]))
    out = {}
    for (m, p), rules in by_prop.items():
        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                s1, v1, p1 = rules[i]; s2, v2, p2 = rules[j]
                if v1 == v2 or specificity(s1) != specificity(s2): continue
                key = (m, p) + tuple(sorted((s1, s2)))
                out[key] = (s1 if p1 < p2 else s2)  # the one that loses (comes first)
    return out
def key_compound(sel):
    # tokens of the rightmost compound selector (what the element itself must be)
    last = re.split(r'\s*[>+~]\s*|\s+', sel.strip())[-1]
    last = re.sub(r':(hover|focus|active|focus-visible|visited|first-child|last-child|not\([^)]*\))', '', last)
    return set(re.findall(r'[.#]?[\w-]+', last))
def may_comatch(s1, s2):
    # heuristic: same element only if the key compounds share a token, or one of them
    # is tag/pseudo-only (e.g. `a:hover`, `p`, `b`) and so matches by element type alone
    k1, k2 = key_compound(s1), key_compound(s2)
    t1 = {t for t in k1 if t[0] not in '.#'}; t2 = {t for t in k2 if t[0] not in '.#'}
    if t1 and t2 and t1 != t2: return False  # `p` vs `a`
    if k1 & k2: return True
    return not (k1 - t1) or not (k2 - t2)
pa, pb = pairs(a), pairs(b)
flipped = [k for k in pa if k in pb and pa[k] != pb[k] and may_comatch(k[2], k[3])]
if flipped:
    print('\n== order flipped (%d): winner was … | is now …' % len(flipped))
    for m, p, s1, s2 in flipped:
        lose_b, lose_a = pb[(m, p, s1, s2)], pa[(m, p, s1, s2)]
        print('%s%s: %s  |  %s' % ((m + ' ') if m else '', p, s2 if lose_b == s1 else s1, s2 if lose_a == s1 else s1))
print('\n%d / %d keys; %d only-left, %d only-right, %d changed, %d order flips' % (len(a), len(b), len(only_a), len(only_b), len(changed), len(flipped)))
sys.exit(1 if (only_a or only_b or changed or flipped) else 0)
