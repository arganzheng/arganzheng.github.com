#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q&A footnotes ([^qN] markers → [^qN]: definitions) in posts.

    tools/qfootnotes.py --check [files…]   # exit 1 on: a marker without a definition,
                                           # a definition without a marker, a duplicate
                                           # or empty definition, numbers out of reading
                                           # order, 「见下文」-style non-answers
    tools/qfootnotes.py --fix   [files…]   # renumber markers q0, q1, … in reading order
                                           # and sort the definition block to match

Markers inside fenced code or inline code are ignored (the blog memo shows the
syntax in backticks). Default file set: every _posts/*.md.
"""
import glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FENCE = re.compile(r'^\s*(```|~~~)')
DEF = re.compile(r'^\[\^q(\d+)\]:\s*(.*)$')
MARK = re.compile(r'\[\^q(\d+)\](?!:)')
INLINE_CODE = re.compile(r'`[^`\n]*`')
NON_ANSWER = re.compile(r'^(见|详见|参见|答案见)(下文|后文|正文|后面)')

def parse(lines):
    """-> (markers [(line_idx, num)], defs {num: (line_idx, text)}, dup_defs [num])"""
    markers, defs, dups, in_fence = [], {}, [], False
    for i, l in enumerate(lines):
        if FENCE.match(l):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = DEF.match(l)
        if m:
            n = int(m.group(1))
            if n in defs: dups.append(n)
            defs[n] = (i, m.group(2).strip())
            continue
        for mm in MARK.finditer(INLINE_CODE.sub('', l)):
            markers.append((i, int(mm.group(1))))
    return markers, defs, dups

def check(path):
    lines = open(path, encoding='utf-8').read().split('\n')
    markers, defs, dups = parse(lines)
    if not markers and not defs:
        return []
    errs = []
    used = [n for _, n in markers]
    for n in dups: errs.append('duplicate definition [^q%d]' % n)
    for i, n in markers:
        if n not in defs: errs.append('line %d: [^q%d] has no definition' % (i + 1, n))
    for n, (i, text) in defs.items():
        if n not in used: errs.append('line %d: [^q%d]: is never referenced' % (i + 1, n))
        if not text: errs.append('line %d: [^q%d]: is empty' % (i + 1, n))
        elif NON_ANSWER.match(re.sub(r'^\*\*|[「」*]', '', text)) and len(text) < 20:
            errs.append('line %d: [^q%d]: is not an answer (%s)' % (i + 1, n, text))
    order = []
    for _, n in markers:
        if n not in order: order.append(n)
    if order != list(range(len(order))):
        errs.append('markers not numbered 0..%d in reading order: %s' % (len(order) - 1, order))
    dl = sorted(defs.items(), key=lambda kv: kv[1][0])
    if [n for n, _ in dl] != sorted(defs):
        errs.append('definitions not in numeric order')
    return errs

def fix(path):
    src = open(path, encoding='utf-8').read()
    lines = src.split('\n')
    markers, defs, dups = parse(lines)
    if dups or not markers:
        return False
    order = []
    for _, n in markers:
        if n not in order: order.append(n)
    if any(n not in defs for n in order):
        return False
    new = {n: i for i, n in enumerate(order)}
    out, in_fence = [], False
    def_idx = sorted(defs.values(), key=lambda v: v[0])
    first_def = def_idx[0][0] if def_idx else None
    new_defs = []
    for n in order:
        i, text = defs[n]
        new_defs.append(re.sub(r'^\[\^q\d+\]:', '[^q%d]:' % new[n], lines[i]))
    skip = set(i for i, _ in defs.values())
    for i, l in enumerate(lines):
        if FENCE.match(l):
            in_fence = not in_fence; out.append(l); continue
        if in_fence:
            out.append(l); continue
        if i in skip:
            if i == first_def: out.extend(new_defs)
            continue
        # protect inline code, renumber the rest
        parts, pos = [], 0
        for c in INLINE_CODE.finditer(l):
            parts.append(MARK.sub(lambda m: '[^q%d]' % new[int(m.group(1))], l[pos:c.start()]))
            parts.append(c.group(0)); pos = c.end()
        parts.append(MARK.sub(lambda m: '[^q%d]' % new[int(m.group(1))], l[pos:]))
        out.append(''.join(parts))
    res = '\n'.join(out)
    if res != src:
        open(path, 'w', encoding='utf-8').write(res)
        return True
    return False

def main(argv):
    mode = '--check'
    files = []
    for a in argv:
        if a in ('--check', '--fix'): mode = a
        else: files.append(a)
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, '_posts', '*.md')))
    bad = 0
    for f in files:
        if mode == '--fix':
            if fix(f): print('fixed', os.path.relpath(f, ROOT))
        errs = check(f)
        if errs:
            bad += 1
            print(os.path.relpath(f, ROOT))
            for e in errs: print('  ' + e)
    if mode == '--check' and not bad:
        n = sum(1 for f in files if parse(open(f, encoding='utf-8').read().split('\n'))[0])
        print('q footnotes ok in %d posts' % n)
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
