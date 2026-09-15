#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flag Liquid-looking `{{` / `{%` inside fenced code blocks that are NOT wrapped
in {% raw %} … {% endraw %}. Jekyll either aborts the whole build on them
("Variable '{{1, 0}' was not properly terminated" — Java/C++ nested
initialisers are the classic case) or silently renders them as empty strings.

    tools/liquid-scan.py _posts/foo.md ...    # exit 1 and list offenders
    tools/liquid-scan.py                      # scan every post / slide / draft
"""
import glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FENCE = re.compile(r'^\s*(```|~~~)')
LIQUID = re.compile(r'\{\{|\{%')
RAW_OPEN, RAW_CLOSE = re.compile(r'\{%-?\s*raw\s*-?%\}'), re.compile(r'\{%-?\s*endraw\s*-?%\}')

def scan(path):
    hits, in_fence, in_raw = [], False, False
    for n, line in enumerate(open(path, encoding='utf-8'), 1):
        if RAW_OPEN.search(line): in_raw = True
        if RAW_CLOSE.search(line): in_raw = False; continue
        if FENCE.match(line): in_fence = not in_fence; continue
        if in_fence and not in_raw and LIQUID.search(line):
            hits.append((n, line.rstrip()[:100]))
    return hits

files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, '_posts/*.md')) + glob.glob(os.path.join(ROOT, '_drafts/*.md')) + glob.glob(os.path.join(ROOT, 'slides/*.md')))
bad = 0
for f in files:
    if not f.endswith(('.md', '.markdown')) or not os.path.exists(f): continue
    for n, line in scan(f):
        bad += 1
        print('%s:%d: %s' % (os.path.relpath(f, ROOT), n, line))
if bad:
    print('\n%d line(s) with {{ / {%% inside code blocks not wrapped in {%% raw %%} … {%% endraw %%}' % bad)
    sys.exit(1)
