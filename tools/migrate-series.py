#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-off: turn the hand-written series quote at the top of a post

    > 本文是[《系列名》](/overview.html)系列的第 N 篇（共X篇）。上一篇：…；下一篇：…

into `series: <key>` front matter (key looked up by overview URL in
_data/series.yml) and delete the quote — _includes/series-nav.html now renders
it. Verifies that N matches the post's date order within the series and refuses
to write anything if a single post disagrees. Safe to re-run (idempotent).
"""
import io, os, re, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTS = os.path.join(ROOT, '_posts')
DATA = os.path.join(ROOT, '_data', 'series.yml')

# minimal YAML: `key:` / `  overview: /x.html`
by_overview = {}
key = None
for line in io.open(DATA, encoding='utf-8'):
    m = re.match(r'^([\w-]+):\s*$', line)
    if m: key = m.group(1); continue
    m = re.match(r'^\s+overview:\s*(\S+)', line)
    if m and key: by_overview[m.group(1)] = key

QUOTE = re.compile(r'^> 本文是\[《[^》]*》\]\(([^)]+)\)系列的第 (\d+) 篇（共[^）]*篇）。[^\n]*\n', re.M)

plan = []   # (path, key, n, new_text)
series_dates = {}
for path in sorted(glob.glob(os.path.join(POSTS, '*.md'))):
    s = io.open(path, encoding='utf-8').read()
    m = QUOTE.search(s)
    fm_key = re.search(r'^series:\s*(\S+)\s*$', s.split('\n---', 2)[0] + '\n', re.M) if s.startswith('---') else None
    if not m and not fm_key: continue
    key = by_overview.get(m.group(1)) if m else fm_key.group(1)
    if not key: sys.exit('no series key for overview %s (%s)' % (m.group(1), path))
    date = os.path.basename(path)[:10]
    series_dates.setdefault(key, []).append((date, path, int(m.group(2)) if m else None))
    if not m: continue
    # front matter: add `series:` after the `layout:` line
    head, sep, body = s.partition('\n---\n')
    if not re.search(r'^series:', head, re.M):
        head = re.sub(r'^(layout:.*)$', r'\1\nseries: ' + key, head, count=1, flags=re.M)
    body = '\n' + QUOTE.sub('', body, count=1).lstrip('\n')
    plan.append((path, key, int(m.group(2)), head + sep + body))

# verify N == date order
bad = []
for k, items in series_dates.items():
    items.sort()
    for i, (date, path, n) in enumerate(items, 1):
        if n is not None and n != i: bad.append('%s: written 第 %d 篇, date order %d' % (os.path.basename(path), n, i))
if bad: sys.exit('order mismatch, nothing written:\n  ' + '\n  '.join(bad))

for path, key, n, text in plan:
    io.open(path, 'w', encoding='utf-8').write(text)
    print('%-70s series=%s #%d' % (os.path.relpath(path, ROOT), key, n))
print('%d posts migrated; series sizes: %s' % (len(plan), {k: len(v) for k, v in series_dates.items()}))
