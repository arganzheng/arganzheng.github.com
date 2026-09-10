#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a post skeleton with the front matter this site understands.

    tools/new-post.py my-slug "标题" [--subtitle "副标题"] [--tags AI,AI-Infra]
                      [--series deep-dive-into-vllm] [--category life|meta] [--date 2026-10-01] [--draft]
                      [--layout post|header-post|keynote] [--header-img img/x.jpg] [--iframe /slides/x.html]

Writes _posts/<date>-<slug>.md (or _drafts/<slug>.md with --draft) and prints the
path. The slug is the URL: /<slug>.html — pick it once, renaming later breaks
links and the comment thread.
"""
import argparse, datetime, io, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('slug'); ap.add_argument('title')
ap.add_argument('--subtitle', default=''); ap.add_argument('--tags', default='')
ap.add_argument('--series', default=''); ap.add_argument('--date', default=datetime.date.today().isoformat())
ap.add_argument('--layout', default='post', choices=['post', 'header-post', 'keynote'])
ap.add_argument('--header-img', default=''); ap.add_argument('--iframe', default='')
ap.add_argument('--category', default='', choices=['', 'life', 'meta'])
ap.add_argument('--draft', action='store_true'); ap.add_argument('--no-catalog', action='store_true')
a = ap.parse_args()

if a.series:
    keys = [l.split(':')[0] for l in io.open(os.path.join(ROOT, '_data', 'series.yml'), encoding='utf-8') if l and l[0].isalpha() and l.rstrip().endswith(':')]
    if a.series not in keys: sys.exit('unknown series %r; add it to _data/series.yml first (known: %s)' % (a.series, ', '.join(keys)))
if a.layout == 'keynote' and not a.iframe: sys.exit('--layout keynote needs --iframe <deck url>')

fm = ['layout: ' + a.layout]
if a.category: fm.append('category: ' + a.category)
if a.series: fm.append('series: ' + a.series)
fm.append('title: "%s"' % a.title.replace('"', '\\"'))
if a.subtitle: fm.append('subtitle: "%s"' % a.subtitle.replace('"', '\\"'))
fm.append('tags: [%s]' % ', '.join(t.strip() for t in a.tags.split(',') if t.strip()))
if not a.no_catalog: fm.append('catalog: true')
if a.header_img: fm.append('header-img: ' + a.header_img)
if a.iframe: fm.append('iframe: "%s"' % a.iframe)
if a.draft:
    path = os.path.join(ROOT, '_drafts', a.slug + '.md')
else:
    path = os.path.join(ROOT, '_posts', '%s-%s.md' % (a.date, a.slug))
if os.path.exists(path): sys.exit('exists: ' + path)
os.makedirs(os.path.dirname(path), exist_ok=True)
io.open(path, 'w', encoding='utf-8').write('---\n' + '\n'.join(fm) + '\n---\n\n')
print(path)
