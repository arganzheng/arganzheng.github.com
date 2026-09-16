#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content health report — the *soft* problems `tools/check.sh` does not block on.
Read-only; prints one section per finding with a count and (up to --limit)
examples, so the numbers can be watched shrinking over time.

    tools/audit.py                  # = npm run audit
    tools/audit.py --limit 50       # longer lists
    tools/audit.py --days 60        # window for "edited but no updated:"

Sections
  tags        posts without tags; tags used by one post only; tags that differ
              only by case (Jekyll treats them as different tags)
  subtitle    posts without subtitle (it is the share-card / SEO description fallback)
  titles      duplicate titles
  drafts      _drafts older than 6 months (publish or delete)
  images      non-WebP images in img/in-post referenced by posts; images > 300 KB
  links       bare http:// links in post bodies
  updated     posts with a substantial edit (>= 20 changed lines) in the last
              --days days, in commits touching < 10 posts (bigger ones are
              mechanical sweeps), but no `updated:` at or after that edit (the header
              「更新于」, JSON-LD dateModified and the 「本文写于 N 年前」 notice
              all key off it). Posts younger than --days are skipped.
"""
import argparse, collections, datetime, glob, os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=15)
ap.add_argument('--days', type=int, default=30)
A = ap.parse_args()

def front_matter(path):
    text = open(path, encoding='utf-8').read()
    m = re.match(r'^---\n(.*?)\n---\n?(.*)$', text, re.S)
    if not m: return {}, text
    fm = {}
    for line in m.group(1).splitlines():
        k = re.match(r'^([A-Za-z_-]+):\s*(.*)$', line)
        if k: fm[k.group(1)] = k.group(2).strip().strip('"')
    return fm, m.group(2)

def tags_of(fm):
    t = fm.get('tags', '')
    t = t.strip('[]')
    return [x.strip().strip('"\'') for x in t.split(',') if x.strip()] if t else []

def section(title, items, note=''):
    print('\n== %s: %d%s' % (title, len(items), (' — ' + note) if note else ''))
    for it in items[:A.limit]: print('   ' + it)
    if len(items) > A.limit: print('   … +%d more (--limit)' % (len(items) - A.limit))

posts = sorted(glob.glob(os.path.join(ROOT, '_posts/*.md')))
data = {}
for p in posts:
    fm, body = front_matter(p)
    data[p] = (fm, body)
rel = lambda p: os.path.relpath(p, ROOT)
today = datetime.date.today()

# ---- tags
no_tags = [rel(p) for p, (fm, _) in data.items() if not tags_of(fm)]
section('posts without tags', no_tags, 'never recommended, absent from HOT TAGS')
df = collections.Counter(t for fm, _ in data.values() for t in tags_of(fm))
singles = sorted(t for t, c in df.items() if c == 1)
section('tags used by a single post', singles, 'fine for specific topics; merge the accidental ones')
by_lower = collections.defaultdict(set)
for t in df: by_lower[t.lower()].add(t)
case_dupes = [' / '.join(sorted(v)) for v in by_lower.values() if len(v) > 1]
section('tags differing only by case', case_dupes, 'Jekyll splits them into separate tag pages')

# ---- subtitle / titles
no_sub = [rel(p) for p, (fm, _) in data.items() if not fm.get('subtitle') and not fm.get('description')]
section('posts without subtitle / description', no_sub, 'share cards and search snippets fall back to the first paragraph')
titles = collections.defaultdict(list)
for p, (fm, _) in data.items(): titles[fm.get('title', '')].append(rel(p))
section('duplicate titles', ['%s: %s' % (t, ', '.join(ps)) for t, ps in titles.items() if len(ps) > 1])

# ---- drafts
def last_commit_date(path):
    out = subprocess.run(['git', 'log', '-1', '--format=%cs', '--', path], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return datetime.date.fromisoformat(out) if out else datetime.date.fromtimestamp(os.path.getmtime(path))
old_drafts = []
for d in sorted(glob.glob(os.path.join(ROOT, '_drafts/*.md'))):
    named = re.match(r'(\d{4}-\d{2}-\d{2})', os.path.basename(d))   # a rename resets the git date; the filename does not lie
    age = (today - (datetime.date.fromisoformat(named.group(1)) if named else last_commit_date(d))).days
    if age > 183: old_drafts.append('%s (untouched %d days)' % (rel(d), age))
section('stale drafts (> 6 months)', old_drafts, 'publish or delete')

# ---- images
IMG = re.compile(r'(?:!\[[^\]]*\]\(|<img[^>]+src=["\'])\s*(/?img/[^)\s"\']+)')
refs = collections.defaultdict(set)
for p, (_, body) in data.items():
    for src in IMG.findall(body): refs[src.lstrip('/')].add(rel(p))
non_webp = sorted(s for s in refs if s.startswith('img/in-post/') and not s.lower().endswith(('.webp', '.svg', '.gif'))
                  and os.path.exists(os.path.join(ROOT, s)) and os.path.getsize(os.path.join(ROOT, s)) >= 20 * 1024)
section('non-WebP images >= 20 KB referenced from posts', non_webp, 'tools/webp-images.py --apply (it skips smaller files on purpose)')
big = []
for s in refs:
    f = os.path.join(ROOT, s)
    if os.path.exists(f) and os.path.getsize(f) > 300 * 1024: big.append('%s (%d KB)' % (s, os.path.getsize(f) // 1024))
section('referenced images > 300 KB', sorted(big), 'header images are allowed to be large; content images are not')

# ---- links
HTTP = re.compile(r'(?<![\w/])http://[^\s)>"\']+')
bare = []
for p, (_, body) in data.items():
    n = len(HTTP.findall(body))
    if n: bare.append('%s (%d)' % (rel(p), n))
section('posts with bare http:// links', sorted(bare, key=lambda s: -int(s.rsplit('(', 1)[1][:-1])), 'lychee treats them fine; readers get a mixed-content warning at best')

# ---- edited recently but no updated:
since = today - datetime.timedelta(days=A.days)
log = subprocess.run(['git', 'log', '--since=%s' % since.isoformat(), '--numstat', '--format=@%cs', '--', '_posts'], cwd=ROOT, capture_output=True, text=True).stdout
BULK = 10   # a commit touching this many posts is a mechanical sweep (tags, WebP, formatting), not a revision
commits, cur = [], None
for line in log.splitlines():
    if line.startswith('@'): cur = [datetime.date.fromisoformat(line[1:]), []]; commits.append(cur); continue
    m = re.match(r'^(\d+|-)\t(\d+|-)\t(_posts/.+\.md)$', line)
    if m and cur and m.group(1) != '-': cur[1].append((m.group(3), int(m.group(1)) + int(m.group(2))))
edits = collections.defaultdict(lambda: [0, None])   # path -> [changed lines, latest date]
for when, files in commits:
    if len(files) >= BULK: continue
    for path, n in files:
        e = edits[path]; e[0] += n
        if e[1] is None or when > e[1]: e[1] = when
stale_updated = []
for path, (lines, when) in sorted(edits.items()):
    full = os.path.join(ROOT, path)
    if lines < 20 or not os.path.exists(full): continue
    fm, _ = data.get(full, ({}, ''))
    post_date = re.match(r'(\d{4}-\d{2}-\d{2})', os.path.basename(path))
    if post_date and datetime.date.fromisoformat(post_date.group(1)) >= since: continue
    added = subprocess.run(['git', 'log', '--diff-filter=A', '--follow', '-1', '--format=%cs', '--', path], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if added and datetime.date.fromisoformat(added) >= since: continue   # written (maybe back-dated) inside the window: new, not revised
    upd = fm.get('updated', '')[:10]
    if not upd or datetime.date.fromisoformat(upd) < when:
        stale_updated.append('%s (%d lines on %s, updated: %s)' % (path, lines, when, upd or '—'))
section('edited in the last %d days without a matching updated:' % A.days, stale_updated, 'add `updated: YYYY-MM-DD` if the change was substantive')
print()
