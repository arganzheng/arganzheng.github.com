#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rendered review of post changes — the site's own rendering (Jekyll build, site
CSS, KaTeX, Mermaid) of the old and new version of every changed post, diffed
block by block with word-level marks, grouped by h2/h3 section, served locally.
In PR mode the PR's review threads are shown next to the section they belong
to and comments can be posted from the page.

    tools/review.py 45                  # PR #45 (merge-base .. head)
    tools/review.py origin/master..HEAD # any commit range
    tools/review.py 4d5c0ef             # one commit (= 4d5c0ef^..4d5c0ef)
    tools/review.py                     # working tree vs HEAD
    tools/review.py 45 --notes .review/notes.md
                                        # post the rationale as one PR review
                                        # (one comment per section) and report
                                        # changed sections that have no note

Options: --port 4100  --no-serve  --no-build (reuse existing builds)  --no-open

Notes file (--notes): Markdown, `## <post path>` per file, `### <section title>`
per section (the h2/h3 the change sits under; text before the first `###` is a
file-level note), body = the reason. Each note lands as a review comment on the
first changed line of that section, so it shows up in the review page and on
GitHub alike.

Output: <head build>/_review/index.html (+ one page per post); builds are
cached per commit sha under _site-review/<sha>/ (worktrees in /tmp/blog-review).
"""
import argparse, difflib, html, json, os, re, shutil, subprocess, sys, threading, webbrowser
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, '_site-review')
WT = '/tmp/blog-review'
WORKTREE = 'WORKTREE'
ASSETS = os.path.join(ROOT, 'tools', 'review')
POST_DIRS = ('_posts/', '_drafts/')

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('target', nargs='?', default='')
ap.add_argument('--notes')
ap.add_argument('--port', type=int, default=4100)
ap.add_argument('--no-serve', action='store_true')
ap.add_argument('--no-build', action='store_true')
ap.add_argument('--no-open', action='store_true')
A = ap.parse_args()


def sh(*cmd, cwd=ROOT, check=True, input=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, input=input)
    if check and r.returncode:
        sys.exit('%s\n%s' % (' '.join(cmd), r.stderr.strip()))
    return r.stdout


def gh_json(*args, input=None):
    return json.loads(sh('gh', *args, input=input))


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- target ----
class Target:
    pr = None          # dict from gh pr view
    base = head = None  # shas (head may be WORKTREE)
    repo = None         # owner/name

    def label(self):
        if self.pr: return '#%d %s' % (self.pr['number'], self.pr['title'])
        return '%s..%s' % (self.base[:7], 'working tree' if self.head == WORKTREE else self.head[:7])


def resolve_target(spec):
    t = Target()
    if not spec:
        t.base, t.head = sh('git', 'rev-parse', 'HEAD').strip(), WORKTREE
    elif re.fullmatch(r'\d{1,6}', spec):
        t.pr = gh_json('pr', 'view', spec, '--json',
                       'number,title,body,url,baseRefName,baseRefOid,headRefName,headRefOid,author,isDraft,state')
        t.repo = gh_json('repo', 'view', '--json', 'nameWithOwner')['nameWithOwner']
        for sha, ref in ((t.pr['headRefOid'], 'pull/%d/head' % t.pr['number']), (t.pr['baseRefOid'], t.pr['baseRefName'])):
            if subprocess.run(['git', 'cat-file', '-e', sha + '^{commit}'], cwd=ROOT, capture_output=True).returncode:
                log('fetching %s' % ref)
                sh('git', 'fetch', '-q', 'origin', ref)
        t.head = t.pr['headRefOid']
        t.base = sh('git', 'merge-base', t.pr['baseRefOid'], t.head).strip()
    elif '..' in spec:
        a, b = spec.split('..', 1)
        t.base, t.head = sh('git', 'rev-parse', a or 'HEAD').strip(), sh('git', 'rev-parse', b or 'HEAD').strip()
    else:
        t.head = sh('git', 'rev-parse', spec).strip()
        t.base = sh('git', 'rev-parse', spec + '^').strip()
    return t


# ----------------------------------------------------------------- build ----
def build(sha):
    if sha == WORKTREE:
        src, out = ROOT, os.path.join(OUT, 'worktree')
        if A.no_build and os.path.isdir(out): return out
    else:
        out = os.path.join(OUT, sha)
        if os.path.exists(os.path.join(out, '.done')): return out
        src = os.path.join(WT, sha)
        if not os.path.isdir(src):
            os.makedirs(WT, exist_ok=True)
            sh('git', 'worktree', 'prune')
            sh('git', 'worktree', 'add', '--detach', '-q', src, sha)
    log('building %s -> %s' % ('working tree' if sha == WORKTREE else sha[:7], os.path.relpath(out, ROOT)))
    r = subprocess.run(['bundle', 'exec', 'jekyll', 'build', '--future', '--unpublished', '--drafts', '-q',
                        '-s', src, '-d', out], cwd=src, capture_output=True, text=True)
    if r.returncode or not os.path.isdir(out):
        sys.exit('jekyll build failed for %s:\n%s' % (sha, (r.stderr or r.stdout)[-3000:]))
    if sha != WORKTREE: open(os.path.join(out, '.done'), 'w').close()
    return out


# ------------------------------------------------------------ changed files --
def changed_files(t):
    """[(status, path, old_path)] — status A/M/D/R."""
    out = []
    if t.head == WORKTREE:
        raw = sh('git', 'diff', '--name-status', '-M', t.base)
        for p in sh('git', 'ls-files', '--others', '--exclude-standard').splitlines():
            out.append(('A', p, None))
    else:
        raw = sh('git', 'diff', '--name-status', '-M', t.base, t.head)
    for line in raw.splitlines():
        parts = line.split('\t')
        st = parts[0][0]
        if st == 'R': out.append(('R', parts[2], parts[1]))
        else: out.append((st, parts[1], None))
    return out


def read_at(sha, path):
    if sha == WORKTREE:
        p = os.path.join(ROOT, path)
        return open(p, encoding='utf-8').read() if os.path.exists(p) else None
    r = subprocess.run(['git', 'show', '%s:%s' % (sha, path)], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def front_matter(text):
    m = re.match(r'^---\n(.*?)\n---\n?', text or '', re.S)
    fm = {}
    if not m: return fm, text or '', 0
    for line in m.group(1).splitlines():
        k = re.match(r'^([A-Za-z_-]+):\s*(.*)$', line)
        if k: fm[k.group(1)] = k.group(2).strip().strip('"\'')
    return fm, (text or '')[m.end():], m.group(0).count('\n')


def post_url(path, fm):
    name = os.path.splitext(os.path.basename(path))[0]
    slug = re.sub(r'^\d{4}-\d{1,2}-\d{1,2}-', '', name)
    perm = fm.get('permalink', '')
    if perm and ':title' not in perm:
        return perm if perm.endswith('.html') else perm.rstrip('/') + '/index.html'
    return '/%s.html' % slug


# ------------------------------------------------------------ html blocks ----
VOID = {'img', 'br', 'hr', 'input', 'meta', 'link', 'source', 'wbr', 'col', 'area', 'base', 'embed', 'param', 'track'}


class Splitter(HTMLParser):
    """Top-level element boundaries of an HTML fragment."""
    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self.text, self.depth, self.blocks, self.start = text, 0, [], None
        self.lines = [0]
        for i, ch in enumerate(text):
            if ch == '\n': self.lines.append(i + 1)
        self.feed(text); self.close()
        if self.start is not None: self.blocks.append((self.start, len(text)))

    def off(self):
        l, c = self.getpos(); return self.lines[l - 1] + c

    def handle_starttag(self, tag, attrs):
        if self.depth == 0: self.start = self.off()
        if tag not in VOID: self.depth += 1
        elif self.depth == 0: self._end(len(self.get_starttag_text() or ''))

    def handle_startendtag(self, tag, attrs):
        if self.depth == 0: self.start = self.off(); self._end(len(self.get_starttag_text() or ''))

    def handle_endtag(self, tag):
        if tag in VOID: return
        self.depth = max(0, self.depth - 1)
        if self.depth == 0 and self.start is not None:
            self._end(len(tag) + 3)

    def _end(self, taglen):
        end = self.off() + taglen
        self.blocks.append((self.start, end)); self.start = None


TAG_RE = re.compile(r'<[^>]+>')
WS_RE = re.compile(r'\s+')


def strip_tags(h):
    return html.unescape(WS_RE.sub(' ', TAG_RE.sub(' ', h))).strip()


def first_tag(h):
    m = re.match(r'\s*<([a-zA-Z0-9]+)([^>]*)>', h)
    return (m.group(1).lower(), m.group(2)) if m else ('', '')


class Block:
    def __init__(self, h):
        self.html = h.strip()
        self.tag, attrs = first_tag(self.html)
        self.text = strip_tags(self.html)
        cls = re.search(r'class="([^"]*)"', attrs)
        cls = cls.group(1) if cls else ''
        if self.tag == 'pre' or 'highlighter-rouge' in cls or 'language-' in cls: self.kind = 'code'
        elif self.tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'): self.kind = 'heading'
        elif self.tag in ('ul', 'ol'): self.kind = 'list'
        elif self.tag == 'table': self.kind = 'table'
        else: self.kind = 'other'
        self.mermaid = self.kind == 'code' and 'language-mermaid' in self.html[:300]
        self.section = ''   # nearest h2/h3 above (text), filled by caller
        self.hid = ''       # heading id when this is a heading


def split_blocks(fragment):
    sp = Splitter(fragment)
    blocks = []
    section = ''
    for a, b in sp.blocks:
        blk = Block(fragment[a:b])
        if not blk.text and blk.tag not in ('img', 'hr', 'figure', 'div', 'svg', 'table', 'iframe', 'video', 'p'):
            continue
        if blk.kind == 'heading' and blk.tag in ('h2', 'h3'):
            section = blk.text
            m = re.search(r'id="([^"]*)"', blk.html[:200]); blk.hid = m.group(1) if m else ''
        blk.section = section if not (blk.kind == 'heading' and blk.tag in ('h2', 'h3')) else blk.text
        blocks.append(blk)
    return blocks


def extract_article(page_html):
    if page_html is None: return None
    m = re.search(r'<!-- article -->(.*?)<!-- /article -->', page_html, re.S)
    if m: return m.group(1)
    # older builds without the markers: cut inside .post-container
    m = re.search(r'<div class="[^"]*post-container">(.*)', page_html, re.S)
    if not m: return None
    body = m.group(1)
    for end in ('<div class="series-toc">', '<div class="post-license">', '<hr style="visibility: hidden;">'):
        i = body.find(end)
        if i != -1: body = body[:i]
    body = re.sub(r'^\s*<div class="post-stale".*?</div>\s*', '', body, flags=re.S)
    body = re.sub(r'^\s*<blockquote class="series-nav">.*?</blockquote>\s*', '', body, flags=re.S)
    return body


# ------------------------------------------------------------- word diff -----
# A token is (tags_before, word). Tags are never diffed; they ride with the
# next word (or the tail) and are emitted from whichever side is being rendered,
# so each side's markup stays balanced. Math \( \) / \[ \] is one atomic token.
TOKEN_RE = re.compile(
    r'(?P<tag><[^>]+>)|(?P<math>\\\(.*?\\\)|\\\[.*?\\\])|(?P<ent>&[#\w]+;)'
    r'|(?P<word>[A-Za-z0-9_]+|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])|(?P<ws>\s+)|(?P<other>.)', re.S)


def tokenize(h):
    toks, pend = [], ''
    for m in TOKEN_RE.finditer(h):
        if m.group('tag'): pend += m.group(0)
        else: toks.append((pend, m.group(0))); pend = ''
    return toks, pend


def word_diff(old_html, new_html):
    """-> (unified, left, right, change_ratio)."""
    ot, otail = tokenize(old_html)
    nt, ntail = tokenize(new_html)
    ow, nw = [w for _, w in ot], [w for _, w in nt]
    sm = difflib.SequenceMatcher(None, ow, nw, autojunk=False)
    uni, left, right = [], [], []
    changed = 0

    def run(tokens, i1, i2, wrap, into, keep_tags=True):
        if i1 >= i2: return
        def emit(buf):
            s = ''.join(buf)
            into.append('<%s>%s</%s>' % (wrap, s, wrap) if wrap and s.strip() else s)
        if keep_tags: into.append(tokens[i1][0])
        buf = []
        for k in range(i1, i2):
            tags, w = tokens[k]
            if k > i1 and tags and keep_tags:
                emit(buf); buf = []; into.append(tags)
            buf.append(w)
        emit(buf)

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for side, toks, a, b in ((uni, nt, j1, j2), (right, nt, j1, j2), (left, ot, i1, i2)):
                run(toks, a, b, '', side)
            continue
        changed += max(i2 - i1, j2 - j1)
        if op in ('delete', 'replace'):
            run(ot, i1, i2, 'del', left)
            run(ot, i1, i2, 'del', uni, keep_tags=False)
        if op in ('insert', 'replace'):
            run(nt, j1, j2, 'ins', right)
            run(nt, j1, j2, 'ins', uni)
    ratio = changed / max(1, max(len(ow), len(nw)))
    return ''.join(uni) + ntail, ''.join(left) + otail, ''.join(right) + ntail, ratio


def line_diff(old_text, new_text):
    ol, nl = old_text.split('\n'), new_text.split('\n')
    sm = difflib.SequenceMatcher(None, ol, nl, autojunk=False)
    uni, left, right = [], [], []
    e = html.escape
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(i1, i2):
                s = e(ol[k]) + '\n'; uni.append(s); left.append(s); right.append(s)
            continue
        for k in range(i1, i2):
            s = '<del>%s</del>\n' % e(ol[k]); uni.append(s); left.append(s)
        for k in range(j1, j2):
            s = '<ins>%s</ins>\n' % e(nl[k]); uni.append(s); right.append(s)
    w = lambda parts: '<pre class="rv-code">%s</pre>' % ''.join(parts)
    return w(uni), w(left), w(right)


def code_text(block_html):
    m = re.search(r'<code[^>]*>(.*?)</code>', block_html, re.S)
    return html.unescape(TAG_RE.sub('', m.group(1) if m else block_html)).rstrip('\n')


def children(h, tag):
    """Direct-ish children <tag>…</tag> of a container as a list of html strings."""
    return re.findall(r'<%s\b[^>]*>.*?</%s>' % (tag, tag), h, re.S)


def diff_rows(old_rows, new_rows, cell_tag):
    """Align rows by text, cells by index. -> (uni, left, right) lists of row html."""
    ot, nt = [strip_tags(r) for r in old_rows], [strip_tags(r) for r in new_rows]
    sm = difflib.SequenceMatcher(None, ot, nt, autojunk=False)
    uni, left, right = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(j1, j2): uni.append(new_rows[k]); right.append(new_rows[k])
            for k in range(i1, i2): left.append(old_rows[k])
        elif op == 'replace' and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                o, n = old_rows[i1 + k], new_rows[j1 + k]
                oc, nc = children(o, cell_tag), children(n, cell_tag)
                if len(oc) != len(nc) or not oc:
                    left.append(mark_row(o, 'rv-row-del')); right.append(mark_row(n, 'rv-row-ins'))
                    uni.append(mark_row(o, 'rv-row-del')); uni.append(mark_row(n, 'rv-row-ins')); continue
                cu, cl, cr = [], [], []
                for a, b in zip(oc, nc):
                    inner_a = re.sub(r'^<[^>]+>|</[^>]+>$', '', a.strip()); inner_b = re.sub(r'^<[^>]+>|</[^>]+>$', '', b.strip())
                    u, l, r, _ = word_diff(inner_a, inner_b)
                    open_b = re.match(r'<[^>]+>', b.strip()).group(0); open_a = re.match(r'<[^>]+>', a.strip()).group(0)
                    cu.append(open_b + u + '</%s>' % cell_tag); cl.append(open_a + l + '</%s>' % cell_tag); cr.append(open_b + r + '</%s>' % cell_tag)
                tr_open_n = re.match(r'<tr[^>]*>', n.strip()).group(0); tr_open_o = re.match(r'<tr[^>]*>', o.strip()).group(0)
                uni.append(tr_open_n + ''.join(cu) + '</tr>'); left.append(tr_open_o + ''.join(cl) + '</tr>'); right.append(tr_open_n + ''.join(cr) + '</tr>')
        else:
            for k in range(i1, i2): left.append(mark_row(old_rows[k], 'rv-row-del')); uni.append(mark_row(old_rows[k], 'rv-row-del'))
            for k in range(j1, j2): right.append(mark_row(new_rows[k], 'rv-row-ins')); uni.append(mark_row(new_rows[k], 'rv-row-ins'))
    return uni, left, right


def mark_row(row, cls):
    return re.sub(r'^<(tr|li)\b', r'<\1 class="%s"' % cls, row.strip(), count=1)


def diff_table(o, n):
    parts = {}
    for side in ('uni', 'left', 'right'): parts[side] = []
    for sec in ('thead', 'tbody'):
        os_, ns = re.search(r'<%s>(.*?)</%s>' % (sec, sec), o, re.S), re.search(r'<%s>(.*?)</%s>' % (sec, sec), n, re.S)
        if not (os_ or ns): continue
        u, l, r = diff_rows(children(os_.group(1), 'tr') if os_ else [], children(ns.group(1), 'tr') if ns else [], 'th' if sec == 'thead' else 'td')
        parts['uni'].append('<%s>%s</%s>' % (sec, ''.join(u), sec))
        parts['left'].append('<%s>%s</%s>' % (sec, ''.join(l), sec))
        parts['right'].append('<%s>%s</%s>' % (sec, ''.join(r), sec))
    wrap = lambda p: '<table>%s</table>' % ''.join(p)
    return wrap(parts['uni']), wrap(parts['left']), wrap(parts['right'])


def diff_list(o, n):
    tag = first_tag(n)[0] or 'ul'
    oi, ni = children(o, 'li'), children(n, 'li')
    ot, nt = [strip_tags(x) for x in oi], [strip_tags(x) for x in ni]
    sm = difflib.SequenceMatcher(None, ot, nt, autojunk=False)
    uni, left, right = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(j1, j2): uni.append(ni[k]); right.append(ni[k])
            for k in range(i1, i2): left.append(oi[k])
        elif op == 'replace':
            pairs = pair_up([Block(x) for x in oi[i1:i2]], [Block(x) for x in ni[j1:j2]])
            for a, b in pairs:
                if a and b:
                    ia = re.sub(r'^<li[^>]*>|</li>$', '', a.html); ib = re.sub(r'^<li[^>]*>|</li>$', '', b.html)
                    u, l, r, _ = word_diff(ia, ib)
                    uni.append('<li>%s</li>' % u); left.append('<li>%s</li>' % l); right.append('<li>%s</li>' % r)
                elif a: left.append(mark_row(a.html, 'rv-row-del')); uni.append(mark_row(a.html, 'rv-row-del'))
                else: right.append(mark_row(b.html, 'rv-row-ins')); uni.append(mark_row(b.html, 'rv-row-ins'))
        else:
            for k in range(i1, i2): left.append(mark_row(oi[k], 'rv-row-del')); uni.append(mark_row(oi[k], 'rv-row-del'))
            for k in range(j1, j2): right.append(mark_row(ni[k], 'rv-row-ins')); uni.append(mark_row(ni[k], 'rv-row-ins'))
    w = lambda p: '<%s>%s</%s>' % (tag, ''.join(p), tag)
    return w(uni), w(left), w(right)


# ------------------------------------------------------------- alignment -----
def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def pair_up(old, new, threshold=0.45):
    """Monotone pairing of two short block lists by text similarity (DP)."""
    n, m = len(old), len(new)
    if not n: return [(None, b) for b in new]
    if not m: return [(a, None) for a in old]
    sim = [[ratio(a.text, b.text) if a.kind == b.kind or {a.kind, b.kind} <= {'other', 'list'} else 0 for b in new] for a in old]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best[i][j] = max(best[i + 1][j], best[i][j + 1],
                             (sim[i][j] + best[i + 1][j + 1]) if sim[i][j] >= threshold else 0)
    out, i, j = [], 0, 0
    while i < n and j < m:
        if sim[i][j] >= threshold and best[i][j] == sim[i][j] + best[i + 1][j + 1]:
            out.append((old[i], new[j])); i += 1; j += 1
        elif best[i][j] == best[i + 1][j]: out.append((old[i], None)); i += 1
        else: out.append((None, new[j])); j += 1
    out += [(a, None) for a in old[i:]] + [(None, b) for b in new[j:]]
    return out


def align(old_blocks, new_blocks):
    """-> list of (kind, old, new): kind in equal/change/delete/insert."""
    sm = difflib.SequenceMatcher(None, [b.text for b in old_blocks], [b.text for b in new_blocks], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1):
                a, b = old_blocks[i1 + k], new_blocks[j1 + k]
                out.append(('equal' if a.html == b.html else 'change', a, b))
        elif op == 'delete': out += [('delete', a, None) for a in old_blocks[i1:i2]]
        elif op == 'insert': out += [('insert', None, b) for b in new_blocks[j1:j2]]
        else:
            for a, b in pair_up(old_blocks[i1:i2], new_blocks[j1:j2]):
                out.append(('change' if a and b else 'delete' if a else 'insert', a, b))
    # a lone delete right before a lone insert is one replacement (old | new on one row in split mode)
    merged, i = [], 0
    while i < len(out):
        if i + 1 < len(out) and out[i][0] == 'delete' and out[i + 1][0] == 'insert':
            merged.append(('replace', out[i][1], out[i + 1][2])); i += 2
        else: merged.append(out[i]); i += 1
    return merged


# ------------------------------------------------------------- rendering -----
def render_pair(kind, a, b):
    """-> (unified_html, left_html, right_html, section)"""
    if kind == 'equal':
        return b.html, a.html, b.html
    if kind == 'delete':
        return '<div class="rv-blk rv-del">%s</div>' % a.html, '<div class="rv-blk rv-del">%s</div>' % a.html, '<div class="rv-blk rv-gap"></div>'
    if kind == 'insert':
        return '<div class="rv-blk rv-ins">%s</div>' % b.html, '<div class="rv-blk rv-gap"></div>', '<div class="rv-blk rv-ins">%s</div>' % b.html
    if kind == 'replace':
        return ('<div class="rv-blk rv-del">%s</div><div class="rv-blk rv-ins">%s</div>' % (a.html, b.html),
                '<div class="rv-blk rv-del">%s</div>' % a.html, '<div class="rv-blk rv-ins">%s</div>' % b.html)
    # change
    if a.kind == 'code' and b.kind == 'code':
        u, l, r = line_diff(code_text(a.html), code_text(b.html))
        if b.mermaid or a.mermaid:
            u = u + '<div class="rv-render rv-render-old">%s</div><div class="rv-render rv-render-new">%s</div>' % (a.html, b.html)
            l = l + '<div class="rv-render">%s</div>' % a.html; r = r + '<div class="rv-render">%s</div>' % b.html
        return wrap_chg(u), wrap_chg(l), wrap_chg(r)
    if a.kind == 'table' and b.kind == 'table':
        u, l, r = diff_table(a.html, b.html); return wrap_chg(u), wrap_chg(l), wrap_chg(r)
    if a.kind == 'list' and b.kind == 'list' and a.tag == b.tag:
        u, l, r = diff_list(a.html, b.html); return wrap_chg(u), wrap_chg(l), wrap_chg(r)
    if a.tag == b.tag and a.kind in ('other', 'heading') and not re.search(r'<(table|ul|ol|pre|figure|svg)\b', a.html + b.html):
        oi = re.sub(r'^<[^>]+>|</[^>]+>$', '', a.html, flags=re.S); ni = re.sub(r'^<[^>]+>|</[^>]+>$', '', b.html, flags=re.S)
        u, l, r, chg = word_diff(oi, ni)
        if chg <= 0.6:
            opn = re.match(r'<[^>]+>', b.html).group(0); opa = re.match(r'<[^>]+>', a.html).group(0); cl = '</%s>' % b.tag
            return wrap_chg(opn + u + cl), wrap_chg(opa + l + cl), wrap_chg(opn + r + cl)
    # whole-block replacement
    return ('<div class="rv-blk rv-del">%s</div><div class="rv-blk rv-ins">%s</div>' % (a.html, b.html),
            '<div class="rv-blk rv-del">%s</div>' % a.html, '<div class="rv-blk rv-ins">%s</div>' % b.html)


def wrap_chg(h):
    return '<div class="rv-blk rv-chg">%s</div>' % h


def norm_heading(s):
    s = re.sub(r'\{#[^}]*\}\s*$', '', s or '')
    s = re.sub(r'[*_`#\[\]()（）:：、,，.。!！?？\s]+', '', s)
    s = re.sub(r'^[一二三四五六七八九十\d]+', '', s)
    return s.lower()


def md_sections(md_body, fm_lines):
    """[(line_no (1-based in file), heading text)] from markdown."""
    out, fence = [], False
    for i, line in enumerate(md_body.split('\n'), start=fm_lines + 1):
        if re.match(r'^\s*(```|~~~)', line): fence = not fence; continue
        if fence: continue
        m = re.match(r'^(##|###)\s+(.*?)\s*$', line)
        if m: out.append((i, m.group(2)))
    return out


def section_of_line(sections, line):
    cur = ''
    for ln, title in sections:
        if ln <= line: cur = title
        else: break
    return cur


def hunks(t, path):
    """Head-side (line, side) anchors of every hunk: [(new_start, new_count, old_start, old_count)]"""
    args = ['git', 'diff', '-U0', t.base] + ([] if t.head == WORKTREE else [t.head]) + ['--', path]
    out = []
    for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', sh(*args, check=False), re.M):
        out.append((int(m.group(3)), int(m.group(4) or 1), int(m.group(1)), int(m.group(2) or 1)))
    return out


def anchor_for_section(t, path, sections, section, fm_lines):
    """First diff hunk inside a section -> (line, side) for a PR line comment."""
    starts = [ln for ln, _ in sections]
    lo, hi = fm_lines + 1, 10 ** 9
    if section:
        for idx, (ln, title) in enumerate(sections):
            if norm_heading(title) == norm_heading(section):
                lo = ln; hi = starts[idx + 1] if idx + 1 < len(starts) else 10 ** 9; break
        else:
            return None
    for ns, nc, os_, oc in hunks(t, path):
        if lo <= ns < hi or (nc == 0 and lo <= ns + 1 < hi):
            return (ns, 'RIGHT') if nc else (os_, 'LEFT')
    return None


# ------------------------------------------------------------ PR threads -----
def load_threads(t):
    if not t.pr: return []
    owner, name = t.repo.split('/')
    q = '''query($owner:String!,$name:String!,$n:Int!){ repository(owner:$owner,name:$name){ pullRequest(number:$n){
      reviewThreads(first:100){ nodes{ id isResolved isOutdated path line originalLine
        comments(first:50){ nodes{ databaseId author{login avatarUrl} body createdAt url } } } } } } }'''
    data = gh_json('api', 'graphql', '-f', 'query=' + q, '-F', 'owner=' + owner, '-F', 'name=' + name, '-F', 'n=%d' % t.pr['number'])
    return data['data']['repository']['pullRequest']['reviewThreads']['nodes']


def md_inline(s):
    s = html.escape(s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = re.sub(r'(?<![">])(https?://[^\s<]+)', r'<a href="\1" target="_blank" rel="noopener">\1</a>', s)
    return '<p>' + s.replace('\n\n', '</p><p>').replace('\n', '<br>') + '</p>'


def thread_html(th):
    cs = th['comments']['nodes']
    if not cs: return ''
    items = ''.join(
        '<div class="rv-c"><img src="%s" alt=""><div><b>%s</b> <time>%s</time>%s</div></div>' % (
            html.escape((c['author'] or {}).get('avatarUrl', '')), html.escape((c['author'] or {}).get('login', '?')),
            c['createdAt'][:10], md_inline(c['body'])) for c in cs)
    state = ('已解决' if th['isResolved'] else '') + (' · 已过时' if th['isOutdated'] else '')
    return ('<div class="rv-thread%s" data-thread="%s" data-first="%s">%s<div class="rv-thread-foot">%s'
            '<a href="%s" target="_blank" rel="noopener">GitHub ↗</a> <button type="button" class="rv-reply">回复</button></div></div>') % (
        ' is-resolved' if th['isResolved'] else '', th['id'], cs[0]['databaseId'], items,
        ('<span class="rv-state">%s</span> ' % state) if state.strip(' ·') else '', html.escape(cs[0]['url']))


# ------------------------------------------------------------- page build ----
def page_shell(head_page_html):
    """<head> of the real built page, minus analytics / search, plus review assets."""
    m = re.search(r'<head>(.*?)</head>', head_page_html, re.S)
    head = m.group(1) if m else ''
    head = re.sub(r'<script[^>]*cloudflareinsights[^>]*></script>', '', head)
    head = re.sub(r'<link rel="canonical"[^>]*>', '', head)
    head = re.sub(r'<script type="application/ld\+json">.*?</script>', '', head, flags=re.S)
    return head


def build_post_page(t, ctx, status, path, old_path):
    base_dir, head_dir = ctx['base_dir'], ctx['head_dir']
    old_md = read_at(t.base, old_path or path) if status != 'A' else None
    new_md = read_at(t.head, path) if status != 'D' else None
    ofm, _, _ = front_matter(old_md); nfm, nbody, fm_lines = front_matter(new_md)
    url_old = post_url(old_path or path, ofm) if old_md is not None else None
    url_new = post_url(path, nfm) if new_md is not None else None
    old_page = open(os.path.join(base_dir, url_old.lstrip('/')), encoding='utf-8').read() if url_old and os.path.exists(os.path.join(base_dir, url_old.lstrip('/'))) else None
    new_page = open(os.path.join(head_dir, url_new.lstrip('/')), encoding='utf-8').read() if url_new and os.path.exists(os.path.join(head_dir, url_new.lstrip('/'))) else None
    if old_md is not None and old_page is None: log('  ! %s: no built page for old version (%s)' % (path, url_old))
    if new_md is not None and new_page is None: log('  ! %s: no built page for new version (%s) — draft or unpublished?' % (path, url_new))
    old_blocks = split_blocks(extract_article(old_page) or '')
    new_blocks = split_blocks(extract_article(new_page) or '')
    pairs = align(old_blocks, new_blocks)

    # sections from markdown (for thread placement) and from html
    sections = md_sections(nbody, fm_lines)
    threads_by_sec = {}
    for th in ctx['threads']:
        if th['path'] != path: continue
        ln = th['line'] or th['originalLine']
        key = norm_heading(section_of_line(sections, ln)) if ln else '__outdated__'
        threads_by_sec.setdefault(key, []).append(th)

    changed_secs, seen = [], set()
    for kind, a, b in pairs:
        if kind == 'equal': continue
        sec = (b or a).section
        if norm_heading(sec) not in seen: seen.add(norm_heading(sec)); changed_secs.append(sec)
    fm_changed = sorted(k for k in set(ofm) | set(nfm) if ofm.get(k) != nfm.get(k))
    if fm_changed and '' not in seen: changed_secs.insert(0, ''); seen.add('')

    # body: fold unchanged runs, keep 1 context block + all h2/h3
    body = []
    n = len(pairs)
    keep = [False] * n
    for i, (kind, a, b) in enumerate(pairs):
        if kind != 'equal':
            for k in (i - 1, i, i + 1):
                if 0 <= k < n: keep[k] = True
        elif (b or a).kind == 'heading' and (b or a).tag in ('h2', 'h3') and norm_heading((b or a).text) in seen: keep[i] = True
    notes_done = set()
    i = 0
    while i < n:
        if not keep[i]:
            j = i
            while j < n and not keep[j]: j += 1
            inner = ''.join('<div class="rv-pair rv-eq"><div class="rv-u">%s</div><div class="rv-l">%s</div><div class="rv-r">%s</div></div>' % render_pair(*pairs[k]) for k in range(i, j))
            body.append('<details class="rv-fold"><summary>… %d 段未变</summary>%s</details>' % (j - i, inner))
            i = j; continue
        kind, a, b = pairs[i]
        sec = (b or a).section
        key = norm_heading(sec)
        if kind != 'equal' and key not in notes_done:
            notes_done.add(key)
            body.append(notes_card(t, path, sec, threads_by_sec.pop(key, []), sections, fm_lines))
        u, l, r = render_pair(kind, a, b)
        body.append('<div class="rv-pair rv-%s" data-section="%s"><div class="rv-u">%s</div><div class="rv-l">%s</div><div class="rv-r">%s</div></div>' % (
            kind[:2] if kind != 'equal' else 'eq', html.escape(sec), u, l, r))
        i += 1

    # front matter + file-level / outdated / unmatched threads at the top
    top = []
    if fm_changed:
        rows = ''.join('<tr><th>%s</th><td><del>%s</del></td><td><ins>%s</ins></td></tr>' % (html.escape(k), html.escape(ofm.get(k, '')), html.escape(nfm.get(k, ''))) for k in fm_changed)
        top.append('<table class="rv-fm"><thead><tr><th>front matter</th><th>旧</th><th>新</th></tr></thead><tbody>%s</tbody></table>' % rows)
    if '' not in notes_done and (threads_by_sec.get('') or fm_changed):
        top.append(notes_card(t, path, '', threads_by_sec.pop('', []), sections, fm_lines)); notes_done.add('')
    leftovers = [th for k, v in threads_by_sec.items() for th in v]
    if leftovers:
        top.append('<div class="rv-notes"><div class="rv-notes-h">其他评论（过时或未能归到章节）</div>%s</div>' % ''.join(thread_html(th) for th in leftovers))

    title = nfm.get('title') or ofm.get('title') or path
    sub = nfm.get('subtitle', '')
    stats = {'path': path, 'title': title, 'status': status, 'url': url_new or url_old,
             'changed': sum(1 for k, _, _ in pairs if k != 'equal'), 'sections': changed_secs,
             'threads': sum(1 for th in ctx['threads'] if th['path'] == path),
             'open': sum(1 for th in ctx['threads'] if th['path'] == path and not th['isResolved']),
             'file': slug_file(path)}
    shell = page_shell(new_page or old_page or '')
    doc = TEMPLATE.format(
        head=shell, title=html.escape(title), sub=html.escape(sub), path=html.escape(path),
        label=html.escape(t.label()), status={'A': '新文章', 'D': '已删除', 'R': '改名', 'M': ''}[status],
        pr=json.dumps({'number': t.pr['number'], 'repo': t.repo, 'head': t.head, 'path': path} if t.pr else None),
        top=''.join(top), body=''.join(body), url=html.escape((url_new or url_old or '')),
        nav='<a href="index.html">← 全部改动</a>', stats_json=json.dumps(stats, ensure_ascii=False), cls='rv-post')
    with open(os.path.join(ctx['out'], stats['file']), 'w', encoding='utf-8') as f: f.write(doc)
    return stats


def slug_file(path):
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', os.path.splitext(os.path.basename(path))[0]) + '.html'


def notes_card(t, path, sec, threads, sections, fm_lines):
    anchor = anchor_for_section(t, path, sections, sec, fm_lines) if t.pr else None
    head = '<span class="rv-sec">%s</span>' % html.escape(sec or '文章开头 / front matter')
    if t.pr and not threads: head += ' <span class="rv-nonote">无说明</span>'
    form = ''
    if t.pr:
        form = ('<form class="rv-form" data-path="%s" data-line="%s" data-side="%s"><textarea rows="2" placeholder="对这一节的意见……（发到 PR 的对应行）"></textarea>'
                '<div><button type="submit"%s>发评论</button><span class="rv-msg"></span></div></form>') % (
            html.escape(path), anchor[0] if anchor else '', anchor[1] if anchor else '', '' if anchor else ' disabled title="这一节在 diff 里没有可挂评论的行"')
    return '<div class="rv-notes" data-section="%s"><div class="rv-notes-h">%s</div>%s%s</div>' % (
        html.escape(sec), head, ''.join(thread_html(th) for th in threads), form)


def build_index(t, ctx, posts, others):
    rows = ''.join(
        '<li><a href="%s">%s</a> <small>%s · %d 处改动 · %d 节%s</small>%s</li>' % (
            s['file'], html.escape(s['title']), {'A': '新文章', 'D': '已删除', 'R': '改名', 'M': '修改'}[s['status']],
            s['changed'], len(s['sections']),
            (' · %d 条讨论（%d 未解决）' % (s['threads'], s['open'])) if t.pr else '',
            ('<div class="rv-idx-secs">%s</div>' % ' · '.join(html.escape(x or '开头') for x in s['sections'])) if s['sections'] else '')
        for s in posts)
    other_html = ''.join('<details><summary><code>%s</code> <small>%s</small></summary><pre class="rv-raw">%s</pre></details>' % (
        html.escape(p), st, html.escape(d)) for st, p, d in others)
    pr = ''
    if t.pr:
        pr = '<div class="rv-pr"><a href="%s" target="_blank" rel="noopener">%s ↗</a> <small>%s → %s · %s%s</small><div class="rv-pr-body">%s</div>' \
             '<div class="rv-pr-actions"><button type="button" data-event="APPROVE">批准</button> <button type="button" data-event="REQUEST_CHANGES">要求修改</button> <span class="rv-msg"></span></div></div>' % (
            html.escape(t.pr['url']), html.escape(t.label()), html.escape(t.pr['headRefName']), html.escape(t.pr['baseRefName']),
            html.escape(t.pr['author']['login']), ' · draft' if t.pr['isDraft'] else '', md_inline(t.pr['body'] or ''))
    shell = page_shell(ctx['shell_page'])
    doc = TEMPLATE.format(
        head=shell, title='Review: ' + html.escape(t.label()), sub='', path='', label=html.escape(t.label()), status='',
        pr=json.dumps({'number': t.pr['number'], 'repo': t.repo, 'head': t.head, 'path': ''} if t.pr else None),
        top=pr, body='<h2>文章（%d）</h2><ul class="rv-idx">%s</ul>%s' % (len(posts), rows, ('<h2>其他文件（%d）</h2>%s' % (len(others), other_html)) if others else ''),
        url='', nav='', stats_json='null', cls='rv-index')
    with open(os.path.join(ctx['out'], 'index.html'), 'w', encoding='utf-8') as f: f.write(doc)


TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN"><head>{head}
<link rel="stylesheet" href="review.css">
<title>{title} · review</title>
</head>
<body class="rv rv-mode-unified {cls}">
<div class="rv-bar">
  <div class="rv-bar-l">{nav} <span class="rv-label">{label}</span></div>
  <div class="rv-bar-r">
    <label class="rv-toggle"><input type="radio" name="mode" value="unified" checked> 修订标注</label>
    <label class="rv-toggle"><input type="radio" name="mode" value="split"> 左右并排</label>
    <label class="rv-toggle"><input type="checkbox" id="rv-expand"> 展开未变段落</label>
  </div>
</div>
<div class="rv-head"><span class="rv-status">{status}</span><h1>{title}</h1><p class="rv-sub">{sub}</p><p class="rv-path"><code>{path}</code> <a class="rv-open" href="{url}" target="_blank">打开新版页面 ↗</a></p></div>
<div class="rv-cols"><div class="rv-colhead"><span>旧</span><span>新</span></div></div>
<div class="post-container rv-body">{top}{body}</div>
<script>window.RV = {{ pr: {pr}, stats: {stats_json} }};</script>
<script src="/js/code-tokens.js"></script>
<script src="/js/inline-popups.js"></script>
<script src="review.js"></script>
</body></html>'''


# ---------------------------------------------------------------- notes ------
def parse_notes(path):
    """-> [(post_path, section or '', body)]"""
    out, cur_file, cur_sec, buf = [], None, '', []
    def flush():
        if cur_file and ''.join(buf).strip(): out.append((cur_file, cur_sec, '\n'.join(buf).strip()))
    for line in open(path, encoding='utf-8').read().split('\n'):
        m2 = re.match(r'^##\s+(?!#)(.+?)\s*$', line); m3 = re.match(r'^###\s+(.+?)\s*$', line)
        if m2: flush(); cur_file, cur_sec, buf = m2.group(1).strip('`'), '', []
        elif m3: flush(); cur_sec, buf = m3.group(1), []
        else: buf.append(line)
    flush()
    return out


def post_notes(t, notes, post_paths):
    comments, problems = [], []
    for path, sec, body in notes:
        if path not in post_paths: problems.append('%s: not a changed post in this PR' % path); continue
        new_md = read_at(t.head, path); _, nbody, fm_lines = front_matter(new_md)
        anchor = anchor_for_section(t, path, md_sections(nbody, fm_lines), sec, fm_lines)
        if not anchor: problems.append('%s › %s: no changed line under that section' % (path, sec or '(top)')); continue
        comments.append({'path': path, 'line': anchor[0], 'side': anchor[1], 'body': body})
    if problems: log('\n'.join('  ! ' + p for p in problems))
    if not comments: sys.exit('no postable notes')
    payload = json.dumps({'commit_id': t.head, 'event': 'COMMENT', 'body': '修改理由（每节一条，由 tools/review.py --notes 发出）', 'comments': comments})
    gh_json('api', 'repos/%s/pulls/%d/reviews' % (t.repo, t.pr['number']), '--input', '-', input=payload)
    log('posted %d note(s) to PR #%d' % (len(comments), t.pr['number']))
    if problems: sys.exit(1)


# --------------------------------------------------------------- server -----
class Handler(SimpleHTTPRequestHandler):
    t = None

    def log_message(self, *a): pass

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(n) or b'{}')
        try:
            if self.path == '/_api/comment': out = self.comment(req)
            elif self.path == '/_api/review': out = self.review(req)
            else: self.send_error(404); return
            body = json.dumps(out).encode()
            self.send_response(200)
        except SystemExit as e:
            body = json.dumps({'error': str(e)}).encode(); self.send_response(500)
        self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def comment(self, req):
        t = self.t
        if not t.pr: sys.exit('not a PR')
        base = 'repos/%s/pulls/%d/comments' % (t.repo, t.pr['number'])
        if req.get('reply_to'):
            return gh_json('api', '%s/%s/replies' % (base, req['reply_to']), '-f', 'body=' + req['body'])
        return gh_json('api', base, '-f', 'body=' + req['body'], '-f', 'commit_id=' + t.head, '-f', 'path=' + req['path'],
                       '-F', 'line=%d' % int(req['line']), '-f', 'side=' + req.get('side', 'RIGHT'))

    def review(self, req):
        t = self.t
        flag = {'APPROVE': '--approve', 'REQUEST_CHANGES': '--request-changes', 'COMMENT': '--comment'}[req['event']]
        sh('gh', 'pr', 'review', str(t.pr['number']), flag, '-b', req.get('body') or '')
        return {'ok': True}


# ----------------------------------------------------------------- main ------
def main():
    t = resolve_target(A.target)
    log('review: %s' % t.label())
    files = changed_files(t)
    posts = [(st, p, op) for st, p, op in files if p.startswith(POST_DIRS) and p.endswith(('.md', '.markdown'))]
    if not posts: log('no changed posts'); 
    if A.notes:
        if not t.pr: sys.exit('--notes needs a PR number')
        post_notes(t, parse_notes(A.notes), {p for _, p, _ in posts})
        return
    base_dir = build(t.base) if posts else None
    head_dir = build(t.head)
    out = os.path.join(head_dir, '_review'); os.makedirs(out, exist_ok=True)
    for f in ('review.css', 'review.js'): shutil.copy(os.path.join(ASSETS, f), out)
    threads = load_threads(t)
    ctx = {'base_dir': base_dir, 'head_dir': head_dir, 'out': out, 'threads': threads, 'shell_page': ''}
    stats = []
    for st, p, op in posts:
        s = build_post_page(t, ctx, st, p, op); stats.append(s)
        if not ctx['shell_page']:
            fp = os.path.join(head_dir if st != 'D' else base_dir, s['url'].lstrip('/'))
            if os.path.exists(fp): ctx['shell_page'] = open(fp, encoding='utf-8').read()
        log('  %s  %d changed block(s) in %d section(s)%s' % (p, s['changed'], len(s['sections']),
            '' if not t.pr else ' · %d thread(s)' % s['threads']))
        if t.pr:
            noted = {norm_heading(section_of_line(md_sections(*front_matter(read_at(t.head, p))[1:]), th['line'] or th['originalLine'] or 0)) for th in threads if th['path'] == p}
            missing = [s2 for s2 in s['sections'] if norm_heading(s2) not in noted]
            if missing: log('    无说明: ' + ' · '.join(x or '(开头)' for x in missing))
    if not ctx['shell_page']:
        any_page = next((os.path.join(head_dir, f) for f in os.listdir(head_dir) if f.endswith('.html') and f not in ('index.html', '404.html')), None)
        ctx['shell_page'] = open(any_page, encoding='utf-8').read() if any_page else ''
    others = []
    for st, p, op in files:
        if (st, p, op) in posts: continue
        args = ['git', 'diff', t.base] + ([] if t.head == WORKTREE else [t.head]) + ['--', p]
        d = sh(*args, check=False)
        if not d and t.head == WORKTREE and os.path.isfile(os.path.join(ROOT, p)):
            d = sh('git', 'diff', '--no-index', '--', '/dev/null', p, check=False)
        if len(d) > 60000: d = d[:60000] + '\n… (truncated)'
        others.append((st, p, d or '(binary)'))
    build_index(t, ctx, stats, others)
    url = 'http://localhost:%d/_review/index.html' % A.port
    log('review pages: %s' % os.path.relpath(out, ROOT))
    if A.no_serve: return
    Handler.t = t
    srv = ThreadingHTTPServer(('127.0.0.1', A.port), partial(Handler, directory=head_dir))
    log('serving %s  (Ctrl-C to stop)' % url)
    if not A.no_open: threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == '__main__':
    main()
