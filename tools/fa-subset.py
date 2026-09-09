#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Font Awesome 4.7 subset the site actually uses.

Scans layouts / includes / js / posts / less for `fa-<name>` classes and for
`content: "\\fXXX"` glyphs in the stylesheets, then writes

    css/font-awesome.min.css        base rules + @font-face + only those icons
    fonts/fontawesome-webfont.woff2 subset font (woff2 only; every browser we care about)

from the full copies kept in tools/fa/. Re-run after using a new icon; CI does
not do it for you (a missing glyph shows as an empty box). Needs
`pip install fonttools brotli`.
"""
import io, os, re, glob, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSS = os.path.join(ROOT, 'tools', 'fa', 'font-awesome-4.7.0.min.css')
SRC_TTF = os.path.join(ROOT, 'tools', 'fa', 'fontawesome-webfont.ttf')
OUT_CSS = os.path.join(ROOT, 'css', 'font-awesome.min.css')
OUT_WOFF2 = os.path.join(ROOT, 'fonts', 'fontawesome-webfont.woff2')
SCAN = ['_layouts/*.html', '_includes/*.html', '*.html', 'js/*.js', 'js/vendor/*.js', '_posts/*.md', '_draft/*.md', 'slides/*.md', 'less/*.less', 'css/argan-blog.css']
MODIFIERS = {'lg', '2x', '3x', '4x', '5x', 'fw', 'ul', 'li', 'spin', 'pulse', 'inverse', 'border', 'pull-left', 'pull-right', 'stack', 'stack-1x', 'stack-2x'}
ALWAYS = {'question-circle', 'external-link', 'circle'}  # used via CSS content / fa-stack

css = io.open(SRC_CSS, encoding='utf-8').read()
# icon -> codepoint, from rules like `.fa-glass:before{content:"\f000"}` (possibly several selectors)
codepoint = {}
for sels, cp in re.findall(r'((?:\.fa-[\w-]+:before,?)+)\{content:"\\(f[0-9a-f]{3})"\}', css):
    for name in re.findall(r'\.fa-([\w-]+):before', sels): codepoint[name] = cp

used, glyphs = set(ALWAYS), set()
for g in SCAN:
    for p in glob.glob(os.path.join(ROOT, g)):
        try: s = io.open(p, encoding='utf-8').read()
        except UnicodeDecodeError: continue
        used.update(n for n in re.findall(r'\bfa-([a-z0-9-]+)', s) if n not in MODIFIERS and not re.match(r'rotate-|flip-', n))
        glyphs.update(re.findall(r'content:\s*"\\(f[0-9a-f]{3})"', s))
unknown = sorted(n for n in used if n not in codepoint)
used = sorted(n for n in used if n in codepoint)
cps = sorted({codepoint[n] for n in used} | glyphs)
if unknown: print('not FA 4.7 icons, ignored:', ', '.join(unknown))
print('%d icons, %d glyphs' % (len(used), len(cps)))

# font
import shutil
pyft = shutil.which('pyftsubset') or os.path.expanduser('~/Library/Python/3.9/bin/pyftsubset')
subprocess.check_call([pyft, SRC_TTF, '--unicodes=' + ','.join('U+' + c for c in cps), '--flavor=woff2',
                       '--output-file=' + OUT_WOFF2, '--no-hinting', '--desubroutinize', '--layout-features='])

# css: header comment, @font-face (woff2 only), every non-icon rule, then the used icons
header = css[:css.index('@font-face')]
face = ("@font-face{font-family:'FontAwesome';src:url('../fonts/fontawesome-webfont.woff2?v=%s') format('woff2');"
        "font-weight:normal;font-style:normal;font-display:block}")
body = css[css.index('}', css.index('@font-face')) + 1:]
body = re.sub(r'@(?:-webkit-)?keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}', '', body)  # fa-spin/fa-pulse animation, nested braces
rules = re.findall(r'[^{}]+\{[^{}]*\}', body)
keep = []
for r in rules:
    if ':before{content:"\\f' in r:
        names = re.findall(r'\.fa-([\w-]+):before', r)
        wanted = [n for n in names if n in used]
        if not wanted: continue
        r = ','.join('.fa-%s:before' % n for n in wanted) + r[r.index('{'):]
    keep.append(r)
import hashlib
stamp = hashlib.sha1(''.join(cps).encode()).hexdigest()[:8]
io.open(OUT_CSS, 'w', encoding='utf-8').write(header + (face % stamp) + ''.join(keep) + '\n')
print('wrote %s (%d bytes), %s (%d bytes)' % (os.path.relpath(OUT_CSS, ROOT), os.path.getsize(OUT_CSS), os.path.relpath(OUT_WOFF2, ROOT), os.path.getsize(OUT_WOFF2)))
