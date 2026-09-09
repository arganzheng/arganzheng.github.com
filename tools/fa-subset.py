#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Font Awesome 4.7 subset the site actually uses.

Scans layouts / includes / js / posts / less for `fa-<name>` classes and for
`content: "\\fXXX"` glyphs in the stylesheets, then writes

    css/font-awesome.min.css        base rules + @font-face + only those icons
    fonts/fontawesome-webfont.woff2 subset font (woff2 only; every browser we care about)

from the full copies kept in tools/fa/. ~150 common icons are always included
(ALWAYS below), so normally nothing to do; `--check` (run by CI) fails when a
post uses an icon outside the subset — then run this script (needs
`pip install fonttools brotli`) or add the icon to ALWAYS.
"""
import io, os, re, glob, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSS = os.path.join(ROOT, 'tools', 'fa', 'font-awesome-4.7.0.min.css')
SRC_TTF = os.path.join(ROOT, 'tools', 'fa', 'fontawesome-webfont.ttf')
OUT_CSS = os.path.join(ROOT, 'css', 'font-awesome.min.css')
OUT_WOFF2 = os.path.join(ROOT, 'fonts', 'fontawesome-webfont.woff2')
SCAN = ['_layouts/*.html', '_includes/*.html', '*.html', 'js/*.js', 'js/vendor/*.js', '_posts/*.md', '_draft/*.md', 'slides/*.md', 'less/*.less', 'css/argan-blog.css']
MODIFIERS = {'lg', '2x', '3x', '4x', '5x', 'fw', 'ul', 'li', 'spin', 'pulse', 'inverse', 'border', 'pull-left', 'pull-right', 'stack', 'stack-1x', 'stack-2x'}
# Always included, whether or not the scan finds them, so a post can use any of
# these without re-running the script. ~150 of FA 4.7's 786 icons; the font is
# still ~5x smaller than the full one. Add to this list rather than remembering.
ALWAYS = set("""
question-circle external-link circle check check-circle check-square-o square-o times times-circle
exclamation-circle exclamation-triangle info-circle info question lightbulb-o bell-o bell warning
star star-o star-half-o heart heart-o thumbs-up thumbs-down thumbs-o-up thumbs-o-down
bookmark bookmark-o flag flag-o tag tags bolt fire trophy gift
arrow-up arrow-down arrow-left arrow-right arrow-circle-up arrow-circle-down arrow-circle-left arrow-circle-right
long-arrow-up long-arrow-down long-arrow-left long-arrow-right exchange refresh repeat undo
caret-up caret-down caret-left caret-right chevron-up chevron-down chevron-left chevron-right
angle-up angle-down angle-left angle-right angle-double-up angle-double-down angle-double-left angle-double-right
plus minus plus-circle minus-circle plus-square minus-square
link unlink external-link-square share share-alt paper-plane paper-plane-o envelope envelope-o
search search-plus search-minus filter sort sort-asc sort-desc list list-ul list-ol th th-list bars ellipsis-h ellipsis-v
home user users user-circle user-o comment comment-o comments comments-o reply reply-all quote-left quote-right
pencil pencil-square-o edit trash trash-o copy clipboard files-o file file-o file-text file-text-o file-code-o file-pdf-o file-image-o folder folder-o folder-open folder-open-o
download upload cloud cloud-download cloud-upload save floppy-o print
book newspaper-o graduation-cap university flask code code-fork terminal keyboard-o laptop desktop mobile tablet server database hdd-o sitemap
cog cogs wrench sliders toggle-on toggle-off lock unlock unlock-alt key shield
clock-o calendar calendar-o history hourglass-o spinner circle-o-notch
eye eye-slash picture-o camera video-camera film image play pause stop play-circle
map-marker map-o globe compass location-arrow road plane rocket
bar-chart line-chart pie-chart area-chart tachometer signal wifi microchip
bold italic underline strikethrough header paragraph align-left align-center align-right align-justify indent outdent table
github github-alt github-square twitter twitter-square facebook facebook-square linkedin linkedin-square weixin weibo qq google stack-overflow rss rss-square youtube-play slack
lightbulb-o coffee beer smile-o frown-o meh-o hand-o-right hand-o-left hand-o-up hand-o-down hand-peace-o
""".split())

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

if '--check' in sys.argv:
    # CI guard: every icon the site uses must be in the committed subset.
    have = set(re.findall(r'\.fa-([\w-]+):before', io.open(OUT_CSS, encoding='utf-8').read()))
    missing = sorted(n for n in used if n not in have)
    if missing:
        print('icons used but missing from css/font-awesome.min.css: %s\n-> run: python3 tools/fa-subset.py (pip install fonttools brotli)' % ', '.join(missing))
        sys.exit(1)
    print('font-awesome subset is complete'); sys.exit(0)

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
