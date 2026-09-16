#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Font Awesome 7 (Free) subset the site actually uses.

Scans layouts / includes / js / posts / less for `fa-<name>` classes and for
`content: "\\fXXX"` glyphs in the stylesheets, resolves them against the icon
metadata shipped in node_modules/@fortawesome/fontawesome-free, and writes

    css/font-awesome.min.css     FA core rules (sizing, fa-fw, fa-stack, spin…),
                                 three @font-face, only the icons we use
    fonts/fa-solid-900.woff2     subset fonts, one per free style
    fonts/fa-regular-400.woff2
    fonts/fa-brands-400.woff2

Markup convention (FA 7): `fa fa-<name>` is solid, `fa fa-regular fa-<name>`
the outline variant, `fa fa-brands fa-<name>` a brand. `.fa` stays on every
icon so the theme's `.fa` selectors and `querySelector('.fa')` keep working.
FA 7 gives every icon a fixed 1.25em width; we reset it to `auto` (natural
width, as in FA 4) — use `fa-fw` when you want the fixed width.

~150 common icons are always included (ALWAYS below), so a post can use any of
them without re-running this. `--check` (run by tools/check.sh and CI, no
node_modules needed) fails when a name used anywhere is not in the committed
CSS — then run this script (needs `npm install` and `pip install fonttools
brotli`), or add the icon to ALWAYS. A FA 4 name that was renamed in FA 5+
(`share-alt`, `thumbs-o-up`, `external-link`…) is reported with its new name
and style, taken from FA's own shims.yml.
"""
import io, os, re, glob, json, hashlib, subprocess, sys, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, 'node_modules', '@fortawesome', 'fontawesome-free')
OUT_CSS = os.path.join(ROOT, 'css', 'font-awesome.min.css')
FONTS = {'solid': ('fa-solid-900.woff2', 900), 'regular': ('fa-regular-400.woff2', 400), 'brands': ('fa-brands-400.woff2', 400)}
FAMILY = {'solid': 'Font Awesome 7 Free', 'regular': 'Font Awesome 7 Free', 'brands': 'Font Awesome 7 Brands'}
SCAN = ['_layouts/*.html', '_includes/*.html', '*.html', 'admin/*.html', 'js/*.js', 'js/vendor/*.js', '_posts/*.md', '_drafts/*.md', 'slides/*.md', 'less/*.less']
# class tokens that start with fa- but are not icons
MODIFIERS = re.compile(r'^(?:solid|regular|brands|classic|sharp|duotone|light|thin|fw|width-auto|width-fixed|ul|li|border|inverse|'
                       r'lg|xs|sm|xl|2xs|2xl|[0-9]+x|spin|spin-pulse|spin-reverse|pulse|beat|beat-fade|bounce|fade|flip|shake|'
                       r'pull-(?:left|right|start|end)|rotate-.*|flip-.*|stack(?:-[12]x)?|layers.*|canvas-.*|sr-only.*)$')
# Always in the subset (FA 7 names). Outline variants come for free: every
# icon here that has a regular style is subset in both fonts.
ALWAYS = set("""
circle-question up-right-from-square arrow-up-right-from-square circle check circle-check square-check square xmark circle-xmark
circle-exclamation triangle-exclamation circle-info info question lightbulb bell
thumbtack star star-half-stroke heart thumbs-up thumbs-down bookmark flag tag tags bolt fire trophy gift
arrow-up arrow-down arrow-left arrow-right circle-arrow-up circle-arrow-down circle-arrow-left circle-arrow-right
arrow-right-arrow-left arrows-rotate rotate-right rotate-left
caret-up caret-down caret-left caret-right chevron-up chevron-down chevron-left chevron-right
angle-up angle-down angle-left angle-right angles-up angles-down angles-left angles-right
plus minus circle-plus circle-minus square-plus square-minus
link link-slash share share-nodes paper-plane envelope
magnifying-glass magnifying-glass-plus magnifying-glass-minus filter sort sort-up sort-down list list-ul list-ol table-cells table-list bars ellipsis ellipsis-vertical
house user users circle-user comment comments reply reply-all quote-left quote-right
pencil pen-to-square trash trash-can copy clipboard file file-lines file-code file-pdf file-image folder folder-open
download upload cloud cloud-arrow-down cloud-arrow-up floppy-disk print
book newspaper graduation-cap building-columns flask code code-branch terminal keyboard laptop desktop mobile tablet server database hard-drive sitemap
gear gears wrench sliders toggle-on toggle-off lock unlock key shield
clock calendar calendar-days clock-rotate-left hourglass spinner circle-notch
eye eye-slash image camera video film play pause stop circle-play
location-dot map globe compass location-arrow road plane rocket
chart-bar chart-line chart-pie chart-area gauge signal wifi microchip
bold italic underline strikethrough heading paragraph align-left align-center align-right align-justify indent outdent table
github github-alt square-github twitter x-twitter square-twitter facebook square-facebook linkedin linkedin-in weixin weibo qq google stack-overflow rss square-rss youtube slack creative-commons
mug-hot beer-mug-empty face-smile face-frown face-meh hand-point-right hand-point-left hand-point-up hand-point-down hand-peace
""".split())

def scan():
    used, glyphs = {}, set()
    for g in SCAN:
        for p in glob.glob(os.path.join(ROOT, g)):
            try: s = io.open(p, encoding='utf-8').read()
            except UnicodeDecodeError: continue
            if p.endswith('.md'):  # icon names quoted in prose (`fa-xxx`) are not uses
                s = re.sub(r'```.*?```|`[^`\n]*`', '', s, flags=re.S)
            for n in re.findall(r'(?<![\w/.-])fa-([a-z0-9-]+)(?![\w.])', s):
                if not MODIFIERS.match(n): used.setdefault(n, set()).add(os.path.relpath(p, ROOT))
            glyphs.update(c.lower() for c in re.findall(r'content:\s*"\\([0-9a-f]{4,5})"', s))
    return used, glyphs

def check():
    css = io.open(OUT_CSS, encoding='utf-8').read()
    have = set(n for sels in re.findall(r'((?:\.fa-[\w-]+,?)+)\{--fa:', css) for n in re.findall(r'\.fa-([\w-]+)', sels))
    have_cp = set(re.findall(r'--fa:"\\([0-9a-f]+)"', css))
    used, glyphs = scan()
    missing = sorted(n for n in used if n not in have)
    missing_cp = sorted(c for c in glyphs if c not in have_cp)
    if missing or missing_cp:
        for n in missing: print('  fa-%-28s used in %s' % (n, ', '.join(sorted(used[n])[:3])))
        for c in missing_cp: print('  glyph \\%s used in less/ is not in the subset' % c)
        print('icons used but missing from css/font-awesome.min.css\n-> run: python3 tools/fa-subset.py  (needs npm install + pip install fonttools brotli)')
        sys.exit(1)
    print('font-awesome subset is complete (%d names, %d glyphs)' % (len(used), len(have_cp)))

def build():
    meta = json.load(io.open(os.path.join(PKG, 'metadata', 'icon-families.json'), encoding='utf-8'))
    shims = {}
    for m in re.finditer(r'^([\w-]+):\n((?:  .*\n)+)', io.open(os.path.join(PKG, 'metadata', 'shims.yml'), encoding='utf-8').read(), re.M):
        body = dict(re.findall(r'  (\w+): (.+)', m.group(2)))
        shims[m.group(1)] = (body.get('prefix', 'fas'), body.get('name', m.group(1)))
    # name -> canonical icon; aliases (FA 5/6 names FA still ships as classes) included
    canon = {}
    for name, ic in meta.items():
        canon[name] = name
        for a in ic.get('aliases', {}).get('names', []): canon[a] = name
    by_cp = {ic['unicode']: name for name, ic in meta.items()}
    styles = {name: [f['style'] for f in ic['familyStylesByLicense']['free']] for name, ic in meta.items()}

    used, glyphs = scan()
    names = set(ALWAYS) | set(used)
    bad = []
    for n in sorted(names):
        if n in canon: continue
        if n in shims:
            prefix, new = shims[n]
            style = {'far': 'fa-regular ', 'fab': 'fa-brands '}.get(prefix, '')
            bad.append('  fa-%-24s is a FA 4 name: write `fa %sfa-%s`%s' % (n, style, new, ' (' + ', '.join(sorted(used.get(n, []))[:3]) + ')' if n in used else ''))
        elif n in used:
            print('not a FA 7 icon, ignored: fa-%s (%s)' % (n, ', '.join(sorted(used[n])[:3])))
    for c in sorted(glyphs):
        if c not in by_cp: bad.append('  glyph \\%s in less/ is not a FA 7 codepoint' % c)
    if bad:
        print('\n'.join(bad)); sys.exit(1)
    icons = sorted({canon[n] for n in names if n in canon} | {by_cp[c] for c in glyphs})
    # class names to emit: canonical + the aliases actually used in the sources
    emit = {ic: sorted({ic} | {n for n in used if canon.get(n) == ic}) for ic in icons}
    per_font = {s: sorted({meta[ic]['unicode'] for ic in icons if s in styles[ic]}) for s in FONTS}
    print('%d icons; glyphs per font: %s' % (len(icons), ', '.join('%s %d' % (s, len(c)) for s, c in per_font.items())))

    pyft = shutil.which('pyftsubset') or os.path.expanduser('~/Library/Python/3.9/bin/pyftsubset')
    faces, stamp = [], {}
    for s, (fname, weight) in FONTS.items():
        out = os.path.join(ROOT, 'fonts', fname)
        subprocess.check_call([pyft, os.path.join(PKG, 'webfonts', fname), '--unicodes=' + ','.join('U+' + c for c in per_font[s]),
                               '--flavor=woff2', '--output-file=' + out, '--no-hinting', '--desubroutinize', '--layout-features='])
        stamp[s] = hashlib.sha1(''.join(per_font[s]).encode()).hexdigest()[:8]
        faces.append('@font-face{font-family:"%s";font-style:normal;font-weight:%d;font-display:block;src:url(../fonts/%s?v=%s) format("woff2")}'
                     % (FAMILY[s], weight, fname, stamp[s]))

    core = io.open(os.path.join(PKG, 'css', 'fontawesome.min.css'), encoding='utf-8').read()
    header = core[:core.index('*/') + 2] + '\n'
    body = core[len(header) - 1:]
    body = re.sub(r'\.fa-[a-z0-9-]+(?:,\.fa-[a-z0-9-]+)*\{--fa:"[^"]*"\}', '', body)  # the 2000 icon rules
    # brands.min.css / regular.min.css / solid.min.css carry the style variables (family + weight); take those, not their @font-face
    style_vars = ''
    for s in FONTS:
        part = io.open(os.path.join(PKG, 'css', s + '.min.css'), encoding='utf-8').read()
        part = part[part.index('*/') + 2:]
        part = re.sub(r'@font-face\{[^}]*\}', '', part)
        style_vars += re.sub(r'\.fa-[a-z0-9-]+(?:,\.fa-[a-z0-9-]+)*\{--fa:"[^"]*"\}', '', part).strip()  # brands.min.css lists every brand icon
    natural = ':where(.fa,.fas,.far,.fab,.fa-solid,.fa-regular,.fa-brands){--fa-width:auto}'  # FA 4 sizing; fa-fw still opts in
    rules = ''.join('%s{--fa:"\\%s"}' % (','.join('.fa-' + n for n in emit[ic]), meta[ic]['unicode']) for ic in icons)
    io.open(OUT_CSS, 'w', encoding='utf-8').write(header + style_vars + ''.join(faces) + body.strip() + natural + rules + '\n')
    print('wrote %s (%d bytes), fonts: %s' % (os.path.relpath(OUT_CSS, ROOT), os.path.getsize(OUT_CSS),
          ', '.join('%s %d B' % (f, os.path.getsize(os.path.join(ROOT, 'fonts', f))) for f, _ in FONTS.values())))

if __name__ == '__main__':
    check() if '--check' in sys.argv else build()
