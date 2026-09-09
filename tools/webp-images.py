#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert raster images under img/in-post/ to WebP and rewrite every reference.

    tools/webp-images.py            # dry run: what would change and how much
    tools/webp-images.py --apply    # convert, rewrite references, delete originals

Rules: only png/jpg/jpeg >= MIN_KB; a WebP is kept only when it saves >= 15 %
(otherwise the original stays); PNGs use quality 90 (screenshots with text),
photos 82. Site-level images in img/*.jpg are left alone — they are the
og:image and some sharing targets still refuse WebP. Needs `cwebp` (brew
install webp). Run `jekyll build` + lychee afterwards to be sure no reference
was missed (CI does both).
"""
import io, os, re, subprocess, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, 'img', 'in-post')
MIN_KB = 20
TEXT_GLOBS = ['_posts/*.md', '_posts/*.markdown', '_draft/*.md', '_layouts/*.html', '_includes/*.html', '*.html', '*.md', '_config.yml', 'slides/*.md', 'slides/*.html']
apply = '--apply' in sys.argv

def size(p): return os.path.getsize(p)

texts = {}
for g in TEXT_GLOBS:
    for p in glob.glob(os.path.join(ROOT, g)):
        try: texts[p] = io.open(p, encoding='utf-8').read()
        except UnicodeDecodeError: pass

total_before = total_after = 0
converted = []
for src in sorted(glob.glob(os.path.join(IMG_DIR, '*'))):
    ext = os.path.splitext(src)[1].lower()
    if ext not in ('.png', '.jpg', '.jpeg') or size(src) < MIN_KB * 1024: continue
    dst = os.path.splitext(src)[0] + '.webp'
    # foo.png and foo.jpg both exist -> the second one keeps its extension in the name
    if os.path.exists(dst) or any(c[1] == dst for c in converted): dst = os.path.splitext(src)[0] + '-' + ext[1:] + '.webp'
    q = '90' if ext == '.png' else '82'
    tmp = dst + '.tmp'
    r = subprocess.run(['cwebp', '-quiet', '-q', q, '-metadata', 'none', src, '-o', tmp], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        print('skip (cwebp failed): %s %s' % (os.path.basename(src), r.stderr.strip()[:80])); continue
    before, after = size(src), size(tmp)
    if after > before * 0.85:
        os.remove(tmp); print('keep  %-60s %5dK -> %5dK (not worth it)' % (os.path.basename(src), before // 1024, after // 1024)); continue
    rel_old = '/img/in-post/' + os.path.basename(src)
    rel_new = '/img/in-post/' + os.path.basename(dst)
    refs = [p for p, s in texts.items() if rel_old in s or rel_old[1:] in s]
    total_before += before; total_after += after
    converted.append((src, dst, tmp, rel_old, rel_new, refs))
    print('webp  %-60s %5dK -> %5dK  refs=%d' % (os.path.basename(src), before // 1024, after // 1024, len(refs)))
    if not apply: os.remove(tmp)

print('\n%d images, %.1f MB -> %.1f MB' % (len(converted), total_before / 1e6, total_after / 1e6))
if not apply:
    print('dry run; add --apply to convert'); sys.exit(0)

for src, dst, tmp, rel_old, rel_new, refs in converted:
    os.replace(tmp, dst)
    os.remove(src)
    for p in refs:
        s = texts[p].replace(rel_old, rel_new).replace(rel_old[1:], rel_new[1:])
        texts[p] = s
        io.open(p, 'w', encoding='utf-8').write(s)
print('done')
