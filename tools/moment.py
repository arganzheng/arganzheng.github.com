#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append one 随笔 (moment) to this month's file, moments/YYYY-MM.md.

    tools/moment.py "文字"                          just text
    tools/moment.py "文字" --at 深圳湾 --img a.jpg b.jpg   place + pictures (copied to img/moments/YYYY/MM/, WebP)
    tools/moment.py "文字" --quote "诗句\\n第二行" --by "苏轼《…》"
    tools/moment.py "文字" --music https://music.163.com/#/song?id=347230
    tools/moment.py --time "2026-09-21 08:02" ...   backdate (default: now)
    tools/moment.py                                 open this month's file in $EDITOR with a fresh heading

The entry format (see _plugins/moments.rb):

    ## 2026-09-21 08:02 @深圳湾
    text

    > quote line 1
    > quote line 2
    > —— by

    ![](/img/moments/2026/09/a.webp)

    https://music.163.com/#/song?id=347230

Images: copied to img/moments/YYYY/MM/<name>.webp (cwebp -q 82, resized to
1600px wide at most); a file already in .webp/.svg/.gif is copied as is.
"""
import argparse, datetime, os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('text', nargs='?', default='')
ap.add_argument('--at', default='', help='place, shown with a pin')
ap.add_argument('--img', nargs='*', default=[], help='image files')
ap.add_argument('--quote', default='', help='quote text; \\n for line breaks')
ap.add_argument('--by', default='', help='who said it')
ap.add_argument('--music', default='', help='网易云 / QQ 音乐 / Spotify / Apple Music URL, or an .mp3 link')
ap.add_argument('--time', default='', help='YYYY-MM-DD [HH:MM], default now')
ap.add_argument('--no-time', action='store_true', help='date only, no clock time')
a = ap.parse_args()

when = datetime.datetime.strptime(a.time, '%Y-%m-%d %H:%M' if ' ' in a.time else '%Y-%m-%d') if a.time else datetime.datetime.now()
if a.time and ' ' not in a.time: a.no_time = True
month_file = os.path.join(ROOT, 'moments', when.strftime('%Y-%m.md'))
os.makedirs(os.path.dirname(month_file), exist_ok=True)
if not os.path.exists(month_file):
    with open(month_file, 'w', encoding='utf-8') as f: f.write('---\nlayout: moments\n---\n')

head = '## ' + when.strftime('%Y-%m-%d' if a.no_time else '%Y-%m-%d %H:%M') + (' @' + a.at.strip() if a.at else '')

if not (a.text or a.img or a.quote or a.music):
    # interactive: open the editor on a fresh heading
    with open(month_file, 'a', encoding='utf-8') as f: f.write('\n' + head + '\n')
    editor = os.environ.get('VISUAL') or os.environ.get('EDITOR') or 'vi'
    os.execvp(editor.split()[0], editor.split() + [month_file])

def slug(name):
    base, ext = os.path.splitext(os.path.basename(name))
    return re.sub(r'[^a-z0-9\u4e00-\u9fff-]+', '-', base.lower()).strip('-') or 'img', ext.lower()

pics = []
if a.img:
    d = os.path.join(ROOT, 'img', 'moments', when.strftime('%Y'), when.strftime('%m'))
    os.makedirs(d, exist_ok=True)
    for src in a.img:
        if not os.path.exists(src): sys.exit('no such image: %s' % src)
        base, ext = slug(src)
        if ext in ('.webp', '.svg', '.gif'):
            out = os.path.join(d, base + ext); shutil.copyfile(src, out)
        else:
            out = os.path.join(d, base + '.webp')
            r = subprocess.run(['cwebp', '-quiet', '-q', '82', '-metadata', 'none', '-resize', '1600', '0', src, '-o', out], capture_output=True, text=True)
            if r.returncode: sys.exit('cwebp failed on %s: %s (brew install webp)' % (src, r.stderr.strip()))
        pics.append('/' + os.path.relpath(out, ROOT).replace(os.sep, '/'))

parts = [head]
if a.text: parts.append(a.text.strip())
if a.quote:
    q = ['> ' + l for l in a.quote.replace('\\n', '\n').strip().splitlines()]
    if a.by: q.append('> —— ' + a.by.strip())
    parts.append('\n'.join(q))
if pics: parts.append('\n'.join('![](%s)' % p for p in pics))
if a.music: parts.append(a.music.strip())

with open(month_file, 'a', encoding='utf-8') as f:
    f.write('\n' + '\n\n'.join(parts) + '\n')
anchor = when.strftime('%Y%m%d' if a.no_time else '%Y%m%d-%H%M')
print('%s  ->  /moments/%s.html#%s' % (os.path.relpath(month_file, ROOT), when.strftime('%Y-%m'), anchor))
