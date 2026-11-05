#!/usr/bin/env python3
"""Conclusion-style lint for series slide decks.

FAILs on:
  concl-prefix  a paragraph starting with 「结论」 / 「**结论**」 — the rule is
                one bare bold takeaway, no 「结论」 header.
  concl-long    a bold conclusion line longer than 40 CJK-equivalent chars
                (CJK char = 1, ASCII word ≈ 0.5, formula/code span = 1 token).
                Framing slides (主张 / 贯穿线 / 误区 / 下一步 / 误区) are exempt
                from the length rule.

Per-post slides are `##`-headed body sections plus their `<!-- v -->` subs.
"""
import re
import sys

FRAMING = re.compile(r'主张|贯穿线|误区|下一步|回顾|自测|总览|导言|预告|蓝图')
MAXLEN = 40.0


def cjk_len(t):
    t = re.sub(r'\$\$[^$]*\$\$|\$[^$]*\$|\\\(.+?\\\)|`[^`]*`', 'F', t)
    t = re.sub(r'[*_#\[\]()!>|\\「」、，。：；？—…· ×→−≈≤≥±/]', ' ', t)
    cjk = len(re.findall(r'[一-鿿]', t))
    words = len(re.findall(r'[A-Za-z0-9][A-Za-z0-9.\-%]*', t))
    return cjk + words * 0.5


def lint_file(path):
    s = open(path, encoding='utf-8').read()
    out = []
    heading = ''
    in_code = False
    in_notes = False
    for i, line in enumerate(s.split('\n'), 1):
        if line.strip().startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue
        if '<aside' in line:
            in_notes = True
        if '</aside>' in line:
            in_notes = False
            continue
        if in_notes:
            continue
        m = re.match(r'^#{2,3}\s+(.*)', line)
        if m:
            heading = m.group(1)
        t = line.strip()
        bare = t.strip('*').strip()
        # 「结论」 prefix on a paragraph
        if bare.startswith('结论') and not FRAMING.search(heading):
            out.append((i, 'concl-prefix', bare[:60]))
            continue
        # fully-bold standalone line longer than the cap
        if (re.match(r'^\*\*[^*]+\*\*\s*$', t) and cjk_len(bare) > MAXLEN
                and not FRAMING.search(heading)):
            out.append((i, 'concl-long %.0f' % cjk_len(bare), bare[:60]))
    return out


def main():
    files = sys.argv[1:]
    total = []
    for f in files:
        for i, kind, txt in lint_file(f):
            total.append((f, i, kind, txt))
    for f, i, kind, txt in total:
        print(f"{f}:{i}: {kind}: {txt}")
    print(f"{len(total)} problem(s) in {len(files)} file(s)")
    sys.exit(1 if total else 0)


if __name__ == '__main__':
    main()
