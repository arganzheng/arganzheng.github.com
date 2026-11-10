#!/usr/bin/env python3
"""Embed subset fonts into a hand-written SVG (in place, idempotent).

    tools/svg-embed-fonts.py img/in-post/foo.svg [more.svg ...]

For every <text>/<tspan> the tool resolves the effective font-family (sans vs
mono) and weight from attributes, inline style, <style> class rules and
ancestors, collects the used characters and injects one
<style id="embedded-fonts"> with WOFF2 @font-face subsets:

    'SvgSans'  400 -> Noto Sans SC Regular   (ordinary text)
               500 -> Noto Sans SC Medium    (500/600/700/bold -> Medium)
    'SvgMono'  400 -> DejaVu Sans Mono       (mono bold also maps to regular)

'SvgSans' / 'SvgMono' is then prepended to every font-family, keeping the
original stack as fallback. A character missing from its font aborts the file
with a clear message.

Font paths: Noto Sans SC *.otf come from a sibling clone of
arganzheng/ai-learning-labs (../ai-learning-labs/fonts relative to this repo)
or from the FIGTEXT_FONTS env var; DejaVuSansMono.ttf from matplotlib's
bundled mpl-data (or DEJAVU_MONO). Missing fonts fail with a clear message.

Requires: fontTools + brotli (pip install fonttools brotli).
"""
import base64
import io
import os
import re
import sys
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
NS = f"{{{SVG}}}"


def _fonts_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("FIGTEXT_FONTS"),
                 os.path.join(here, "..", "fonts"),
                 os.path.join(here, "..", "..", "ai-learning-labs", "fonts")):
        if cand and os.path.isfile(os.path.join(cand, "NotoSansSC-Regular.otf")):
            return os.path.abspath(cand)
    raise SystemExit(
        "找不到 Noto Sans SC 字体：需要 arganzheng/ai-learning-labs 仓库 "
        "fonts/NotoSansSC-{Regular,Medium}.otf，或设置 FIGTEXT_FONTS=<dir>")

FONTS_DIR = _fonts_dir()
SANS_REG = os.path.join(FONTS_DIR, "NotoSansSC-Regular.otf")
SANS_MED = os.path.join(FONTS_DIR, "NotoSansSC-Medium.otf")


def _mono_font():
    try:
        import matplotlib
        cand = os.path.join(os.path.dirname(matplotlib.__file__),
                            "mpl-data", "fonts", "ttf", "DejaVuSansMono.ttf")
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    cand = os.environ.get("DEJAVU_MONO")
    if cand and os.path.isfile(cand):
        return cand
    raise SystemExit(
        "找不到 DejaVuSansMono.ttf：装 matplotlib，或设置 DEJAVU_MONO=<path>")

MONO_FONT = _mono_font()

_MONO_HINT = re.compile(
    r"mono|menlo|courier|consol|sfmono|monaco|jetbrains|inconsolata|source\s*code",
    re.I)


def _css_rules(root):
    """[(compound selector, {prop:value})] from <style> blocks (mirrors
    svg-mathtext._css_rules — rightmost compound matches the element)."""
    rules = []
    for style in root.iter(f"{NS}style"):
        if style.get("id") == "embedded-fonts":
            continue
        for sel, body in re.findall(r"([^{}]+)\{([^}]*)\}", style.text or ""):
            props = dict(p.split(":", 1) for p in
                         (x.strip() for x in body.split(";")) if ":" in p)
            _expand_font(props)
            for s in sel.split(","):
                s = s.strip()
                if s and ":" not in s and "@" not in s:
                    rules.append((s, props))
    return rules


def _expand_font(props):
    """CSS `font:` shorthand -> longhands: [style] [weight] size[/line-height]
    family-list (see svg-mathtext._expand_font)."""
    font = props.get("font")
    if font:
        m = re.match(r"(?:(?:normal|italic|oblique)\s+)?"
                     r"(?:(normal|bold|bolder|lighter|[1-9]00)\s+)?"
                     r"([0-9.]+(?:px|pt))(?:\s*/\s*[\w.%]+)?\s+(.+)",
                     font.strip())
        if m:
            if m.group(1):
                props.setdefault("font-weight", m.group(1))
            props.setdefault("font-size", m.group(2))
            props.setdefault("font-family", m.group(3))


def _match_compound(el, compound):
    tag = el.tag.rpartition("}")[2]
    for part in re.findall(r"(?:^|[\s])([a-zA-Z*][\w-]*|[.#][\w-]+)", " " + compound):
        pass
    comps = compound.split()[-1]
    elem = re.match(r"^[a-zA-Z*][\w-]*", comps)
    if elem and elem.group(0) not in (tag, "*"):
        return False
    for cls in re.findall(r"\.([\w-]+)", comps):
        if cls not in (el.get("class") or "").split():
            return False
    for i in re.findall(r"#([\w-]+)", comps):
        if el.get("id") != i:
            return False
    return True


def _selector_matches(el, selector, parent_map):
    compounds = selector.split()
    if not compounds or not _match_compound(el, compounds[-1]):
        return False
    cur = parent_map.get(el)
    for comp in reversed(compounds[:-1]):
        while cur is not None and not _match_compound(cur, comp):
            cur = parent_map.get(cur)
        if cur is None:
            return False
        cur = parent_map.get(cur)
    return True


def _resolve(el, parent_map, prop, rules):
    """Attr > inline style > matching CSS rule > ancestor (inherit)."""
    v = el.get(prop)
    if v:
        return v.strip()
    st = el.get("style")
    if st:
        props = dict(p.split(":", 1) for p in
                     (x.strip() for x in st.split(";")) if ":" in p)
        _expand_font(props)
        if props.get(prop):
            return props[prop].strip()
    out = None
    for sel, props in rules:
        if prop in props and _selector_matches(el, sel, parent_map):
            out = props[prop].strip()   # later rules win (document order)
    if out:
        return out
    par = parent_map.get(el)
    if par is not None:
        return _resolve(par, parent_map, prop, rules)
    return None


def _norm_weight(w):
    if not w or w in ("normal", "lighter"):
        return 400
    if w == "bold":
        return 700
    try:
        return int(re.sub(r"\D", "", w))
    except Exception:
        return 400


def _tspan_texts(el):
    """All visible characters in <text> (including nested tspan)."""
    out = []
    for t in el.iter():
        if t.tag in (f"{NS}text", f"{NS}tspan") and t.text:
            out.append(t.text)
        if t.tail:
            pass  # tail text belongs to parent, counted via .text above
    return "".join(out)


_CMAPS = {}


def _cmap(path):
    if path not in _CMAPS:
        from fontTools.ttLib import TTFont
        _CMAPS[path] = TTFont(path).getBestCmap()
    return _CMAPS[path]


def process(path):
    from fontTools import subset
    from fontTools.ttLib import TTFont

    src = open(path, encoding="utf-8").read()
    tree = ET.fromstring(src)
    # strip previous injection → idempotent
    parent_map = {c: p for p in tree.iter() for c in p}
    for st in list(tree.iter(f"{NS}style")):
        if st.get("id") == "embedded-fonts":
            parent_map[st].remove(st)

    rules = _css_rules(tree)
    parent_map = {c: p for p in tree.iter() for c in p}

    def _in_mathtext(el):
        node = el
        while node is not None:
            if node.get("class") == "mathtext":
                return True
            node = parent_map.get(node)
        return False

    # (slot) -> chars;  slot: 'sans400'|'sans500'|'mono'
    used = {"sans400": set(), "sans500": set(), "mono": set()}
    affected = []
    for el in tree.iter():
        if el.tag not in (f"{NS}text", f"{NS}tspan"):
            continue
        if el.get("data-tex") is not None or _in_mathtext(el):
            continue  # mathtext groups keep their own fonts
        # el.text is collected under THIS element's family/weight; each
        # child's .tail too (it belongs to the parent's text run). Descendant
        # elements' own .text is collected when they are visited on their own,
        # under their own resolved family/weight.
        content = (el.text or "") + "".join(c.tail or "" for c in el)
        if not content.strip():
            continue
        fam = _resolve(el, parent_map, "font-family", rules) or ""
        weight = _norm_weight(_resolve(el, parent_map, "font-weight", rules))
        content = "".join(c for c in content if ord(c) >= 0x20
                          and ord(c) != 0x7f)
        if _MONO_HINT.search(fam):
            mono = _cmap(MONO_FONT)
            sans = _cmap(SANS_MED if weight >= 500 else SANS_REG)
            slot = "sans500" if weight >= 500 else "sans400"
            missing = []
            for c in content:
                if ord(c) in mono:
                    used["mono"].add(c)
                elif ord(c) in sans:
                    used[slot].add(c)   # CJK etc. -> Noto subset via 'SvgSans'
                else:
                    missing.append(c)
            if missing:
                raise SystemExit(
                    f"{os.path.basename(path)}: 字形在 DejaVu Sans Mono 和 "
                    f"Noto Sans SC 都没有 {sorted(set(missing))!r}")
            affected.append((el, "SvgMono", "mono"))
        else:
            slot = "sans500" if weight >= 500 else "sans400"
            used[slot].update(content)
            affected.append((el, "SvgSans", slot))

    if not any(used.values()):
        return 0, 0

    faces = []
    plans = [("SvgSans", "400", SANS_REG, used["sans400"]),
             ("SvgSans", "500", SANS_MED, used["sans500"]),
             ("SvgMono", "400", MONO_FONT, used["mono"])]
    for fam_name, weight, f, chars in plans:
        if not chars:
            continue
        font = TTFont(f)
        cmap = font.getBestCmap()
        missing = sorted(c for c in chars if ord(c) not in cmap)
        if missing:
            raise SystemExit(
                f"{os.path.basename(path)}: {os.path.basename(f)} 缺字形 "
                f"{missing!r}（{fam_name} {weight}）— 改用已有字符或加字体")
        opts = subset.Options(flavor="woff2")
        ss = subset.Subsetter(opts)
        ss.populate(text="".join(sorted(chars)))
        try:
            ss.subset(font)
        except Exception:
            opts.flavor = "woff"
            font = TTFont(f)
            ss = subset.Subsetter(opts)
            ss.populate(text="".join(sorted(chars)))
            ss.subset(font)
        buf = io.BytesIO()
        font.save(buf)
        mime = "woff2" if opts.flavor == "woff2" else "woff"
        faces.append(
            f"@font-face{{font-family:'{fam_name}';font-weight:{weight};"
            f"src:url(data:font/{mime};base64,"
            f"{base64.b64encode(buf.getvalue()).decode()})}}")

    # prepend SvgSans / SvgMono to every resolved font-family:
    #   - inline font-family attr/style on text & tspan
    #   - <style> rules (non-embedded-fonts)
    def _prefix(v):
        """'SvgSans'/'SvgMono' for one family stack (mono also falls back to
        'SvgSans' for CJK chars DejaVu lacks)."""
        v = v.strip()
        if v.lstrip().startswith(("'Svg", '"Svg')):
            return v
        return (f"'SvgMono','SvgSans', {v}" if _MONO_HINT.search(v)
                else f"'SvgSans', {v}")

    for el in list(tree.iter()):
        if el.tag not in (f"{NS}svg", f"{NS}text", f"{NS}tspan", f"{NS}g"):
            continue
        if _in_mathtext(el):
            continue
        fam = el.get("font-family")
        if fam and not fam.lstrip().startswith(("'Svg", '"Svg')):
            el.set("font-family", _prefix(fam))
        st = el.get("style")
        if st and "font-family" in st:
            st = re.sub(r"font-family:\s*([^;]+)",
                        lambda m: f"font-family:{_prefix(m.group(1))}", st)
            el.set("style", st)
    for st in tree.iter(f"{NS}style"):
        if st.get("id") in ("embedded-fonts", "mathtext-fonts") or not st.text:
            continue
        st.text = re.sub(r"font-family:\s*([^;}]+)",
                         lambda m: f"font-family:{_prefix(m.group(1))}",
                         st.text)

    new_src = ET.tostring(tree, encoding="unicode")
    style = (f'<style id="embedded-fonts" type="text/css"><![CDATA['
             + "".join(faces) + "]]></style>")
    m = re.search(r"<svg[^>]*>", new_src)
    new_src = new_src[:m.end()] + "\n" + style + new_src[m.end():]
    before = len(src.encode("utf-8"))
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(new_src)
    return before, len(new_src.encode("utf-8")), len(faces)


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        sys.exit(2)
    for f in files:
        try:
            out = process(f)
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"{f}: {e}")
        if not out:
            print(f"{f}: 没有 text")
        else:
            b, a, n = out
            print(f"{f}: {b} -> {a} 字节（{n} 个 @font-face）")


if __name__ == "__main__":
    main()
