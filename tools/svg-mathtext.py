#!/usr/bin/env python3
"""Typeset `data-tex` text elements in a hand-written SVG as real text.

In place:

    tools/svg-mathtext.py img/in-post/foo.svg [more.svg ...]

Every `<text data-tex="…">` is replaced by `<g class="mathtext" data-tex="…"
data-x data-y data-size data-anchor data-fill data-family>` whose children are
real `<text>`/`<tspan>` runs positioned per glyph (Computer Modern mathtext for
`$...$` parts, the CJK/system font for the rest — same look as the posts'
KaTeX), plus `<rect>`s for fraction bars. The `data-tex` string is the
editable source: rerunning the tool re-renders every
`<g class="mathtext" data-tex>` from its stored attributes, so edit the string,
not the markup. A `$...$` that doesn't parse aborts the file with an error.

After typesetting, every font file actually used (cmr10/cmmi10/cmsy10/cmex10/
STIX from matplotlib's mpl-data, plus the prose font) is subset to the used
glyphs and embedded as WOFF2 in a `<style id="embedded-fonts">` @font-face
block — the file renders without the reader's fonts. CM/STIX glyphs get a
same-colour ~0.3px stroke (scaled from the viewBox) to match KaTeX weight.
Reruns are idempotent: the embedded block is rebuilt from the current glyphs.

Baseline sits at the original `y`; `text-anchor` start/middle/end is honored.
Fill comes from the text's `fill`, inline style, `<style>` class rule, or
ancestor/parent `fill` (default #222). Non-math text uses the resolved
font-family (mono classes → monospace stack; CJK → Hiragino Sans GB etc.).
"""
import base64
import copy
import io
import os
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.mathtext import MathTextParser  # noqa: E402

plt.rcParams.update({"mathtext.fontset": "cm"})

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
NS = f"{{{SVG}}}"

CJK_FAMILY = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
MONO_FAMILY = ["Menlo", "Courier New", "monospace"]
_PARSER = MathTextParser("path")
_MATH_FONT_HINT = ("cmmi", "cmr", "cmsy", "cmex", "cmss", "cmti", "cmtt",
                   "cmb", "STIX")


def _css_rules(root):
    """[(compound selector, {prop: value})] from every <style> block.

    Compound selector = optional element name + .class / #id parts, e.g.
    `text`, `.t`, `text.t`, `g.annot text`. Only the rightmost compound must
    match the element itself; earlier compounds (if any) must match ancestors
    in order (descendant combinator).
    """
    rules = []
    for style in root.iter(f"{NS}style"):
        for sel, body in re.findall(r"([^{}]+)\{([^}]*)\}", style.text or ""):
            props = dict(p.split(":", 1) for p in
                         (x.strip() for x in body.split(";")) if ":" in p)
            _expand_font(props)
            for s in sel.split(","):
                s = s.strip()
                if s and ":" not in s:  # skip pseudo selectors we can't match
                    rules.append((s, props))
    return rules


def _match_compound(el, compound):
    """Does a compound selector like `text`, `.t`, `text.t`, `#id`, `g.a` match
    this element?"""
    tag = el.tag.rsplit("}", 1)[-1]
    m = re.fullmatch(r"(\*|[\w-]+)?((?:[.#][\w-]+)*)", compound)
    if not m or compound == "":
        return False
    name, rest = m.group(1), m.group(2)
    if name and name != "*" and name != tag:
        return False
    classes = set((el.get("class") or "").split())
    for tok in re.findall(r"[.#][\w-]+", rest):
        if tok.startswith(".") and tok[1:] not in classes:
            return False
        if tok.startswith("#") and tok[1:] != el.get("id"):
            return False
    return bool(name or rest)


def _selector_matches(el, selector, parent_map):
    parts = selector.split()
    if not _match_compound(el, parts[-1]):
        return False
    node = parent_map.get(el)
    for part in reversed(parts[:-1]):
        while node is not None and not _match_compound(node, part):
            node = parent_map.get(node)
        if node is None:
            return False
        node = parent_map.get(node)
    return True


def _rule_props(el, rules, parent_map):
    """Merged props from all matching rules, in stylesheet order (later wins)."""
    out = {}
    for sel, props in rules:
        if _selector_matches(el, sel, parent_map):
            out.update(props)
    return out


def _expand_font(props):
    font = props.get("font")
    if font:
        m = re.match(r"(?:(normal|bold|[1-9]00)\s+)?([0-9.]+(?:px|pt))\s+(.+)", font.strip())
        if m:
            if m.group(1):
                props.setdefault("font-weight", m.group(1))
            props.setdefault("font-size", m.group(2))
            props.setdefault("font-family", m.group(3))


def _style_props(el):
    """Inline style="k:v; k:v" -> dict."""
    out = {}
    for part in (el.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    _expand_font(out)
    return out


def _resolve(el, parent_map, prop, rules):
    """Resolve a CSS-ish property: attr > inline style > matching <style> rule,
    then the same three sources on each ancestor (SVG text properties
    inherit), else None."""
    node = el
    while node is not None:
        if node.get(prop) is not None:
            return node.get(prop)
        inline = _style_props(node)
        if prop in inline:
            return inline[prop]
        rp = _rule_props(node, rules, parent_map)
        if prop in rp:
            return rp[prop]
        node = parent_map.get(node)
    return None


def _font_name(ft2font):
    """@font-face family name for a matplotlib FT2Font — stable, quoted in CSS."""
    return os.path.splitext(os.path.basename(ft2font.fname))[0]


def _render(tex, size, weight, family, used):
    """Mixed CJK/$...$ string -> (children for a <g>, total advance px).

    Children are one <text> holding <tspan>s (one per glyph, grouped by font
    style — same trick as matplotlib's svg backend) plus <rect>s for fraction
    bars. `used` accumulates {font_file: {codepoint}} for later subsetting.
    MathTextParser units are points at dpi=72 → treat as SVG px 1:1.
    """
    fams = MONO_FAMILY if family and "mono" in family else CJK_FAMILY
    prop = FontProperties(family=fams)
    if weight and weight not in ("normal", "400"):
        try:
            prop.set_weight(int(weight) if str(weight).isdigit() else weight)
        except Exception:
            prop.set_weight("bold")
    tex = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", tex)
    tex = re.sub(r"\\le(?![A-Za-z])", r"\\leq", tex)
    try:
        _w, _h, _d, glyphs, rects = _PARSER.parse(tex, 72, prop)
    except Exception as e:
        raise ValueError(f"mathtext parse failed for {tex!r}: {e}")
    spans = {}   # style-attr dict key -> [(x, y, ccode)]
    meta = {}    # style key -> (fontfile, name, font_style_dict)
    for font, fontsize, ccode, _gidx, gx, gy in glyphs:
        if ccode == 32:
            ccode = 0xa0
        entry = font_manager.ttfFontProperty(font)
        fs = {"font-size": f"{fontsize:g}px",
              "font-family": f"'{_font_name(font)}'"}
        if entry.style != "normal":
            fs["font-style"] = entry.style
        if entry.weight != 400:
            fs["font-weight"] = f"{entry.weight}"
        if entry.variant != "normal":
            fs["font-variant"] = entry.variant
        if entry.stretch != "normal":
            fs["font-stretch"] = entry.stretch
        key = ";".join(f"{k}:{v}" for k, v in sorted(fs.items()))
        spans.setdefault(key, []).append((gx, -gy, ccode))
        meta[key] = font.fname
        used.setdefault(font.fname, set()).add(chr(ccode))
    children = []
    text_el = ET.Element(f"{NS}text")
    for key, chars in spans.items():
        chars.sort()
        for gx, gy, ccode in chars:
            ts = ET.SubElement(text_el, f"{NS}tspan",
                               {"x": f"{gx:.3f}", "y": f"{gy:.3f}",
                                "style": key})
            ts.text = chr(ccode)
    children.append(text_el)
    for rx, ry, rw, rh in rects:
        children.append(ET.Element(f"{NS}rect", {
            "x": f"{rx:.3f}", "y": f"{-ry - 1:.3f}",
            "width": f"{rw:.3f}", "height": f"{rh:.3f}"}))
    return children, _w


def _embed_style(used, stroke_px):
    """{font_file: {chars}} -> <style> with WOFF2 subsets; math fonts get
    a same-colour stroke via CSS on their tspans? No — stroke is applied
    per-tspan in _render instead. Returns ET <style> element."""
    from fontTools import subset
    from fontTools.ttLib import TTFont
    def _open(f):
        try:
            return TTFont(f)
        except Exception:
            from fontTools.ttLib import TTCollection
            return TTCollection(f).fonts[0]

    faces = []
    for f, chars in sorted(used.items()):
        font = _open(f)
        cmap = font.getBestCmap()
        missing = {c for c in chars if ord(c) not in cmap}
        if missing:
            raise SystemExit(f"{os.path.basename(f)} 缺字形 {sorted(missing)!r}")
        name = os.path.splitext(os.path.basename(f))[0]
        opts = subset.Options(flavor="woff2")
        ss = subset.Subsetter(opts)
        ss.populate(text="".join(sorted(chars)))
        try:
            ss.subset(font)
        except Exception:
            opts.flavor = "woff"
            font = _open(f)
            ss = subset.Subsetter(opts)
            ss.populate(text="".join(sorted(chars)))
            ss.subset(font)
        buf = io.BytesIO()
        font.save(buf)
        mime = "woff2" if opts.flavor == "woff2" else "woff"
        faces.append(
            f"@font-face{{font-family:'{name}';"
            f"src:url(data:font/{mime};base64,"
            f"{base64.b64encode(buf.getvalue()).decode()})}}")
    st = ET.Element(f"{NS}style", {"id": "mathtext-fonts",
                                   "type": "text/css"})
    st.text = "".join(faces)
    return st


def _make_group(tex, x, y, size, anchor, fill, family, weight, used,
                original=None):
    children, width = _render(tex, size, weight, family, used)
    g = ET.Element(f"{NS}g", {
        "class": "mathtext",
        "data-tex": tex,
        "data-x": f"{x:g}",
        "data-y": f"{y:g}",
        "data-size": f"{size:g}",
        "data-anchor": anchor,
        "data-fill": fill,
        "fill": fill,
    })
    if family:
        g.set("data-family", family)
    if weight:
        g.set("data-weight", weight)
    if original is not None:
        for attr in ("id", "role", "aria-label", "aria-labelledby", "tabindex"):
            if original.get(attr) is not None:
                g.set(attr, original.get(attr))
        if not g.get("aria-label"):
            g.set("aria-label", "".join(original.itertext()).strip() or tex)
        for child in original:
            if child.tag == f"{NS}title":
                g.append(copy.deepcopy(child))
    x0 = x if anchor == "start" else x - width / 2 if anchor == "middle" else x - width
    wrap = ET.SubElement(g, f"{NS}g",
                         {"transform": f"translate({x0:.3f} {y:.3f})"})
    for c in children:
        wrap.append(c)
    return g


def _apply_stroke(group, stroke_px, fill):
    """CM/STIX glyphs are thinner than Noto — add a same-colour stroke."""
    if stroke_px <= 0:
        return
    for ts in group.iter(f"{NS}tspan"):
        st = ts.get("style", "")
        fam = re.search(r"font-family:\s*'([^']+)'", st)
        if fam and any(h in fam.group(1) for h in _MATH_FONT_HINT):
            ts.set("style", st + f";stroke:{fill};stroke-width:{stroke_px:.3f}px;"
                                f"paint-order:stroke fill")


def _stroke_width(root):
    vb = root.get("viewBox") or ""
    m = re.split(r"[\s,]+", vb.strip())
    try:
        w = float(m[2])
    except Exception:
        return 0.3
    return 0.3 * w / 741


def process(path):
    tree = ET.parse(path)
    root = tree.getroot()
    # drop our previous font block → idempotent
    parent_map0 = {c: p for p in root.iter() for c in p}
    for st in list(root.iter(f"{NS}style")):
        if st.get("id") == "mathtext-fonts":
            parent_map0[st].remove(st)
    rules = _css_rules(root)
    parent_map = {c: p for p in root.iter() for c in p}
    changed = 0
    used = {}
    sw = _stroke_width(root)

    # pass 1: re-render existing mathtext groups (idempotent source of truth)
    for g in list(root.iter(f"{NS}g")):
        if g.get("class") != "mathtext" or not g.get("data-tex"):
            continue
        tex = g.get("data-tex")
        x, y = float(g.get("data-x")), float(g.get("data-y"))
        size = float(g.get("data-size"))
        anchor = g.get("data-anchor", "middle")
        fill = g.get("data-fill", "#222222")
        family, weight = g.get("data-family"), g.get("data-weight")
        for child in list(g):
            if child.tag != f"{NS}title":
                g.remove(child)
        children, width = _render(tex, size, weight, family, used)
        x0 = x if anchor == "start" else x - width / 2 if anchor == "middle" else x - width
        wrap = ET.SubElement(g, f"{NS}g",
                             {"transform": f"translate({x0:.3f} {y:.3f})"})
        for c in children:
            wrap.append(c)
        _apply_stroke(g, sw, fill)
        changed += 1

    # pass 2: <text data-tex> -> mathtext group, in place
    for text in list(root.iter(f"{NS}text")):
        if not text.get("data-tex"):
            continue
        tex = text.get("data-tex")
        x = float(text.get("x", "0"))
        y = float(text.get("y", "0"))
        size_value = (_resolve(text, parent_map, "font-size", rules) or "12").strip()
        size = float(re.sub(r"(?:px|pt)$", "", size_value))
        if size_value.endswith("pt"):
            size *= 96 / 72
        anchor = _resolve(text, parent_map, "text-anchor", rules) or "start"
        fill = _resolve(text, parent_map, "fill", rules) or "#222222"
        weight = _resolve(text, parent_map, "font-weight", rules) or "normal"
        family = _resolve(text, parent_map, "font-family", rules)
        try:
            new = _make_group(tex, x, y, size, anchor, fill, family, weight,
                              used, text)
        except ValueError as e:
            sys.exit(f"{path}: {e}")
        _apply_stroke(new, sw, fill)
        parent = parent_map[text]
        idx = list(parent).index(text)
        if text.get("transform"):
            wrap = ET.Element(f"{NS}g", {"transform": text.get("transform")})
            wrap.append(new)
            parent.remove(text)
            parent.insert(idx, wrap)
        else:
            parent.remove(text)
            parent.insert(idx, new)
        changed += 1

    if used:
        root.insert(0, _embed_style(used, sw))
    if changed or used:
        ET.indent(tree, space=" ")
        tree.write(path, encoding="unicode", xml_declaration=False)
    print(f"{path}: {changed} mathtext group(s), "
          f"{len(used)} embedded font file(s)")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        process(p)


if __name__ == "__main__":
    main()
