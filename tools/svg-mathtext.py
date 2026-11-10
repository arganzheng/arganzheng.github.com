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
import unicodedata
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


_FONTS_DIR = _fonts_dir()
for _f in ("NotoSansSC-Regular.otf", "NotoSansSC-Medium.otf"):
    font_manager.fontManager.addfont(os.path.join(_FONTS_DIR, _f))

CJK_FAMILY = ["Noto Sans SC", "Hiragino Sans GB", "Heiti SC",
              "Arial Unicode MS", "DejaVu Sans"]
CJK_MEDIUM = ["Noto Sans SC Medium", "Noto Sans SC"]
MONO_FAMILY = ["Menlo", "Courier New", "monospace"]
_PARSER = MathTextParser("path")
_MATH_FONT_HINT = ("cmmi", "cmr", "cmsy", "cmex", "cmss", "cmti", "cmtt",
                   "cmb", "STIX")
MATH_SCALE = 1.21   # KaTeX .katex{font-size:1.21em} — CM math runs larger

# KaTeX inter-atom spacing: \; (5/18em) around relations, \: (4/18em) around
# binary operators. Applied by rewriting the $...$ source before parsing.
_REL = re.compile(
    r"\\(?:leq?|geq?|neq|approx|propto|parallel|perp|simeq|cong|equiv|"
    r"subseteq|supseteq|subset|supset|ni|to|rightarrow|leftarrow|mapsto|"
    r"ll|gg|in)(?![A-Za-z])|(?<!\\)[<>]|=")
_BIN = re.compile(
    r"\\(?:times|cdot|div|pm|mp|oplus|ominus|otimes|odot|cup|cap|setminus|"
    r"wedge|vee|circ|ast|star|oplus|otimes)(?![A-Za-z])|(?<![\\\ue000])[+*]")
_UNARY_PREV = set("^_({[=,:;/+-*<>")   # chars after which +-*/ are unary/arg


def _space_math(seg):
    """Insert KaTeX-style spacing inside one $...$ segment (dollars kept)."""
    body = seg[1:-1]
    # unary +,-,* first — mark with a private-use guard so the later passes
    # can't touch them. An operator right after an opening token is unary
    # (start, or after ^ _ { ( [ = , : ; / + - * < > or a space after those).
    out = []
    for i, c in enumerate(body):
        if c in "+-*":
            j = i - 1
            while j >= 0 and body[j] == " ":
                j -= 1
            prev = body[j] if j >= 0 else ""
            if j < 0 or prev in _UNARY_PREV:
                out.append("\ue000" + c)   # private-use sentinel
                continue
        out.append(c)
    body = "".join(out)
    # TeX inserts no inter-atom spacing in sub/superscripts — shield them.
    scripts = []
    def _stash(m):
        scripts.append(m.group(0))
        return f"\ue001{len(scripts)-1}\ue002"
    body = re.sub(r"[_^](?:\{[^{}]*\}|[A-Za-z0-9\\]+)", _stash, body)
    body = _REL.sub(lambda m: "\\;" + m.group(0) + "\\;", body)
    body = _BIN.sub(lambda m: "\\:" + m.group(0) + "\\:", body)
    # binary - / + (not marked unary, not after an opening token or an
    # already-inserted \; \:): medium space both sides
    body = re.sub(r"(?<![\ue000^_{(\[=,:;/+\-*<>\\;])-", r"\\:-\\:", body)
    body = re.sub(r"(?<![\ue000^_{(\[=,:;/+\-*<>\\;])\+", r"\\:+\\:", body)
    # restore protected scripts BEFORE stripping \ue000: stashed script
    # text can itself carry the unary marker (e.g. `^{-4}` -> `{\ue000-4}`),
    # which must still be removed below
    body = re.sub(r"\ue001(\d+)\ue002",
                  lambda m: scripts[int(m.group(1))], body)
    body = body.replace("\ue000", "")
    assert "\ue000" not in body, f"U+E000 left in {body!r}"
    return "$" + body + "$"


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
    """CSS `font:` shorthand -> longhands: [style] [weight] size[/line-height]
    family-list. A later longhand wins per normal cascade (callers merge rules
    in document order), so setdefault only fills what the shorthand gives."""
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


# Bakoma cmex10/cmsys cmaps situate some delimiter glyphs at codepoints
# that are real but *invisible* Unicode characters — e.g. `\langle` is
# emitted at U+00AD SOFT HYPHEN, which browsers never paint. Emit a visible
# display character instead (the true Unicode delimiter when we can name
# it, else a private-use alias) and add the alias to the embedded subset's
# cmap so it still resolves to the same glyph.
_REMAP = {}      # font file -> {display codepoint: source codepoint}
_PUA_NEXT = {}   # font file -> next private-use codepoint
_CMAPS = {}      # font file -> source cmap (lazy)
_DELIM_UV = {"angbracketleftbig": 0x27E8, "angbracketrightbig": 0x27E9,
             "bracketleftbig": 0x5B, "bracketrightbig": 0x5D,
             "braceleftbig": 0x7B, "bracerightbig": 0x7D,
             "parenleftBig": 0x28, "parenrightBig": 0x29,
             "floorleftbig": 0x230A, "floorrightbig": 0x230B,
             "ceilingleftbig": 0x2308, "ceilingrightbig": 0x2309,
             "slashbig": 0x2F, "backslashbig": 0x5C,
             "vextendsingle": 0x2223, "vextenddouble": 0x2225,
             "integraldisplay": 0x222B, "integraltext": 0x222B}


def _display_ccode(font, ccode):
    """Visible codepoint to emit for a glyph the parser addressed via an
    invisible/control codepoint (Cc/Cf); registers a cmap alias."""
    if unicodedata.category(chr(ccode)) not in ("Cc", "Cf"):
        return ccode
    fname = font.fname
    if fname not in _CMAPS:
        from fontTools.ttLib import TTFont
        try:
            _CMAPS[fname] = TTFont(fname).getBestCmap() or {}
        except Exception:
            _CMAPS[fname] = {}
    dst = _DELIM_UV.get(_CMAPS[fname].get(ccode) or "")
    if dst is None:
        dst = _PUA_NEXT.get(fname, 0xE4FF) + 1
        _PUA_NEXT[fname] = dst
    _REMAP.setdefault(fname, {})[dst] = ccode
    return dst


def _prop(size, weight, family):
    weight = (weight or "").strip()
    heavy = bool(weight) and weight not in ("normal", "400")
    fams = MONO_FAMILY if family and "mono" in family else (
        CJK_MEDIUM if heavy else CJK_FAMILY)
    prop = FontProperties(family=fams, size=size)
    if heavy:
        try:
            prop.set_weight(int(weight) if str(weight).isdigit() else weight)
        except Exception:
            prop.set_weight("bold")
    return prop


def _parse_seg(seg, prop):
    seg = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", seg)
    seg = re.sub(r"\\le(?![A-Za-z])", r"\\leq", seg)
    try:
        return _PARSER.parse(seg, 72, prop)
    except Exception as e:
        raise ValueError(f"mathtext parse failed for {seg!r}: {e}")


def _render(tex, size, weight, family, used):
    """Mixed CJK/$...$ string -> (children for a <g>, total advance px).

    One <text font-size="0"> (its own size is irrelevant — every tspan carries
    its own font-size; the 0 keeps the element's bbox at the real glyph
    extents) holding <tspan>s, plus <rect>s for fraction bars. The string is
    split on $...$ and each segment typeset separately so CM math runs at
    1.21× the prose size (KaTeX's convention) and KaTeX-style inter-atom
    spacing can be injected into math source before parsing.
    `used` accumulates {font_file: {codepoint}} for later subsetting.
    """
    text_prop = _prop(size, weight, family)
    math_prop = FontProperties(family="DejaVu Sans", size=size * MATH_SCALE)
    glyphs, rects, adv = [], [], 0.0
    for seg in re.split(r"(\$[^$]*\$)", tex):
        if not seg:
            continue
        is_math = seg.startswith("$") and seg.endswith("$")
        src = _space_math(seg) if is_math else seg
        w, _h, _d, gl, rc = _parse_seg(src, math_prop if is_math else text_prop)
        for font, fontsize, ccode, _gidx, gx, gy in gl:
            ccode = _display_ccode(font, ccode)
            glyphs.append((font, fontsize, ccode, gx + adv, gy))
            used.setdefault(font.fname, set()).add(
                chr(0xa0) if ccode == 32 else chr(ccode))
        for rx, ry, rw, rh in rc:
            rects.append((rx + adv, ry, rw, rh))
        adv += w
    spans = {}   # style key -> [(x, y, ccode)]
    for font, fontsize, ccode, gx, gy in glyphs:
        if ccode == 32:
            ccode = 0xa0
        entry = font_manager.ttfFontProperty(font)
        # text-anchor inside the style key: a CSS rule matching text/tspan
        # (e.g. `text{text-anchor:middle}`) must not re-anchor these runs
        fs = {"font-size": f"{fontsize:g}px",
              "font-family": f"'{_font_name(font)}'",
              "text-anchor": "start"}
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
    children = []
    # explicit start anchoring: an inherited text-anchor (ancestor <g>, CSS)
    # must not re-anchor the absolute-x tspans
    text_el = ET.Element(f"{NS}text", {"font-size": "0",
                                      "text-anchor": "start",
                                      "style": "text-anchor:start"})
    for key, chars in spans.items():
        chars.sort()
        for gx, gy, ccode in chars:
            ts = ET.SubElement(text_el, f"{NS}tspan",
                               {"x": f"{gx:.3f}", "y": f"{gy:.3f}",
                                "style": key, "text-anchor": "start"})
            ts.text = chr(ccode)
    children.append(text_el)
    for rx, ry, rw, rh in rects:
        children.append(ET.Element(f"{NS}rect", {
            "x": f"{rx:.3f}", "y": f"{-ry - 1:.3f}",
            "width": f"{rw:.3f}", "height": f"{rh:.3f}"}))
    return children, adv


# cmr10 vertical metrics, per upem — browsers size a <text> line box from
# the font's ascent/descent, and cmsy10/cmex10 ship descenders of ~2em/~6em
# (the big delimiters), inflating every math run's getBBox until svg-collide
# flags neighbours. Normalise embedded math subsets to cmr10's proportions;
# glyph outlines are untouched, so rendered pixels don't change.
_CMR10_UPEM = 2048
_CMR10_METRICS = {"ascent": 1444, "descent": -442,
                  "sTypoAscender": 1444, "sTypoDescender": -422,
                  "usWinAscent": 1536, "usWinDescent": 512}


def _normalise_metrics(font, fname):
    name = os.path.basename(fname)
    if not any(h in name for h in _MATH_FONT_HINT):
        return
    upem = font["head"].unitsPerEm
    s = upem / _CMR10_UPEM
    hhea = font["hhea"]
    hhea.ascent = round(_CMR10_METRICS["ascent"] * s)
    hhea.descent = round(_CMR10_METRICS["descent"] * s)
    os2 = font["OS/2"]
    os2.sTypoAscender = round(_CMR10_METRICS["sTypoAscender"] * s)
    os2.sTypoDescender = round(_CMR10_METRICS["sTypoDescender"] * s)
    os2.usWinAscent = round(_CMR10_METRICS["usWinAscent"] * s)
    os2.usWinDescent = round(_CMR10_METRICS["usWinDescent"] * s)


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
        rem = _REMAP.get(f, {})
        # display aliases are added to the cmap after subsetting; the
        # subset itself must keep the *source* codepoint so the glyph and
        # its cmap entry survive.
        missing = {c for c in chars if ord(c) not in cmap
                   and ord(c) not in rem}
        if missing:
            raise SystemExit(f"{os.path.basename(f)} 缺字形 {sorted(missing)!r}")
        chars = chars | {chr(s) for s in rem.values()}
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
        if rem:
            best = font.getBestCmap() or {}
            for t in font["cmap"].tables:
                if t.isUnicode():
                    for disp, src in rem.items():
                        gn = best.get(src)
                        if gn and disp not in t.cmap:
                            t.cmap[disp] = gn
        _normalise_metrics(font, f)
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
    anchor = (anchor or "middle").strip()
    fill = (fill or "#222222").strip()
    family = family.strip() if family else None
    weight = weight.strip() if weight else None
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
        anchor = (g.get("data-anchor") or "middle").strip()
        fill = (g.get("data-fill") or "#222222").strip()
        family = (g.get("data-family") or "").strip() or None
        weight = (g.get("data-weight") or "").strip() or None
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
        anchor = (_resolve(text, parent_map, "text-anchor", rules) or "start").strip()
        fill = (_resolve(text, parent_map, "fill", rules) or "#222222").strip()
        weight = (_resolve(text, parent_map, "font-weight", rules) or "normal").strip()
        family = (_resolve(text, parent_map, "font-family", rules) or "").strip() or None
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
