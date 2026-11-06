#!/usr/bin/env python3
"""Typeset `data-tex` text elements in a hand-written SVG as LaTeX-style paths.

In place:

    tools/svg-mathtext.py img/in-post/foo.svg [more.svg ...]

Every `<text data-tex="…">` is replaced by `<g class="mathtext" data-tex="…"
data-x data-y data-size data-anchor data-fill data-family>` whose children are
`<path>`s rendered by matplotlib (Computer Modern mathtext for `$...$` parts,
the CJK/system font for the rest — same look as the posts' KaTeX). The
`data-tex` string is the editable source: rerunning the tool re-renders every
`<g class="mathtext" data-tex>` from its stored attributes, so edit the string,
not the paths. A `$...$` that doesn't parse aborts the file with an error.

Baseline sits at the original `y`; `text-anchor` start/middle/end is honored.
Fill comes from the text's `fill`, inline style, `<style>` class rule, or
ancestor/parent `fill` (default #222). Non-math text uses the resolved
font-family (mono classes → monospace stack; CJK → Hiragino Sans GB etc.).
"""
import copy
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

plt.rcParams.update({"mathtext.fontset": "cm"})

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
NS = f"{{{SVG}}}"

CJK_FAMILY = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
MONO_FAMILY = ["Menlo", "Courier New", "monospace"]


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


def _path_d(tp):
    """matplotlib Path -> SVG path data (vertex space is typographic points;
    flip y since SVG y grows downward)."""
    from matplotlib.path import Path as MplPath
    parts, i = [], 0
    verts, codes = tp.vertices, tp.codes
    n = len(codes)
    while i < n:
        c, v = codes[i], verts[i]
        if c == MplPath.MOVETO:
            parts.append(f"M {v[0]:.3f} {-v[1]:.3f}")
            i += 1
        elif c == MplPath.LINETO:
            parts.append(f"L {v[0]:.3f} {-v[1]:.3f}")
            i += 1
        elif c == MplPath.CURVE3:
            v2 = verts[i + 1]
            parts.append(f"Q {v[0]:.3f} {-v[1]:.3f} {v2[0]:.3f} {-v2[1]:.3f}")
            i += 2
        elif c == MplPath.CURVE4:
            v2, v3 = verts[i + 1], verts[i + 2]
            parts.append(f"C {v[0]:.3f} {-v[1]:.3f} {v2[0]:.3f} {-v2[1]:.3f} "
                         f"{v3[0]:.3f} {-v3[1]:.3f}")
            i += 3
        elif c == MplPath.CLOSEPOLY:
            parts.append("z")
            i += 1
        else:
            i += 1
    return "\n".join(parts)


def _render(tex, size, weight, family):
    """Mixed CJK/$...$ string -> (path-d list, total advance width in px).

    TextPath units are points at dpi=72 → treat as SVG px 1:1.
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
        tp = TextPath((0, 0), tex, size=size, prop=prop)
    except Exception as e:
        raise ValueError(f"mathtext parse failed for {tex!r}: {e}")
    ext = tp.get_extents()
    return _path_d(tp), ext.width


def _make_group(tex, x, y, size, anchor, fill, family, weight, original=None):
    d, width = _render(tex, size, weight, family)
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
    p = ET.SubElement(g, f"{NS}path", {
        "d": d,
        "transform": f"translate({x0:.3f} {y:.3f})",
    })
    return g


def process(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rules = _css_rules(root)
    parent_map = {c: p for p in root.iter() for c in p}
    changed = 0

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
        titles = [copy.deepcopy(child) for child in list(g) if child.tag == f"{NS}title"]
        for child in list(g):
            if child.tag != f"{NS}title":
                g.remove(child)
        d, width = _render(tex, size, weight, family)
        x0 = x if anchor == "start" else x - width / 2 if anchor == "middle" else x - width
        ET.SubElement(g, f"{NS}path", {
            "d": d, "transform": f"translate({x0:.3f} {y:.3f})"})
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
            new = _make_group(tex, x, y, size, anchor, fill, family, weight, text)
        except ValueError as e:
            sys.exit(f"{path}: {e}")
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

    if changed:
        ET.indent(tree, space=" ")
        tree.write(path, encoding="unicode", xml_declaration=False)
    print(f"{path}: {changed} mathtext group(s)")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        process(p)


if __name__ == "__main__":
    main()
