#!/usr/bin/env python3
"""Static lint for ```mermaid flowchart/graph blocks in slide/post sources.

FAILs on:
  bare-id    an ID used in an edge (or class list) that never got a label —
             mermaid renders it with its raw ID as the label.
  isolated   a labelled node that never appears in any edge (in a flowchart
             that has edges).
  dead-end   a decision {} node with no outgoing edge.

Non-flowchart diagrams (sequenceDiagram, pie, etc.) are skipped — different
syntax. Quoted strings are stripped before parsing so label text can never
be mistaken for node IDs.
"""
import re
import sys

KEYWORDS = {'subgraph', 'end', 'direction', 'classDef', 'class', 'style',
            'linkStyle', 'flowchart', 'graph', 'click', 'invis'}

ARROW_SPLIT = re.compile(r'-{1,}\.{0,}-{0,}\|?[^|]*\|?\.?-{0,}>|<-{1,}\.{0,}-{0,}\|?[^|]*\|?\.?-{0,}|===>|~~>|<-->|--x')
# we use a simpler split below after normalisation


def strip_quotes(line):
    line = re.sub(r'"[^"]*"', '', line)
    return re.sub(r'`[^`]*`', '', line)


def tokens(part):
    """Node IDs in a chain part like 'A & B["x"]' (quotes already stripped)."""
    ids = re.findall(r'\b([A-Za-z_][\w.]*)\b', part)
    return [i for i in ids if i not in KEYWORDS]


def lint_block(block, fname, idx):
    # only lint flowchart/graph
    head = block.lstrip().split('\n')
    first = next((l for l in head if l.strip() and not l.strip().startswith('%%')), '')
    if not re.match(r'\s*(flowchart|graph)\b', first):
        return []

    # A quoted label may span lines (`A["line 1` + newline + `line 2"]`); the
    # per-line passes below would then read its second line as code and
    # report words in it as bare ids / isolated nodes. Drop whole-line `%%`
    # comments first (`%%{init: {"…"}}%%` has quotes of its own), then blank
    # every quoted / backtick string across lines, keeping the empty quotes
    # so `A[""]` still counts as a labelled node.
    block = '\n'.join(l for l in block.split('\n') if not l.strip().startswith('%%'))
    block = re.sub(r'"[^"]*"', '""', block)
    block = re.sub(r'`[^`]*`', '``', block)
    # `A["…"]:::cls` attaches a classDef; the class name is not a node
    block = re.sub(r':::[\w-]+', '', block)
    # unquoted text on an edge: `A -. sleep / wake .-> B`, `A -- text --> B`,
    # `A == text ==> B` — the words are a label, not node ids
    block = re.sub(r'(<?)-\.[ \t]+[^\n]*?[ \t]+\.->', r'\1-.->', block)
    block = re.sub(r'(<?)--[ \t]+[^\n>]*?[ \t]+-->', r'\1-->', block)
    block = re.sub(r'(<?)==[ \t]+[^\n>]*?[ \t]+==>', r'\1==>', block)

    labelled = set()
    decision = set()
    subgraphs = set()
    edge_out = {}
    edge_ids = set()
    has_edges = False
    issues = []
    in_subgraph = set()      # node ids declared inside any subgraph
    substack = []

    # labels can span multiple lines — collect decls on the whole block
    full = strip_quotes(re.sub(r'%%[^\n]*', '', block))
    for m in re.finditer(r'\b([A-Za-z_][\w.]*)\s*(\[[^\]]*\]|\(\[[^\]]*\]\)|\([^)]*\)|\{[^}]*\})', full):
        labelled.add(m.group(1))
        if m.group(2).startswith('{'):
            decision.add(m.group(1))

    for raw in block.split('\n'):
        line = raw.split('%%')[0]
        if not line.strip():
            continue
        sm = re.match(r'\s*subgraph\s+([A-Za-z_][\w.]*)', line)
        if sm:
            subgraphs.add(sm.group(1))
            substack.append(sm.group(1))
            continue
        if re.match(r'\s*end\b', line):
            if substack:
                substack.pop()
            continue
        if re.match(r'\s*(flowchart|graph|direction|classDef|style|linkStyle)\b', line):
            continue
        cm = re.match(r'\s*class\s+([\w,]+)\s+\w+', line)
        if cm:
            for cid in cm.group(1).split(','):
                cid = cid.strip()
                if cid:
                    issues.append(('class-ref', cid))
            continue

        body = strip_quotes(line)
        # collect labelled / decision decls from whole line
        for m in re.finditer(r'\b([A-Za-z_][\w.]*)\s*(\[[^\]]*\]|\(\[[^\]]*\]\)|\([^)]*\)|\{[^}]*\})', body):
            labelled.add(m.group(1))
            if substack:
                in_subgraph.add(m.group(1))
            if m.group(2).startswith('{'):
                decision.add(m.group(1))
        # bare decision decl like X{...}
        for m in re.finditer(r'\b([A-Za-z_][\w.]*)\s*\{', body):
            decision.add(m.group(1))

        # normalise arrows: -->, -. "t" .->, -.->, ==>, --x, ~~~, <-->
        norm = strip_quotes(body)
        norm = norm.replace('-.', '--').replace('.->', '-->').replace('~~~', '-->')
        norm = re.sub(r'--\|[^|]*\|', '-->', norm)
        norm = re.sub(r'==+>', '-->', norm)
        norm = re.sub(r'-{2,}x', '-->', norm)
        norm = re.sub(r'-{2,}>', '-->', norm)
        norm = re.sub(r'-{3,}', '-->', norm)  # A --- B undirected edge
        norm = re.sub(r'<--', '-->', norm)   # direction-agnostic for connectivity
        if '-->' not in norm:
            continue
        has_edges = True
        parts = norm.split('-->')
        prev = None
        for part in parts:
            part = re.sub(r'\|[^|]*\|', '', part)  # edge labels between |..|
            part = re.sub(r'\[[^\]]*\]|\([^)]*\)|\{[^}]*\}', '', part)  # strip shape text
            ids = tokens(part)
            if ids:
                if prev is not None:
                    for s_ in prev:
                        for d_ in ids:
                            edge_out.setdefault(s_, []).append(d_)
                            edge_ids.update((s_, d_))
                prev = ids

    # subgraph ids count as labelled nodes if used in edges
    labelled |= subgraphs

    out = []
    for kind, cid in issues:
        if cid not in labelled:
            out.append((fname, idx, cid, 'bare-id(class)'))
    if has_edges:
        for nid in sorted(edge_ids):
            if nid not in labelled and nid not in decision:
                out.append((fname, idx, nid, 'bare-id(edge)'))
        for nid in sorted(labelled | decision):
            if nid not in edge_ids and nid not in in_subgraph and nid not in subgraphs:
                out.append((fname, idx, nid, 'isolated'))
        for nid in sorted(decision):
            if nid in edge_ids and not edge_out.get(nid):
                out.append((fname, idx, nid, 'dead-end decision'))
    return out


def lint_file(path):
    s = open(path, encoding='utf-8').read()
    problems = []
    for i, m in enumerate(re.finditer(r'```mermaid\n(.*?)```', s, re.S), 1):
        problems += lint_block(m.group(1), path, i)
    return problems


def main():
    files = sys.argv[1:]
    total = []
    for f in files:
        total += lint_file(f)
    for f, idx, nid, kind in total:
        print(f"{f}:mermaid#{idx}: {kind}: {nid}")
    print(f"{len(total)} problem(s) in {len(files)} file(s)")
    sys.exit(1 if total else 0)


if __name__ == '__main__':
    main()
