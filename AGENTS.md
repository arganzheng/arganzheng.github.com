# AGENTS.md

## Build / preview

Jekyll 4.4.1 is installed against Homebrew's Ruby.
Homebrew Ruby and Gem paths are configured in `~/.zshrc`:
`/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin`

A `Gemfile` is also present at repo root:

```bash
jekyll build            # or bundle exec jekyll build -> _site/
jekyll serve            # or bundle exec jekyll serve -> http://localhost:4000
```

The pre-existing `Conflict: ... the-productive-programmer-on-windows.html`
warning is harmless.

`Gruntfile.js` compiles `less/ -> css/argan-blog{,.min}.css` and minifies
`js/hux-blog.js`. Edit the `.less`/`.js` sources, not the generated CSS:

```bash
npm install && npx grunt        # or `npx grunt watch`
```

**WARNING — CSS drift:** `npx grunt` currently fails (grunt-cli is not
installed and the Gruntfile expects a non-existent `less/argan-blog.less`;
the real entry point is `less/blog.less`). More importantly,
`css/argan-blog{,.min}.css` contain ~485 lines of hand-appended rules
(from `.page-header .title` onwards) that are NOT in the `.less` sources,
so a full recompile destroys them. To change styles, edit the `.less`
source for reference, then append the corresponding compiled CSS to the
END of both `css/argan-blog.css` and `css/argan-blog.min.css` by hand
(compile a fragment with `node_modules/.bin/lessc` if helpful).

## Layout

- `_posts/` — blog posts, `layout: post`, permalink `/:title.html`
- `slides/` — reveal.js decks, `layout: slides` (or set in front matter),
  URL `/slides/:name.html`, indexed by `slides.html` (`/slides/`)
- `_includes/rich-content.html` — Mermaid + KaTeX loaders, shared by
  `_includes/head.html` and `_layouts/slides.html`. Both renderers are lazy:
  they only fetch their bundle if the page actually contains a diagram/formula,
  and they only look inside `.post-container` and `.reveal .slides`.
- `_includes/analytics.html` — GA + Baidu Tongji, shared by `footer.html` and
  `_layouts/slides.html`

## Writing a deck

Decks are ordinary Markdown; kramdown renders the file and `_layouts/slides.html`
splits the HTML on every `<hr>` into reveal.js `<section>`s.

- Create `slides/my-talk.md` with `layout: slides` and `permalink: /slides/my-talk.html` in the front matter.
- Separate slides with `---` and **always leave a blank line before it**,
  otherwise Markdown reads it as a setext `<h2>` underline and the slide is
  not split.
- `<!-- v -->` inside a slide creates vertical (nested) sub-slides.
- `<!-- .slide: data-background="#1c1f26" -->` puts reveal.js attributes on the
  current `<section>`.
- `{: .fragments}` on a list reveals it one item at a time; `{: .fragment}`
  reveals any single element.
- Speaker notes: `<aside class="notes" markdown="1">...</aside>`, shown with `S`.
- Export: append `?print-pdf` and print from the browser.

`slides/reveal-demo.md` is a live demo of all of the above.

## Writing AI-Infra series posts

- **Version/date rule:** a post may only cite software versions (and facts about
  them) released *before* the post's front-matter date. Check tag dates with
  `git log -1 --format=%cs <tag>` before pinning a version; pin an explicit tag
  (e.g. "NCCL 2.28.9", "PyTorch 2.12", "vLLM v0.23.0"), never "近期主线" or a
  local checkout. Pinned checkouts used so far live next to this repo:
  `../nccl` (v2.28.9-1), `../nccl-tests` (v2.18.3), `../pytorch-v2.12.0`,
  `../vllm-v0.23.0` (git worktrees of `../pytorch` / `../vllm`); series 7 adds
  `../pytorch-v2.13.0`, `../Megatron-LM` (core_v0.18.0), `../DeepSpeed`
  (v0.19.2), `../torchtitan` (v0.3.0), `../torchft` (v0.2.0),
  `../nvidia-resiliency-ext` (v0.6.0); series 6 (MoE post) adds `../DeepEP`
  (v1.2.1); series 8 uses `../vllm-v0.27.1`; series 10 uses
  `../pytorch-v2.14.0` and `../vllm-v0.28.0`. Series 2 pins PyTorch v2.10.0 /
  vLLM v0.15.0 and series 5 pins vLLM v0.20.0 but have no local worktree —
  add one (`git -C ../vllm worktree add ../vllm-v0.20.0 v0.20.0`) before
  re-verifying their source citations.
- **Series-nav quote** is the first line after front matter:
  `> 本文是[《系列名》](/overview.html)系列的第 N 篇（共X篇）。上一篇：[…](/slug.html)；下一篇：[…](/slug.html)`
  (Arabic N, Chinese total; first post has no 上一篇, last has no 下一篇).
- **Update-note exception:** when no usable version predates the post (no tag,
  or the only tag is months stale), a post may cite a newer version *if* it
  carries a note right after the series-nav quote:
  `> **更新 @YYYY-MM-DD**：本文 X 部分基于 vA 刷新；其余源码引用仍以 … 为准。`
  Use it sparingly, list only the projects actually refreshed, and keep one
  version set per project per post — refreshing means re-verifying every claim
  about that project, never mixing two versions in one article.
- Cite source as path + function/class name, never line numbers.
- Length is not a target; rigor and organisation are. Structure: nav quote →
  intro with the post's core question → `## 一、总览` (ending with 本文的章节安排)
  → body (`##` Chinese numerals, `###` Arabic) → `## N、本文小结` → `## 下一篇`.
- Series are independent: no links to posts of other series.
- `{%`/`{{` inside code (PTX asm, printf formats, regexes) must be wrapped in
  `{% raw %}` … `{% endraw %}` or the Liquid pass fails the build.
- **图文并茂，一图胜千言.** Posts must not be walls of text. Whenever a
  concept is about *structure, flow, layout, or state over time* (architecture,
  execution hierarchy, memory layout, timelines, decision trees, algorithm
  state evolution), draw it — a diagram is the primary explanation and the prose
  supports it, not the other way round. Before finishing a post, re-read it
  section by section and ask "would a reader understand this faster from a
  picture?"; if yes, add one. Each diagram must carry an explanation the text
  cannot easily give — no decorative figures.
  - **The test is gain, not coverage.** Before drawing, ask what the diagram
    shows that the surrounding text/list/table does not: a mapping (index ↔
    storage), a concurrency relation (two streams, who waits for whom), a
    branching decision, a state that evolves. If the diagram would just
    restate a list with arrows between the items, don't draw it. Concrete
    things that do *not* deserve a diagram (learned the hard way — the
    series-overview post's Mermaid figures were all removed as 画蛇添足):
    reading-order suggestions, chapter outlines, "capability ladders" of
    questions, series roadmaps, anything whose content is a linear list or is
    already a table. Overview/navigation posts usually need a table at most.
  - When a batch of diagrams is added (e.g. by parallel agents working from a
    checklist), re-read each one afterwards with the same question; a
    checklist item is a hypothesis, not a mandate.
  - Structure / flow / timelines / decisions → ```` ```mermaid ```` (rendered by
    `_includes/rich-content.html`, Mermaid 10.9.1: `~~~` invisible links to force
    row/column order, `classDef` colours, `<br/>` in quoted labels; horizontal
    layouts shrink to unreadable size at 755 px width, so favour `flowchart TB`
    and split overly tall graphs).
  - Cell-exact layouts (byte/sector maps, bank mappings, reduction trees,
    thread→address tables) → monospace ASCII in a fenced `text` block; exact
    alignment matters more than styling here and Mermaid renders them badly.
  - Anything neither handles well (log-axis plots, precise geometry, dense
    grids) → generate an SVG/PNG into `img/in-post/<post-slug>-<name>.{svg,png}`
    and embed with `![alt](/img/in-post/...)`.
  - Verify rendering in a real browser (`jekyll serve` + check `.mermaid-error`
    and eyeball each SVG's size), not just `jekyll build`. Use the checker
    script for this:

    ```bash
    jekyll serve --future &                              # http://localhost:4000
    ~/.claude/skills/browser/scripts/start.cjs           # Chrome with CDP on :9222
    node tools/check-render.cjs <post-slug> [<post-slug> ...]
    ```

    `check-render.cjs` opens each `/<slug>.html` in its own Chrome tab, waits
    for Mermaid to finish, prints `[PASS]`/`[FAIL]` with `mermaid/ok/errs`
    counts, the first lines of any failing diagram source (`errTexts`), each
    SVG's rendered size (`sizes`, flag anything > 1600 px tall or squeezed
    < 450 px wide), `/img/in-post/` image sizes, broken images and `<pre>`
    blocks that overflow horizontally (`widePre`), then closes the tab. Safe
    to run from several agents in parallel. Post slug = file name without the
    date prefix and `.md`. It needs only the `ws` module, which it loads from
    `~/.claude/skills/browser/node_modules` (the browser skill's install); if
    that path moves, edit the `require` at the top of the script. Hand-drawn SVGs still need an eyeball pass for
    label collisions: `curl -s -X PUT "localhost:9222/json/new?http://localhost:4000/img/in-post/<name>.svg"`
    then `~/.claude/skills/browser/scripts/screenshot.cjs` and view the PNG.
  - Mermaid 10.9.1 pitfalls seen so far: reserved words as node IDs (`end`,
    `call`, `click`, `style`, `class`, `default`, `o`, `x`) break parsing;
    always quote labels and write literal `[`/`]`/`{`/`}` as `#91;`/`#93;`/
    `#123;`/`#125;`; one message per line in `sequenceDiagram`, no `;` inside.
    Brief used for the diagram pass: `tools/diagram-brief.md`.
