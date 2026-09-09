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

## Deploy

`.github/workflows/deploy.yml` builds with the Gemfile's Jekyll (4.4) and
publishes via `actions/deploy-pages` (Pages `build_type: workflow`; the legacy
GitHub-side Jekyll 3.10 build is off). Triggers: push to `master`, daily at
00:05 Beijing (so future-dated posts go live on their date — the build has no
`--future`), or manually. `timezone: Asia/Shanghai` in `_config.yml` decides
what "future" means. `Gemfile.lock` carries the `x86_64-linux` platform for the
runners (`bundle lock --add-platform x86_64-linux` after changing gems).
Pages has `https_enforced` on.

## CI

- `.github/workflows/check.yml` (push / PR touching posts, layouts, includes,
  data, img, js, css): `jekyll build --strict_front_matter`, then **lychee
  offline** over `_site/**/*.html` (every internal link, image and `#fragment`
  must resolve — this is what caught the phantom `/pwa/manifest.json`), then
  `tools/check-render.cjs` in headless Chrome for the posts changed in the
  push (or the 15 newest when only infrastructure changed; `workflow_dispatch`
  takes explicit slugs). `check-render.cjs` reads `SITE` / `CDP` from the
  environment and falls back to the browser skill's `ws` when there is no
  local one. Run the link check locally with `brew install lychee` and
  `lychee --offline --root-dir $PWD/_site --include-fragments --exclude '/tags/?#' '_site/**/*.html'`
  (`/tags/#x` anchors are excluded: lychee cannot check fragments on a
  directory index). Gotcha: lychee skips `<pre>`/`<code>` content and treats an
  unclosed `<pre>` *anywhere* — even inside a JS comment — as "rest of the
  document is verbatim", silently checking nothing after it. Never write a
  literal `<pre>` in inline script comments.
- Images: `img/in-post/` is WebP (`tools/webp-images.py` converted the old
  png/jpg in bulk and rewrote references; run it again for new large images,
  `--apply` to write). Site-level `img/*.jpg` stay JPEG (og:image targets).
- `.github/workflows/links.yml` (Mondays, or manual): external links, never
  blocking; opens/updates an issue labelled `dead-links`. Set the repo
  variable `DEAD_LINKS_ISSUE` to an issue number to keep updating one issue.

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
- `_includes/seo.html` — Open Graph / Twitter Card / JSON-LD (`BlogPosting` for
  the three post layouts, `WebSite` elsewhere), included from `head.html`.
  Description priority: `page.description` > `page.subtitle` > excerpt.
  `og:image` = `page.header-img` or `site.header-img`.
- `_includes/post-meta.html` — the "Posted by … | date (· 更新于) | 约 N 分钟 ·
  X.Xk 字 | N 次阅读" line under the title, shared by the three post layouts.
  Reading time = HTML-stripped body without `<pre>` blocks / 450 chars per
  minute. Optional front matter: `updated: YYYY-MM-DD` (also feeds
  `dateModified` / `article:modified_time`), `description:` (SEO text).
- Post layouts pipe `content` through `replace: '<img src=', '<img
  loading="lazy" decoding="async" src='` — every content image is lazy.
  `js/diagram-zoom.js` opens Mermaid diagrams *and* content images (>= 200 px
  natural width, not inside `<a>`) in the zoom/pan lightbox.
- Local CSS/JS in `head.html` / `footer.html` carry `?v=<build time>` for
  cache busting (GitHub Pages serves `max-age=600`; the old `no-cache` meta
  tags were removed). Font Awesome 4.7 is a self-hosted **subset**:
  `tools/fa-subset.py` scans templates/js/posts/less for `fa-*` classes and
  `content:"\fXXX"` glyphs and writes `css/font-awesome.min.css` +
  `fonts/fontawesome-webfont.woff2` (~5 KB each) from the full copies in
  `tools/fa/`. **Re-run it after using a new icon**, or it renders as a blank
  box (needs `pip install fonttools brotli`). `sw.js` is disabled via
  `service-worker: false`.
- `_includes/comments.html` — comment section (GitHub Discussions), used by
  `_layouts/post.html`, `header-post.html` and `keynote.html`. It is an empty
  `section.comment > .annotation-comments` shell with data attributes; the
  list and editor are rendered by `js/annotations.js` (same code as the
  highlight comments, see below). Repo/category IDs live in the `giscus:`
  block of `_config.yml` (the giscus GitHub App is still the login broker and
  the thread mapping is giscus-compatible: one discussion per `page.url`, so
  renaming a permalink orphans its comments). No giscus iframe is loaded any
  more — it could not be given edit/delete/toolbar/issue controls. Disqus was
  removed even earlier: blocked in mainland China and it injected VigLink
  affiliate links (`a.vglnk`) into article text.
- `js/annotations.js` — comments, reader highlight comments ("划线评论"), likes /
  votes and page views, see below.

## Comments & highlight comments (js/annotations.js)

User-facing wording is always 评论 (划线评论 for passage-level ones, 回复 for
replies) — never 标注 / 批注. "annotation" survives only in identifiers, file
names and the W3C selector terminology.

Reader-facing demo/manual: `_posts/2026-08-03-highlight-annotations-demo.md`
(`/highlight-annotations-demo.html`); the author-side features (footnotes,
tips, external links) have their own demo post, `popup-footnotes-and-inline-tips-demo`.

One data model, two views. `comments` holds every top-level comment of the
post's discussion (`parseComment`); those whose body starts with the quote
header get `selector`/`noteHTML` and are the `annotations` highlighted in the
article. `syncViews()` re-renders both the highlights/panel
(`applyHighlights`) and the bottom section (`renderCommentSection`) after
every mutation. `commentEl` and `renderEditor` are shared, so plain comments
and passage notes have identical reply / edit / delete / 「同时提交 Issue」
controls. The bottom section (`.annotation-comments`): header bar
(`renderLikeBar`: 「有用」 like button, page views, count, GitHub link), `.ac-group` per top-level comment with its replies, inline reply
editor (`openInlineReply`), and a persistent editor (`.ac-editor`,
`clearOnSubmit`, draft in `sessionStorage`). Plain comments filed with an
issue start with `<sub>[⚑ Issue #N](url)</sub>` (recognised by
`parseBodyHeader`). Soft-deleted comments (GitHub keeps a comment that has
replies, `deletedAt` set) render as 「此评论已删除」 with their replies.

Code-review / WeChat-reading style, no hover popups. Readers select text in
`.post-container` → floating toolbar (`评论` / `复制链接`) → an **in-flow editor
panel** is inserted right after the paragraph (取消 / 提交评论 bottom-right).
The note is posted as a **normal comment** of the post's giscus Discussion:

```markdown
> quoted passage
>
> <sub>[§ 原文位置](https://arganzheng.life/<slug>.html#annot-<fnv1a of quote>)</sub>

note (Markdown)
```

Further notes on the *same passage* are either new top-level comments with the
same quote (grouped into one thread by identical anchor range; the panel's
editor defaults to this) or replies to a specific comment
(`addDiscussionComment` with `replyToId`, via each comment's 「回复」 button).
When a selection lies inside an existing annotation's range the toolbar's
「评论」 opens that passage's thread instead of a new editor, so a passage has
exactly one thread on GitHub too. The editor has a small Markdown toolbar
(`applyFormat`) and its submit button is disabled while empty.

- **「同时提交 Issue」** (checkbox next to the buttons, passage-level notes only,
  shown when `_config.yml` `annotations.issues: true`): code-review style
  "this needs fixing". The worker's `POST /issues` files a GitHub Issue
  (label `划线评论`) *first*, then the comment is posted with
  `· [⚑ Issue #N](url)` appended inside the `<sub>` line; `parseComment` reads
  that link back and the panel shows the note with a red flag badge and left
  stripe (`.has-issue`). The reader's giscus token cannot open issues (the
  giscus GitHub App only has the Discussions permission), so the worker acts
  as *our own* GitHub App (Issues: write, installed on the repo): it signs an
  RS256 JWT with `GITHUB_APP_PRIVATE_KEY` (secret; PKCS#1 or PKCS#8 PEM) for
  `GITHUB_APP_ID` (var), exchanges it for a cached 1 h installation token,
  verifies the reader via `GET /user` with *their* token and credits them in
  the issue body. No expiring credentials. Without the key the route returns
  501 and the client hides the checkbox. Setup steps: worker README.
- **Own comments** (author login == `viewer.login`) get 编辑 / 删除 in the
  meta row. Edit fetches the raw body (`node(id){ body }`), strips the quote
  header for top-level notes (`stripQuoteHeader`), reuses `renderEditor` with
  `inline: true` inside the comment, and saves with `updateDiscussionComment`
  re-wrapping via `buildCommentBody` (selector + issue link preserved).
  Delete = `deleteDiscussionComment` after `confirm`; state is patched locally
  and `applyHighlights()` re-renders/closes the panel. GitHub does *not*
  cascade: a top-level comment with replies is soft-deleted (`deletedAt` set,
  replies kept, shown as "This comment was deleted" on GitHub and in giscus);
  `parseComment` skips `deletedAt` comments, so such a thread vanishes from
  the article on reload — the confirm text says so. The thread re-renders
  once the viewer query returns so the buttons appear on first open.
- **Likes / votes** are plain GitHub reactions, no own storage: the post's
  「有用」 is `THUMBS_UP` on the Discussion (`toggleLike`, creates the
  discussion first for an uncommented post), each comment's ▲ score ▼
  (`.ap-vote`, in the meta row) is `THUMBS_UP` / `THUMBS_DOWN` on that
  comment (`toggleVote`, optimistic, switching sides removes the other
  reaction first; `addReaction` / `removeReaction`). `parseVotes` reads both
  giscus' `{THUMBS_UP:{count,viewerHasReacted}}` and GraphQL
  `reactionGroups`. The relay's payload is anonymous, so `loadViewerReactions`
  re-queries the discussion with the reader's token once the viewer is known
  and patches `votes.mine` / `likes.mine` in place (`updateVoteEls`). GitHub's
  native discussion-comment `upvote` is deliberately not used (top-level only,
  no downvote). Don't re-add the giscus iframe for reactions.
- **最受关注的段落** (`renderHotPassages`, `.ac-hot` above the comment list):
  anchored passages ranked by `2 × net votes + comments`, shown only when
  there are 2+ passages, max 3; clicking scrolls to the passage and opens its
  thread. Re-ranked on every vote (`updateVoteEls`).
- **Page views**: `loadViews()` → `POST /views {path}` once per browser per
  post per day (`localStorage["viewed:<path>"]`), otherwise `GET /views`;
  localhost never increments. The worker keeps `views(path, count)` in a D1
  database (binding `DB` in `wrangler.toml`; without it the route is 501 and
  the counter is simply not shown). Rendered in the head bar and into
  `.post-views` in the post header (all three post layouts have the span).
- **Spacing gotcha**: the theme's `.post-container img { margin: 1.5em auto
  1.6em }` hits every `<img>` inside the in-flow panel — avatar rules must
  reset `margin: 0` or replies get ~40 px of phantom whitespace.

On load the thread is fetched, comments of that shape are parsed into a W3C
`TextQuoteSelector` (exact = blockquote text), anchored exactly or fuzzily
(`js/vendor/approx-string-match.js`, MIT, Hypothesis' algorithm) and wrapped in
`<mark class="annotation-hl">` (one mark per text piece carrying every covering
id — overlaps never nest). Each distinct passage gets a `.annotation-marker`
(comment icon + count) after its last mark; clicking it or the highlight
toggles the **thread panel** below the paragraph: notes, replies, and an editor
to join. Anchors that no longer match are listed under the comment hint as
"未能定位". Only one panel exists (`panelState`); it is re-rendered after
re-anchoring.

- **Links**: `#annot-<hash>` opens a thread, `#hl=<readable text>` (from
  `复制链接`) flashes a passage for 2.5 s. Both are handled by the script on
  load *and* on `hashchange` (giscus opens `§ 原文位置` in the same tab). No
  Text Fragment (`:~:text=`) is emitted any more: browsers hide it from
  `location.hash`, its native matcher breaks on footnote markers, and its purple
  `::target-text` highlight never goes away. Old links with `:~:text=` are
  still parsed.
- **Data path**: the browser cannot call `giscus.app/api/*` (CORS is limited
  to giscus' own origin) or GitHub GraphQL anonymously, so
  `tools/annotations-worker/` is a thin Cloudflare Worker that relays
  `GET /discussions?term=` (giscus public API, 60 s edge cache, `&t=` bypasses),
  `POST /token` (giscus session → GitHub token), `POST /discussions`
  (create the thread for a post nobody commented on yet), the optional
  `POST /issues` (needs the GitHub App credentials, above) and the optional
  `GET`/`POST /views` page counter (needs the D1 binding). Deploy with
  `wrangler deploy` (see its README) and put the URL in `_config.yml`
  `annotations.api`; an empty `api` disables the feature.
- **Login** = redirect to `giscus.app/api/oauth/authorize?redirect_uri=<page>`;
  giscus comes back with `?giscus=<session>`, which `takeSessionFromUrl()`
  stores in `localStorage["giscus-session"]` (JSON string, same as giscus'
  client.js did) and strips from the URL. The worker exchanges the session for
  a GitHub token and the browser calls `api.github.com/graphql` directly. The
  viewer is fetched once (`fetchViewer`, `onViewerKnown` re-renders every
  editor and list); 「退出」 clears the session (`logout`). This relies on
  giscus' OAuth/API endpoints, which are stable but not a public contract —
  any failure degrades to a "复制内容" button (paste on GitHub). Editor drafts
  survive the login round-trip in `sessionStorage`.
- **Local dev**: `cd tools/annotations-worker && npx wrangler dev` (no login
  needed, talks to the real giscus API), then in the console
  `localStorage.annotationsApi = 'http://localhost:8787'`. A static JSON with
  the `{discussion:{comments:[…]}}` shape served with CORS also works for
  testing anchoring. `window.BlogAnnotations` exposes
  `reload/anchor/buildIndex/annotHash/threadLink/shareLink/buildCommentBody/
  parseComment/parseVotes/openThread/closePanel/logout/list/comments`.
- **Interplay with footnotes / tips**: the text index excludes footnote
  markers (`sup[id^=fnref]`, `a.footnote`, `.footnotes`), KaTeX, Mermaid,
  markers/panels and the comment section; `<mark>` wraps text nodes only, so
  bound events and `data-tip` on `.inline-tip` / `sup` survive. Hover on a
  tip still shows the tip popover (annotations have no hover UI); clicking a
  highlight inside a tip opens the thread panel with `stopPropagation`;
  highlights inside `a.external-link` let the link navigate (use the marker).
  `window.InlinePopover` (exported by `js/inline-popups.js`) is only used for
  `scrollToTargetWithOffset` / `renderMathIfPresent`.
- Anchoring waits for `richcontent:rendered` when the post has Mermaid/KaTeX
  (6 s fallback) and re-anchors on every later event. Styles live in
  `less/annotations.less` (appended by hand to both CSS bundles).

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
  (v1.2.1); series 8 uses `../vllm-v0.27.1`; series 9 and 10 use
  `../vllm-v0.28.0` (series 9 only for CLI flags / metric names / OpenAI
  protocol fields; its platform components are pinned to their Aug-2026
  releases, local checkouts `../kueue`, `../volcano`, `../kserve`, `../llm-d`,
  `../llm-d-router`, `../gpu-operator` etc.); series 10 also uses
  `../pytorch-v2.14.0`. Series 2 pins PyTorch v2.10.0 /
  vLLM v0.15.0 and series 5 pins vLLM v0.20.0 but have no local worktree —
  add one (`git -C ../vllm worktree add ../vllm-v0.20.0 v0.20.0`) before
  re-verifying their source citations.
- **Series membership** is front matter, not prose: `series: <key>` where
  `<key>` is an entry in `_data/series.yml` (`name`, `overview` URL). Members
  are ordered by date; the layouts render the top quote (`本文是《…》系列的第
  N 篇（共X篇）。上一篇：…；下一篇：…`, `_includes/series-nav.html`), the
  bottom table of contents (`series-toc.html`) and a series-aware pager
  (`post-pager.html`, chronological Previous/Next for non-series posts). Do
  **not** hand-write the nav quote any more; the overview post itself has no
  `series:` key. Titles are `系列名（NN）：副标题` — the nav/TOC show the part
  after `）：`. New series: add the key to `_data/series.yml` first.
  `tools/migrate-series.py` converted the old hand-written quotes (idempotent).
- **Update-note exception:** when no usable version predates the post (no tag,
  or the only tag is months stale), a post may cite a newer version *if* it
  carries a note as the first line of the body:
  `> **更新 @YYYY-MM-DD**：本文 X 部分基于 vA 刷新；其余源码引用仍以 … 为准。`
  Use it sparingly, list only the projects actually refreshed, and keep one
  version set per project per post — refreshing means re-verifying every claim
  about that project, never mixing two versions in one article.
- Cite source as path + function/class name, never line numbers.
- Length is not a target; rigor and organisation are. Structure: (update note) →
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
  - The converse also holds: "the prose doesn't cover it" is not a reason to
    drop a valuable diagram. If the concept belongs to the post's topic, add
    the prose *and* the diagram together; only skip when the concept is out
    of scope for that post or the diagram adds nothing.
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

## Writing Popup Footnotes & Inline Tips

Supported in all posts within `.post-container` via `js/inline-popups.js`:

1. **External links**: Any `http://` or `https://` link pointing outside the blog
   automatically gets `class="external-link"`, `target="_blank"`, `rel="noopener noreferrer"`,
   a dashed underline, and a top-right `↗` icon (`fa-external-link`).
2. **Standard Markdown Footnotes (Popup Footnotes)**:
   - Write standard Kramdown footnotes: `概念[^name]` and `[^name]: 解释内容（支持完整 Markdown/链接/加粗/代码块）`。
   - Readers hovering or clicking `[1]` will see the explanation pop up right beside the reference without jumping to the bottom.
3. **Inline Tips (原地行内解释)**:
   - For short 1-2 sentence concept explanations where you don't want to navigate to the bottom:
     - Pure Markdown link: `[概念](# "tip: 解释文案，支持 **加粗**、\`代码\`")`
     - Kramdown IAL: `[概念](#){: .tip data-tip="解释文案"}`
     - Liquid Include: `{% include tip.html text="概念" tip="解释文案" url="可选更多链接" %}`
     - Inline HTML: `<span class="inline-tip" data-tip="解释文案">概念</span>`
   - Renders with a dashed underline and top-right `?` icon (`fa-question-circle`), popping up a floating card on hover/click.

