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
- `.github/workflows/feedback-queue.yml` (Mondays 09:00 Beijing, or manual with
  `threshold` / `only` / `dry_run`): `tools/feedback-queue.cjs` builds every
  post's 修订简报 with `js/feedback-brief.js` (the module the dashboard uses —
  Node gets a DOM from `linkedom@0.18.13`, installed `--no-save` in the job)
  from worker `GET /feedback` (all posts), GraphQL (the Comments category, every
  discussion with comments — needs `discussions: read`), REST (`划线评论`
  issues + open `待修订` ones) and the live pages, then opens / updates / closes
  **one issue labelled `待修订` per post** (body = the brief, marker
  `<!-- feedback-queue: /slug.html -->`; updated only when the brief changed
  apart from the date; closed with a comment once the score drops under the
  threshold, default 3). Coding agents pick the issue up as-is; `Fixes #N`
  closes it. Local: `DRY_RUN=1 API=http://localhost:8788 GITHUB_TOKEN=$(gh auth
  token) node tools/feedback-queue.cjs` (`ONLY=/slug.html` also prints the brief).

## Layout

- Categories: front matter `category:` is `life` (essays; nav **Life** →
  cards on `/life/` = `life.html`, `[Life]` in the archive) or absent (tech;
  nav **Tech** = the paginated home). `_plugins/home_flow.rb` sets
  `hidden: true` on life + pinned posts so jekyll-paginate leaves them out of
  the home flow without gaps (site.posts / archive / tags / feed still include
  them). No `meta` category any more — docs about the blog are plain posts
  (the two manuals are also linked from the footer). Feed items carry
  `<category>` (`tech` / `life`).
- `/admin/stats.html` + `js/dashboard.js`: author dashboard — 阅读趋势
  (worker `GET /views/daily?days=`, per-day bars from `views_daily`, Beijing
  dates), 文章榜 (`GET /stats/top`: views · 点赞 · 点赞率 · 分享 + comment counts
  via `/stats?paths=` in chunks of 20; click a `th[data-sort]` to sort; TOP 10 by default, `.dash-toggle` expands), 读者划出
  来的句子 (`GET /reactions/top?kind=doubt|up`), recent comments via GraphQL
  with the giscus session (alias the `comments(last:3)` field — a response key
  named `comments` twice is a GraphQL validation error and looked like a login
  failure), open issues via REST. `sitemap: false`, `noindex: true`
  (`head.html` emits the robots meta for `page.noindex`). Its styles are
  hand-appended to the CSS files (`.dash*`), there is no Less source.
- `index.html`: posts with `pinned: true` lead page 1 (badge `.post-pin`) and
  are skipped in the paginated flow. Sidebar (`_layouts/page.html`): HOT TAGS
  threshold is `site.featured-condition-size`; RECOMMEND renders
  `site.recommends` (`title`/`href`/`desc`) as external links. Styles for these
  live in `less/extras.less` (inserted before the series block in both CSS
  bundles, same hand-compile procedure).
- Post layouts: `post` (default, text header), `header-post` (hero image;
  front matter `header-img`, `header-bg-css`, `header-mask`,
  `header-img-credit(-href)`), `keynote` (header is an `iframe` of a slide
  deck sized to the viewport, `navcolor: invert` for light decks; same body
  as the others — catalog, pager, related, comments; demo post
  `keynote-layout-demo`). `redirect_from:` works
  (jekyll-redirect-from). `tools/new-post.py` scaffolds a post/draft.
- `_posts/` — blog posts, `layout: post`, permalink `/:title.html`
- `slides/` — reveal.js decks, `layout: slides` (or set in front matter),
  URL `/slides/:name.html`, indexed by `slides.html` (`/slides/`)
  Decks are pages, so Jekyll itself takes no date from their filename;
  `_plugins/slides_date.rb` (`:site, :post_read` hook) fills `page.date` from a
  `YYYY-MM-DD-` filename prefix when front matter has no `date:`, so decks
  follow the post convention. `archive.html` still parks a deck with neither
  under a 未注明日期 bucket at the bottom (it used to float to the top with an
  empty year). Posts never need `date:` (only to order several posts on the
  same day).
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
- `_includes/post-license.html` — license notice after the body (all three post
  layouts); text from `_config.yml` `license: {name, url, note}` (CC BY 4.0;
  `%url%` in `note` = the article's link — Liquid can't take `{url}` inside
  `{{ }}`), a post can set `license: false` or its own map. Excluded from the WeChat export.
- Sidebar readability overrides (darker text, bigger/bolder section titles,
  RECOMMEND title/desc contrast; the theme's `<hr>` rhythm is kept) live at the
  end of `less/extras.less`, not in `sidebar.less`. Footer: RSS + GitHub Star (count via anonymous REST, cached
  a day in localStorage); no article links there.
- `_includes/post-meta.html` — the "Posted by … | date (· 更新于) | 约 N 分钟 ·
  X.Xk 字 | N 次阅读" line under the title, shared by the three post layouts.
  Reading time = HTML-stripped body without `<pre>` blocks / 450 chars per
  minute. Optional front matter: `updated: YYYY-MM-DD` (also feeds
  `dateModified` / `article:modified_time`), `description:` (SEO text).
- Post layouts pipe `content` through `replace: '<img src=', '<img
  loading="lazy" decoding="async" src='` — every content image is lazy.
  `js/diagram-zoom.js` opens Mermaid diagrams *and* content images (>= 200 px
  natural width, not inside `<a>`) in the zoom/pan lightbox.
- **No jQuery / Bootstrap JS.** `footer.html` loads only our own scripts;
  `js/argan-blog.js` (→ `.min.js` via `node_modules/.bin/uglifyjs js/argan-blog.js
  -c -m --comments '/^!/' -o js/argan-blog.min.js`) does the theme bits in plain
  DOM (wrap tables in `.table-responsive` + `.table`, wrap YouTube/Vimeo
  iframes, navbar hide-on-scroll-down `.is-fixed/.is-visible`, `.side-catalog.fixed`),
  the mobile navbar toggle is inline in `nav.html`, `js/tagcloud.js` colours
  `#tag_cloud a[rel]` on `/tags/` (loaded there only). FastClick and the
  `data-toggle="tooltip"` pager attributes are dead. **Bootstrap 3 CSS stays**
  (`css/bootstrap.min.css`; grid, navbar, tables, `.embed-responsive`,
  `.visible-*/.hidden-*` are all in use) — don't swap in Bootstrap 5.
- Local CSS/JS in `head.html` / `footer.html` carry `?v=<build time>` for
  cache busting (GitHub Pages serves `max-age=600`; the old `no-cache` meta
  tags were removed). Font Awesome 4.7 is a self-hosted **subset**:
  `tools/fa-subset.py` scans templates/js/posts/less for `fa-*` classes and
  `content:"\fXXX"` glyphs and writes `css/font-awesome.min.css` +
  `fonts/fontawesome-webfont.woff2` (~20 KB, vs 77 KB full) from the full
  copies in `tools/fa/`. ~150 common icons are always included (`ALWAYS` in
  the script) so new posts rarely need anything; CI runs `--check` and fails
  with the missing icon names if they do — then run the script (needs
  `pip install fonttools brotli`). `sw.js` is disabled via
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
- `_includes/post-actions.html` + `js/share.js` (+ `less/share.less`) — action bar
  「♥ 点赞 N · 分享 · [复制为公众号格式]」 right above `comments.html` in all
  three post layouts. **「点赞」 is anonymous**: worker `GET/POST /votes` keeps
  `votes(path, up, down)` in D1 (only `up` is used now — no downvote), the
  browser remembers its choice in `localStorage["vote:<path>"]` and sends the
  transition `{path, dir, prev}`; localhost never posts. No GitHub login (the
  old discussion-THUMBS_UP 「点赞」 is gone; comment votes stay reactions).
  Counters are one `.post-stats` **text strip** (`data-path`): 阅读 · 点赞 ·
  评论 · 分享, each rendered as a text label followed by its count. `share.js` paints it
  (`paintStrip`, partial patches merged per element). On the post page the
  strip sits in the header meta line (`post-meta.html`): 点赞 + 分享 come from
  `GET /votes`, views / comment count from `annotations.js`, which dispatches
  `blog:stats` (`{views}` / `{comments}`) from `renderLikeBar`. List pages
  (`index.html`, `life.html`) have **no buttons**, only the strip per post,
  filled from one `GET /stats?paths=…` (views + votes + shares from D1,
  comments / discussion id from the giscus API, 120 s edge cache). Every
  completed share (system sheet resolved, Weibo/X/LinkedIn opened, QR shown,
  link copied) is one `POST /shares {path}` → D1 `shares(path, count)`;
  localhost never posts. annotations.js runs `takeSessionFromUrl()` and exposes
  `BlogAnnotations.core` (`api`, `graphql`, `getSession`, `login`,
  `ensureToken`) even on non-post pages. 「分享」 opens a
  single body-level `.pa-share-pop` menu: Web Share API (only when supported),
  Weibo / X / LinkedIn intent URLs built in JS, WeChat QR
  (`js/vendor/qrcode.min.js`, qrcode-generator 1.4.4 MIT, lazy-loaded), copy
  link. No third-party script. The author-only 「复制为公众号格式」 button is
  shown when the GitHub viewer equals `site.github_username`: annotations.js
  dispatches `blog:viewer` (detail = viewer or null on logout) from
  `onViewerKnown()` / `logout()` and exposes `BlogAnnotations.viewer()`.
  Clicking lazy-loads `js/wechat-export.js`, which clones `.post-container`,
  strips chrome (series nav/TOC, pager, related, comments, highlights, copy
  buttons), inlines styles per tag (Rouge token colours read from the live DOM
  via getComputedStyle), turns external links / footnotes / inline tips into a
  numbered 「参考与脚注」 list (WeChat strips links), replaces KaTeX with
  codecogs images (Zhihu: `zhihu.com/equation`, `target: 'zhihu'`), re-renders
  Mermaid with `htmlLabels:false` (foreignObject taints the canvas) into PNG
  data URLs, converts same-origin `.webp` to JPEG data URLs (<= 1280 px), and
  writes `text/html` + `text/plain` via `ClipboardItem` (contenteditable +
  execCommand fallback). Relative URLs resolve against the canonical page URL,
  not localhost. Whether the WeChat editor accepts base64 images on paste is
  only verified by pasting.

## Comments & highlight comments (js/annotations.js)

User-facing wording is always 评论 (划线评论 for passage-level ones, 回复 for
replies) — never 标注 / 批注. "annotation" survives only in identifiers, file
names and the W3C selector terminology.

Two manuals only: reader-facing `_posts/2026-08-03-a-letter-to-readers.md`
(《致读者的一封信》, `/a-letter-to-readers.html`: what the features are and
how to use them, casual tone, FAQ — no implementation detail) and the
maintainer memo `_posts/2026-09-09-blog-memo.md` (《博客备忘录》,
`/blog-memo.html`: hosting, config, writing conventions incl. live footnote /
tips examples, CI, ops, FAQ and a Releases log — bump it when shipping a
feature). Old URLs `highlight-annotations-demo`, `blog-user-manual`,
`popup-footnotes-and-inline-tips-demo` redirect via `jekyll-redirect-from`.
Keep both up to date when features change.

One data model, two views. `comments` holds every top-level comment of the
post's discussion (`parseComment`); those whose body starts with the quote
header get `selector`/`noteHTML` and are the `annotations` highlighted in the
article. `syncViews()` re-renders both the highlights/panel
(`applyHighlights`) and the bottom section (`renderCommentSection`) after
every mutation. `commentEl` and `renderEditor` are shared, so plain comments
and passage notes have identical reply / edit / delete / 「同时提交 Issue」
controls. The bottom section (`.annotation-comments`): header bar
(`renderLikeBar`: count + GitHub link; it also paints the `.post-actions` bar above the section), `.ac-group` per top-level comment with its replies, inline reply
editor (`openInlineReply`), and a persistent editor (`.ac-editor`,
`clearOnSubmit`, draft in `sessionStorage`). Plain comments filed with an
issue start with `<sub>[⚑ Issue #N](url)</sub>` (recognised by
`parseBodyHeader`). Soft-deleted comments (GitHub keeps a comment that has
replies, `deletedAt` set) render as 「此评论已删除」 with their replies.

Code-review / WeChat-reading style, no hover popups. Readers select text in
`.post-container` → floating toolbar (`点赞` / `存疑` / `评论` / `复制` / `搜一搜` / `分享`) → an **in-flow editor
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
- **Comment votes** are plain GitHub reactions, no own storage (the article-level
  「点赞」 is the anonymous worker counter described above): each comment's ▲ score ▼
  (`.ap-vote`, in the meta row) is `THUMBS_UP` / `THUMBS_DOWN` on that
  comment (`toggleVote`, optimistic, switching sides removes the other
  reaction first; `addReaction` / `removeReaction`). `parseVotes` reads both
  giscus' `{THUMBS_UP:{count,viewerHasReacted}}` and GraphQL
  `reactionGroups`. The relay's payload is anonymous, so `loadViewerReactions`
  re-queries the discussion with the reader's token once the viewer is known
  and patches `votes.mine` / `likes.mine` in place (`updateVoteEls`). GitHub's
  native discussion-comment `upvote` is deliberately not used (top-level only,
  no downvote). Don't re-add the giscus iframe for reactions.
- **Passage 赞 / 存疑** (`react`, `reactions` map, toolbar buttons
  `.annotation-tb-up/-doubt`, panel row `.ap-react`): anonymous counters, no
  login, like the article 「点赞」. Worker `GET/POST /reactions` keeps
  `passage_reactions(path, hash, quote, up, doubt, share, reasons, section)` in D1; `hash` =
  `annotHash(exact)` (the `#annot-<hash>` id), `quote` lets `applyHighlights`
  anchor and underline a passage nobody commented on (mark ids `r:<hash>`,
  same `mark.annotation-hl`; `.has-doubt` = red dotted line). One reader's
  choices live in `localStorage["react:<path>:<hash>:<kind>"]`. The unit of
  everything passage-level is `passages()` / `passageFor(ids)` (`{ ids, list,
  hash, exact, reaction, marks }`): markers (`markerHtml`: 💬 · 👍 · ❓),
  `openThread`, `passageContaining(offsets)` (a selection inside an
  underlined passage joins it — comment or reaction), `renderHotPassages`.
  `refreshReactionViews` repaints marker / panel row in place and only
  re-anchors when an underline must appear or vanish. Toolbar 存疑 opens the
  passage panel (its 「说说哪里不对 →」 focuses the editor); 赞 just flashes +
  toasts. Local previews post only when `localStorage.annotationsApi` is set.
- **Passage 分享** (`sharePassage`, toolbar `.annotation-tb-share`, panel
  `.ap-react-share`): opens the article's share popover — `js/share.js` exposes
  `window.BlogShare.open(btn, { url, title, text, onShared, toast })` — with
  `threadLink` when the passage has comments, else `shareLink` (`#hl=`), and
  the quote as text. Each completed share = `countPassageShare` → `POST
  /reactions kind:'share'` (+1, no toggle, `share` column; the worker also
  bumps the article's `shares` row and the response's `shares` is forwarded as
  a `blog:stats` event for the header badge). A share alone does not underline
  a passage (only up / doubt do); the count shows in the panel row and the
  marker title. Dashboard 读者划出来的句子 has a 分享最多 tab.
- **Quote escaping gotcha**: `escapeMarkdown` must produce `1\.5px`, not
  `\1.5px` — a backslash before a digit is literal in GFM, the parsed quote
  gets an extra `\` and its hash no longer matches the `§ 原文位置` link.
- **Feedback loop (存疑原因 / 章节 / 已修正 / 修订简报)** — the
  point is to turn reader signals into work an AI can execute:
  - `DOUBT_REASONS` (`wrong|unclear|outdated|example|conflict`, labels in
    both `annotations.js` and `dashboard.js`, whitelist in the worker): after
    存疑 the panel's `.ap-react` gets a `.ap-doubt-why` chip row (`setReason`,
    `POST /reactions kind:'reason' {reason, prev}`; `prev === reason` clears;
    one pick per browser in `localStorage["react:<path>:<hash>:reason"]`;
    un-doubting sends the clear too). Counts live in `passage_reactions.reasons`
    (JSON) and show in the 存疑 button title / marker title / dashboard.
  - `section` = nearest `h2/h3` above the passage (`sectionForOffsets`, set on
    every selector by `selectorFromOffsets`, sent with every reaction POST,
    stored once per row). Comment header gets ` · 位于「…」` (`sectionNote`,
    parsed back by `parseBodyHeader` → `selector.section`); the Issue body too.
  - The former `建议修改` quick action has been removed; readers now use the
    normal `评论` action when they want to explain a problem or propose a change.
    Historical comments remain readable as ordinary comments.
  - `resolved` (`markResolved`, run in `syncViews`): a note **with an Issue**
    is resolved iff that Issue is closed (`loadIssueStates`: anonymous REST
    `GET /repos/{repo}/issues/{n}` per distinct number, cached in
    `localStorage["issueState:<repo>#<n>"]` — open 10 min, closed 1 day; only
    source of truth for issue-backed notes). Without an Issue: an owner reply
    matching `RESOLVED_RE` (已修正 / 已修复 / 已更正 / 已改正 / 已订正 / 已采纳) or a
    🎉 `HOORAY` on the note (`parseVotes` now carries `hooray`). Effects:
    `mark.is-resolved` (green solid line, beats `.has-doubt`),
    `.annotation-marker.is-resolved` + ✓, `.ap-comment.is-resolved` +
    `.ap-resolved-badge`, `renderHotPassages` skips `passageResolved(p)`.
    Reaction-only passages have no GitHub object: editing the text orphans
    them and they simply stop rendering. No new storage anywhere.
  - Orphans (`renderOrphans`) show the first 24 chars of the quote + author
    (title = full quote + section) under 「N 条划线评论对应的原文已修改」.
  - **Figures** (`js/figures.js`, loaded before annotations.js): every `p > img`
    becomes `figure.post-figure > span.fig-media > (img + div.fig-tools >
    button.code-copy.fig-feedback) + figcaption.post-figcaption (.fig-no 「图 N」 +
    .fig-title = alt)`; every rendered `.mermaid` gets its `svg` wrapped in the same
    `.fig-media` (inline-block, `width` = the svg's `max-width`, so it shrink-wraps
    the drawing) with a `.fig-tools` strip that also holds code-copy's button, and
    the caption as the `.mermaid`'s next sibling (title = Mermaid front matter
    `title:` or a first-line `%% 图：…` comment). The strip must sit on the
    **picture's own** top-right corner — anchored to the block it floated in blank
    space hundreds of px right of a centred picture, which is why it was "hard to
    hit". 32 px targets, 6 px padded dead zone whose click handler stops propagation
    so near-misses don't open the zoom lightbox; hover-only on `.fig-media` (it
    overlaps the drawing), always shown on phones. `code-copy.js`'s duplicate check
    looks inside `.fig-tools` / `.fig-media` too — moving its button out again would
    loop the two MutationObservers. Code blocks (`.highlighter-rouge` / `pre`) get the
    same strip with the copy button and the same handle, which selects the whole
    `<code>` (`pick(code)`) — nothing mode-specific: the normal toolbar then offers
    点赞 / 存疑 / 评论 / 复制 / 搜一搜 / 分享 on the block, whose passage is
    its full text (any edit orphans old notes, which is the intended signal). A
    「跑不通？」 pill with a pre-filled 环境 / 报错 template was tried and dropped:
    the block handle should be generic, like the figure one. The figure button
    only *selects* the caption title (scrolling the caption to the viewport centre
    first when it is off-screen, focusing it, flashing `.is-picked`) —
    annotations.js' `selectionchange` shows the normal toolbar, so a picture's
    passage is its caption text (stable while the alt is). `figcaption` is in
    `BLOCK_SELECTOR` (panel goes right under it); a mark inside a caption sets
    `.has-note` on it and CSS `:has()` outlines the figure. `feedback-brief.js`'s
    `articleFromHtml` turns `img[alt]` into text so those quotes still anchor in
    the static HTML. Writing rule: **alt is required and reads as a caption**;
    「图 N」 alone is the fallback and re-numbers when a figure is inserted.
  - **Tables** (`js/code-copy.js` + `js/figures.js`): every table in the article
    body gets a copy button and a feedback handle, excluding comments, annotation
    panels, series TOC, and related posts. The copy menu offers TSV, Markdown,
    and HTML. A stable feedback passage comes from a native `<caption>` or a
    paragraph immediately before the table matching `表：标题`, `表1：标题`,
    `表: 标题`, or `表1: 标题` (spaces before the number are allowed); an
    an explicit number is kept. Without a title, the handle selects the
    `<thead>` row directly; only a table with no header falls back to selecting
    the whole table. Explicit caption nodes and title paragraphs are included
    in `BLOCK_SELECTOR` so the feedback panel is mounted after the table rather
    than after the caption node.
  - **Section-level 点赞 / 没看懂** (`renderChapterBars`, `.sec-react` appended
    inside every article heading `h2`–`h6`, two `.sec-react-btn`s; `chapters` map): anonymous
    like passage reactions, no selection needed. Same worker route and table,
    `quote = '§ ' + heading` (`CHAPTER_PREFIX`), `section = heading`, `up` = 点赞,
    `doubt` = 没看懂; `loadReactions` splits `§ ` rows into `chapters` so they are
    never anchored as passages. `.sec-react` is in `EXCLUDE_SELECTOR`, in
    wechat-export's `REMOVE`, and `headingText()` strips it (and `.anchorjs-link`)
    wherever a heading's text is read (`sectionForOffsets`). The dashboard tags
    such rows 「章节」 and the brief has a 「章节热度」 table.
  - **修订简报** lives only in `/admin/stats.html` (`#brief=/slug.html`, a
    `<select>` of `window.DASH_POSTS` and a 「简报」 link per 文章榜 row — one
    URL to remember, no CLI twin). The aggregation is **`js/feedback-brief.js`**
    (UMD: `window.FeedbackBrief` in the browser, `require()` in Node — shared
    with `tools/feedback-queue.cjs`, keep it I/O-free and DOM-agnostic: callers
    pass `dom.parse(html) -> Document`): `parseNote` (mirrors `parseBodyHeader`),
    `articleFromHtml` (`.post-container` text + h2/h3 offsets), `sectionAt`,
    `analyze` → `{ todo, done, lost, plain, chapters, score… }`, `render` → Markdown.
    `openBrief` fetches worker `GET /feedback?path=` (D1: reactions + reasons +
    section + chapter rows, views, 点赞, shares), the Discussion via the worker's
    anonymous `/discussions?term=` relay, the repo's `划线评论` issues (REST,
    `state=all`, filtered by `body` containing the path — gives 已修正) and the
    live article. Passages are ordered by `2×doubt + 0.5×up + Σ(1 + ▲ + 3×suggest)`;
    sections: 章节热度 / 待处理 / 普通评论 / 其他 open Issue / 未定位（原文已改）/
    已修正 / 给 AI 的修订指令. `analysis.score` (todo + unresolved plain comments +
    chapter 没看懂) is what the weekly Action thresholds on. Output is a `<pre>` +
    「复制 Markdown」. Links use `siteUrl` (local builds show localhost). The
    dashboard's 待处理 block lists open `待修订` issues first (with a 「简报」 link
    parsed from the marker), then `划线评论`, then `dead-links`.
- **最受关注的段落** (`renderHotPassages`, `.ac-hot` above the comment list):
  passages ranked by `赞 + 2 × 存疑 + 2 × net comment votes + comments`, shown
  only when there are 2+ scored passages, max 3; clicking scrolls to the
  passage and opens its thread. Re-ranked on every vote / reaction.
- **Page views**: `loadViews()` → `POST /views {path}` once per browser per
  post per day (`localStorage["viewed:<path>"]`), otherwise `GET /views`;
  localhost never increments. The worker keeps `views(path, count)` in a D1
  database (binding `DB` in `wrangler.toml`; without it the route is 501 and
  the counter is simply not shown) and, per Beijing day, `views_daily(path,
  day, count)` for the dashboard trend. Handed to `share.js` via `blog:stats` for
  the `.post-stats` badge in the post header (all three post layouts).
- **Spacing gotcha**: the theme's `.post-container img { margin: 1.5em auto
  1.6em }` hits every `<img>` inside the in-flow panel — avatar rules must
  reset `margin: 0` or replies get ~40 px of phantom whitespace.

On load the thread is fetched, comments of that shape are parsed into a W3C
`TextQuoteSelector` (exact = blockquote text), anchored exactly or fuzzily
(`js/vendor/approx-string-match.js`, MIT, Hypothesis' algorithm) and wrapped in
`<mark class="annotation-hl">` (one mark per text piece carrying every covering
id — overlaps never nest). Marks are a dashed amber underline, no background
(WeChat-reading style; `.is-multi` = solid line, wash on hover only) — teal
dashed is reserved for `.inline-tip`. Each distinct passage gets a `.annotation-marker`
(comment icon + count) after its last mark; clicking it or the highlight
toggles the **thread panel** below the paragraph: notes, replies, and an editor
to join. Anchors that no longer match are listed under the comment hint as
"未能定位". Only one panel exists (`panelState`); it is re-rendered after
re-anchoring.

- **Links**: `#annot-<hash>` opens a thread, `#hl=<readable text>` (from
  passage `分享` of an uncommented passage) flashes a passage for 2.5 s. Both are handled by the script on
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
  (v1.2.1); series 8 uses `../vllm-v0.27.1`; series 10 (平台) and 11 (开源贡献) use
  `../vllm-v0.28.0` (series 10 only for CLI flags / metric names / OpenAI
  protocol fields; its platform components are pinned to their Aug-2026
  releases, local checkouts `../kueue`, `../volcano`, `../kserve`, `../llm-d`,
  `../llm-d-router`, `../gpu-operator` etc.); series 11 also uses
  `../pytorch-v2.14.0`. Series 9 (RL 后训练基础设施) will pin verl v0.9.0 and
  reuse `../vllm-v0.27.1` / `../pytorch-v2.13.0` / `../Megatron-LM`. Series 2 pins PyTorch v2.10.0 /
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
- Series 4 (`transformer-and-llm`) has 8 posts (the cost table, dated
  2026-04-02 … 04-09); post 08 closes with 「系列总结（八篇）」. The former
  posts 09–12 (预训练补篇) were split out on reader request (discussion #24)
  into their own series `pretraining` (《预训练：从 tokenizer 到训练配方》,
  overview `2026-04-09-pretraining-from-tokenizer-to-training-recipe.md`,
  posts 01–04 dated 2026-04-10 … 04-13, algorithm roadmap L4 only, not
  shared with the Infra roadmap). Slugs/URLs did not change, only titles and
  `series:`. Cross-references inside the pretraining posts to the cost table
  are written 「《Transformer 与 LLM》第 N 篇」; their companion scripts stay
  in `ai-learning-labs/transformer-and-llm` (`llm_cost_09` … `_12`). Keep
  math out of `##`/`###` headings — the sidebar OUTLINE shows raw `\(…\)`.
- **No bare `|` inside inline `$$…$$` in a paragraph** (`|A \cap B|`,
  `\sum |w_i|`, `\mathbb{R}^{|\theta|}`): kramdown turns the whole paragraph
  into a header-less one-row table and the formula is shredded (reader
  reports #33 / #36, 2026-09-14; fixed in 5 posts). Write `\lvert … \rvert` /
  `\lVert … \rVert` (escaped `\|` also survives). Audit:
  `rg -l --multiline '<table>\s*<tbody>' _site --glob '*.html'` after a
  build — the hits should only be old posts with intentional header-less tables.
- Series 09 `rl-post-training-infra` (《RL 后训练基础设施：rollout 与训练如何共享
  一组 GPU》, overview `2026-08-26-rl-post-training-infrastructure.md`, Infra
  roadmap L4 alongside 07/08, promoted from 选修 on 2026-09-14) is complete: overview + 8 posts dated 2026-08-27 … 09-03 (written 09-14, all
  linked from the overview's 章节目录); vLLM was compressed to daily (08-11 … 08-25 — it cannot start
  earlier: its pin vLLM v0.27.1 is tagged 2026-08-11), 平台 and 开源贡献 were
  renumbered 10 / 11 and re-dated to 09-04 … 09-12 and 09-13 … 09-17 to keep
  publication order = reading order.
  verl is the single deep-dive framework (源码线 + 8 卡实践); slime / AReaL appear
  only as 对照 in post 7. Planned next: a short 选修 series on diffusion-model
  inference infra (5 posts). Version baseline in the overview: verl v0.9.0, slime v0.3.0, OpenRLHF
  v0.11.0, AReaL paper/docs, vLLM v0.27.1, PyTorch 2.13.0, Megatron Core 0.18.0.
  Only post 1 has a companion script (`ai-learning-labs/rl-post-training-infra/rl_ledger.py`);
  posts 2–8 deliberately have none — the author asked to stop writing
  companion experiments (they cost time and thinned the articles); each post
  ends with a prose 实践建议 instead. Sources were read from shallow clones of
  verl v0.9.0 / vLLM v0.27.1 / slime v0.3.0 / AReaL; cite paths + function
  names, never line numbers. Overview posts that link to future-dated posts
  fail lychee until those dates — build locally with `--future` to check.
- Companion code lives in `../ai-learning-labs` (git repo, pushed by the
  user). Its `.venv/` (Python 3.12 via `~/.local/bin/python3.12`, torch CPU,
  numpy, tiktoken, tokenizers) is gitignored; recreate with
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
  Every script writes its full run to `<series>/expected/<script>.txt`; the
  article quotes those numbers, so re-run and refresh `expected/` when a
  script changes. SVG figures for posts are generated by
  `transformer-and-llm/tools/gen_*_svg.py` into `img/in-post/` here; eyeball
  them with `qlmanage -t -s 1480 -o /tmp <svg>` (renders to `/tmp/<name>.png`).
- Series `post-training` (《后训练：从 SFT 到可验证奖励》, overview
  `2026-04-15-post-training-from-sft-to-verifiable-rewards.md`, 8 posts dated
  2026-04-16 … 04-23, algorithm roadmap L5): organised around the RLHF 三件套
  (策略 / 奖励 / 参考); post 3 = 在线 RL, post 4 = 离线 RL (DPO family), post 6 =
  Agent RL. Only post 1 has a run lab with measured numbers; posts 2–8 are
  推导 → 算账 → 公开配方 with a "动手" section giving a `trl` skeleton and the
  curves to watch — they cite no unrun numbers (the overview labels these
  "动手（建议）"). Keep it that way unless a lab is actually run. Depth
  baseline for all L4/L5 posts is the original 04 series (~35K+ chars: every
  section has derivation with intermediate steps, a real-model account, the
  common misconception, and the public-recipe form). Labs in
  `../ai-learning-labs/post-training/` use `trl` on
  Qwen2.5-0.5B/1.5B; run them on MPS (`ptlab.device()`), ~5 s per 4×512 step
  for a 0.5B full fine-tune, and with `HF_HUB_OFFLINE=1` after the first
  download (anonymous Hub requests have hung for minutes). Python stdout is
  block-buffered when redirected — the `expected/*.txt` only appears at exit.
  Future-dated posts need `jekyll serve --future --port 4001 -d /tmp/_site4001`
  + `SITE=http://localhost:4001 node tools/check-render.cjs <slug>` to verify.
- Series `efficient-inference` (《高效推理与压缩（算法侧）》, L6, 6 posts) and
  `multimodal` (《多模态：从视觉编码器到扩散模型》, L7, 7 posts), plus the
  横切 导读 `experimental-methodology-for-ai-algorithm-engineers`, complete the
  algorithm roadmap (written 2026-09-14, no labs — same "动手（建议）" rule as
  post-training 2–8; numbers come from papers / tech reports only). L6 builds
  on 04-07 (量化 / 投机 / LoRA 的账) and 04-03/04-04 (KV) and must not re-derive
  them; L7 builds on 04-08 (多模态成本) and L3-05 (ViT). Both series link to
  04 / L5 posts by design — the "series are independent" rule below applies to
  the Infra series, whereas algorithm-map series cite each other through the
  map's layer structure. L6 post 6 and L7 post 7 carry the 系列总结 (no
  hand-written 目录). Time anchors: nothing later than 2025 (posts are dated
  Apr–May 2026); model refs go up to Qwen2.5-VL / Gemma 3 / BAGEL / gpt-oss.
- Post dates encode the reading order of the three roadmaps and were re-dated
  on 2026-09-14 (permalinks are `/:title.html`, so dates are free to move):
  01-01 《AI 全栈学习地图》(overview of the three, pinned) → 01-02 算法地图 →
  01-03 Infra 地图 → 01-04 应用地图 → 算法 L0 数学 (01-07 overview, 01-08 … 01-15,
  series `math-for-ai`) → L1 工具箱 (01-16 overview, 01-17 … 01-21,
  `algorithm-tooling`) → Infra 01 Python (01-22 … 01-29, shared: L1 深入篇) →
  Infra 02 C++ (02-02 … 02-15) → Infra 03 PyTorch (02-16 … 02-26, shared: L1
  深入篇) → L2 经典机器学习 (02-27 overview, 02-28 … 03-05, `classical-ml`) →
  L3 (03-23 … 03-29) → 04 Transformer 与 LLM (04-01 … 04-13, shared L4)
  → 后训练 (04-15 … 04-23) → 横切 实验方法论 (04-24, one 导读) → L6
  高效推理与压缩 (04-25 overview, 04-26 … 05-01) → L7 多模态 (05-02 overview,
  05-03 … 05-09) → Infra 05–10 (GPU Kernel was moved from 05-06…05-30 to
  05-10 … 05-20 on 2026-09-14 to make room; 通信 starts 06-01 unchanged) → 07
  大规模训练 (07-13 … 07-29) → 08 vLLM (08-11 … 08-25, daily) → 09 RL 后训练基础设施
  (08-26 overview, posts 08-27 … 09-03) → 10 平台 (09-04 … 09-12) → 11 开源贡献
  (09-13 … 09-17).
  Keep a series contiguous (daily posts are fine); do not interleave two maps'
  series except at the shared 01 / 03 / 04 series. The three L0–L2 series were
  expanded from three 导读 on 2026-09-14 after reader feedback (overviews keep
  the old URLs `/math-for-ai-algorithm-engineers.html`,
  `/tooling-for-ai-algorithm-engineers.html`,
  `/classical-machine-learning-in-the-llm-era.html`); they target readers who
  can program but have forgotten university maths — every concept is defined,
  every claim gets a number from a real model, each post ends with 自测. L1 and
  L2 posts have one CPU script each in `ai-learning-labs/algorithm-tooling/`
  and `classical-ml/` (numbers in the posts come from `expected/`); L0 has
  none. L1 does not teach Python itself — Infra 01 / 03 are its 深入篇. When
  other posts cite these layers, write 「L0 数学系列第 N 篇」 etc., never
  「L0 导读第 N 章」 (the 导读 chapters no longer exist). Roadmaps link forward to series published later —
  that is the established convention. Series 收尾篇 must NOT carry a
  hand-written 「系列目录」: the layout generates it from `series:`.
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

