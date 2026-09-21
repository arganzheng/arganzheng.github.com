# AGENTS.md

## Build / preview

Jekyll 4.4.1 is installed against Homebrew's Ruby.
Homebrew Ruby (4.0) and Gem paths are configured in `~/.zshrc`; CI uses the same `ruby-version: '4.0'`.
Jekyll 4.4.1 pins `liquid ~> 4`, `rouge < 5`, `json ~> 2.6` — `bundle outdated` will keep listing those three until Jekyll 5.

`/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin`

A `Gemfile` is also present at repo root:

```bash
jekyll build            # or bundle exec jekyll build -> _site/
jekyll serve            # or bundle exec jekyll serve -> http://localhost:4000
```

### CSS and JS are build output — edit `less/` and `js/*.js`, then rebuild

```bash
npm install          # less 4 + clean-css-cli + uglify-js (devDependencies)
npm run css          # less/argan-blog.less -> css/argan-blog.css + css/argan-blog.min.css
npm run js           # js/*.js (post-page scripts) -> js/blog.min.js (+ .map)
npm run build        # both
npm run check        # tools/check.sh — everything CI checks, locally (see below)
npm run hooks        # once per clone: git config core.hooksPath .githooks (pre-push = npm run check)
npm run serve        # jekyll serve --future on http://localhost:4000
```

`less/argan-blog.less` is the single entry point; its `@import` order is the
cascade order (`blog` → `theme-overrides` → `inline-popups` → `comments` →
`extras` → `series` → `annotations` → `share` → `dashboard`). Every rule has a
Less source now — `theme-overrides.less` (page header, [TOC], outline panel,
wide-screen grid, post-layout navbar, diagram zoom, code copy) and
`dashboard.less` (`/admin/stats.html`) are the former hand-appended CSS,
kept as plain CSS. **Never edit `css/argan-blog*.css` or `js/blog.min.js` by
hand**: `tools/check.sh` (and so the pre-push hook) fails if they are not
byte-identical to what the sources produce (`npm run css -- --check`,
`npm run js -- --check`). `tools/css-compare.py a.css b.css` is a
cascade-aware semantic diff (final value per selector/property, plus
order-sensitive pairs) — use it when you touch the import order or migrate
rules, instead of eyeballing a textual diff.

History, for when an old commit looks odd: until 2026-09-15 the two CSS
files were hand-maintained and had drifted from each other *and* from Less
(e.g. the 2026-09-14 「字体优化」 body font stack / `font-weight: 450` reached
`argan-blog.css` but never the served `.min.css`); the Less-built bundle is
what has shipped since.

**Verifying a style/JS change means checking `_site/`, not the source.** The
browser (local `jekyll serve` and the user's eyes) reads `_site/css/*.css`;
that copy is only refreshed by a *successful* `jekyll build`. If the build
aborts (typically a `Liquid Exception` from an unrelated draft post), `_site`
silently keeps the previous output and the edit looks like it "didn't work".
Before reporting a CSS change as done: run `jekyll build`, confirm it printed
`done in …`, then `grep` the changed selector in `_site/css/argan-blog.min.css`
(or `curl` it from the local server). Never conclude from `lessc` compiling or
from the source file alone.

### `tools/check.sh` (= `npm run check` = the pre-push hook)

Mirrors `.github/workflows/check.yml` so a bad push is caught locally
(~20 s): `tools/liquid-scan.py` (unescaped `{{` / `{%` inside fenced *or
inline* code in any post / draft / slide — inline backticks bite exactly the
same way), `css/` and `js/blog.min.js` up to date with their sources,
`jekyll build --future --unpublished --strict_front_matter -d _site-check`
printed `done in` (unpublished too, so a `published: false` post cannot park
a Liquid error or a dead anchor that surfaces the day it is published;
`_site-check/` is gitignored and leaves the `jekyll serve` `_site/` alone),
`tools/fa-subset.py --check`, lychee offline over `_site-check` (skipped when
lychee is not installed, or `SKIP_LINKS=1`), `git diff --check`. Bypass once with
`git push --no-verify`. `tools/check-render.cjs` (headless Chrome) is not part
of it — run it by hand for posts with diagrams. `.githooks/pre-push` prepends
Homebrew's ruby / gems / bin to `PATH` before calling `check.sh`: GUI Git
clients (the IDE's push button) run hooks with a bare PATH, so `bundle`,
`lychee`, `node`, `rg` were not found and every push from the IDE failed
while the terminal passed (2026-09-16).

### Revising posts = a PR, reviewed rendered (`tools/review.py`)

Content revisions by an agent (fixing a post after reader feedback, a
待修订 issue, a sweep over a series) never go straight to `master`. The
workflow, mirroring code review:

1. Work in a **separate worktree** so the user's checkout is untouched:
   `git worktree add ../arganzheng.github.com-rev-<topic> -b rev/<topic>`
   (we once had the user's commits land on an agent's branch because both
   used the same working tree). Commit there, `gh pr create`; the PR body is
   the summary and the sources (which passages / issues / instructions).
2. **Every changed section gets a reason.** Write `.review/notes.md`
   (gitignored) — `## _posts/<file>.md`, then `### <h2/h3 title>` + the
   reason (what was wrong, what the change does, what reader signal it
   answers); text before the first `###` is a file-level note — and run
   `tools/review.py <PR#> --notes .review/notes.md`. It posts one review
   with a comment on the first changed line of each section and prints the
   changed sections that still have **no** note (non-zero exit) — fix those
   before handing over. The same anchor rule (section → first hunk under
   that heading, `RIGHT` side, `LEFT` for pure deletions) is how the user's
   comments from the review page land on GitHub, so both sides of the
   conversation sit on the same lines.
3. The user runs `npm run review -- <PR#>` (see below), comments per section
   from the rendered page (→ PR line comments), approves / requests changes.
4. Read the threads (`gh api graphql` `reviewThreads`, or the page), reply
   on each with 采纳（见 commit …）or 不采纳 + why, push, and tell the user to
   re-run the page. Merge = publish (deploy runs on `master`).

`tools/review.py` (= `npm run review --`): `<PR#>` (merge-base..head,
fetches `pull/N/head` if needed), `A..B`, a single commit (`X^..X`), or no
argument (working tree vs HEAD). It `jekyll build`s both sides
(`--future --unpublished --drafts`, worktrees in `/tmp/blog-review/<sha>`,
output cached in `_site-review/<sha>/` — the working tree is always rebuilt),
cuts each changed post's article out of the built page between the
`<!-- article -->` / `<!-- /article -->` markers the three post layouts
emit (older builds fall back to `.post-container` heuristics), splits it
into top-level blocks, aligns them by text (`difflib`, then a similarity DP
inside replace runs), and renders each pair three ways — unified (new
markup with `<del>` text spliced in), left (old + `<del>`), right (new +
`<ins>`): word-level for prose (tags ride with the next word so each
side's markup stays balanced; `\( \)` / `\[ \]` math is one token so KaTeX
still renders it), line-level for code / Mermaid (plus the rendered
diagrams), row/cell-level for tables, item-level for lists; a block whose
text changed > 60 % is shown as old | new. Unchanged runs fold to 「… N
段未变」, one context block on each side, h2/h3 shown only when their
section changed. Pages go to `<head build>/_review/` and are served from
`http://localhost:4100/` (so `/css`, `/js`, `/img` resolve against the new
build); the page reuses the built post's `<head>` (site CSS + the
`rich-content` KaTeX / Mermaid loaders) and loads only `code-tokens.js` and
`inline-popups.js`, not `blog.min.js` (annotations / views / figures would
run against the diff). Its own chrome is `tools/review/review.{css,js}` —
not part of `less/` or the JS bundle. In PR mode the server also answers
`POST /_api/comment` (new line comment or `reply_to`) and `POST /_api/review`
(`gh pr review --approve|--request-changes`), all through the local `gh`
login. Rendering is faithful because it *is* the site's rendering; the
price is ~20 s per side to build.

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
  png/jpg/bmp in bulk and rewrote references; run it again for new large images,
  `--apply` to write). Site-level `img/*.jpg` stay JPEG (og:image targets).
  It sniffs the real format (a `.png` that is really a
  JPEG gets JPEG quality; 32-bit BMPs go through `sips` because `cwebp`
  rejects them). Screenshots wider than ~1600 px are worth an extra
  `cwebp -resize 1600 0` — the article column is 750 px. Site-level `img/*.jpg` stay JPEG (og:image targets).
- `.github/workflows/d1-backup.yml` (Sundays, or manual): `wrangler d1 export`
  of the worker's `blog-views` database (views, views_daily, votes, shares,
  passage_reactions) as a 90-day workflow artifact — the only copy of those
  counters. Needs secrets `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN`
  (Account · D1 · Edit). Restore: `wrangler d1 execute blog-views --remote
  --file blog-views.sql` (the dump has `CREATE TABLE` + `INSERT`s).
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
  failure), open issues via REST, and 值得翻新的老文章 (`GET /stats/top?limit=500`
  joined with `window.DASH_META` — per-post `[date, updated, source path,
  category]` emitted at build time; tech posts whose `updated`/date is >= 3
  years old, ranked by views × log(age), with 简报 + GitHub 编辑 links; a post
  leaves the list once `updated:` is set), and 待修订的文章 above the 修订简报
  picker (worker `GET /feedback` with no path = every post's D1 rows in one
  call, joined with the open `划线评论` / `待修订` issues; a post is listed when
  it has 存疑 / 章节没看懂 / an open issue, ranked by `FeedbackBrief.analyze`
  score — Discussion comments are not fetched for the list, only for the
  brief itself). `sitemap: false`, `noindex: true`
  (`head.html` emits the robots meta for `page.noindex`). Its styles are
  `less/dashboard.less` (`.dash*`).
- `index.html`: posts with `pinned: true` lead page 1 (badge `.post-pin`) and
  are skipped in the paginated flow. Sidebar (`_layouts/page.html`): HOT TAGS
  is `_includes/tag-cloud.html` (`min=site.featured-condition-size`); `/tags/`
  renders the same include with `min=0` (`id="tag_cloud"`, one size step
  larger). Weight tiers `tag-weight-s/a/b/c/d` by post count, styled
  `.tag-cloud` in `less/extras.less` — pure CSS, the old `tagcloud.js` colour
  interpolation is gone. RECOMMEND renders
  `site.recommends` (`title`/`href`/`desc`) as external links. Styles for these
  live in `less/extras.less`.
- Post layouts: `post` (default, text header), `header-post` (hero image;
  front matter `header-img`, `header-bg-css`, `header-mask`,
  `header-img-credit(-href)`), `keynote` (header is an `iframe` of a slide
  deck sized to the viewport, `navcolor: invert` for light decks; same body
  as the others — catalog, pager, related, comments; demo post
  `keynote-layout-demo`). `redirect_from:` works
  (jekyll-redirect-from). `tools/new-post.py` scaffolds a post/draft.
- Related posts (`_includes/related-posts.html`, "YOU MIGHT ALSO LIKE") come
  from `_plugins/related_posts.rb` (`:site, :post_read`), which fills
  `post.data['related']`: shared tags weighted by rarity (IDF), normalised by
  both posts' tag counts; own-series members excluded (they have the pager +
  series TOC), max one post per other series, same category only, ties to the
  newer post, topped up with newest same-category posts when fewer than
  `related_posts_threshold` share a tag. Each entry carries `shared_tags`
  (rarest first) and the include prints up to three as the reason. This is
  why every post needs tags — an untagged post is never recommended.
- `_includes/post-stale.html` (all three post layouts, first thing in the
  post column): 「本文写于 / 最后更新于 N 年前，部分内容可能已经过时」 when
  `updated` (else `date`) is >= 3 full years before `site.time` — the daily
  deploy keeps the number current. Skipped for `category: life` and for
  `stale: false`. Styled `.post-stale` in `less/extras.less`; stripped from
  the WeChat export.
- `tools/audit.py` (= `npm run audit`): the *soft* content report `check.sh`
  does not block on — untagged posts, single-use / case-variant tags, missing
  subtitle, duplicate titles, stale drafts, non-WebP or > 300 KB content
  images, bare `http://` links, and posts with a substantive edit (>= 20 lines
  in a commit touching < 10 posts, i.e. not a mechanical sweep) in the last
  30 days without a matching `updated:`. Read-only; `--limit`, `--days`.
- Site search (`_includes/search-overlay.html` + `js/search.js`, index from
  `_plugins/search_index.rb` at `:site, :post_render`): `/search/meta.json`
  (one `[url, title, date, tags]` per post, 68 KB), `/search/idx/<0..255>.json`
  (inverted index, key → id deltas, ~8 KB gzipped each) and
  `/search/doc/<slug>.txt` (plain text per post). Keys: ASCII words
  `[a-z0-9_]+` (whole-word, = JS `\b`), `~part` for the `_`-separated parts of
  identifiers, every CJK char and every CJK bigram. The client ANDs a term's
  keys, fetches only those buckets, ranks by title hits (+20 whole query, +6
  per term; ties → newer), then fetches the text of the 10 posts it shows for
  snippet / highlight and re-checks the real match there (3+ char CJK terms
  and punctuated ASCII are bigram/word approximations, shown as 「约 N 篇」).
  Result set / order / snippets were verified identical to the old
  all-in-memory search over 27 zh/en queries. `bucketOf` in JS and
  `SearchIndex.bucket_of` in Ruby must stay in sync. **Pagefind was tried and
  rejected** (2026-09-16): word-based segmentation is wrong for CJK substring
  search (参数服务器 → 148 hits via 参数 + 服务器; 一致性哈希 missed 6 of 7).
- Heading anchors: `js/toc.js` appends an empty `a.heading-anchor` to every
  heading in `.post-container` (glyph via CSS in `less/extras.less`, so the
  heading's textContent — what highlight comments anchor to — is unchanged);
  click copies the section URL via `window.BlogCopy` (exported by
  `js/code-copy.js`, same focus/secure-context fallback) and scrolls with the
  navbar offset. Removed from the WeChat export. This replaced the theme's
  AnchorJS (cdnjs, `anchorjs: true` in `_config.yml`) — for a week both ran
  and every heading had two `#`s; do not bring AnchorJS back.
- `<head>` load order (`_includes/head.html`): `bootstrap`, `argan-blog`,
  `github-markdown` are blocking (they set layout / body typography);
  `syntax.css` and the Font Awesome subset are colours and glyphs only and load
  via `media="print" onload="this.media='all'"` (+ `<noscript>` fallback).
  `js/search.js` is `defer` — `search-overlay.html` waits for
  `DOMContentLoaded` before touching `BLOG_SEARCH`. CDN URLs are explicit
  `https://` (protocol-relative ones showed up as mixed content in local
  audits). `/tags/` lists ~1400 entries: keep its per-entry markup lean and
  free of HTML comments (it used to be 1.2 MB).
- Colours: the brand teal is `@brand-primary: #00788f` (was `#0085a1`, 4.31:1
  on white — just under WCAG AA 4.5; the same hex is repeated as a literal in
  `less/dashboard.less`, `less/theme-overrides.less`, `archive.html`,
  `slides.html`, `_layouts/slides.html`, `_includes/search-overlay.html`,
  `js/wechat-export.js`, so grep before changing it again). Muted text is
  `#6a6a6a` (`@gray`, `.text-muted`) / `#59636e` (GitHub-style, `@annot-faint`);
  nothing lighter on white — Lighthouse contrast is at 0 failures.
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
- `_includes/rich-content.html` — Mermaid (11.17.2) + KaTeX (0.18.7, only the
  public `.katex` / `.katex-display` classes are referenced from our code, so
  0.18's internal class prefixing did not matter) loaders, shared by
  `_includes/head.html` and `_layouts/slides.html`. Both renderers are lazy:
  they only fetch their bundle if the page actually contains a diagram/formula,
  and they only look inside `.post-container` and `.reveal .slides`.
- `_includes/analytics.html` — GA4 gtag (`ga_track_id: G-…`; the old
  `analytics.js` + `UA-` id only kept working through Google's UA→GA4
  "connected site tag" forwarding) + Baidu Tongji, shared by `footer.html` and
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
  `js/diagram-zoom.js` is the zoom/pan lightbox for Mermaid diagrams *and*
  content images (>= 200 px natural width, not inside `<a>`), exposed as
  `window.DiagramZoom.open(el)` and opened **only** from the 放大 button
  `js/figures.js` puts in each figure's `.fig-tools` strip — not by clicking the
  picture (a drag that selected a caption ended in a click that opened the
  lightbox over the 划线 toolbar), and no `zoom-in` cursor.
- **No jQuery / Bootstrap JS.** `footer.html` loads one bundle,
  `js/blog.min.js` (`npm run js` = `tools/build-js.sh`, uglify-js; the source
  list and order live in that script — `figures.js` must precede
  `annotations.js`). In it: `js/argan-blog.js` does the theme bits in plain
  DOM (wrap tables in `.table-responsive` + `.table`, wrap YouTube/Vimeo
  iframes, navbar hide-on-scroll-down `.is-fixed/.is-visible`, `.side-catalog.fixed`),
  then toc, diagram-zoom, code-copy, code-tabs, figures, code-tokens,
  inline-popups, vendor/approx-string-match, annotations, share. Still
  separate: `js/search.js` (head, every page), `js/wechat-export.js` (lazy, author
  only), `js/dashboard.js` + `js/feedback-brief.js` (`/admin/`). The mobile
  navbar toggle is inline in `nav.html`. FastClick and the
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
  「♥ 点赞 N · 分享 · [复制为公众号格式] · [编辑文章]」 right above
  `comments.html` (the GitHub edit link appears only for the author on the
  post-like layouts that include this bar). `slides.html` has its own direct
  「编辑幻灯片」 link because it has no comments/action bar. **「点赞」 is
  anonymous**: worker `GET/POST /votes` keeps
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
  same `mark.annotation-hl`; `.has-doubt` = red dotted line; `.has-issue` =
  red solid line on a faint red wash + ⚑ in the marker, when a live note there
  has an *open* GitHub Issue — `passageIssues(p)`). One reader's
  choices live in `localStorage["react:<path>:<hash>:<kind>"]`. The unit of
  everything passage-level is `passages()` / `passageFor(ids)` (`{ ids, list,
  hash, exact, reaction, marks }`): markers (`markerHtml`: 💬 · 👍 · ❓),
  `openThread`, `passageContaining(offsets)` (a selection inside an
  underlined passage joins it — comment or reaction), `renderHotPassages`.
  The toolbar's 评论 button is relabelled per selection (`updateCommentButton`):
  「编辑评论」 when the viewer's own note is on that passage (click =
  `editMyComment`: open the thread and start `startEdit` on it — re-selecting
  your own quote means "fix my note", not "add a second one"), 「加入讨论」
  when others' notes are, plain 「评论」 otherwise. Posting, replying or saving
  an edit *in the panel* closes it (`closePanel`; the flash + toast confirm);
  the bottom comment section's editors stay put. 搜一搜 opens a three-item
  menu (`openSearchMenu`, reuses `.pa-share-pop` styles): 站内搜索 →
  `window.openSearchOverlay(q)` (search-overlay.html pre-fills and runs the
  query), Google, and Google AI 模式 → `search?udm=50&q=` with `aiPrompt()`
  — the passage wrapped in a Chinese prompt naming the article (og:title) and
  chapter (`sectionForOffsets`); a URL has no system prompt, the instructions
  ride in `q`.
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
    multi-select: each chip toggles on its own — `POST /reactions kind:'reason'
    {reason}` counts, `{reason, prev: reason}` un-counts; my picks are the
    comma-joined keys in `localStorage["react:<path>:<hash>:reason"]`;
    un-doubting un-counts every pick, one call each, sequentially because the
    worker read-modify-writes the JSON). Counts live in `passage_reactions.reasons`
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
    becomes `figure.post-figure > span.fig-media > img + div.fig-tools >
    (button.code-copy.fig-zoom + button.code-copy.fig-feedback) +
    figcaption.post-figcaption (.fig-no 「图 N：」 + .fig-title = alt)`; every rendered
    `.mermaid` gets its `svg` wrapped in the same `.fig-media` (inline-block,
    `width` = the svg's `max-width`, so it shrink-wraps the drawing), a `.fig-tools`
    strip (code-copy's button · 放大 · feedback) and the caption as the `.mermaid`'s
    next sibling (title = Mermaid front matter `title:` or a first-line `%% 图：…`
    comment). The strip is **always visible** (GitHub-style) on the block's
    top-right corner, like on code blocks — it used to be hover-only on the
    picture's own corner, which covered a narrow diagram's top node once it stayed
    on. 32 px targets, click handler stops propagation. `code-copy.js`'s duplicate check
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
    and HTML. Every table also gets a caption under it, like a figure's:
    `<div class="post-figcaption table-caption">` 「表 N：标题」 (「表 N」 when
    untitled), tables numbered in document order, independently of 图 N. The
    title is written **Pandoc style — a paragraph right after the table
    starting with `Table:` or `表：`** (`表1：` also works, the number is
    dropped; inline markup kept):

    ```
    | a | b |
    |---|---|

    Table: 各调度器对比
    ```

    `_plugins/table_captions.rb` (`:documents, :post_render`, posts only)
    folds that paragraph into the table as a real `<caption>` at build time,
    so the static HTML / feed / WeChat export / review pages carry it without
    JS (CSS `caption-side: bottom` while it is still native); HTML tables can
    write `<caption>` directly. `figures.js` then moves the caption's nodes
    into the `.fig-title` under the table (it also accepts the raw `Table:`
    paragraph on pages the plugin does not process). Feedback handle: titled →
    selects the title (the passage, stable while the title is); untitled →
    the `<thead>` row (「表 N」 renumbers), a header-less table → the whole
    table. `.table-caption` is in `BLOCK_SELECTOR` and `blockFor` redirects a
    table cell's panel to the caption, so a note on a table lands under its
    caption and `:has()` outlines the table.
  - **Section-level 点赞 / 没看懂** (`renderChapterBars`, `.sec-react` appended
    inside every article heading `h2`–`h6`, two `.sec-react-btn`s; `chapters` map): anonymous
    like passage reactions, no selection needed. Same worker route and table,
    `quote = '§ ' + heading` (`CHAPTER_PREFIX`), `section = heading`, `up` = 点赞,
    `doubt` = 没看懂; `loadReactions` splits `§ ` rows into `chapters` so they are
    never anchored as passages. `.sec-react` is in `EXCLUDE_SELECTOR`, in
    wechat-export's `REMOVE`, and `headingText()` strips it (and `.heading-anchor`)
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
- **Interplay with footnotes / tips / formulas**: the text index excludes footnote
  markers (`sup[id^=fnref]`, `a.footnote`, `.reversefootnote` — footnote *bodies*
  are indexed and can be annotated), Mermaid,
  markers/panels and the comment section. **KaTeX**: `.katex-html` (the glyphs)
  is excluded, but the TeX source in the MathML `<annotation>` is indexed
  (`isExcluded`), so a formula is a passage whose `exact` is `s = 8192` /
  `\frac{a}{b}` as written; a selection boundary inside a formula takes the
  whole formula (`currentRange`). The `<mark>` then sits in the hidden MathML,
  so `markHost(mark)` (= the `.katex`) carries the visible classes
  (`.katex.has-note` + `.has-doubt/.has-issue/.is-resolved/.is-new`), the
  click handler and the marker (`insertMarkers` puts it after the formula).
  Any selected text >= 1 char shows the toolbar (single characters allowed).
  `<mark>` wraps text nodes only, so
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

Decks are ordinary Markdown; kramdown renders the file and the layouts split
the HTML on every `<hr>` into slides. One file gives two pages:

- `/slides/my-talk.html` — `_layouts/slides.html`, the **landing page**, laid
  out like a web PowerPoint: a thumbnail rail on the left (one miniature per
  page — the slide HTML in a 1280×720 box scaled by CSS transform, styled
  after the white theme, `data-background*` honoured; current page marked
  and kept in view; click = jump) and the deck playing large on the right
  (16:9 iframe + a bar: 上一页 / 下一页 / `n / N` / PDF / 新窗口 / 全屏;
  `js/slides-player.js` drives the iframe's `Reveal` directly — same
  origin). Above: the post-style header meta (author · date · pages · 阅读 /
  点赞 / 评论 via `.post-stats`); below: the action bar and the comments.
  Views / likes / the Discussion are keyed on this URL like a post. Position
  in the URL is `#/N` (replaceState). Under 992 px the rail becomes a
  horizontal strip beneath the player. Styles: `less/slides.less`.
- `/slides/my-talk/play.html` — `_layouts/deck.html`, the bare reveal.js
  presentation, generated by `_plugins/slides_deck.rb` from the same file
  (front matter copied, `sitemap: false`, `noindex`, canonical → landing
  page). `?embed` (what the player loads) hides the back link, reveal's
  controls / slide number and leaves the hash alone. Use this URL for
  `keynote` posts' `iframe:`, for `?print-pdf`, and for presenting.

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
- Export: open `/slides/my-talk/play.html?print-pdf` (the PDF button) and print from the browser.

`slides/reveal-demo.md` is a live demo of all of the above. `/slides/` lists
the decks (cards link to the landing page; 全屏播放 / PDF go to `play.html`).

## 随笔 / Moments (`/moments/`)

Short notes in a 朋友圈-style timeline — the third content type next to posts
and decks. **One file per month**, `moments/YYYY-MM.md` (layout `moments` from
the `_config.yml` defaults; no other front matter needed), entries under dated
headings, any order, rendered newest first:

```markdown
## 2026-09-21 08:02 @深圳湾      date · optional HH:MM · optional @place
早起跑了五公里。                   plain Markdown

![](/img/moments/2026/09/a.webp)   image-only lines in a row = one gallery
![](/img/moments/2026/09/b.webp)   (1 large · 2 / 4 two columns · 3+ a 3-col grid)

> 人生到处知何似，应似飞鸿踏雪泥。   blockquote = quote card, line breaks kept;
> —— 苏轼《和子由渑池怀旧》          a last line starting —— / — / -- is the attribution

https://music.163.com/#/song?id=347230   a line that is only a URL of 网易云 /
                                         QQ 音乐 / Spotify / Apple Music, or an
                                         .mp3/.m4a/.ogg link = a player card
```

`tools/moment.py "文字" [--at 地点] [--img a.jpg …] [--quote "…" --by "…"]
[--music URL] [--time "YYYY-MM-DD HH:MM"]` appends an entry (images →
`img/moments/YYYY/MM/*.webp` via cwebp, ≤ 1600 px); with no arguments it opens
the month file in `$EDITOR` under a fresh heading.

How it is built (`_plugins/moments.rb`):
- `:site, :post_read` gives each month page `permalink: /moments/YYYY-MM.html`
  (pages do not get `.html` from the site's `/:title.html` style, and the
  worker's `VIEW_PATH` wants it), `date` (the 1st), `month`, `title`.
- `Moments::Generator` (`:low`) splits `page.content` on the `## YYYY-MM-DD…`
  headings, renders each entry with the site's kramdown converter (no Liquid),
  applies the gallery / quote / music rewrites and stores
  `page.moments = [{id, title, time, has_time, place, html, text}]` (id
  `YYYYMMDD[-HHMM]`, `-2` … on collision; title = `2026-09-21 08:02`, the
  section / reaction quote). `site.data.moments = {months, entries}` feeds
  `moments.xml` (one `<item>` per entry, 30 newest, guid = month URL + `#id`),
  `archive.html` (`[Moments]` rows) and the month picker. `/moments/` is a
  `PageWithoutAFile` copy of the newest month (`canonical` → month URL, which
  `head.html` now honours, `sitemap: false`).
- `_layouts/moments.html` iterates `page.moments` — never `{{ content }}`.
  Markup per entry: `li.moment#id[data-title] > a.moment-when > time` +
  `.moment-body` (place, html, `.moment-foot`: `.sec-react` placeholder ·
  评论 → `#comments` · 链接). `comments.html` / `.post-stats` use
  `page.comments_path` (= the month URL) so the `/moments/` copy shares the
  month's Discussion, views and reactions.
- Reader interaction reuses the post machinery unchanged: the container is
  `.post-container.moments`, so 划线评论 work; `sectionForOffsets` returns the
  enclosing `.moment[data-title]` as the section; the per-entry ♡ is a
  section reaction (`renderChapterBars` fills any empty
  `.sec-react[data-title]` placeholder — `data-kinds="up"`,
  `data-icon`/`data-icon-on` swap the glyph, toast says 已点赞 not 这一章; quote
  `§ 2026-09-21 08:02` in `passage_reactions`). `figures.js` skips captions
  in `.moments` and opens `DiagramZoom` on a `.moment-pic` click.
  `EXCLUDE_SELECTOR` has `.moment-when, .moment-foot, .moment-music`.
- Search: one document per month page (`search_index.rb`, doc file
  `<month>.txt` because js/search.js derives it from the URL's last segment).
- Styles `less/moments.less`, all under `.post-container.moments` to outrank
  `css/github-markdown.css` (loaded after our bundle); do not use `<footer>`
  inside an entry — `blog.less` styles the tag for the site footer.

## Writing AI-Infra series posts

- **Every figure and every table has a title** (applies to all posts, not
  only the series). Images: the alt *is* the title (`![学习率调度与 batch 增长](…)`),
  never empty or a file name. Mermaid: first source line `%% 图：…` (after any
  `%%{init}%%`), or `title:` in Mermaid front matter. Tables: a paragraph
  right after the table, `Table: …` (or `表：…`), see the Tables bullet under
  the annotations section. The page shows 「图 N：title」 / 「表 N：title」 under
  the block; a bare 「图 N」 / 「表 N」 in a rendered post means a missing title
  and is a defect to fix. Titles are a noun phrase saying *what the reader is
  looking at* (「三种并行方式的通信量对比」), not a sentence, not the section
  heading repeated, no trailing period; the number is never hand-written (it
  is generated and would change when a figure / table is inserted).
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
  (v1.2.1); series 8 uses `../vllm-v0.27.1`; series 11 (平台) and 12 (开源贡献) use
  `../vllm-v0.28.0` (series 11 only for CLI flags / metric names / OpenAI
  protocol fields; its platform components are pinned to their Aug-2026
  releases, local checkouts `../kueue`, `../volcano`, `../kserve`, `../llm-d`,
  `../llm-d-router`, `../gpu-operator` etc.); series 12 also uses
  `../pytorch-v2.14.0`. Series 9 (RL 后训练基础设施) will pin verl v0.9.0 and
  reuse `../vllm-v0.27.1` / `../pytorch-v2.13.0` / `../Megatron-LM`. Series 2 pins PyTorch v2.10.0 /
  vLLM v0.15.0 and series 5 pins vLLM v0.20.0 but have no local worktree —
  add one (`git -C ../vllm worktree add ../vllm-v0.20.0 v0.20.0`) before
  re-verifying their source citations.
  Series 13 (选修《ML 编译器内部》, key `ml-compilers`, dates 2026-11-21 …
  12-04) pins Triton **v3.8.0** (`../triton-v3.8.0`, detached at the tag;
  `.venv` = python3.12 with `pip install -e .` built on macOS — `triton-opt`,
  `triton-tensor-layout` and the gtest binaries live under
  `build/cmake.macosx-11.0-arm64-cpython-3.12/`; the LLVM pin is auto-downloaded to
  `~/.triton/llvm/`), LLVM/MLIR **23.1.1** (Homebrew `llvm`: `mlir-opt`,
  `mlir-tblgen`, `mlir-runner`, `opt`, `llc` with `nvptx64`/`amdgcn` targets),
  TVM **v0.26.0** (`../tvm-v0.26.0`, built in `build/` with Apple clang + Homebrew
  LLVM; use `PYTHONPATH=python`, never `pip install -e`). All IR in the posts
  was generated locally without a GPU: NVIDIA compiles run with
  `TRITON_PTXAS_PATH=/tmp/mlc/fakebin/ptxas` (a script answering `--version`
  with `release 12.9, V12.9.86` and touching the `-o` file), giving real
  ttir/ttgir/llir/ptx and a 0-byte cubin; the AMD path (`GPUTarget("hip",
  "gfx942", 64)`) needs nothing. Scratch scripts and dumps are in `/tmp/mlc/`
  (`triton/compile_matmul.py`, `compile_tma.py`, `compile_amd.py`,
  `compile_small.py`, `plugin/MulToShift.cpp`). Rebuild the fake ptxas if
  `/tmp` was wiped. Do **not** `cd` into `../tvm-v0.26.0`, `../pytorch-*` or
  `/opt/homebrew` from the blog shell — they ship their own AGENTS.md whose
  rules then leak into the session; run their commands with `workdir` or
  absolute paths. `lit` must be `< 20` in the Triton venv (Homebrew's lit 23
  rejects Triton's `lit.cfg.py`).
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
  publication order = reading order (and again to 11 / 12, 09-14 … 09-27, when
  series 10 扩散 was inserted on 2026-09-15).
  verl is the single deep-dive framework (源码线 + 8 卡实践); slime / AReaL appear
  only as 对照 in post 7. Version baseline in the overview: verl v0.9.0, slime v0.3.0, OpenRLHF
  v0.11.0, AReaL paper/docs, vLLM v0.27.1, PyTorch 2.13.0, Megatron Core 0.18.0.
  Only post 1 has a companion script (`ai-learning-labs/rl-post-training-infra/rl_ledger.py`);
  posts 2–8 deliberately have none — the author asked to stop writing
  companion experiments (they cost time and thinned the articles); each post
  ends with a prose 实践建议 instead. Sources were read from shallow clones of
  verl v0.9.0 / vLLM v0.27.1 / slime v0.3.0 / AReaL; cite paths + function
  names, never line numbers. Overview posts that link to future-dated posts
  fail lychee until those dates — build locally with `--future` to check.
- Series 10 `diffusion-inference-infra` (《扩散模型推理基础设施：从一次去噪到一个生成服务》,
  overview `2026-09-04-diffusion-model-inference-infrastructure.md`, Infra
  roadmap **L4 alongside 07/08/09** — promoted from the planned 选修 on
  2026-09-15 with the same argument as 09: the workload is mainstream now
  that SGLang and vLLM both ship diffusion serving, and it is the other half
  of the inference mainline) is complete: overview + 9 posts dated 09-05 …
  09-13 (written 09-15). To keep publication order = reading order, 平台 and
  开源贡献 were renumbered again to **11 / 12** and re-dated +10 days to
  09-14 … 09-22 and 09-23 … 09-27 (ten already-live posts went offline for up
  to 12 days — accepted by the author). 开源贡献 stays 横切 (贡献者路径), not
  选修; the only 选修 left is ML 编译器. No single deep-dive framework
  (author's choice): mechanisms first, each post ends with an 「实现对照」
  table over SGLang Diffusion v0.5.19
  (`../sglang-v0.5.19/python/sglang/multimodal_gen/`), vLLM-Omni v0.28.0
  (`../vllm-omni-v0.28.0/vllm_omni/diffusion/`), xDiT commit `07572e7`
  (2026-09-02, the last commit before the overview's date; checked out in
  `../xDiT`, the project has no tags) and diffusers v0.40.0
  (`../diffusers-v0.40.0`); post 8 walks one request through all three. Only
  post 1 has a lab (`ai-learning-labs/diffusion-inference-infra/diffusion_ledger.py`,
  stdlib); FLOPs use `P_tok` (parameters a token actually passes: FLUX 6.45B of
  11.9B), MFU 0.45 default (xDiT measured eager 0.31 / compile 0.49). Inline
  math must be `$$…$$` (kramdown turns it into `\(…\)`; single `$` is literal —
  the first draft of this series had 660 of them). Literal prices are `\$2.5`.
  When a whole series is drafted first, give the drafts a provisional `date:`
  (drafts have no filename date, so the series nav would otherwise order by
  mtime) and preview with `jekyll build --drafts --future -d /tmp/_site_drafts`
  + a static server + `SITE=… node tools/check-render.cjs`; remove `date:` on
  publish.
- Series `coding-interview` (《面试手撕代码：从 LeetCode 中等题到 Transformer 组件》,
  overview `2025-12-01-coding-interview.md`, **not part of any roadmap** —
  the author wanted it kept out of the full-stack map; only the overview's
  last paragraph points to the maps) is complete: overview + 19 posts dated
  **2025-12-02 … 12-20**, i.e. before the roadmaps, so the 2026 timeline is
  untouched (written 2026-09-15). 01–13 are LeetCode-medium pattern posts
  (每篇：识别信号 → 模板 → 3–5 主讲题逐题推演 → 变式追问 → 两种语言的坑 →
  题单 → 自测; DP 11/12 marked 可选), 14–19 are AI-role 手撕 (attention,
  Transformer block + backprop, tokenizer + decoding, losses + training
  algorithms, classical ML + metrics, Infra concurrency/systems). Main
  problems were picked by scoring 高频 / 模板代表性 / follow-up 空间 (table in
  `ai-learning-labs/coding-interview/README.md`); problem statements are
  paraphrased, never copied. Every algorithm snippet is Python **and** Java
  in one `<div class="code-tabs" markdown="1">` (Infra post: Python + C++);
  the code is copied verbatim from the labs, which all have tests
  (`python/` unittest, `java/` `make test` with `-ea`, `ai/*.py --check`
  against torch, `infra/` `make run`). Java is formatted with
  google-java-format `--aosp` (4-space indent, one statement per line,
  blank line between members) — both the labs files and every `java`
  fence in the posts (wrap a fence in `class __W { … }`, format, unwrap,
  dedent 4; a fence containing `...` pseudo-code is hand-formatted the same
  way). Never write `{ a; b; }` one-liners. Version rule for a 2025-12 date:
  Python 3.12 / Java 21 syntax / NumPy 2 / PyTorch 2.5 only; the overview
  and posts 14–19 link forward to 2026 series and therefore carry
  `updated: 2026-09-15`. Mermaid: fan-outs (one node → 4+ children) get
  squeezed at 755 px — write DFS traces as vertical chains, and use
  `flowchart LR` only for a tree whose leaves should stack vertically.
- `code-tabs` (added for that series, `js/code-tabs.js` loaded after
  `code-copy.js`): direct `.highlighter-rouge` children of `.code-tabs` become
  panels labelled from `language-xxx`; the choice is page-wide and persisted
  in `localStorage["code-tab-lang"]`; groups lacking the preferred language
  show their first panel; no-JS stacks the panels with a language label
  (`less/extras.less` `.code-tabs`);
  `wechat-export.js` flattens groups into labelled blocks. Highlight
  comments anchored in a hidden panel simply stay hidden until that tab is
  chosen — no special handling.
- Line numbers (`_plugins/code_lines.rb`, `:documents, :post_render` on
  posts): every `<pre><code>` line becomes `<span class="line">` (Rouge spans
  that cross lines are closed and reopened, textContent unchanged) and the
  `<pre>` gets `data-lines="N"`; blocks that show numbers get `pre.lineno`
  (+ `lineno-3` / `lineno-4` for 100+ / 1000+ lines, gutter width). Default:
  a real language (`language-xxx`, not `text` / untyped) and >= 2 lines —
  ```text and bare fences are output / logs / ASCII art. Per block: IAL on
  the line before the fence, `{:.lineno}` forces, `{:.no-lineno}` hides.
  The numbers are CSS counters (`less/theme-overrides.less`, `.line::before`,
  sticky so they stay while a long line scrolls), so they are never text:
  copy, 划线 quotes, the search index and the WeChat export do not see them.
  Not Rouge's `line_numbers`: global, table layout, numbers in innerText,
  and ~70 % of the fences never reach Rouge. Mermaid sources are skipped.
  Prose that refers to lines by number now has something to point at;
  `第 N 行` is still plain text (no auto-linking — it also means table rows).
- Code refs (same plugin, `js/code-refs.js`, styles next to the lineno rules):
  the Code Hike "code mentions" model. In the fence, a comment line **of its
  own** `# !ref name` (`+N` covers N more lines; `//` `--` `;` `%` `/* */`
  `<!-- -->` all work, name `[A-Za-z][\w-]*`) is *dropped* from the output —
  not counted, not copied, not indexed — and the next line(s) become
  `span.line.ref-line[data-ref=name]`, the first one with `id=name`. In the
  prose, a plain kramdown link `[text](#name)` becomes
  `a.code-ref[data-ref][data-line=N]`. A block with refs always shows line
  numbers: the blue gutter number *is* the marker. JS: hover a prose ref →
  lines `.is-active`; click → scroll (nav offset) + flash, `pushState`;
  hover / tap the gutter (x-coordinate test against the `::before` width, the
  pseudo-element cannot take events) → the ref's paragraph in the shared
  `InlinePopover` with the ref `<mark>`ed and a 「查看说明」 jump. Without JS
  it is an anchor jump with `.line:target`. `wechat-export.js` appends
  「（第 N 行）」. Names are per post; a name used in several blocks
  (code-tabs panels) gets its id on the first only, JS picks the visible
  panel. `!ref` never linked → build warning; link to a missing ref → lychee
  fails `npm run check`. A directive inside a multi-line string / block
  comment (a Rouge span still open at the line start) is left as text;
  `{:.no-refs}` on a block keeps directives verbatim (the memo shows the
  syntax that way). Trailing-comment form (`x = 1  # !ref a`) is deliberately
  unsupported — stripping half a Rouge span is not worth it. The 34 posts
  that explain code with ①②③ in comments keep working as plain text; new
  posts should use `!ref` when a paragraph explains specific lines.
- Companion code lives in `../ai-learning-labs` (git repo, pushed by the
  user). Its `.venv/` (Python 3.12 via `~/.local/bin/python3.12`, torch CPU,
  numpy, tiktoken, tokenizers) is gitignored; recreate with
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
  Every script writes its full run to `<series>/expected/<script>.txt`; the
  article quotes those numbers, so re-run and refresh `expected/` when a
  script changes. SVG figures for posts are generated by
  `transformer-and-llm/tools/gen_*_svg.py` into `img/in-post/` here; eyeball
  them with `qlmanage -t -s 1480 -o /tmp <svg>` (renders to `/tmp/<name>.png`).
  **Distributed toys without a GPU**: `torch.distributed` with the `gloo`
  backend + `mp.spawn(…, nprocs=4)` runs real all-reduce / reduce-scatter /
  all-gather / all-to-all / isend-recv between 4 CPU processes (~1–2 s a
  spawn). `large-scale-training/02_parallelism_toys.py` (2026-09-21, for the
  并行篇 after reader #74 asked to cross-pollinate with vLLM 08) does ZeRO /
  TP (f, g as `autograd.Function`) / Ring Attention / GPipe / EP (all-to-all
  whose backward is the split-swapped all-to-all) this way and checks each
  against a single-process reference; the article's 「亲手验证」 blocks quote
  its `expected/`. Use the same pattern for any future 通信 / 并行 toy
  instead of pseudo-code.
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
  `multimodal` (《多模态：从视觉编码器到扩散模型》, L7, 9 posts + recap NN=10),
  plus the 横切 导读 `experimental-methodology-for-ai-algorithm-engineers`,
  complete the algorithm roadmap (written 2026-09-14; L6 has no labs — same
  "动手（建议）" rule as post-training 2–8, numbers from papers / tech reports
  only). L6 builds on 04-07 (量化 / 投机 / LoRA 的账) and 04-03/04-04 (KV) and
  must not re-derive them; L7 builds on 04-08 (多模态成本) and L3-05 (ViT). Both
  series link to 04 / L5 posts by design — the "series are independent" rule
  below applies to the Infra series, whereas algorithm-map series cite each
  other through the map's layer structure. Time anchors: nothing later than
  2025 (posts are dated Apr–May 2026); model refs go up to Qwen2.5-VL / Gemma 3
  / BAGEL / gpt-oss.
  **L7 was expanded 2026-09-21 (reader #67 「太简略，要图文并茂、有实际例子」) to
  the L2 depth standard**: 语音 and 扩散 each split into 上 / 下 (the 上 keeps
  the old slug, the 下 is a new file on the same date with `date: … 20:00:00
  +0800` — 05-06 `speech-understanding-generation-and-full-duplex`, 05-07
  `score-matching-flow-matching-and-classifier-free-guidance`; recap moved to
  NN=10; roadmaps say 9 篇 / 7h). Every body post now has a CPU toy in
  `ai-learning-labs/multimodal/NN_*.py` (numbers / figures in the post come
  from `expected/` and `out/`, copied to `img/in-post/multimodal-NN-*.svg`;
  `_plot.save` rasterizes scatter / quiver so a 4000-point plot stays ~100 KB);
  the 「动手（建议）」 sections stay for real-model reproduction on a GPU. 06 /
  07 / 08 share `_diffusion_toy.py` (two-moons data, MLP, DDPM schedule) and 07
  / 08 load the model 06 saves to `out/06_ddpm_model.pt`. Honest toy results
  are kept as teaching points (flow-matching trajectories are *curvier* than
  DDIM before reflow, 0.49 vs 0.74; a count-based next-token model produces
  half-recognizable digits).
- Post dates encode the reading order of the three roadmaps and were re-dated
  on 2026-09-14 (permalinks are `/:title.html`, so dates are free to move):
  01-01 《AI 全栈学习地图》(overview of the three, pinned) → 01-02 算法地图 →
  01-03 Infra 地图 → 01-04 应用地图 → 算法 L0 数学 (01-07 overview, 01-08 … 01-15,
  series `math-for-ai`) → L1 工具箱 (01-16 overview, 01-17 … 01-22,
  `algorithm-tooling`; 01-17 Python 使用层 was added 2026-09-15 and everything
  up to 01-30 shifted a day) → Infra 01 Python (01-23 … 01-30, shared: L1 深入篇) →
  Infra 02 C++ (02-02 … 02-15) → Infra 03 PyTorch (02-16 … 02-26, shared: L1
  深入篇) → L2 经典机器学习 (02-27 overview, 02-28 … 03-09, `classical-ml`; expanded
  from 6 to 10 body posts on 2026-09-21 — see below) →
  L3 (03-23 … 03-29) → 04 Transformer 与 LLM (04-01 … 04-13, shared L4)
  → 后训练 (04-15 … 04-23) → 横切 实验方法论 (04-24, one 导读) → L6
  高效推理与压缩 (04-25 overview, 04-26 … 05-01) → L7 多模态 (05-02 overview,
  05-03 … 05-09, two same-day 下篇 at 20:00, recap 05-09 20:00) → Infra 05–10 (GPU Kernel was moved from 05-06…05-30 to
  05-10 … 05-20 on 2026-09-14 to make room; 通信 starts 06-01 unchanged) → 07
  大规模训练 (07-13 … 07-29) → 08 vLLM (08-11 … 08-25, daily) → 09 RL 后训练基础设施
  (08-26 overview, posts 08-27 … 09-03) → 10 扩散模型推理基础设施 (09-04 overview,
  posts 09-05 … 09-13) → 11 平台 (09-14 … 09-22) → 12 开源贡献 (09-23 … 09-27).
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
  none. L1 does not teach Python itself — Infra 01 / 03 are its 深入篇.
- **Split posts (2026-09-21, reader #14 / #52 / #56 「太长」)**: Infra 01 Python
  post 1 → 上 `python-execution-model-scopes-imports-and-exceptions` (执行模型 · 作用域 ·
  导入 · 异常) + 下 `python-object-model-protocols-decorators-and-generators`
  (对象模型 · 协议 · 装饰器 · 生成器 · 上下文 · 完整追踪, same day 20:00); post 2 →
  上 `python-type-expression-and-the-typing-toolbox` (§二 类型表达 + typing 速查表) +
  中 `python-type-information-distribution-and-consumption` (存根 / mypy / 运行时读注解,
  12:00) + 下 `python-data-contract-design-dataclass-pydantic-and-settings` (数据契约,
  20:00); Infra 02 C++ post 1 → 上 `cpp-compilation-model-from-cpp-to-shared-object` (四阶段 ·
  翻译单元 · ODR · 符号 · 动态链接 · 实践一) + 下
  `cpp-project-layout-namespaces-libraries-and-cmake` (命名空间 · 库分层 · CMake ·
  mini-c10 · 建议, 20:00). Series 讲 numbering is unchanged (titles say 「01 上 / 01 下」,
  「02 上 / 中 / 下」; recaps still say 第一篇 / 第二篇 and link all parts); slugs follow
  the content (the user: 「slug 要跟标题一致，没有不能变的」) — the three old slugs are
  `redirect_from:` on the 上篇 files; the giscus threads stayed on the old
  pathnames (#14 / #52 / #56 were answered there). The split
  was a pure move: chapters renumbered, 「第X章」 references rewritten to
  `[上篇第N章](/slug.html#anchor)` across parts (helper `/tmp/split_lib.py` at the
  time — kramdown ids = strip backticks and punctuation, spaces → `-`, lowercase),
  each part got its own 总览 / 小结 / 自测 / 下一篇; body text untouched. Roadmap
  counts: Python 7 → 9, C++ 8 → 9, Infra total 102 → 105 (hours unchanged).
  **L2 was expanded 2026-09-21** after reader feedback (#58 / #59 「整个系列
  太走马观花」): 6 → 10 body posts (01 什么是学习 · 02 线性回归 (new,
  `linear-regression-least-squares-ridge-and-lasso`) · 03 逻辑回归与奖励模型
  (old `linear-and-logistic-regression-…` slug kept) · 04 三个基础分类器 (old
  `a-family-of-classifiers-…` slug kept) · 05 SVM 与核方法 (new) · 06 集成 (new,
  `ensembles-random-forest-and-gradient-boosting`) · 07 聚类 (old
  `unsupervised-learning-…` slug kept) · 08 降维 (new,
  `dimensionality-reduction-pca-svd-tsne-and-umap`) · 09 去重 · 10 评估; recap
  NN=11 at 03-09 20:00). Slugs with giscus discussions were never renamed.
  The depth standard the user set for this series (and asked for series-wide):
  「小白能看懂」 — every mechanism goes 具体小例子 → 图 → 逐符号公式 → 10–40
  行手写实现（带 ①②③ 行标、与 scikit-learn 对数、贴真实输出）→ 在 LLM 里哪出现.
  Second pass (user: 「务必以小白不需要借助其他资料就可以看懂为目标」) added
  two more rules that apply to every expanded series: (a) a prerequisite the
  post relies on (导数、梯度、转置、特征值、期望、方差、似然、标准误、hash …)
  is explained **in place the first time it is used** — a sentence or a
  `tip:` — never 「见 L0 第 N 篇」 (cross-series pointers are fine only for
  *further* reading); (b) every formula gets a **hand-checkable example with
  3–10 numbers** (a table the reader can redo on paper) *before* the code
  runs it on real data, and the same toy numbers are reused across the post
  where possible (02 uses the 3 points (1,3)(2,5)(3,8) from 残差 through
  正规方程 to Ridge). Verify toy numbers with a one-off NumPy run.
  Every number and every figure comes from `ai-learning-labs/classical-ml/NN_*.py`
  (figures are matplotlib SVGs written by `_plot.py` — Heiti SC, `svg.fonttype
  none`, 7.6 in wide, constrained layout — copied to `img/in-post/classical-ml-NN-*.svg`;
  keep scatter plots subsampled so an SVG stays under ~150 KB). Posts 07 / 08 / 09
  use a 78-sentence corpus embedded with the locally cached Qwen2.5-0.5B
  (`_sentences.py`, `HF_HUB_OFFLINE=1`). When
  other posts cite these layers, write 「L0 数学系列第 N 篇」 etc., never
  「L0 导读第 N 章」 (the 导读 chapters no longer exist). Roadmaps link forward to series published later —
  that is the established convention. Series 收尾篇 must NOT carry a
  hand-written 「系列目录」: the layout generates it from `series:`.
- **Application-roadmap series** (started 2026-09-16; the map
  `2026-01-04-ai-application-engineer-learning-roadmap.md` lists them in
  「已有的文章与系列」, one series per layer L1 → L7, dates continue after the
  Infra series: 09-28 onwards, contiguous, no interleaving). Rules that differ
  from the Infra series: **no labs** — every body post ends with a prose
  「实践建议」 section before 本文小结 (the author asked for practicality and
  real industry cases instead); numbers come from vendor docs / pricing pages,
  papers, court records and news reports, each dated. Version rule applies to
  API fields, model names and prices: cite only what was public before the
  post's date and label prices 「2026 年 9 月价目页」. Literal prices in prose
  are `\$10 / \$50` (escaped, outside math); formulas stay in `$$…$$`. Cross
  links between application series and other series are allowed via
  *overviews and maps only*. Series 01 `model-as-component`
  (《模型作为组件：契约、失效模式与选型》, overview
  `2026-09-28-model-as-a-component.md`, posts 09-29 … 10-04, recap
  `2026-10-04-model-as-component-series-recap-and-self-test.md`) is the
  template: intro with `[^q0–3]`, `## 一、总览` (table + one Mermaid where the
  structure is real: where failure modes enter a request, the tool-call
  sequence, a reasoning turn, the latency timeline, the client checkpoints),
  body chapters each with 机制 / 证据 / 检测 / 应对所在的层, `## 实践建议`,
  本文小结, 自测 (5), 下一篇, footnotes. Baseline facts used there (Sept 2026):
  OpenAI GPT-6 Astra (09-03) / GPT-5.6 Sol · Terra · Luna (07-09), Responses
  API recommended, Assistants API closed 08-26, Agents API beta 09-10;
  Anthropic Fable 5.1 (09-01) / Opus 5 (07-24) / Sonnet 5 (06-30, new
  tokenizer +30 %, sampling params → 400) / Haiku 4.5; Gemini 3.8 Flash
  (09-02), Interactions API GA 06, `outputs → steps` 05-26 / 06-08; DeepSeek
  V4.1 Flash (09-10, MIT), V4-Pro routed to it from 09-14, peak/off-peak
  pricing. All seven application series landed 2026-09-16 (dates run daily,
  overview → body → recap on the last body day at 20:00):
  02 `context-engineering` 10-05 … 10-11 (6 body: 七层解剖、模式 vs 措辞、
  结构化输出与约束解码、预算与卸载 / 清理 / 压缩、prompt caching 与排列、
  prompt 当代码管 + AGENTS.md / SKILL.md + 上下文 vs 检索);
  03 `retrieval-and-knowledge` 10-12 … 10-19 (7 body: 进上下文还是进权重、
  三类检索 + coding agent 的 grep / 索引分歧、解析与分块、索引 / 混合 / rerank、
  流水线到 agentic、SQL / 本体 / GraphRAG、评测与运营);
  04 `agent-and-harness` 10-20 … 10-29 (9 body: 最小循环、MCP 2026-07-28 +
  tool search + PTC、运行时与会话日志、上下文与子 agent、权限 / 沙箱 / 安全边界、
  **源码级对照 Codex `codex-rs` / DeepSeek Harness `dsh` / Claude Code /
  OpenHarness (Python)** — twelve-dimension table + three delivery forms、多 agent
  与 A2A、memory 与 human-in-the-loop、可靠性与轨迹评测). Source facts come
  from shallow clones at `/Users/argan/Code/codex` (2026-09-16) and
  `/Users/argan/Code/deepseek-harness` (2026-09-15); cite crate / package /
  module names, never line numbers; DeepSeek Harness is "developer preview,
  compatibility-breaking changes expected". Both clones ship their own
  AGENTS.md — do not `cd` into them from a blog shell or their rules leak in;
  05 `evals-and-observability` 10-30 … 11-06 (7 body: 评测集、judge 校准
  (2026 studies: kappa deflation 33–41 pp, style bias dominant, mid-tier +
  debias beats frontier 15× cheaper)、指标矩阵、门禁 / 静默升级 / 在线、trace +
  OTel GenAI semconv (all Development, moved to `semantic-conventions-genai`
  2026-06)、录制回放 / 决策点 / 失败分类 / 反馈绑定、运行时与物理世界仿真);
  06 `production-and-operations` 11-07 … 11-14 (7 body: 网关、成本、延迟、安全
  (EchoLeak CVE-2025-32711, Cursor MCPoison / CurXecute / DuneSlide, MCP
  description poisoning 2026-06-30, OWASP LLM Top 10 2025)、治理与合规 (EU AI
  Act: Digital Omnibus Reg. 2026/1744 in force 2026-07-27, GPAI enforceable
  2026-08-02, Annex III → 2027-12-02, Annex I → 2028-08-02; 中国标识办法
  2025-09-01)、发布工程、飞轮与物理世界 OTA);
  07 `product-and-experience` 11-15 … 11-20 (5 body: 场景选择 (jagged frontier
  758-consultant study, Klarna / IBM / Duolingo reversals)、形态与后台 agent、
  信任校准、呈现与非对话形态、指标与回流). The 应用地图 has a per-layer
  「这一层已写成系列…」 pointer under each L1–L7 heading plus the 已有的文章与系列
  table; the 全栈 map's three former 「系列待写」 spots say 七层齐. Body posts
  run 10–16k chars (tables and worked examples, no padding); the user asked for
  everything to be written first and reviewed together afterwards.
- **Every series ends with a 「系列总结与通关自测」 post** (added 2026-09-16
  for all 21 series): `_posts/<last-post-date>-<series-key>-series-recap-and-self-test.md`,
  `date: <same day> 20:00:00` so it sorts after the last body post without
  moving the timeline. Bare timestamps like that are UTC by the YAML spec —
  Jekyll rendered them as 04:00 the *next* day Beijing, after the following
  series' overview (filename date = 00:00 local), so every recap showed up
  inside the next series (2026-09-16). `_plugins/local_dates.rb` now
  reinterprets a zero-offset `date:` / `updated:` as wall-clock time in
  `site.timezone`, so write times without an offset like everything else.
  When the next overview shares the day with the previous series' last post
  (pretraining overview on 04-09), give the overview `date: … 22:00:00`. Title `系列名（NN）：系列总结与通关自测` with NN = body
  posts + 1, tags copied from the overview. Fixed structure: intro with three
  `[^q0–2]` questions → `## 一、总览` (one table 篇 | 问题 | 一句话结论 | 必记
  + 章节安排) → `## 二、逐篇回顾` (per post: 核心问题 / 结论 / 必记 / 常见误解)
  → `## 三、贯穿全系列的几条线` (+ concept table; at most one Mermaid, only for
  real dependencies between quantities) → `## 四、常见误区` table →
  `## 五、通关自测` (A 判断与计算 10 · B 跨篇综合 5 · C 面试题 6–8 with 答案要点 /
  追问方向 / 好答案与一般答案的区别 · D 掌握判据) → `## 六、下一步` (links only
  to other series' *overviews* and the maps) → `## 七、延伸阅读` (the
  「本系列的边界」 paragraphs that used to live in the overview: what this
  series deliberately does not cover and which series does, 2026-09-20)
  → footnotes. **「二、逐篇回顾」 must answer, per post, the questions the
  overview's 分章导读 raises for that post** (reader feedback 2026-09-17/20:
  「总纲每一篇都抛几个问题，系列总结务必回应」) — copy the questions into the
  核心问题 cell and give the checkable answer. No 本文小结, no
  下一篇, no lab. Every number must come from the series' own posts. The old
  「系列总结」 sections in the last body posts were removed (本文小结 kept,
  chapter numbers / 章节安排 rows renumbered); overviews got a 分章导读
  subsection and a 章节目录 row; the maps' 篇数 / 时长 count body posts only and
  say so. Adding a body post to a series later means renumbering the recap's
  NN and adding a row to its tables.
- Cite source as path + function/class name, never line numbers.
- Length is not a target; rigor and organisation are. Structure: (update note) →
  intro with the post's core question → `## 一、总览` (ending with 本文的章节安排)
  → body (`##` Chinese numerals, `###` Arabic) → `## N、本文小结` →
  `## N+1、自测` → `## 下一篇` → `[^qN]:` footnote definitions.
  - **Opening questions are answered in footnotes** (`[^q0]`, `[^q1]`, …, in
    reading order). Every question in the intro gets its own marker — the bold
    core question is split at each 「？」 (marker right after the 「？」, inside
    the `**…**`), and each 「为什么……？」 bullet gets one at line end. The
    definitions go after 「下一篇」 (kramdown renders them at the very end
    anyway): a checkable answer of one to several sentences, numbers included,
    ending with 「详见[第 N 章](#anchor)」 links to the chapters that carry the
    detail (anchor = the `<h2 id>` kramdown generates; check `_site`). A post
    with a chapter titled 「回答核心问题」 must match that chapter's numbers.
    The `q` prefix matters: `js/inline-popups.js` skips `fn:q…` footnotes — no
    hover card, click only jumps to the bottom (Q&A is meant to be read after
    the article; hover cards are for explanatory footnotes) — and
    `.footnotes:has(li[id^="fn:q"])::before` labels the list 「文首问题的答案」
    instead of 「脚注」 (styles at the end of `less/extras.less`). The former folded 「核心问题的答案」 `<details>` block after
    小结 is gone (2026-09); do not add it to new posts. All 142 body posts of the
    algorithm and Infra series carry the footnotes.
  - 自测: 3–5 questions per body post (overview posts have none), each with a
    checkable answer (a number, a shape, a yes/no with one reason) — no open
    questions. Every answer sits in its own
    `<details markdown="1"><summary>答案</summary> … </details>` directly under
    the question (a list item's continuation, indented 3 spaces), so the reader
    can try first. `markdown="1"` is required for KaTeX / lists inside. Styles
    for `details` live at the end of `less/extras.less`.
- **Series overview (总纲) template** (unified over all 27 overviews on
  2026-09-20 after reader feedback on the 工具箱 overview: 「作为一篇总纲太啰嗦」):
  内容简介 → 为什么写这个系列 → 系列的整体主线 → 章节结构与分章导读 (each post
  gets its questions) → 贯穿全系列的实践线 / 源码阅读线 (tables with links)
  → 前置要求与说明 → 最终目标. **Not** in an overview: 「阅读路径建议」,
  「怎么学 / 材料 / 顺序」, a 「下一步做什么」 paragraph at the end of 最终目标
  (the recap's 下一步 owns that), 「本系列的边界」 (moved to the recap's
  延伸阅读), a history paragraph such as 「这一层原来是一篇导读 / 这个系列的
  前身」 (no information for the reader), and a 前置要求 row for something the
  series itself teaches (L1 first post teaches the Python subset → Python is
  not a prerequisite). Version baselines (C++17, PyTorch v2.10.0, vLLM
  v0.15.0 …) are stated once, in the overview.
- **Body-post 「一、总览」**: keep the 本文的章节安排 table, but open the section
  with a short paragraph on *why the post is organised this way* (the
  organising axis, e.g. 「按类型信息的流动：表达 → 分发 → 消费」). Do not
  repeat overview material there — no 「本文的读者与读法」, no 「语言标准与
  版本基线」 (removed from the C++ body posts 2026-09-20).
- **A post must be self-contained; the companion script is for reproducing,
  not for understanding** (reader, 2026-09-17: 「看懂文章不应依赖脚本，只有
  需要重现实验才去实验仓拉脚本」). Never write 「脚本里的 X」, 「见 train.py
  第 N 行」 or point the reader at the lab to learn what a thing is; paste the
  code the prose needs (the 20-line loop, the model definition, the
  `get_batch`) into the post and refer to *those* blocks. The lab path
  appears once, in a 「配套代码」 paragraph after 本文小结 — nowhere earlier.
- **「几点」 are written as points** (reader, 2026-09-17, several posts): when
  the prose announces a count — 三件事, 两个习惯, 五个对象, 多出来的四样 — what
  follows is a numbered list starting at 1 (short items) or numbered
  sub-sections (long items), never one sentence with the items joined by
  commas. Do not introduce an item the count did not announce (the 工具箱
  overview said 四块 and then produced a 「第五块」).
- **Table headers carry meaning**: no one-character headers (「错」 →
  「错误类型」, 「概念」 → 「对象」 when the column holds objects).
- **Abbreviations get full name + Chinese on first use**: `peft（Parameter-
  Efficient Fine-Tuning，参数高效微调）`, `trl（Transformer Reinforcement
  Learning）`; a term the passage is not about (Jinja, PPL, ridge point,
  Jacobian, SP) gets an inline tip (see below) at its first appearance.
- **No 「本机」**: 「本机没有 libtorch，输出标注为预期」, 「（本机就有）」, 「本机
  macOS ld64 的实际输出」 — the reader is not on the author's machine. Either
  set the environment up and show real output, or show no output; never a
  「预期」 transcript.
- **No AI / lecturer voice** (reader flagged 「AI 味太浓」 on: 「一个现实的标准」,
  「出现时就是越界的信号」, 「……就过关了；遇到……也是回到这二十行想」, 「本篇写出它」,
  「一切机制都建立在一个事实上」, 「读的时候带着 L0 第八篇的置信区间」). Write the
  concrete statement instead: what the thing is, what number, where it is
  used. Cut sentences that only grade the reader or announce what the text
  is about to do.
- **Code and prose must be linkable**: in a multi-line block that the prose
  discusses piece by piece, mark the lines (`# ①`, `# ②` … or trailing
  comments) and refer to the marks. 「逐行解释」 is a code block followed by a
  numbered list ① ② … — not a table. A block that defines classes/functions
  and is followed by printed output must also show the calling code that
  produced the output.
- **Space-aligned `text` blocks → tables, series-wide**: when a post is
  touched for any reason, scan its remaining ```` ```text ```` blocks; about
  70 Infra posts still have column-aligned ones (2026-09-20). `text` stays
  only for box-drawing figures, cell-exact memory/thread maps, terminal
  output and multi-line calculations.
- Series are independent: no links to posts of other series.
- `{%`/`{{` inside code (PTX asm, printf formats, regexes, **Java / C++ nested
  array initializers like `int[][] DIRS = {{1, 0}, {-1, 0}}`**, Go/Jinja
  templates) must be wrapped in `{% raw %}` … `{% endraw %}` or the Liquid pass
  fails the build. Inside a `<div class="code-tabs" markdown="1">` put the
  `raw` pair around the offending fenced block only (`raw` cannot nest). Since
  one bad post aborts the *whole* build, grep new posts for `{{` before
  building: `rg -n '\{\{|\{%' _posts/<new>.md`.
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
    `_includes/rich-content.html`, Mermaid 11.17.2 (was 10.9.1 until 2026-09-16; all 144 diagram posts re-checked with `tools/check-render.cjs`): `~~~` invisible links to force
    row/column order, `classDef` colours, `<br/>` in quoted labels; horizontal
    layouts shrink to unreadable size at 755 px width, so favour `flowchart TB`
    and split overly tall graphs).
  - Cell-exact layouts (byte/sector maps, bank mappings, reduction trees,
    thread→address tables) → monospace ASCII in a fenced `text` block; exact
    alignment matters more than styling here and Mermaid renders them badly.
  - **A fenced `text` block whose lines are space-aligned columns is a table
    — write a Markdown table.** (Reader feedback 2026-09-15/16: 「行列式格式
    固定的 text block 直接用表格表示更清晰」.) That covers concept ↔ post
    outlines, name / formula / where-used lists, numeric lookups
    (loss ↔ PPL, V ↔ ln V), parameter-count / FLOPs sums, reading paths,
    shape ↔ meaning legends. Keep `text` only for things a table cannot hold:
    box-drawing diagrams, cell-exact memory/thread maps, terminal output,
    multi-line calculations. Numeric columns right-aligned (`---:`); formulas
    in cells as `$$…$$` with `\lVert \rVert` / `\Vert` / `\mid` — a bare `|`
    inside a cell ends the cell. Boxes drawn with `┌─┐│` that are really a
    *figure* (e.g. the matmul row-i × column-j picture) belong in an SVG
    under `img/in-post/`, not ASCII.
  - **Parenthetical glosses of a term become inline tips**, not inline
    parentheses: write `[总变差距离](# "tip: total variation distance，…")`
    instead of `总变差距离（total variation distance：…）`. Same for a term
    that is used before its own post explains it (name the post in the tip).
    Keep the parenthesis when it is part of the argument (a number, a formula
    step), not a definition. Tips must not contain `"` or, inside a table
    cell, `|` (use `∣` U+2223 for absolute values).
  - **When to use an inline tip — the one test (author, 2026-09-17): is the
    concept the *subject* of this passage, or a *bystander*?** A tip is for a
    term that merely appears in the context of what is being explained — the
    reader may not know it, but this passage is not about it (总变差距离 while
    explaining KL; 感受野 while listing CNN topics on the map). If the passage
    *is* explaining the concept — its definition, an analogy for it, why it
    works — that goes in body text, never a tip, however short it is: the
    tensor-as-flat-buffer paragraph in L0 一, the dropout / chaos-engineering
    comparison in L3 四 are body text because those sections are about
    tensors and dropout. Corollary: the same term is a tip in one post and
    body text in the post that owns it; and answers to a post's own core
    questions are `[^qN]` footnotes (jump, no popup), not tips.
  - **Key derivations get the loop next to the formula** (2026-09-17, from a
    reader-perspective review): a `∑` is an inner `for`; `2mnk`, `∂L/∂W = XᵀG`
    (why the transpose: `dW[r][j] += X[i][r]*G[i][j]` sums over the shared
    index `i`, i.e. reads a *column* of X), `softmax(QKᵀ)V` are written as
    3–8 lines of plain Python loops in L0 一 / 七, L3 一, 04 一. Do this for
    new derivations whose formula hides an index being summed over. Java
    analogies are used only when they are exact (tensor = flat buffer +
    strides, broadcasting = stride 0, backprop = error attribution down the
    call stack, dropout = chaos engineering); gradient descent ≠ PID / rate
    limiter and weight decay ≠ pool eviction — do not add those.
  - **Hardware honesty paragraph**: each map's 配套代码 section (and the labs
    README) states what runs on a laptop CPU / MPS and what truly needs an
    NVIDIA card (05 CUDA, 06 NCCL) or many cards (07, 09 verl). "千卡 / H100"
    figures in posts are accounting, not requirements — say so where it
    matters instead of a generic "no GPU needed" reassurance.
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
  - Mermaid pitfalls seen so far (10.x and 11.x alike): reserved words as node IDs (`end`,
    `call`, `click`, `style`, `class`, `default`, `graph`, `o`, `x`) break parsing —
    also as `classDef` names (`classDef graph …` killed a diagram in 2026-09);
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

