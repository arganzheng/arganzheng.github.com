# blog-annotations worker

Cloudflare Worker that relays the giscus API for `js/annotations.js` (the post
comment section and the highlight comments) and keeps the page-view counter.
See `worker.js` header for the routes. Likes and votes need nothing here: they
are GitHub reactions written by the browser with the reader's token.

## Deploy (once, free tier)

```bash
npm i -g wrangler
wrangler login                      # opens the browser
cd tools/annotations-worker
wrangler deploy                     # prints https://blog-annotations.<subdomain>.workers.dev
```

Put the printed URL into `_config.yml`:

```yaml
annotations:
  api: https://blog-annotations.<subdomain>.workers.dev
```

No tokens or secrets are needed for reading and commenting: reads go through
giscus' public API, and writes use the reader's own giscus session (GitHub
login) exchanged for a token.

## Optional: 「同时提交 Issue」 (POST /issues)

The editor has a checkbox that also files a GitHub Issue for the note, so
problems readers flag in the text show up in the repo's issue tracker (label
`划线评论`, configurable via `ISSUE_LABEL`). The reader's giscus token cannot do
this — the giscus GitHub App only has the *Discussions* permission — so the
worker files the issue itself, acting as **our own GitHub App**, and credits
the reader in the body. App credentials never expire (the worker mints a
10-minute JWT and trades it for a 1-hour installation token as needed), so
this is a one-off setup:

1. GitHub → Settings → Developer settings → GitHub Apps → New GitHub App:
   any name (issues will appear as `<name>[bot]`), homepage = the blog,
   webhook **inactive**, repository permission **Issues: Read and write**
   only, "Where can this app be installed" = only this account.
2. On the App page: note the **App ID**; **Generate a private key** — a
   `*.private-key.pem` downloads (PKCS#1, `BEGIN RSA PRIVATE KEY`; the worker
   accepts PKCS#8 too).
3. **Install App** → your account → only `arganzheng.github.com`.
4. Put the App ID in `wrangler.toml` (`GITHUB_APP_ID = "123456"`), then
   ```bash
   cd tools/annotations-worker
   wrangler secret put GITHUB_APP_PRIVATE_KEY < ~/Downloads/<app>.private-key.pem
   wrangler deploy
   ```
   Delete the downloaded `.pem` afterwards; it is the only copy that matters.
5. Smoke test (needs a giscus login in the browser — the request is refused
   without a valid reader token): tick 「同时提交 Issue」 on any post and submit.
   A 501 means the key/App ID is missing; "GitHub App 未安装到 …" means step 3.

Without the key the route answers 501 and the client simply reports that the
feature is off. The worker only accepts the request when the
`Authorization: Bearer <reader token>` header resolves via `GET /user`, i.e.
from readers signed in through giscus.

## Optional: page views (GET/POST /views)

One row per post in a Cloudflare D1 database (free tier is plenty: the browser
increments at most once per post per day per browser). One-off setup:

```bash
cd tools/annotations-worker
wrangler d1 create blog-views       # prints database_id
```

Paste the id into the `[[d1_databases]]` block of `wrangler.toml` (replace
`REPLACE_WITH_DATABASE_ID`), then `wrangler deploy`. The table is created on
first use, no migration to run. Without the binding the route answers 501 and
the client hides the counter. Smoke test:

```bash
curl -H 'Origin: http://localhost:4000' 'https://blog-annotations.<subdomain>.workers.dev/views?path=/a-letter-to-readers.html'
```

`GET /views/top?limit=50&order=count|recent` feeds the author dashboard.

## Article votes (GET/POST /votes)

Anonymous 「有用」 on a post, same trust model as views: one D1 row per post
(`votes(path, up, down)` — the UI only uses `up` today), the browser keeps its
own choice in `localStorage["vote:<path>"]` and sends the transition
(`POST {path, dir, prev}` with `up` / `null`), `GET /votes?path=` reads the
counts (plus `shares`, below). No GitHub login involved (comment votes are
still GitHub reactions). Needs the D1 binding; 501 without it.

## Share counter (POST /shares)

`POST /shares { path }` adds one to `shares(path, count)` and returns
`{ shares }`. `js/share.js` calls it whenever a reader actually uses the share
menu (system share sheet completed, Weibo / X / LinkedIn opened, WeChat QR
shown, link copied); localhost previews don't count. Same D1 binding.

## Passage 赞 / 存疑 / 分享 (GET/POST /reactions)

Anonymous per-passage reactions, same trust model. One row per
`(path, hash)` in `passage_reactions(path, hash, quote, up, doubt, share)` —
`hash` is the FNV-1a id `js/annotations.js` already uses for `#annot-<hash>`
links, `quote` the exact text (≤ 600 chars) so the browser can re-anchor and
underline a passage that has reactions but no comment. `POST {path, hash,
quote, kind: 'up'|'doubt', on: true|false}` toggles one reader's reaction (the
browser remembers its own in `localStorage["react:<path>:<hash>"]`); `kind:
'share'` is a plain +1 (no toggle) and also bumps the article's `shares` row
(the response carries `shares`). `GET /reactions?path=` lists the post's
passages with any count. The `share`, `reasons` and `section` columns are added
to existing tables by `ALTER TABLE` on first use.

**Why a passage is doubted** — `kind: 'reason'` with `reason` one of `wrong |
unclear | outdated | example | conflict` (有错误 / 没看懂 / 版本过时 / 缺例子 /
与前文矛盾) bumps that key in the row's `reasons` JSON object; pass `prev` to
switch (un-counts the old pick) and `prev === reason` to clear. Every POST may
carry `section` (nearest heading above the passage, ≤ 120 chars) which is stored
once per row (`COALESCE`). Both come back in every reactions response.

**Chapter-level 有用 / 没看懂** reuse the same route and table: the browser
posts `kind: 'up' | 'doubt'` with `quote = '§ ' + <heading text>` (h2–h6) and `section` =
the heading. The `§ ` prefix is how readers of the table (dashboard, brief,
`js/annotations.js`) tell a chapter row from a passage row.

## Dashboard reads (/stats/top, /views/daily, /reactions/top, /feedback)

`GET /stats/top?limit=100` joins views / votes / shares per post; `GET
/views/daily?days=30` returns per-day totals (Beijing dates, from the
`views_daily(path, day, count)` table that every `POST /views` also writes) plus
the posts read most in the window; `GET /reactions/top?kind=doubt|up|share&limit=50`
lists the most doubted / liked / shared passages. All public, cached 1–5 min.
`GET /feedback?path=` returns everything D1 holds about one post — its
`passage_reactions` rows (with `reasons` / `section`), views, 有用 and shares —
for the dashboard's 修订简报 (the Discussion and Issues are fetched by the
browser from GitHub). Without `path` it returns every post at once —
`{ posts: { "/slug.html": { reactions, views, up, shares } } }` — for the weekly
`feedback-queue` GitHub Action (`tools/feedback-queue.cjs`), which sends
`Origin: <site url>` to pass the CORS allow-list. Not cached.

## List-page counters (GET /stats)

`GET /stats?paths=/a.html,/b.html` (up to 20) returns, per path, `views`,
`up` / `down` and `shares` (D1, 0 without the binding), and from the giscus
public API `comments` (comments + replies), `id` and `url` of the discussion
(`null` when nobody has commented yet). Each path is cached 120 s at the edge,
so the home page (10 posts) costs at most 10 giscus lookups every two minutes.
Used by `js/share.js` for the 阅读 / 有用 / 评论 / 分享 badges in every list
entry's meta line.

## Local development

```bash
wrangler dev                        # http://localhost:8787
```

Then in the browser console on http://localhost:4000:

```js
localStorage.annotationsApi = 'http://localhost:8787'
```

Smoke test:

```bash
curl -H 'Origin: http://localhost:4000' 'http://localhost:8787/discussions?term=/a-letter-to-readers.html'
```
