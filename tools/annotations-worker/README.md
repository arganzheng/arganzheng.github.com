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
   webhook **inactive**, repository permission **Issues: Read and write** only, "Where can this app be
   installed" = only this account.
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

## 随笔 from the phone (POST / GET / PUT / DELETE /moments)

`/moments/post.html` (layout `bare`, `js/moment-post.js`; 「添加到主屏幕」 installs
it as an app via `moments/post.webmanifest`) is the author's 发布页: text with
`#标签`, up to 9 pictures (shrunk to ≤ 1600 px / WebP in the browser), place,
quote + attribution, music URL, time. It POSTs

```json
{ "text": "…", "place": "深圳湾", "tags": ["跑步", "读书/开源"], "quote": "…", "by": "…",
  "music": "https://…", "time": "2026-10-01 20:15", "images": [{ "name": "", "type": "image/webp", "data": "<base64>" }] }
```

with the giscus reader token (`Authorization: Bearer …`, same login as the
comments). The worker accepts it only when `GET /user` is the owner of `REPO`
(like `/reactions/resolve`), then — acting as **our GitHub App**, installation
token with `contents: write` — makes **one commit on `MOMENTS_BRANCH`
(default `master`)** through the Git Data API: the pictures as
`img/moments/YYYY/MM/<YYYYMMDD-HHMM-n>.webp` plus the entry appended to
`moments/YYYY-MM.md` exactly as `tools/moment.py` writes it (`## YYYY-MM-DD
HH:MM @place`, text + `#tags`, `> quote` / `> —— by`, `![](…)`, URL). The
commit's author is the blog author, the committer the App; the push runs the
deploy workflow, so the entry is live a minute or two later. Answers
`201 { url: "/moments/YYYY-MM.html#YYYYMMDD-HHMM", commit, file, images }`;
`401` not signed in, `403` not the owner, `400` validation (empty entry, bad
tag / time / image type), `413` picture > 3 MB, `501` no App key; a ref
update that loses a race with another push is retried once. Times are
Beijing (`timezone: Asia/Shanghai`), as from the CLI.

### Optional API-key capture

For Shortcuts, Telegram bridges or scripts, set the optional `MOMENT_KEY`
secret. The worker compares SHA-256 digests and never logs the key. This
credential is accepted only by `POST /moments`; edits, deletes, pinning and tag
renames still require the owner's GitHub login.

```bash
cd tools/annotations-worker
wrangler secret put MOMENT_KEY
```

JSON requests can include base64 pictures:

```bash
curl -X POST "$API/moments" \
  -H "Authorization: Bearer $MOMENT_KEY" \
  -H 'Content-Type: application/json' \
  --data '{"text":"散步时看到海了","time":"2026-10-01 20:15","tags":["散步"],"images":[{"name":"sea.jpg","type":"image/jpeg","data":"<base64>"}]}'
```

For a text-only request, `text/plain` is the body; the worker supplies the
current Beijing time. A JSON request can omit `time` for the same default:

```bash
printf '%s' '散步时看到海了 #散步' | curl -X POST "$API/moments" \
  -H "Authorization: Bearer $MOMENT_KEY" -H 'Content-Type: text/plain' --data-binary @-
```

Form uploads can send repeated `image` fields:

```bash
curl -X POST "$API/moments" \
  -H "Authorization: Bearer $MOMENT_KEY" \
  -F 'text=散步时看到海了 #散步' -F image=@sea.jpg -F image=@sky.jpg
```

iOS 快捷指令：在「获取 URL 内容」中设置 URL = `$API/moments`、方法 = POST，
请求头添加 `Authorization: Bearer <MOMENT_KEY>`；请求体选择 **表单**，
添加 `text`（文本 = 提供的输入）和 `image`（文件 =
转换后的照片，可重复添加多张）。先依次「选择照片」→「调整图像大小」
（宽 1600）→「转换图像」（JPEG）；iPhone HEIC 不受支持，原图也可能超过
3 MB。不要把密钥放在日志、截图或共享的快捷指令中。

Editing and deleting (the 编辑 / 删除 links a signed-in author sees on every card
of a month page, `js/moments.js`; the 发布页 opens as `/moments/post.html?edit=YYYY-MM/<id>`):

- `GET /moments?month=2026-09&id=20260921-0802` → `{ month, id, time, place, text, quote, by, music, images: ["/img/moments/…"], raw }`
  — the entry parsed back into the page's fields (ids as `_plugins/moments.rb`
  assigns them, `-2` for a second entry at the same minute). `raw: true` when the
  body is not in the canonical text → quote → pictures → URL order (then `text`
  is the whole Markdown body).
- `PUT /moments { month, id, …the POST fields, images: [{ url } | { type, data }] }`
  rewrites that entry's block in place (`{ url }` keeps a picture already in the
  repo, in the new order; a new picture uploads as with POST). A changed date
  that lands in another month moves the entry to that month's file in the same
  commit. Pictures the entry no longer shows — and nothing else in the file
  does — are deleted from `img/moments/`. Answers `200 { url, commit, … }`.
- `DELETE /moments { month, id }` removes the block and its pictures the same
  way → `200 { commit, month, file }`.

- `POST /moments/pin { id, pinned }` rewrites `_data/moments.yml` in one
  commit. The generated `/moments/` front door shows configured ids in a
  separate pinned block; their month stream remains unchanged.
- `POST /moments/tags { from, to }` validates both tag names and rewrites every
  matching `moments/*.md` entry in one commit. Children such as `#from/child`
  follow the rename; code and URL text is left alone. An existing `to` tag is
  merged. The response is `{ changed, files }`; no matches returns 404.

GET/PUT/DELETE still require the owner's GitHub login; `404` when the id is
not in the file. Pin and tag-management routes also require that login.

The publisher's `/moments/sw.js` is registered only by `post.html`. It caches
the publisher shell for offline opening, handles Android image/text shares,
and stores processed picture drafts and network-failed posts in IndexedDB.
Queued posts retry in order when online; pending cards appear locally until
the site deploys.

Setup on top of the `/issues` App: give the App repository permission
**Contents: Read and write** (App settings → Permissions & events → save,
then accept the new permission under Settings → Applications → Installed
GitHub Apps), and `wrangler deploy`. Until then the route answers
`installation token … 缺少该权限` (502).

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
与前文矛盾) bumps that key in the row's `reasons` JSON object; `prev === reason`
un-counts it instead (the browser lets a reader pick several reasons, one POST
per toggle; `prev` ≠ `reason` still switches in one call). Every POST may
carry `section` (nearest heading above the passage, ≤ 120 chars) which is stored
once per row (`COALESCE`). Both come back in every reactions response.

**Section-level reactions** (the ♡ under a 随笔 entry; until 2026-09-30 also
点赞 / 没看懂 on every h2–h6, whose rows remain) reuse the same route and table:
the browser posts `kind: 'up' | 'doubt'` with `quote = '§ ' + <section title>`
and `section` = the title. The `§ ` prefix is how readers of the table (dashboard, brief,
`js/annotations.js`) tell a chapter row from a passage row.

**The author answers a 存疑** — `POST /reactions/resolve {path, hash, action}`
with the reader's giscus token in `Authorization`; the worker calls `GET /user`
and only accepts the owner of `REPO` (403 otherwise; no extra secret).
`action: 'resolve'` sets `resolved_at = now, resolved_doubt = doubt` — the count
is kept, the browser shows the passage green (✓ 作者已修正) and counts only
doubts above `resolved_doubt` as open, so a doubt raised after the fix turns it
red again with the delta; `'reopen'` clears the stamp; `'clear'` zeroes `doubt`
and `reasons`. Answers the row like every other reactions response
(`up, doubt, share, reasons, resolved_at, resolved_doubt`); 404 when the row
does not exist. Both columns are `ALTER TABLE`d in on first use.

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
public API `comments` (comments + replies, **minus the author's own top-level
comments and their replies** — those are the author's private working notes
for the AI revision loop; `js/annotations.js` hides them from everyone but the
logged-in author, and the badge must not count what the page will not show),
`id` and `url` of the discussion
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
