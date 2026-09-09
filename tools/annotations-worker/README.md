# blog-annotations worker

Cloudflare Worker that relays the giscus API for `js/annotations.js` (the post
comment section and the highlight annotations). See `worker.js` header for the routes.

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
curl -H 'Origin: http://localhost:4000' 'http://localhost:8787/discussions?term=/popup-footnotes-and-inline-tips-demo.html'
```
