# blog-annotations worker

Secret-free Cloudflare Worker that relays the giscus API for `js/annotations.js`
(highlight annotations on blog posts). See `worker.js` header for the routes.

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
worker files the issue itself and credits the reader in the body. That needs
one secret:

1. GitHub → Settings → Developer settings → Fine-grained tokens → generate:
   resource owner = you, repository access = only `arganzheng.github.com`,
   repository permissions = **Issues: Read and write** (nothing else). Pick an
   expiry you are happy to renew.
2. `cd tools/annotations-worker && wrangler secret put GITHUB_TOKEN` (paste it).
3. `wrangler deploy`.

Without the secret the route answers 501 and the client simply reports that
the feature is off. The worker only accepts the request when the
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
