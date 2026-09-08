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

No tokens or secrets are needed: reads go through giscus' public API, and
writes use the reader's own giscus session (GitHub login) exchanged for a token.

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
