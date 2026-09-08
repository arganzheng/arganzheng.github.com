/**
 * blog-annotations — a tiny, secret-free relay in front of the giscus API.
 *
 * Why: js/annotations.js needs to read a post's GitHub Discussion (to anchor
 * highlight annotations) and to exchange the reader's giscus session for a
 * GitHub token (to post annotations). giscus.app exposes both, but only with
 * CORS for its own origin. CORS is a browser rule, so this worker fetches on
 * behalf of the blog and re-serves the JSON with our origin allowed.
 *
 * Routes:
 *   GET  /discussions?term=/slug.html[&t=…]  -> all comments of that post (paginated), cached 60s; `t` bypasses cache
 *   POST /token        { session }             -> { token }   (giscus /api/oauth/token)
 *   POST /discussions  { input }               -> { id }      (giscus /api/discussions, Authorization passthrough)
 *
 * repo / category are fixed via wrangler.toml [vars]; the worker never accepts them from the request.
 */

const GISCUS = 'https://giscus.app/api';
const PAGE_SIZE = 100;
const MAX_PAGES = 5;

export default {
  async fetch(request, env, ctx) {
    const origin = request.headers.get('Origin') || '';
    const cors = corsHeaders(origin, env);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
    if (!cors) return json({ error: 'Origin not allowed' }, 403, {});

    const url = new URL(request.url);
    try {
      if (request.method === 'GET' && url.pathname === '/discussions') return await getDiscussion(url, env, ctx, cors);
      if (request.method === 'POST' && url.pathname === '/token') return await relay(`${GISCUS}/oauth/token`, request, cors);
      if (request.method === 'POST' && url.pathname === '/discussions') return await createDiscussion(request, env, cors);
    } catch (err) {
      return json({ error: err.message || String(err) }, 502, cors);
    }
    return json({ error: 'Not found' }, 404, cors);
  },
};

function corsHeaders(origin, env) {
  const allowed = String(env.ALLOWED_ORIGINS || '').split(',').map((s) => s.trim()).filter(Boolean);
  if (!allowed.includes(origin)) return null;
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
  };
}

function json(body, status, headers) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', ...headers },
  });
}

async function getDiscussion(url, env, ctx, cors) {
  const term = url.searchParams.get('term');
  if (!term) return json({ error: '`term` is required' }, 400, cors);

  const bypass = url.searchParams.has('t');
  const cacheKey = new Request(`${url.origin}/discussions?term=${encodeURIComponent(term)}`);
  const cache = caches.default;
  if (!bypass) {
    const hit = await cache.match(cacheKey);
    if (hit) {
      const res = new Response(hit.body, hit);
      Object.entries(cors).forEach(([k, v]) => res.headers.set(k, v));
      res.headers.set('X-Cache', 'HIT');
      return res;
    }
  }

  const params = { repo: env.REPO, category: env.CATEGORY, term, first: String(PAGE_SIZE) };
  let discussion = null;
  let after = '';
  for (let page = 0; page < MAX_PAGES; page++) {
    const qs = new URLSearchParams(params);
    if (after) qs.set('after', after);
    const r = await fetch(`${GISCUS}/discussions?${qs}`, { headers: { Accept: 'application/json' } });
    const data = await r.json();
    if (!r.ok) {
      // 404 "Discussion not found" is a normal state for a post nobody has commented on yet.
      const status = r.status === 404 ? 200 : r.status;
      const body = r.status === 404 ? { discussion: null } : data;
      return json(body, status, { ...cors, 'Cache-Control': 'public, max-age=30' });
    }
    if (!discussion) discussion = data.discussion;
    else discussion.comments = discussion.comments.concat(data.discussion.comments);
    const info = data.discussion && data.discussion.pageInfo;
    if (!info || !info.hasNextPage) break;
    after = info.endCursor;
  }

  const res = json({ discussion }, 200, {
    ...cors,
    'Cache-Control': 'public, max-age=30, s-maxage=60',
    'X-Cache': 'MISS',
  });
  ctx.waitUntil(cache.put(cacheKey, res.clone()));
  return res;
}

async function relay(target, request, cors) {
  const headers = { 'Content-Type': 'application/json', Accept: 'application/json' };
  const auth = request.headers.get('Authorization');
  if (auth) headers.Authorization = auth;
  const r = await fetch(target, { method: 'POST', headers, body: await request.text() });
  return new Response(r.body, {
    status: r.status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', ...cors },
  });
}

async function createDiscussion(request, env, cors) {
  // Only the discussion *input* comes from the client; repo is pinned so this
  // cannot be used to create discussions elsewhere.
  const { input } = await request.json();
  if (!input || !input.title) return json({ error: '`input.title` is required' }, 400, cors);
  const forwarded = new Request(request, { body: JSON.stringify({ repo: env.REPO, input }) });
  return relay(`${GISCUS}/discussions`, forwarded, cors);
}
