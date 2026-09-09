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
 *   POST /issues       { title, body }         -> { number, url }  (optional, see below)
 *
 * repo / category are fixed via wrangler.toml [vars]; the worker never accepts them from the request.
 *
 * /issues is the one route that needs a secret. The reader's token comes from the
 * giscus GitHub App, whose only permission is Discussions: read & write, so it
 * cannot open issues. The worker therefore verifies the reader (GET /user with
 * their token) and files the issue itself with GITHUB_TOKEN — a fine-grained PAT
 * limited to this repo with Issues: read & write — crediting the reader in the
 * body. Without GITHUB_TOKEN the route answers 501 and the client hides the
 * checkbox after the first attempt.
 */

const GISCUS = 'https://giscus.app/api';
const GITHUB = 'https://api.github.com';
const ISSUE_TITLE_MAX = 200;
const ISSUE_BODY_MAX = 20000;
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
      if (request.method === 'POST' && url.pathname === '/issues') return await createIssue(request, env, cors);
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

function githubHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'blog-annotations-worker',
    'Content-Type': 'application/json',
  };
}

async function createIssue(request, env, cors) {
  if (!env.GITHUB_TOKEN) return json({ error: 'Issue 功能未启用（worker 未配置 GITHUB_TOKEN）' }, 501, cors);

  // Who is asking? Only signed-in giscus users may file issues, and the issue
  // credits them; the GitHub App user token is enough to answer GET /user.
  const auth = request.headers.get('Authorization') || '';
  const userToken = auth.replace(/^Bearer\s+/i, '');
  if (!userToken) return json({ error: '需要登录' }, 401, cors);
  const who = await fetch(`${GITHUB}/user`, { headers: githubHeaders(userToken) });
  if (who.status === 401) return json({ error: '登录已过期，请重新登录 GitHub' }, 401, cors);
  if (!who.ok) return json({ error: `GitHub /user: HTTP ${who.status}` }, 502, cors);
  const user = await who.json();

  const { title, body } = await request.json();
  if (typeof title !== 'string' || !title.trim()) return json({ error: '`title` is required' }, 400, cors);
  if (typeof body !== 'string') return json({ error: '`body` is required' }, 400, cors);

  const label = env.ISSUE_LABEL || '划线评论';
  const repo = `${GITHUB}/repos/${env.REPO}`;
  const headers = githubHeaders(env.GITHUB_TOKEN);
  // Make sure the label exists (422 = already there).
  const lr = await fetch(`${repo}/labels`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ name: label, color: 'd1242f', description: 'Reader annotations (划线评论) flagged as problems from the blog' }),
  });
  if (!lr.ok && lr.status !== 422) return json({ error: `create label: HTTP ${lr.status}` }, 502, cors);

  const r = await fetch(`${repo}/issues`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      title: title.trim().slice(0, ISSUE_TITLE_MAX),
      body: `_由 [@${user.login}](${user.html_url}) 通过博客划线评论提出_\n\n${body.slice(0, ISSUE_BODY_MAX)}`,
      labels: [label],
    }),
  });
  const data = await r.json();
  if (!r.ok) return json({ error: data.message || `create issue: HTTP ${r.status}` }, r.status === 403 ? 502 : r.status, cors);
  return json({ number: data.number, url: data.html_url }, 201, { ...cors, 'Cache-Control': 'no-store' });
}
