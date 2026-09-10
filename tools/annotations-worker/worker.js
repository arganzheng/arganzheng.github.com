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
 *   GET  /views?path=/slug.html                -> { views }       (optional, needs the D1 binding)
 *   POST /views        { path }                -> { views }       increments, then returns the count
 *   GET  /views/top?limit=50[&order=recent]   -> { rows: [{ path, views, updated_at }] }  for the author's dashboard
 *
 * repo / category are fixed via wrangler.toml [vars]; the worker never accepts them from the request.
 *
 * Likes and votes are NOT here: they are GitHub reactions (THUMBS_UP on the
 * post's Discussion, THUMBS_UP / THUMBS_DOWN on its comments) and the browser
 * reads them from the giscus payload and writes them via GitHub GraphQL with the
 * reader's token. Only the page-view counter needs our own storage — a D1 table
 * `views(path, count)`. The browser increments at most once per path per day
 * (localStorage), the worker only accepts paths that look like a post URL. Good
 * enough for a blog; not an analytics product.
 *
 * /issues is the one route that needs secrets. The reader's token comes from the
 * giscus GitHub App, whose only permission is Discussions: read & write, so it
 * cannot open issues. The worker therefore verifies the reader (GET /user with
 * their token) and files the issue itself *as our own GitHub App* (Issues: read
 * & write on this repo), crediting the reader in the body. App auth never
 * expires: the worker signs a 10-minute RS256 JWT with the App's private key
 * (GITHUB_APP_PRIVATE_KEY secret, PEM — PKCS#1 as downloaded from GitHub or
 * PKCS#8 both work) and trades it for a 1-hour installation token, cached in
 * the isolate. Needs GITHUB_APP_ID (var) and the App installed on the repo.
 * Without the key the route answers 501 and the client hides the checkbox.
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
      if (url.pathname === '/views' && (request.method === 'GET' || request.method === 'POST')) return await views(request, url, env, cors);
      if (url.pathname === '/views/top' && request.method === 'GET') return await viewsTop(url, env, cors);
      if (url.pathname === '/stats' && request.method === 'GET') return await stats(url, env, ctx, cors);
      if (url.pathname === '/votes' && (request.method === 'GET' || request.method === 'POST')) return await votes(request, url, env, cors);
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
  if (!env.GITHUB_APP_PRIVATE_KEY || !env.GITHUB_APP_ID) return json({ error: 'Issue 功能未启用（worker 未配置 GitHub App）' }, 501, cors);

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
  const headers = githubHeaders(await installationToken(env));
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

// ---- page views (D1) -------------------------------------------------------

const VIEW_PATH = /^\/[A-Za-z0-9_\-./]{1,200}\.html$/;
let viewsTableReady = null;

async function views(request, url, env, cors) {
  if (!env.DB) return json({ error: '阅读数未启用（worker 未绑定 D1）' }, 501, cors);
  const path = request.method === 'GET' ? url.searchParams.get('path') : (await request.json().catch(() => ({}))).path;
  if (typeof path !== 'string' || !VIEW_PATH.test(path) || path.includes('..')) return json({ error: '`path` must be a post URL' }, 400, cors);

  if (!viewsTableReady) viewsTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS views (path TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at TEXT)');
  await viewsTableReady;

  let row;
  if (request.method === 'POST') {
    row = await env.DB.prepare(
      'INSERT INTO views (path, count, updated_at) VALUES (?1, 1, ?2) ' +
      'ON CONFLICT(path) DO UPDATE SET count = count + 1, updated_at = ?2 RETURNING count'
    ).bind(path, new Date().toISOString()).first();
  } else {
    row = await env.DB.prepare('SELECT count FROM views WHERE path = ?1').bind(path).first();
  }
  return json({ views: (row && row.count) || 0 }, 200, { ...cors, 'Cache-Control': 'no-store' });
}

async function viewsTop(url, env, cors) {
  if (!env.DB) return json({ error: '阅读数未启用（worker 未绑定 D1）' }, 501, cors);
  if (!viewsTableReady) viewsTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS views (path TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at TEXT)');
  await viewsTableReady;
  const limit = Math.min(500, Math.max(1, parseInt(url.searchParams.get('limit') || '50', 10) || 50));
  const order = url.searchParams.get('order') === 'recent' ? 'updated_at DESC' : 'count DESC';
  const { results } = await env.DB.prepare(`SELECT path, count AS views, updated_at FROM views ORDER BY ${order} LIMIT ?1`).bind(limit).all();
  return json({ rows: results || [] }, 200, { ...cors, 'Cache-Control': 'public, max-age=60' });
}

// Article-level 赞同 / 反对, anonymous (no GitHub login), one row per post in D1.
// GET  /votes?path=…                      -> { up, down }
// POST /votes { path, dir, prev }         dir/prev ∈ 'up' | 'down' | null: the
//   browser remembers its own vote in localStorage and sends the transition
//   (prev -> dir); we add/subtract accordingly. Same trust level as page views.
let votesTableReady = null;
function ensureVotesTable(env) {
  if (!votesTableReady) votesTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS votes (path TEXT PRIMARY KEY, up INTEGER NOT NULL DEFAULT 0, down INTEGER NOT NULL DEFAULT 0, updated_at TEXT)');
  return votesTableReady;
}
const DIRS = new Set(['up', 'down']);
async function votes(request, url, env, cors) {
  if (!env.DB) return json({ error: '投票未启用（worker 未绑定 D1）' }, 501, cors);
  const body = request.method === 'POST' ? await request.json().catch(() => ({})) : {};
  const path = request.method === 'GET' ? url.searchParams.get('path') : body.path;
  if (typeof path !== 'string' || !VIEW_PATH.test(path) || path.includes('..')) return json({ error: '`path` must be a post URL' }, 400, cors);
  await ensureVotesTable(env);
  let row;
  if (request.method === 'POST') {
    const dir = DIRS.has(body.dir) ? body.dir : null, prev = DIRS.has(body.prev) ? body.prev : null;
    if (dir === prev) return json({ error: 'nothing to change' }, 400, cors);
    const dUp = (dir === 'up' ? 1 : 0) - (prev === 'up' ? 1 : 0);
    const dDown = (dir === 'down' ? 1 : 0) - (prev === 'down' ? 1 : 0);
    row = await env.DB.prepare(
      'INSERT INTO votes (path, up, down, updated_at) VALUES (?1, MAX(0, ?2), MAX(0, ?3), ?4) ' +
      'ON CONFLICT(path) DO UPDATE SET up = MAX(0, up + ?2), down = MAX(0, down + ?3), updated_at = ?4 RETURNING up, down'
    ).bind(path, dUp, dDown, new Date().toISOString()).first();
  } else {
    row = await env.DB.prepare('SELECT up, down FROM votes WHERE path = ?1').bind(path).first();
  }
  return json({ up: (row && row.up) || 0, down: (row && row.down) || 0 }, 200, { ...cors, 'Cache-Control': 'no-store' });
}

// GET /stats?paths=/a.html,/b.html  (<= 20) -> per-post counters for list pages:
// { items: { "/a.html": { views, comments, up, down, id, url } } }
// views and up/down from D1 (0 without the binding), comments / discussion id
// from the giscus public API (one lookup per path, first page only).
// Cached 120 s at the edge per path so a listing of 10 posts is cheap.
const STATS_MAX = 20;
async function stats(url, env, ctx, cors) {
  const paths = String(url.searchParams.get('paths') || '').split(',').map((s) => s.trim()).filter(Boolean).slice(0, STATS_MAX);
  if (!paths.length) return json({ error: '`paths` is required' }, 400, cors);
  for (const p of paths) if (!VIEW_PATH.test(p) || p.includes('..')) return json({ error: `bad path ${p}` }, 400, cors);

  let viewsByPath = {}, votesByPath = {};
  if (env.DB) {
    if (!viewsTableReady) viewsTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS views (path TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at TEXT)');
    await Promise.all([viewsTableReady, ensureVotesTable(env)]);
    const marks = paths.map((_, i) => `?${i + 1}`).join(',');
    const [v, w] = await Promise.all([
      env.DB.prepare(`SELECT path, count FROM views WHERE path IN (${marks})`).bind(...paths).all(),
      env.DB.prepare(`SELECT path, up, down FROM votes WHERE path IN (${marks})`).bind(...paths).all(),
    ]);
    for (const r of v.results || []) viewsByPath[r.path] = r.count;
    for (const r of w.results || []) votesByPath[r.path] = { up: r.up, down: r.down };
  }

  const cache = caches.default;
  const items = {};
  await Promise.all(paths.map(async (p) => {
    const key = new Request(`${url.origin}/stats/one?path=${encodeURIComponent(p)}`);
    let d = null;
    const hit = await cache.match(key);
    if (hit) d = await hit.json();
    else {
      const qs = new URLSearchParams({ repo: env.REPO, category: env.CATEGORY, term: p, first: '1' });
      const r = await fetch(`${GISCUS}/discussions?${qs}`, { headers: { Accept: 'application/json' } });
      const data = r.ok ? await r.json() : null;
      const disc = data && data.discussion;
      d = disc ? { id: disc.id, url: disc.url, comments: (disc.totalCommentCount || 0) + (disc.totalReplyCount || 0) }
               : { id: null, url: null, comments: 0 };
      if (r.ok || r.status === 404) ctx.waitUntil(cache.put(key, json(d, 200, { 'Cache-Control': 'public, s-maxage=120' })));
    }
    items[p] = { ...d, views: viewsByPath[p] || 0, up: (votesByPath[p] || {}).up || 0, down: (votesByPath[p] || {}).down || 0 };
  }));
  return json({ items }, 200, { ...cors, 'Cache-Control': 'public, max-age=60' });
}

// ---- GitHub App authentication --------------------------------------------

let cachedInstallation = null; // { token, expiresAt } — per isolate, so a warm worker reuses it

async function installationToken(env) {
  if (cachedInstallation && cachedInstallation.expiresAt - Date.now() > 60_000) return cachedInstallation.token;
  const jwt = await appJwt(env.GITHUB_APP_ID, env.GITHUB_APP_PRIVATE_KEY);
  const appHeaders = githubHeaders(jwt);

  let installationId = env.GITHUB_INSTALLATION_ID;
  if (!installationId) {
    const ir = await fetch(`${GITHUB}/repos/${env.REPO}/installation`, { headers: appHeaders });
    if (!ir.ok) throw new Error(`GitHub App 未安装到 ${env.REPO}（HTTP ${ir.status}）`);
    installationId = (await ir.json()).id;
  }
  const tr = await fetch(`${GITHUB}/app/installations/${installationId}/access_tokens`, {
    method: 'POST',
    headers: appHeaders,
    body: JSON.stringify({ permissions: { issues: 'write' } }),
  });
  const data = await tr.json();
  if (!tr.ok) throw new Error(`installation token: ${data.message || tr.status}`);
  cachedInstallation = { token: data.token, expiresAt: Date.parse(data.expires_at) };
  return data.token;
}

async function appJwt(appId, pem) {
  const now = Math.floor(Date.now() / 1000);
  const enc = (o) => b64url(new TextEncoder().encode(JSON.stringify(o)));
  const unsigned = `${enc({ alg: 'RS256', typ: 'JWT' })}.${enc({ iat: now - 60, exp: now + 540, iss: String(appId) })}`;
  const key = await crypto.subtle.importKey('pkcs8', pemToPkcs8(pem), { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['sign']);
  const sig = await crypto.subtle.sign('RSASSA-PKCS1-v1_5', key, new TextEncoder().encode(unsigned));
  return `${unsigned}.${b64url(new Uint8Array(sig))}`;
}

function b64url(bytes) {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

// GitHub hands out PKCS#1 keys ("BEGIN RSA PRIVATE KEY"); WebCrypto only imports
// PKCS#8, which is just the PKCS#1 blob wrapped in an AlgorithmIdentifier.
function pemToPkcs8(pem) {
  const body = pem.replace(/-----[^-]+-----/g, '').replace(/\s+/g, '');
  const der = Uint8Array.from(atob(body), (c) => c.charCodeAt(0));
  if (!/BEGIN RSA PRIVATE KEY/.test(pem)) return der; // already PKCS#8
  const rsaOid = [0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01, 0x05, 0x00];
  const octet = derTLV(0x04, der);
  return derTLV(0x30, concat([0x02, 0x01, 0x00], rsaOid, octet));
}

function derTLV(tag, content) {
  const len = content.length;
  let lenBytes;
  if (len < 0x80) lenBytes = [len];
  else {
    const bytes = [];
    for (let n = len; n > 0; n >>= 8) bytes.unshift(n & 0xff);
    lenBytes = [0x80 | bytes.length, ...bytes];
  }
  return concat([tag, ...lenBytes], content);
}

function concat(...parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}
