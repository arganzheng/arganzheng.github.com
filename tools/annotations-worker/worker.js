import { rewriteMomentTags } from './moment-tags.mjs';

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
 *   GET  /votes?path=/slug.html                -> { up, down, shares }   anonymous article 「有用」 (D1)
 *   POST /votes        { path, dir, prev }     -> { up, down, shares }
 *   POST /shares       { path }                -> { shares }    one more share (any channel of the share menu)
 *   GET  /stats?paths=/a.html,/b.html          -> { items: { path: { views, comments, up, down, shares, id, url } } }
 *   GET  /reactions?path=/slug.html            -> { items: [{ hash, quote, section, up, doubt, share, reasons, resolved_at, resolved_doubt }] }   passage-level 赞 / 存疑 (anonymous)
 *   POST /reactions    { path, hash, quote, kind, on, section?, reason?, prev? } -> the same row
 *                                                 kind up | doubt (toggle) | share (+1) | reason (why 存疑)
 *   POST /reactions/resolve { path, hash, action: resolve | reopen | clear }  author only (Authorization = giscus token, GET /user must be REPO's owner)
 *   dashboard: GET /stats/top, /views/daily?days=30, /reactions/top?kind=doubt|up, /feedback?path= (修订简报)
 *   GET /feedback (no path) -> { posts: { path: { reactions, views, up, shares } } }  every post, for the weekly 待修订 Action
 *   POST /moments      { text, place, tags, quote, by, music, time, images: […] | video: { src, poster? } } -> { url, commit }
 *                                                 author only (same check as /reactions/resolve); one commit on the
 *                                                 default branch that appends the entry to moments/YYYY-MM.md and adds
 *                                                 pictures/posters under img/moments/YYYY/MM/ (same format as tools/moment.py).
 *                                                 Needs the GitHub App with *Contents: read & write* (501 without the key).
 *   POST /moments/media  raw video bytes       -> { src }       author/API-key only; Content-Length required; streams ≤ 50 MB to R2
 *   GET    /moments?month=YYYY-MM&id=…   -> the entry's fields (text, place, tags in text, quote, by, music, images, video, raw)
 *   PUT    /moments    { month, id, …POST fields, images: [{ url } | { type, data }], video? } -> { url, commit } rewrites the entry
 *   DELETE /moments    { month, id }           -> { commit }    removes the entry, pictures/poster, and R2 video
 *
 * repo / category are fixed via wrangler.toml [vars]; the worker never accepts them from the request.
 *
 * Comment votes are NOT here: they are GitHub reactions (THUMBS_UP / THUMBS_DOWN
 * on a comment) and the browser reads them from the giscus payload and writes
 * them via GitHub GraphQL with the reader's token. The anonymous per-article
 * counters — page views, 「有用」 and shares — live in three D1 tables
 * (`views`, `votes`, `shares`, one row per path). The browser rate-limits itself
 * (views once per day, one 「有用」 per browser, both in localStorage), the worker
 * only accepts paths that look like a post URL. Good enough for a blog; not an
 * analytics product.
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
    const url = new URL(request.url);
    const origin = request.headers.get('Origin') || '';
    let cors = corsHeaders(origin, env);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
    if (!cors) {
      if (request.method === 'POST' && ['/moments', '/moments/media'].includes(url.pathname) && await hasMomentKey(request, env)) cors = {};
      else return json({ error: 'Origin not allowed' }, 403, {});
    }

    try {
      if (request.method === 'GET' && url.pathname === '/discussions') return await getDiscussion(url, env, ctx, cors);
      if (request.method === 'POST' && url.pathname === '/token') return await relay(`${GISCUS}/oauth/token`, request, cors);
      if (request.method === 'POST' && url.pathname === '/discussions') return await createDiscussion(request, env, cors);
      if (request.method === 'POST' && url.pathname === '/issues') return await createIssue(request, env, cors);
      if (url.pathname === '/views' && (request.method === 'GET' || request.method === 'POST')) return await views(request, url, env, cors);
      if (url.pathname === '/views/top' && request.method === 'GET') return await viewsTop(url, env, cors);
      if (url.pathname === '/views/daily' && request.method === 'GET') return await viewsDaily(url, env, cors);
      if (url.pathname === '/stats' && request.method === 'GET') return await stats(url, env, ctx, cors);
      if (url.pathname === '/stats/top' && request.method === 'GET') return await statsTop(url, env, cors);
      if (url.pathname === '/votes' && (request.method === 'GET' || request.method === 'POST')) return await votes(request, url, env, cors);
      if (url.pathname === '/shares' && request.method === 'POST') return await shares(request, env, cors);
      if (url.pathname === '/reactions' && (request.method === 'GET' || request.method === 'POST')) return await reactions(request, url, env, cors);
      if (url.pathname === '/reactions/top' && request.method === 'GET') return await reactionsTop(url, env, cors);
      if (url.pathname === '/reactions/resolve' && request.method === 'POST') return await resolveReaction(request, env, cors);
      if (url.pathname === '/feedback' && request.method === 'GET') return await feedback(url, env, cors);
      if (url.pathname === '/moments' && request.method === 'POST') return await publishMoment(request, env, cors);
      if (url.pathname === '/moments/media' && request.method === 'POST') return await uploadMomentMedia(request, env, cors);
      if (url.pathname === '/moments/pin' && request.method === 'POST') return await pinMoment(request, env, cors);
      if (url.pathname === '/moments/tags' && request.method === 'POST') return await renameMomentTag(request, env, cors);
      if (url.pathname === '/moments' && request.method === 'GET') return await readMoment(url, request, env, cors);
      if (url.pathname === '/moments' && request.method === 'PUT') return await editMoment(request, env, cors);
      if (url.pathname === '/moments' && request.method === 'DELETE') return await deleteMoment(request, env, cors);
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
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
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

  const got = await fetchDiscussion(env, term);
  if (got.error) return json(got.error, got.status, { ...cors, 'Cache-Control': 'public, max-age=30' });
  const discussion = got.discussion;

  const res = json({ discussion }, 200, {
    ...cors,
    'Cache-Control': 'public, max-age=30, s-maxage=60',
    'X-Cache': 'MISS',
  });
  ctx.waitUntil(cache.put(cacheKey, res.clone()));
  return res;
}

// The whole discussion of one post from the giscus public API (all pages of
// top-level comments concatenated). Resolves to { discussion } (null when
// nobody has commented yet — giscus answers 404 for that, a normal state) or
// { error, status } for a real failure.
async function fetchDiscussion(env, term) {
  const params = { repo: env.REPO, category: env.CATEGORY, term, first: String(PAGE_SIZE) };
  let discussion = null;
  let after = '';
  for (let page = 0; page < MAX_PAGES; page++) {
    const qs = new URLSearchParams(params);
    if (after) qs.set('after', after);
    const r = await fetch(`${GISCUS}/discussions?${qs}`, { headers: { Accept: 'application/json' } });
    const data = await r.json();
    if (!r.ok) {
      if (r.status === 404) return { discussion: null };
      return { error: data, status: r.status };
    }
    if (!discussion) discussion = data.discussion;
    else discussion.comments = discussion.comments.concat(data.discussion.comments);
    const info = data.discussion && data.discussion.pageInfo;
    if (!info || !info.hasNextPage) break;
    after = info.endCursor;
  }
  return { discussion };
}

// What readers get to see: the author's own top-level comments are private
// working notes (js/annotations.js hides them unless the author is logged
// in), so they and their replies do not count.
function publicCommentCount(disc) {
  const list = (disc && disc.comments) || [];
  let n = 0;
  for (const c of list) {
    if (c.authorAssociation === 'OWNER' || c.isMinimized) continue;
    n += 1 + (c.replyCount || (c.replies && c.replies.length) || 0);
  }
  return n;
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
    await ensureDailyTable(env);
    const now = new Date().toISOString();
    const rs = await env.DB.batch([
      env.DB.prepare(
        'INSERT INTO views (path, count, updated_at) VALUES (?1, 1, ?2) ' +
        'ON CONFLICT(path) DO UPDATE SET count = count + 1, updated_at = ?2 RETURNING count'
      ).bind(path, now),
      env.DB.prepare(
        'INSERT INTO views_daily (path, day, count) VALUES (?1, ?2, 1) ' +
        'ON CONFLICT(path, day) DO UPDATE SET count = count + 1'
      ).bind(path, beijingDay()),
    ]);
    row = rs[0].results && rs[0].results[0];
  } else {
    row = await env.DB.prepare('SELECT count FROM views WHERE path = ?1').bind(path).first();
  }
  return json({ views: (row && row.count) || 0 }, 200, { ...cors, 'Cache-Control': 'no-store' });
}

// Per-day counts (Beijing dates, the blog's timezone) for the dashboard's trend
// view: `views_daily(path, day, count)`, written alongside `views` on every POST.
// GET /views/daily?days=30 -> { days: [{ day, views }], paths: [{ path, views }] }
// (totals per day, and the posts read most in that window).
let dailyTableReady = null;
function ensureDailyTable(env) {
  if (!dailyTableReady) dailyTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS views_daily (path TEXT NOT NULL, day TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (path, day))');
  return dailyTableReady;
}
function beijingDay(offsetDays = 0) {
  return new Date(Date.now() + 8 * 3600e3 - offsetDays * 86400e3).toISOString().slice(0, 10);
}
async function viewsDaily(url, env, cors) {
  if (!env.DB) return json({ error: '阅读数未启用（worker 未绑定 D1）' }, 501, cors);
  await ensureDailyTable(env);
  const days = Math.min(365, Math.max(1, parseInt(url.searchParams.get('days') || '30', 10) || 30));
  const since = beijingDay(days - 1);
  const [d, p] = await env.DB.batch([
    env.DB.prepare('SELECT day, SUM(count) AS views FROM views_daily WHERE day >= ?1 GROUP BY day ORDER BY day').bind(since),
    env.DB.prepare('SELECT path, SUM(count) AS views FROM views_daily WHERE day >= ?1 GROUP BY path ORDER BY views DESC LIMIT 30').bind(since),
  ]);
  return json({ since, days: d.results || [], paths: p.results || [] }, 200, { ...cors, 'Cache-Control': 'public, max-age=300' });
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
  await Promise.all([ensureVotesTable(env), ensureSharesTable(env)]);
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
  const sh = await env.DB.prepare('SELECT count FROM shares WHERE path = ?1').bind(path).first();
  return json({ up: (row && row.up) || 0, down: (row && row.down) || 0, shares: (sh && sh.count) || 0 }, 200, { ...cors, 'Cache-Control': 'no-store' });
}

// Dashboard: every post's counters in one table.
// GET /stats/top?limit=100 -> { rows: [{ path, views, up, shares, updated_at }] }
async function statsTop(url, env, cors) {
  if (!env.DB) return json({ error: '未启用（worker 未绑定 D1）' }, 501, cors);
  if (!viewsTableReady) viewsTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS views (path TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at TEXT)');
  await Promise.all([viewsTableReady, ensureVotesTable(env), ensureSharesTable(env)]);
  const limit = Math.min(500, Math.max(1, parseInt(url.searchParams.get('limit') || '100', 10) || 100));
  const { results } = await env.DB.prepare(
    'SELECT v.path, v.count AS views, v.updated_at, COALESCE(o.up, 0) AS up, COALESCE(s.count, 0) AS shares ' +
    'FROM views v LEFT JOIN votes o ON o.path = v.path LEFT JOIN shares s ON s.path = v.path ORDER BY v.count DESC LIMIT ?1'
  ).bind(limit).all();
  return json({ rows: results || [] }, 200, { ...cors, 'Cache-Control': 'public, max-age=60' });
}

// Passage-level 赞 / 存疑, anonymous like the article counters. One row per
// (post, passage): `hash` is the FNV-1a id js/annotations.js already uses for
// #annot-<hash> links, `quote` the exact text so the browser can draw the
// underline on a passage nobody has commented on (it re-anchors the quote).
// GET  /reactions?path=…                          -> { items: [{ hash, quote, section, up, doubt, share, reasons }] }
// POST /reactions { path, hash, quote, kind, on, section?, reason? }
//                                                  kind 'up' | 'doubt' (on true/false = toggle)
//                                                  or 'share' (always +1, also bumps the
//                                                  article's `shares` row) -> { up, doubt, share, reasons, shares? }
//                                                  kind 'reason' { reason, prev? }: the 存疑 reader picks *why*
//                                                  (one of DOUBT_REASONS; `prev` is un-counted when switching)
// GET  /reactions/top?kind=doubt|up|share&limit=50 -> { rows: [{ path, hash, quote, section, up, doubt, share, reasons, updated_at }] }
// GET  /feedback?path=…                           -> { reactions: [rows], views, up, shares } for the dashboard's 修订简报
// POST /reactions/resolve { path, hash, action }   author only (Authorization: the giscus GitHub token;
//                                                  GET /user must be the owner of REPO). action
//                                                  'resolve' stamps resolved_at + resolved_doubt = doubt
//                                                  (the 存疑 stays in the row, the page shows ✓ until
//                                                  *new* doubts arrive: doubt > resolved_doubt);
//                                                  'reopen' clears the stamp; 'clear' zeroes doubt +
//                                                  reasons (history gone). -> the row as GET returns it
//
// `section` is the nearest heading above the passage (browser-supplied, ≤ 120
// chars) and `reasons` a JSON object of reason -> count, e.g. {"unclear":3}.
const HASH = /^[0-9a-f]{8}$/;
const QUOTE_MAX = 600;
const SECTION_MAX = 120;
const REACTION_KINDS = ['up', 'doubt', 'share', 'reason'];
const DOUBT_REASONS = ['wrong', 'unclear', 'outdated', 'example', 'conflict'];
const RESOLVE_ACTIONS = ['resolve', 'reopen', 'clear'];
const ROW_COLS = 'up, doubt, share, reasons, resolved_at, resolved_doubt';
let reactionsTableReady = null;
function ensureReactionsTable(env) {
  if (!reactionsTableReady) {
    reactionsTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS passage_reactions (path TEXT NOT NULL, hash TEXT NOT NULL, quote TEXT NOT NULL, up INTEGER NOT NULL DEFAULT 0, doubt INTEGER NOT NULL DEFAULT 0, share INTEGER NOT NULL DEFAULT 0, reasons TEXT, section TEXT, updated_at TEXT, resolved_at TEXT, resolved_doubt INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (path, hash))')
      // columns added later: migrate tables created without them (D1 has no ADD COLUMN IF NOT EXISTS)
      .then(() => env.DB.exec('ALTER TABLE passage_reactions ADD COLUMN share INTEGER NOT NULL DEFAULT 0').catch(() => {}))
      .then(() => env.DB.exec('ALTER TABLE passage_reactions ADD COLUMN reasons TEXT').catch(() => {}))
      .then(() => env.DB.exec('ALTER TABLE passage_reactions ADD COLUMN section TEXT').catch(() => {}))
      .then(() => env.DB.exec('ALTER TABLE passage_reactions ADD COLUMN resolved_at TEXT').catch(() => {}))
      .then(() => env.DB.exec('ALTER TABLE passage_reactions ADD COLUMN resolved_doubt INTEGER NOT NULL DEFAULT 0').catch(() => {}));
  }
  return reactionsTableReady;
}
function parseReasons(s) {
  try { const o = JSON.parse(s || '{}'); return o && typeof o === 'object' ? o : {}; } catch (e) { return {}; }
}
function withReasons(row) {
  return { ...row, reasons: parseReasons(row.reasons) };
}
function rowOut(row) {
  return { up: (row && row.up) || 0, doubt: (row && row.doubt) || 0, share: (row && row.share) || 0, reasons: parseReasons(row && row.reasons), resolved_at: (row && row.resolved_at) || null, resolved_doubt: (row && row.resolved_doubt) || 0 };
}
async function reactions(request, url, env, cors) {
  if (!env.DB) return json({ error: '段落点赞未启用（worker 未绑定 D1）' }, 501, cors);
  await ensureReactionsTable(env);
  if (request.method === 'GET') {
    const path = url.searchParams.get('path');
    if (typeof path !== 'string' || !VIEW_PATH.test(path) || path.includes('..')) return json({ error: '`path` must be a post URL' }, 400, cors);
    const { results } = await env.DB.prepare(`SELECT hash, quote, section, ${ROW_COLS} FROM passage_reactions WHERE path = ?1 AND (up > 0 OR doubt > 0 OR share > 0)`).bind(path).all();
    return json({ items: (results || []).map(withReasons) }, 200, { ...cors, 'Cache-Control': 'no-store' });
  }
  const b = await request.json().catch(() => ({}));
  if (typeof b.path !== 'string' || !VIEW_PATH.test(b.path) || b.path.includes('..')) return json({ error: '`path` must be a post URL' }, 400, cors);
  if (typeof b.hash !== 'string' || !HASH.test(b.hash)) return json({ error: '`hash` must be 8 hex chars' }, 400, cors);
  if (!REACTION_KINDS.includes(b.kind)) return json({ error: '`kind` must be up | doubt | share | reason' }, 400, cors);
  const quote = String(b.quote || '').replace(/\s+/g, ' ').trim().slice(0, QUOTE_MAX);
  if (!quote) return json({ error: '`quote` is required' }, 400, cors);
  const section = String(b.section || '').replace(/\s+/g, ' ').trim().slice(0, SECTION_MAX) || null;
  const now = new Date().toISOString();
  let row;
  if (b.kind === 'reason') {
    if (!DOUBT_REASONS.includes(b.reason)) return json({ error: '`reason` must be one of ' + DOUBT_REASONS.join(' | ') }, 400, cors);
    const prev = DOUBT_REASONS.includes(b.prev) ? b.prev : null;
    const cur = await env.DB.prepare('SELECT reasons FROM passage_reactions WHERE path = ?1 AND hash = ?2').bind(b.path, b.hash).first();
    const reasons = parseReasons(cur && cur.reasons);
    if (prev && reasons[prev] > 0) { reasons[prev] -= 1; if (!reasons[prev]) delete reasons[prev]; }
    if (prev !== b.reason) reasons[b.reason] = (reasons[b.reason] || 0) + 1;
    row = await env.DB.prepare(
      'INSERT INTO passage_reactions (path, hash, quote, section, reasons, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6) ' +
      `ON CONFLICT(path, hash) DO UPDATE SET reasons = ?5, section = COALESCE(section, ?4), updated_at = ?6 RETURNING ${ROW_COLS}`
    ).bind(b.path, b.hash, quote, section, JSON.stringify(reasons), now).first();
  } else {
    const col = b.kind, delta = col === 'share' ? 1 : (b.on === false ? -1 : 1);
    row = await env.DB.prepare(
      `INSERT INTO passage_reactions (path, hash, quote, section, ${col}, updated_at) VALUES (?1, ?2, ?3, ?4, MAX(0, ?5), ?6) ` +
      `ON CONFLICT(path, hash) DO UPDATE SET ${col} = MAX(0, ${col} + ?5), section = COALESCE(section, ?4), updated_at = ?6 RETURNING ${ROW_COLS}`
    ).bind(b.path, b.hash, quote, section, delta, now).first();
  }
  const out = rowOut(row);
  if (b.kind === 'share') out.shares = await bumpShares(env, b.path, now); // a passage share is an article share too
  return json(out, 200, { ...cors, 'Cache-Control': 'no-store' });
}
// The author answers an anonymous 存疑 (see the route comment above). The
// giscus token only proves *who* asks; the owner of REPO is the one allowed.
// The blog's author = the owner of REPO, signed in through giscus like any
// reader: `Authorization: Bearer <reader token>` must answer GET /user with
// that login. Returns { user } or { error: Response }.
async function requireAuthor(request, env, cors, what) {
  const userToken = (request.headers.get('Authorization') || '').replace(/^Bearer\s+/i, '');
  if (!userToken) return { error: json({ error: '需要登录' }, 401, cors) };
  const who = await fetch(`${GITHUB}/user`, { headers: githubHeaders(userToken) });
  if (who.status === 401) return { error: json({ error: '登录已过期，请重新登录 GitHub' }, 401, cors) };
  if (!who.ok) return { error: json({ error: `GitHub /user: HTTP ${who.status}` }, 502, cors) };
  const user = await who.json();
  const owner = String(env.REPO || '').split('/')[0].toLowerCase();
  if (!owner || String(user.login || '').toLowerCase() !== owner) return { error: json({ error: `只有博客作者可以${what}` }, 403, cors) };
  return { user };
}

async function requireMomentPoster(request, env, cors, what) {
  if (await hasMomentKey(request, env)) {
    const login = String(env.REPO || '').split('/')[0];
    return { user: { login, name: login, email: `${login}@users.noreply.github.com`, via: 'api' } };
  }
  return requireAuthor(request, env, cors, what);
}

async function hasMomentKey(request, env) {
  if (!env.MOMENT_KEY) return false;
  const match = (request.headers.get('Authorization') || '').match(/^Bearer\s+(.+)$/i);
  return !!match && await sameSecret(match[1], env.MOMENT_KEY);
}

async function sameSecret(a, b) {
  const encoder = new TextEncoder();
  const [left, right] = await Promise.all([
    crypto.subtle.digest('SHA-256', encoder.encode(a)),
    crypto.subtle.digest('SHA-256', encoder.encode(b)),
  ]);
  const x = new Uint8Array(left), y = new Uint8Array(right);
  let difference = 0;
  for (let i = 0; i < x.length; i++) difference |= x[i] ^ y[i];
  return difference === 0;
}

async function resolveReaction(request, env, cors) {
  if (!env.DB) return json({ error: '段落点赞未启用（worker 未绑定 D1）' }, 501, cors);
  const author = await requireAuthor(request, env, cors, '处理存疑');
  if (author.error) return author.error;

  const b = await request.json().catch(() => ({}));
  if (typeof b.path !== 'string' || !VIEW_PATH.test(b.path) || b.path.includes('..')) return json({ error: '`path` must be a post URL' }, 400, cors);
  if (typeof b.hash !== 'string' || !HASH.test(b.hash)) return json({ error: '`hash` must be 8 hex chars' }, 400, cors);
  if (!RESOLVE_ACTIONS.includes(b.action)) return json({ error: '`action` must be ' + RESOLVE_ACTIONS.join(' | ') }, 400, cors);
  await ensureReactionsTable(env);
  const now = new Date().toISOString();
  const set = b.action === 'resolve' ? 'resolved_at = ?3, resolved_doubt = doubt'
    : b.action === 'reopen' ? 'resolved_at = NULL, resolved_doubt = 0'
    : 'doubt = 0, reasons = NULL, resolved_at = NULL, resolved_doubt = 0';
  const row = await env.DB.prepare(`UPDATE passage_reactions SET ${set}, updated_at = ?3 WHERE path = ?1 AND hash = ?2 RETURNING ${ROW_COLS}`).bind(b.path, b.hash, now).first();
  if (!row) return json({ error: '没有这条记录' }, 404, cors);
  return json(rowOut(row), 200, { ...cors, 'Cache-Control': 'no-store' });
}
async function reactionsTop(url, env, cors) {
  if (!env.DB) return json({ error: '段落点赞未启用（worker 未绑定 D1）' }, 501, cors);
  await ensureReactionsTable(env);
  const want = url.searchParams.get('kind');
  const kind = ['up', 'doubt', 'share'].includes(want) ? want : 'doubt';
  const limit = Math.min(200, Math.max(1, parseInt(url.searchParams.get('limit') || '50', 10) || 50));
  const { results } = await env.DB.prepare(`SELECT path, hash, quote, section, ${ROW_COLS}, updated_at FROM passage_reactions WHERE ${kind} > 0 ORDER BY ${kind} DESC, updated_at DESC LIMIT ?1`).bind(limit).all();
  return json({ rows: (results || []).map(withReasons) }, 200, { ...cors, 'Cache-Control': 'public, max-age=60' });
}

// Everything D1 knows about one post, for the dashboard's 修订简报 (the
// Discussion comments and Issues are fetched by the browser from GitHub).
// Without `path`: every post at once — { posts: { path: { reactions, views, up, shares } } } —
// for the weekly tools/feedback-queue.cjs Action (one call instead of one per post).
async function feedback(url, env, cors) {
  if (!env.DB) return json({ error: '未启用（worker 未绑定 D1）' }, 501, cors);
  if (!viewsTableReady) viewsTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS views (path TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at TEXT)');
  await Promise.all([viewsTableReady, ensureVotesTable(env), ensureSharesTable(env), ensureReactionsTable(env)]);
  const path = url.searchParams.get('path');
  if (path == null) {
    const [r, v, vo, sh] = await Promise.all([
      env.DB.prepare(`SELECT path, hash, quote, section, ${ROW_COLS}, updated_at FROM passage_reactions WHERE up > 0 OR doubt > 0 OR share > 0 ORDER BY path, doubt DESC, up DESC`).all(),
      env.DB.prepare('SELECT path, count FROM views').all(),
      env.DB.prepare('SELECT path, up FROM votes').all(),
      env.DB.prepare('SELECT path, count FROM shares').all(),
    ]);
    const posts = {};
    const at = (p) => posts[p] || (posts[p] = { reactions: [], views: 0, up: 0, shares: 0 });
    for (const row of r.results || []) { const { path: p, ...rest } = row; at(p).reactions.push(withReasons(rest)); }
    for (const row of v.results || []) at(row.path).views = row.count;
    for (const row of vo.results || []) at(row.path).up = row.up;
    for (const row of sh.results || []) at(row.path).shares = row.count;
    return json({ posts }, 200, { ...cors, 'Cache-Control': 'no-store' });
  }
  if (typeof path !== 'string' || !VIEW_PATH.test(path) || path.includes('..')) return json({ error: '`path` must be a post URL' }, 400, cors);
  const [r, v, vo, sh] = await Promise.all([
    env.DB.prepare(`SELECT hash, quote, section, ${ROW_COLS}, updated_at FROM passage_reactions WHERE path = ?1 AND (up > 0 OR doubt > 0 OR share > 0) ORDER BY doubt DESC, up DESC`).bind(path).all(),
    env.DB.prepare('SELECT count FROM views WHERE path = ?1').bind(path).first(),
    env.DB.prepare('SELECT up FROM votes WHERE path = ?1').bind(path).first(),
    env.DB.prepare('SELECT count FROM shares WHERE path = ?1').bind(path).first()
  ]);
  return json({
    path, reactions: (r.results || []).map(withReasons),
    views: (v && v.count) || 0, up: (vo && vo.up) || 0, shares: (sh && sh.count) || 0
  }, 200, { ...cors, 'Cache-Control': 'no-store' });
}

// Share counter: POST /shares { path } -> { shares }. Bumped by js/share.js
// whenever a reader uses the share menu (system sheet, Weibo/X/LinkedIn, WeChat
// QR, copy link). Same trust level as views; `shares(path, count)` in D1.
let sharesTableReady = null;
function ensureSharesTable(env) {
  if (!sharesTableReady) sharesTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS shares (path TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at TEXT)');
  return sharesTableReady;
}
async function shares(request, env, cors) {
  if (!env.DB) return json({ error: '分享计数未启用（worker 未绑定 D1）' }, 501, cors);
  const body = await request.json().catch(() => ({}));
  const path = body.path;
  if (typeof path !== 'string' || !VIEW_PATH.test(path) || path.includes('..')) return json({ error: '`path` must be a post URL' }, 400, cors);
  return json({ shares: await bumpShares(env, path, new Date().toISOString()) }, 200, { ...cors, 'Cache-Control': 'no-store' });
}
async function bumpShares(env, path, now) {
  await ensureSharesTable(env);
  const row = await env.DB.prepare(
    'INSERT INTO shares (path, count, updated_at) VALUES (?1, 1, ?2) ' +
    'ON CONFLICT(path) DO UPDATE SET count = count + 1, updated_at = ?2 RETURNING count'
  ).bind(path, now).first();
  return (row && row.count) || 0;
}

// GET /stats?paths=/a.html,/b.html  (<= 20) -> per-post counters for list pages:
// { items: { "/a.html": { views, comments, up, down, shares, id, url } } }
// views, up/down and shares from D1 (0 without the binding), comments / discussion id
// from the giscus public API (one lookup per path; the author's private notes
// are left out, see publicCommentCount).
// Cached 120 s at the edge per path so a listing of 10 posts is cheap.
const STATS_MAX = 20;
async function stats(url, env, ctx, cors) {
  const paths = String(url.searchParams.get('paths') || '').split(',').map((s) => s.trim()).filter(Boolean).slice(0, STATS_MAX);
  if (!paths.length) return json({ error: '`paths` is required' }, 400, cors);
  for (const p of paths) if (!VIEW_PATH.test(p) || p.includes('..')) return json({ error: `bad path ${p}` }, 400, cors);

  let viewsByPath = {}, votesByPath = {}, sharesByPath = {};
  if (env.DB) {
    if (!viewsTableReady) viewsTableReady = env.DB.exec('CREATE TABLE IF NOT EXISTS views (path TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, updated_at TEXT)');
    await Promise.all([viewsTableReady, ensureVotesTable(env), ensureSharesTable(env)]);
    const marks = paths.map((_, i) => `?${i + 1}`).join(',');
    const [v, w, s] = await Promise.all([
      env.DB.prepare(`SELECT path, count FROM views WHERE path IN (${marks})`).bind(...paths).all(),
      env.DB.prepare(`SELECT path, up, down FROM votes WHERE path IN (${marks})`).bind(...paths).all(),
      env.DB.prepare(`SELECT path, count FROM shares WHERE path IN (${marks})`).bind(...paths).all(),
    ]);
    for (const r of v.results || []) viewsByPath[r.path] = r.count;
    for (const r of w.results || []) votesByPath[r.path] = { up: r.up, down: r.down };
    for (const r of s.results || []) sharesByPath[r.path] = r.count;
  }

  const cache = caches.default;
  const items = {};
  await Promise.all(paths.map(async (p) => {
    const key = new Request(`${url.origin}/stats/one?path=${encodeURIComponent(p)}`);
    let d = null;
    const hit = await cache.match(key);
    if (hit) d = await hit.json();
    else {
      const got = await fetchDiscussion(env, p);
      const disc = got.discussion;
      d = disc ? { id: disc.id, url: disc.url, comments: publicCommentCount(disc) }
               : { id: null, url: null, comments: 0 };
      if (!got.error) ctx.waitUntil(cache.put(key, json(d, 200, { 'Cache-Control': 'public, s-maxage=120' })));
    }
    items[p] = { ...d, views: viewsByPath[p] || 0, up: (votesByPath[p] || {}).up || 0, down: (votesByPath[p] || {}).down || 0, shares: sharesByPath[p] || 0 };
  }));
  return json({ items }, 200, { ...cors, 'Cache-Control': 'public, max-age=60' });
}

// ---- 随笔: POST / GET / PUT / DELETE /moments (the phone "发布页", moments/post.html)
// The author writes a 随笔 on the phone; the worker turns it into exactly what
// tools/moment.py would have written — `## YYYY-MM-DD HH:MM @place` + text with
// `#标签` + quote + pictures + music URL appended to moments/YYYY-MM.md — and
// commits it together with the pictures in ONE commit on the default branch
// through the Git Data API (blobs → tree → commit → ref), acting as our GitHub
// App (installation token with contents: write; the commit's *author* is the
// blog author, the committer the App). The push triggers the deploy workflow.
// Editing (PUT) rewrites that entry's block in place (or moves it to another
// month file when the date changed), deleting (DELETE) drops the block; both
// also remove unused pictures/posters under img/moments/ and best-effort delete
// the replaced/deleted R2 video. Entries are addressed by { month: "YYYY-MM", id }
// with the ids of _plugins/moments.rb (YYYYMMDD[-HHMM][-n] for duplicates).
// Times are Beijing (the site's `timezone: Asia/Shanghai`), like the CLI.

const MOMENT_TZ_MINUTES = 8 * 60;
const MOMENT_TEXT_MAX = 5000;
const MOMENT_IMAGES_MAX = 9;
const MOMENT_IMAGE_BYTES_MAX = 3 * 1024 * 1024;   // the page resizes to ≤ 1600px before uploading
const MOMENT_IMAGE_TYPES = { 'image/webp': 'webp', 'image/jpeg': 'jpg', 'image/png': 'png', 'image/gif': 'gif' };
const MOMENT_VIDEO_BYTES_MAX = 50 * 1024 * 1024;
const MOMENT_VIDEO_TYPES = { 'video/mp4': 'mp4', 'video/quicktime': 'mov', 'video/x-m4v': 'm4v', 'video/webm': 'webm' };
const MOMENT_VIDEO_EXT_TYPES = { mp4: 'video/mp4', mov: 'video/quicktime', m4v: 'video/x-m4v', webm: 'video/webm' };
const MOMENT_TAG = /^[\p{L}_][\p{L}\p{N}_\-·]*(?:\/[\p{L}\p{N}_\-·]+)*$/u;   // = Moments::TAG in _plugins/moments.rb
const MOMENT_TIME = /^(\d{4})-(\d{2})-(\d{2})(?: (\d{2}):(\d{2}))?$/;
const MOMENT_HEAD = /^##\s+(\d{4})-(\d{2})-(\d{2})(?:\s+(\d{1,2}):(\d{2}))?(?:\s+@\s*(.+?))?\s*$/;   // = Moments::HEAD
const MOMENT_MONTH = /^\d{4}-\d{2}$/;
const MOMENT_ID = /^\d{8}(?:-\d{4})?(?:-\d+)?$/;
const MOMENT_IMG_URL = /^\/img\/moments\/\d{4}\/\d{2}\/[A-Za-z0-9\u4e00-\u9fff_.-]+\.(?:webp|jpg|jpeg|png|gif)$/;
const MOMENT_VIDEO_URL = /\.(?:mp4|mov|m4v|webm)(?:[?#]|$)/i;
const MD_IMAGE = /!\[[^\]]*\]\(([^)\s]+)(?:\s+"([^"]*)")?\)/g;

function pad2(n) { return String(n).padStart(2, '0'); }

// { y, m, d, hh, mm, hasTime } — `time` as "YYYY-MM-DD[ HH:MM]" (Beijing), default now.
function momentWhen(time) {
  if (time) {
    const m = MOMENT_TIME.exec(time);
    if (!m) return null;
    return { y: m[1], m: m[2], d: m[3], hh: m[4] || '', mm: m[5] || '', hasTime: !!m[4] };
  }
  const t = new Date(Date.now() + MOMENT_TZ_MINUTES * 60_000);
  return { y: String(t.getUTCFullYear()), m: pad2(t.getUTCMonth() + 1), d: pad2(t.getUTCDate()), hh: pad2(t.getUTCHours()), mm: pad2(t.getUTCMinutes()), hasTime: true };
}
function momentStamp(w) { return `${w.y}-${w.m}-${w.d}${w.hasTime ? ' ' + w.hh + ':' + w.mm : ''}`; }
function momentId(w) { return `${w.y}${w.m}${w.d}${w.hasTime ? '-' + w.hh + w.mm : ''}`; }

function momentMediaBase(env) {
  return String(env && env.MEDIA_BASE || '').replace(/\/+$/, '');
}

function momentError(message, status = 400) {
  const err = new Error(message);
  err.status = status;
  return err;
}

export async function putMomentVideo(env, body, type, size) {
  const base = momentMediaBase(env);
  if (!env.MEDIA || !base) throw momentError('还没配置视频存储（R2）', 501);
  const mediaType = String(type || '').split(';')[0].trim().toLowerCase();
  const ext = MOMENT_VIDEO_TYPES[mediaType];
  if (!ext) throw momentError(`视频格式不支持：${type || ''}`);
  if (size > MOMENT_VIDEO_BYTES_MAX) throw momentError(`视频太大（> ${MOMENT_VIDEO_BYTES_MAX / 1024 / 1024} MB）`, 413);
  const w = momentWhen();
  const random = new Uint8Array(5);
  crypto.getRandomValues(random);
  const hex = Array.from(random, (byte) => byte.toString(16).padStart(2, '0')).join('');
  const key = `moments/${w.y}/${w.m}/${hex}.${ext}`;
  await env.MEDIA.put(key, body, {
    httpMetadata: { contentType: mediaType, cacheControl: 'public, max-age=31536000, immutable' },
  });
  return `${base}/${key}`;
}

function imageSlug(name) {
  const base = String(name || '').replace(/\.[^.]*$/, '').toLowerCase().replace(/[^a-z0-9\u4e00-\u9fff-]+/g, '-').replace(/^-+|-+$/g, '');
  return base || 'img';
}

function b64ToBytes(s) {
  const bin = atob(s.replace(/\s+/g, ''));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function base64PrefixBytes(data) {
  let prefix = data.replace(/\s+/g, '').slice(0, 16);
  if (!prefix) return new Uint8Array();
  prefix = prefix.padEnd(Math.ceil(prefix.length / 4) * 4, '=');
  try {
    const binary = atob(prefix);
    return Uint8Array.from(binary, (char) => char.charCodeAt(0));
  } catch {
    return new Uint8Array();
  }
}

function sniffMomentImageType(bytes) {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg';
  if (bytes.length >= 4 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return 'image/png';
  if (bytes.length >= 4 && String.fromCharCode.apply(null, bytes.subarray(0, 4)) === 'GIF8') return 'image/gif';
  if (bytes.length >= 12 &&
    String.fromCharCode.apply(null, bytes.subarray(0, 4)) === 'RIFF' &&
    String.fromCharCode.apply(null, bytes.subarray(8, 12)) === 'WEBP') return 'image/webp';
  return '';
}

function bytesToBase64(bytes) {
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

async function formImage(value, number, label = `第 ${number} 张图`) {
  if (typeof value === 'string') {
    if (!value) return null;
    return { name: '', type: sniffMomentImageType(base64PrefixBytes(value)), data: value };
  }
  if (!value || typeof value.arrayBuffer !== 'function' || typeof value.size !== 'number') return null;
  if (value.size > MOMENT_IMAGE_BYTES_MAX) throw momentError(`${label}太大（> ${MOMENT_IMAGE_BYTES_MAX / 1024 / 1024} MB）`, 413);
  const bytes = new Uint8Array(await value.arrayBuffer());
  const declaredType = typeof value.type === 'string' ? value.type : '';
  const type = Object.prototype.hasOwnProperty.call(MOMENT_IMAGE_TYPES, declaredType)
    ? declaredType
    : sniffMomentImageType(bytes) || declaredType;
  return {
    name: typeof value.name === 'string' ? value.name : '',
    type,
    data: bytesToBase64(bytes),
  };
}

function videoFileType(file) {
  const declared = String(file && file.type || '').split(';')[0].trim().toLowerCase();
  if (MOMENT_VIDEO_TYPES[declared]) return declared;
  if (declared && declared !== 'application/octet-stream') return declared;
  const match = String(file && file.name || '').toLowerCase().match(/\.([a-z0-9]+)$/);
  return match && MOMENT_VIDEO_EXT_TYPES[match[1]] || declared;
}

async function momentBody(request, env) {
  const contentType = (request.headers.get('Content-Type') || '').split(';')[0].trim().toLowerCase();
  if (contentType === 'text/plain') return { body: { text: await request.text() }, uploadedVideo: null };
  if (contentType !== 'multipart/form-data' && contentType !== 'application/x-www-form-urlencoded') {
    return { body: await request.json().catch(() => null), uploadedVideo: null };
  }

  const fd = await request.formData();
  const body = {};
  for (const key of ['text', 'place', 'time', 'quote', 'by', 'music']) {
    const value = fd.get(key);
    if (typeof value === 'string') body[key] = value;
  }
  const tags = fd.getAll('tags')
    .filter((value) => typeof value === 'string')
    .flatMap((value) => value.split(/[,，\s]+/))
    .filter(Boolean);
  if (tags.length) body.tags = tags;

  body.images = [];
  for (const value of [...fd.getAll('image'), ...fd.getAll('images')]) {
    const image = await formImage(value, body.images.length + 1);
    if (image) body.images.push(image);
  }

  const posters = fd.getAll('poster').filter((value) => typeof value !== 'string' || value);
  if (posters.length > 1) throw momentError('只能上传一张视频封面');
  let poster = posters.length ? await formImage(posters[0], 1, '视频封面') : null;
  if (poster) {
    poster = { type: poster.type, data: poster.data };
  }

  const videos = fd.getAll('video').filter((value) => typeof value !== 'string' || value);
  if (videos.length > 1) throw momentError('一条随笔只能上传一个视频');
  let uploadedVideo = null;
  if (videos.length) {
    const video = videos[0];
    if (typeof video === 'string') {
      body.video = { src: video, ...(poster ? { poster } : {}) };
    } else {
      uploadedVideo = await putMomentVideo(env, video.stream(), videoFileType(video), video.size);
      body.video = { src: uploadedVideo, ...(poster ? { poster } : {}) };
    }
  } else if (poster) {
    throw momentError('视频封面必须和视频一起上传');
  }
  return { body, uploadedVideo };
}

// A month file → { pre: [lines before the first entry], entries: [{ id, stamp, place, head, body }] }
// with the same ids as the plugin (a second entry at the same minute gets -2).
function parseMonth(content) {
  const lines = content.split('\n');
  const pre = [], entries = [];
  let cur = null;
  for (const line of lines) {
    const m = MOMENT_HEAD.exec(line);
    if (m) {
      const w = { y: m[1], m: m[2], d: m[3], hh: m[4] ? pad2(+m[4]) : '', mm: m[5] || '', hasTime: !!m[4] };
      cur = { base: momentId(w), stamp: momentStamp(w), place: m[6] || '', head: line, lines: [] };
      entries.push(cur);
    } else if (cur) cur.lines.push(line);
    else pre.push(line);
  }
  const seen = {};
  for (const e of entries) {
    seen[e.base] = (seen[e.base] || 0) + 1;
    e.id = seen[e.base] > 1 ? `${e.base}-${seen[e.base]}` : e.base;
    e.body = e.lines.join('\n').replace(/^\s*\n/, '').replace(/\s+$/, '');
    delete e.lines;
  }
  return { pre, entries };
}
function monthText(parsed) {
  const pre = parsed.pre.join('\n').replace(/\s+$/, '') || '---\nlayout: moments\n---';
  const blocks = parsed.entries.map((e) => (e.head + '\n' + e.body).replace(/\s+$/, ''));
  return pre + '\n' + (blocks.length ? '\n' + blocks.join('\n\n') + '\n' : '');
}
function entryMedia(body) {
  const images = [];
  let video = null;
  for (const match of body.matchAll(MD_IMAGE)) {
    if (MOMENT_VIDEO_URL.test(match[1])) {
      if (!video) video = { src: match[1], poster: match[2] || null };
    } else {
      images.push(match[1]);
    }
  }
  return { images, video };
}
function entryImages(body) { return entryMedia(body).images; }
function entryPictureFiles(body) {
  const media = entryMedia(body);
  return media.video && media.video.poster ? [...media.images, media.video.poster] : media.images;
}

// An entry's body back into the 发布页's fields. Canonical = text paragraphs, then
// at most one quote, one run of pictures, one URL — what POST writes. Anything
// else (text after a quote, two quotes…) is handed over as raw Markdown.
export function entryFields(e) {
  const media = entryMedia(e.body);
  const out = { text: '', quote: '', by: '', music: '', images: media.images, video: media.video, raw: false };
  const paras = e.body ? e.body.split(/\n{2,}/) : [];
  const kind = (p) => (/^(?:!\[[^\]]*\]\([^)\s]+(?:\s+"[^"]*")?\)\s*)+$/.test(p) ? 'media' : /^>/.test(p) ? 'quote' : /^https?:\/\/\S+$/.test(p) ? 'music' : 'text');
  const rank = { text: 0, quote: 1, media: 2, music: 3 };
  let stage = 0;
  const textParas = [];
  for (const p of paras) {
    const k = kind(p);
    if (rank[k] < stage || (k !== 'text' && rank[k] === stage)) { out.raw = true; break; }
    stage = rank[k];
    if (k === 'text') textParas.push(p);
    else if (k === 'quote') {
      const ls = p.split('\n').map((l) => l.replace(/^>\s?/, ''));
      if (ls.length > 1 && /^——\s*/.test(ls[ls.length - 1])) out.by = ls.pop().replace(/^——\s*/, '');
      out.quote = ls.join('\n');
    } else if (k === 'music') out.music = p.trim();
  }
  if (out.raw) { out.text = e.body; out.quote = out.by = out.music = ''; }
  else out.text = textParas.join('\n\n');
  return out;
}

// The request body → validated fields (throws { status, message }).
export function momentInput(b, env = {}) {
  const bad = momentError;
  if (!b || typeof b !== 'object') throw bad('JSON body required');
  const str = (k, max) => {
    const v = b[k] == null ? '' : b[k];
    if (typeof v !== 'string') throw bad(`\`${k}\` must be a string`);
    if (v.length > max) throw bad(`\`${k}\` is too long (> ${max})`);
    return v.trim();
  };
  const f = {
    text: str('text', MOMENT_TEXT_MAX),
    place: str('place', 80).replace(/\s+/g, ' '),
    quote: str('quote', 2000),
    by: str('by', 120).replace(/\s+/g, ' '),
    music: str('music', 500),
    time: str('time', 16),
  };
  if (f.music && !/^https?:\/\/\S+$/.test(f.music)) throw bad('`music` must be a URL');
  let tags = Array.isArray(b.tags) ? b.tags : [];
  if (tags.length > 20) throw bad('too many tags');
  tags = tags.map((t) => String(t || '').trim().replace(/^#/, '')).filter(Boolean);
  for (const t of tags) if (t.length > 40 || !MOMENT_TAG.test(t)) throw bad(`标签不合法：${t}`);
  f.tags = tags;
  const images = Array.isArray(b.images) ? b.images : [];
  if (images.length > MOMENT_IMAGES_MAX) throw bad(`最多 ${MOMENT_IMAGES_MAX} 张图`);
  f.when = momentWhen(f.time);
  if (!f.when) throw bad('`time` must be YYYY-MM-DD[ HH:MM]');
  // Pictures: { url } keeps one already in the repo (editing), { type, data } uploads a new one.
  const picture = (im, label, base) => {
    im = im || {};
    if (typeof im.url === 'string') {
      if (!MOMENT_IMG_URL.test(im.url)) throw bad(`${label}的地址不合法`);
      return { url: im.url };
    }
    const ext = MOMENT_IMAGE_TYPES[im.type];
    if (!ext) throw bad(`${label}的类型不支持：${im.type || '?'}`);
    if (typeof im.data !== 'string' || !im.data) throw bad(`${label}没有数据`);
    if (im.data.length > MOMENT_IMAGE_BYTES_MAX * 4 / 3 + 16) throw bad(`${label}太大（> ${MOMENT_IMAGE_BYTES_MAX / 1024 / 1024} MB）`, 413);
    return { base, ext, data: im.data.replace(/\s+/g, '') };
  };
  f.pics = [];
  images.forEach((im, i) => {
    const slug = imageSlug(im && im.name);
    f.pics.push(picture(im, `第 ${i + 1} 张图`, slug === 'img' ? `${momentId(f.when)}-${i + 1}` : slug));
  });
  f.video = null;
  if (b.video != null) {
    if (!env.MEDIA || !momentMediaBase(env)) throw bad('还没配置视频存储（R2）', 501);
    if (!b.video || typeof b.video !== 'object' || Array.isArray(b.video)) throw bad('视频地址不合法');
    const src = typeof b.video.src === 'string' ? b.video.src.trim() : '';
    const prefix = `${momentMediaBase(env)}/moments/`;
    if (!src.startsWith(prefix) || !MOMENT_VIDEO_URL.test(src)) throw bad('视频地址不合法');
    const poster = b.video.poster == null ? null : picture(b.video.poster, '视频封面', `${momentId(f.when)}-poster`);
    f.video = { src, poster };
  }
  if (f.video && f.pics.length) throw bad('视频和图片不能同时发');
  if (!f.text && !tags.length && !f.quote && !f.pics.length && !f.video && !f.music) throw bad('写点什么吧');
  return f;
}

// The entry, exactly as tools/moment.py writes it: { head, body }.
export function renderEntry(f, picUrls, posterUrl = null) {
  const head = `## ${momentStamp(f.when)}${f.place ? ' @' + f.place : ''}`;
  const parts = [];
  const tagLine = f.tags.map((t) => '#' + t).join(' ');
  if (f.text || tagLine) parts.push((f.text + ' ' + tagLine).trim());
  if (f.quote) {
    const q = f.quote.split(/\r?\n/).map((l) => '> ' + l);
    if (f.by) q.push('> —— ' + f.by);
    parts.push(q.join('\n'));
  }
  if (picUrls.length) parts.push(picUrls.map((u) => `![](${u})`).join('\n'));
  if (f.video) parts.push(`![视频](${f.video.src}${posterUrl ? ` "${posterUrl}"` : ''})`);
  if (f.music) parts.push(f.music);
  return { head, body: parts.join('\n\n') };
}

function momentTarget(b) {
  const month = String(b && b.month || ''), id = String(b && b.id || '');
  if (!MOMENT_MONTH.test(month)) return { error: '`month` must be YYYY-MM' };
  if (!MOMENT_ID.test(id)) return { error: '`id` must be YYYYMMDD[-HHMM][-n]' };
  return { month, id, path: `moments/${month}.md` };
}

// One commit on the branch built by `build(gh, headSha)` → { tree, message }
// (tree entries in Git Data API form; `sha: null` deletes). Two attempts:
// somebody (the CLI, a CI bot) may push between our read and our ref update.
async function momentCommit(env, user, build) {
  const branch = env.MOMENTS_BRANCH || 'master';
  const repo = `${GITHUB}/repos/${env.REPO}`;
  const headers = githubHeaders(await installationToken(env, { contents: 'write' }));
  const gh = async (path, init) => {
    const r = await fetch(`${repo}${path}`, { ...init, headers: { ...headers, ...(init && init.headers) } });
    if (r.status === 404 && (!init || !init.method || init.method === 'GET')) return null;
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(`GitHub ${init && init.method || 'GET'} ${path}: ${data.message || r.status}`);
    return data;
  };
  for (let attempt = 0; attempt < 2; attempt++) {
    const ref = await gh(`/git/ref/heads/${branch}`);
    if (!ref) throw new Error(`分支 ${branch} 不存在`);
    const headSha = ref.object.sha;
    const headCommit = await gh(`/git/commits/${headSha}`);
    const built = await build(gh, headSha);
    if (built.response) return built.response;
    const newTree = await gh('/git/trees', { method: 'POST', body: JSON.stringify({ base_tree: headCommit.tree.sha, tree: built.tree }) });
    const commit = await gh('/git/commits', {
      method: 'POST',
      body: JSON.stringify({
        message: built.message,
        tree: newTree.sha,
        parents: [headSha],
        author: { name: user.name || user.login, email: user.email || `${user.id}+${user.login}@users.noreply.github.com`, date: new Date().toISOString() },
      }),
    });
    const upd = await fetch(`${repo}/git/refs/heads/${branch}`, { method: 'PATCH', headers, body: JSON.stringify({ sha: commit.sha, force: false }) });
    if (upd.ok) return { commit: commit.sha, ...built.result };
    if (upd.status !== 422 || attempt === 1) {
      const data = await upd.json().catch(() => ({}));
      throw new Error(`update ref: ${data.message || upd.status}`);
    }
  }
  throw new Error('update ref: raced twice');
}

// Reads moments/YYYY-MM.md at `sha` → parsed (a missing file = an empty month).
async function readMonth(gh, path, sha) {
  const file = await gh(`/contents/${path}?ref=${sha}`);
  return parseMonth(file ? new TextDecoder().decode(b64ToBytes(file.content)) : '---\nlayout: moments\n---\n');
}
// Uploads the new pictures of `f.pics` into img/moments/YYYY/MM (names made unique
// against the directory) → the gallery URLs in order; pushes blobs onto `tree`.
async function uploadPics(gh, headSha, f, tree) {
  const imgDir = `img/moments/${f.when.y}/${f.when.m}`;
  let existing = null;
  const urls = [];
  for (const p of f.pics) {
    if (p.url) { urls.push(p.url); continue; }
    if (!existing) existing = new Set(((await gh(`/contents/${imgDir}?ref=${headSha}`)) || []).map((x) => x.name));
    let name = `${p.base}.${p.ext}`;
    for (let k = 2; existing.has(name); k++) name = `${p.base}-${k}.${p.ext}`;
    existing.add(name);
    const blob = await gh('/git/blobs', { method: 'POST', body: JSON.stringify({ content: p.data, encoding: 'base64' }) });
    tree.push({ path: `${imgDir}/${name}`, mode: '100644', type: 'blob', sha: blob.sha });
    urls.push(`/${imgDir}/${name}`);
  }
  return urls;
}
async function uploadEntryPictures(gh, headSha, f, tree) {
  const pics = f.video && f.video.poster ? [...f.pics, f.video.poster] : f.pics;
  const urls = await uploadPics(gh, headSha, { when: f.when, pics }, tree);
  return {
    images: urls.slice(0, f.pics.length),
    poster: f.video && f.video.poster ? urls[f.pics.length] : null,
  };
}
// Deletes the pictures of an old entry that no other entry (in the files we are
// about to write) still shows — only those that really exist at `headSha`.
async function dropPics(gh, headSha, oldUrls, keepTexts, tree) {
  const still = keepTexts.join('\n');
  const dirs = {};
  for (const u of new Set(oldUrls)) {
    if (!MOMENT_IMG_URL.test(u) || still.includes(u)) continue;
    const path = u.slice(1), dir = path.replace(/\/[^/]+$/, '');
    if (!dirs[dir]) dirs[dir] = new Set(((await gh(`/contents/${dir}?ref=${headSha}`)) || []).map((x) => x.name));
    if (dirs[dir].has(path.slice(dir.length + 1))) tree.push({ path, mode: '100644', type: 'blob', sha: null });
  }
}
function momentVideoKey(env, src) {
  const base = momentMediaBase(env);
  if (!base || typeof src !== 'string' || !src.startsWith(`${base}/moments/`)) return null;
  const key = src.slice(base.length + 1).split(/[?#]/, 1)[0];
  return key.startsWith('moments/') ? key : null;
}
async function dropMomentVideo(env, src) {
  const key = momentVideoKey(env, src);
  if (!env.MEDIA || !key) return;
  try { await env.MEDIA.delete(key); } catch {}
}
function momentUrl(month, id) { return `/moments/${month}.html#${id}`; }

async function pinMoment(request, env, cors) {
  if (!env.GITHUB_APP_PRIVATE_KEY || !env.GITHUB_APP_ID) return json({ error: '发布功能未启用（worker 未配置 GitHub App）' }, 501, cors);
  const author = await requireAuthor(request, env, cors, '置顶随笔');
  if (author.error) return author.error;
  const body = await request.json().catch(() => null);
  const id = String(body && body.id || '');
  if (!MOMENT_ID.test(id) || !body || typeof body.pinned !== 'boolean') return json({ error: '`id` or `pinned` is invalid' }, 400, cors);

  const result = await momentCommit(env, author.user, async (gh, headSha) => {
    const month = `${id.slice(0, 4)}-${id.slice(4, 6)}`;
    const file = await gh(`/contents/moments/${month}.md?ref=${headSha}`);
    if (!file) return { response: json({ error: '没有这条随笔' }, 404, cors) };
    const parsed = parseMonth(new TextDecoder().decode(b64ToBytes(file.content)));
    if (!parsed.entries.some((entry) => entry.id === id)) return { response: json({ error: '没有这条随笔' }, 404, cors) };
    const dataFile = await gh(`/contents/_data/moments.yml?ref=${headSha}`);
    const ids = readPinnedIds(dataFile ? new TextDecoder().decode(b64ToBytes(dataFile.content)) : '');
    const next = body.pinned ? Array.from(new Set([...ids, id])) : ids.filter((value) => value !== id);
    if (next.length === ids.length && next.every((value, i) => value === ids[i])) {
      return { response: json({ id, pinned: body.pinned }, 200, cors) };
    }
    return {
      tree: [{ path: '_data/moments.yml', mode: '100644', type: 'blob', content: writePinnedIds(next) }],
      message: `随笔: ${body.pinned ? '置顶' : '取消置顶'} ${id}`,
      result: { id, pinned: body.pinned },
    };
  });
  return result instanceof Response ? result : json(result, 200, { ...cors, 'Cache-Control': 'no-store' });
}

export function readPinnedIds(text) {
  const match = /^pinned:[ \t]*(.*)$/m.exec(text);
  if (!match) return [];
  const inline = /^\[(.*)\]$/.exec(match[1].trim());
  const values = inline ? inline[1].split(',') : text.slice(match.index + match[0].length).match(/^\s+-\s+[\w-]+/gm) || [];
  return values.map((value) => (inline ? value.trim() : value.replace(/^\s+-\s+/, '')).replace(/^['"]|['"]$/g, '')).filter((id) => MOMENT_ID.test(id));
}

export function writePinnedIds(ids) {
  return ids.length ? `pinned:\n${ids.map((id) => `  - ${id}`).join('\n')}\n` : 'pinned: []\n';
}

async function renameMomentTag(request, env, cors) {
  if (!env.GITHUB_APP_PRIVATE_KEY || !env.GITHUB_APP_ID) return json({ error: '发布功能未启用（worker 未配置 GitHub App）' }, 501, cors);
  const author = await requireAuthor(request, env, cors, '重命名标签');
  if (author.error) return author.error;
  const body = await request.json().catch(() => null);
  const from = String(body && body.from || ''), to = String(body && body.to || '');
  if (!MOMENT_TAG.test(from) || !MOMENT_TAG.test(to) || from === to) return json({ error: '`from` and `to` must be different valid tags' }, 400, cors);

  const result = await momentCommit(env, author.user, async (gh, headSha) => {
    const entries = await gh(`/contents/moments?ref=${headSha}`);
    const tree = [], files = [];
    let changed = 0;
    for (const item of (entries || []).filter((entry) => entry.type === 'file' && /^\d{4}-\d{2}\.md$/.test(entry.name))) {
      const file = await gh(`/contents/moments/${item.name}?ref=${headSha}`);
      if (!file) continue;
      const path = `moments/${item.name}`;
      const result = rewriteMomentTags(new TextDecoder().decode(b64ToBytes(file.content)), from, to);
      if (!result.changed) continue;
      changed += result.changed;
      files.push(path);
      tree.push({ path, mode: '100644', type: 'blob', content: result.content });
    }
    if (!changed) return { response: json({ error: '没有用到这个标签' }, 404, cors) };
    return {
      tree,
      message: `随笔: 标签 #${from} → #${to}（${changed} 条）`,
      result: { changed, files },
    };
  });
  return result instanceof Response ? result : json({ changed: result.changed, files: result.files }, 200, { ...cors, 'Cache-Control': 'no-store' });
}

async function uploadMomentMedia(request, env, cors) {
  const author = await requireMomentPoster(request, env, cors, '上传随笔视频');
  if (author.error) return author.error;
  const length = request.headers.get('Content-Length');
  if (length == null) return json({ error: '缺少 Content-Length' }, 411, cors);
  if (!/^\d+$/.test(length)) return json({ error: 'Content-Length 不合法' }, 400, cors);
  try {
    const src = await putMomentVideo(env, request.body, request.headers.get('Content-Type') || '', Number(length));
    return json({ src }, 201, { ...cors, 'Cache-Control': 'no-store' });
  } catch (e) {
    return json({ error: e.message }, e.status || 400, cors);
  }
}

async function publishMoment(request, env, cors) {
  const author = await requireMomentPoster(request, env, cors, '发布随笔');
  if (author.error) return author.error;
  if (!env.GITHUB_APP_PRIVATE_KEY || !env.GITHUB_APP_ID) return json({ error: '发布功能未启用（worker 未配置 GitHub App）' }, 501, cors);
  let f, uploadedVideo = null;
  try {
    const parsed = await momentBody(request, env);
    const body = parsed.body;
    uploadedVideo = parsed.uploadedVideo;
    if (body && typeof body === 'object' && !Array.isArray(body) && !body.time) {
      body.time = new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString().slice(0, 16).replace('T', ' ');
    }
    f = momentInput(body, env);
  } catch (e) {
    await dropMomentVideo(env, uploadedVideo);
    return json({ error: e.message }, e.status || 400, cors);
  }
  if (f.pics.some((p) => p.url)) return json({ error: '新随笔的图片要上传，不能引用已有图片' }, 400, cors);

  const month = `${f.when.y}-${f.when.m}`, path = `moments/${month}.md`;
  let result;
  try {
    result = await momentCommit(env, author.user, async (gh, headSha) => {
      const parsed = await readMonth(gh, path, headSha);
      const tree = [];
      const media = await uploadEntryPictures(gh, headSha, f, tree);
      const entry = renderEntry(f, media.images, media.poster);
      parsed.entries.push(entry);
      tree.push({ path, mode: '100644', type: 'blob', content: monthText(parsed) });
      return {
        tree,
        message: `随笔: ${momentStamp(f.when)}${f.place ? ' @' + f.place : ''}\n\n${(f.text || f.quote || f.music).slice(0, 200)}\n\n${author.user.via === 'api' ? '(posted via API)' : '(posted from /moments/post.html)'}`,
        result: {
          url: momentUrl(month, momentId(f.when)),
          month: `/moments/${month}.html`,
          file: path,
          images: media.images,
          video: f.video ? { src: f.video.src, poster: media.poster } : null,
        },
      };
    });
  } catch (e) {
    await dropMomentVideo(env, uploadedVideo);
    throw e;
  }
  return json(result, 201, { ...cors, 'Cache-Control': 'no-store' });
}

// GET /moments?month=YYYY-MM&id=… → the entry's fields for the 发布页's edit mode.
async function readMoment(url, request, env, cors) {
  if (!env.GITHUB_APP_PRIVATE_KEY || !env.GITHUB_APP_ID) return json({ error: '发布功能未启用（worker 未配置 GitHub App）' }, 501, cors);
  const author = await requireAuthor(request, env, cors, '编辑随笔');
  if (author.error) return author.error;
  const t = momentTarget({ month: url.searchParams.get('month'), id: url.searchParams.get('id') });
  if (t.error) return json({ error: t.error }, 400, cors);
  const headers = githubHeaders(await installationToken(env, { contents: 'write' }));
  const r = await fetch(`${GITHUB}/repos/${env.REPO}/contents/${t.path}?ref=${env.MOMENTS_BRANCH || 'master'}`, { headers });
  if (r.status === 404) return json({ error: '没有这个月的随笔' }, 404, cors);
  if (!r.ok) return json({ error: `GitHub: HTTP ${r.status}` }, 502, cors);
  const parsed = parseMonth(new TextDecoder().decode(b64ToBytes((await r.json()).content)));
  const e = parsed.entries.find((x) => x.id === t.id);
  if (!e) return json({ error: '没有这条随笔' }, 404, cors);
  return json({ month: t.month, id: e.id, time: e.stamp, place: e.place, ...entryFields(e) }, 200, { ...cors, 'Cache-Control': 'no-store' });
}

// PUT /moments { month, id, …the POST fields, images: [{ url } | { type, data }] }
async function editMoment(request, env, cors) {
  if (!env.GITHUB_APP_PRIVATE_KEY || !env.GITHUB_APP_ID) return json({ error: '发布功能未启用（worker 未配置 GitHub App）' }, 501, cors);
  const author = await requireAuthor(request, env, cors, '编辑随笔');
  if (author.error) return author.error;
  const b = await request.json().catch(() => null);
  const t = momentTarget(b);
  if (t.error) return json({ error: t.error }, 400, cors);
  let f;
  try { f = momentInput(b, env); } catch (e) { return json({ error: e.message }, e.status || 400, cors); }

  const newMonth = `${f.when.y}-${f.when.m}`, newPath = `moments/${newMonth}.md`;
  let oldVideoSrc = null;
  const result = await momentCommit(env, author.user, async (gh, headSha) => {
    const from = await readMonth(gh, t.path, headSha);
    const i = from.entries.findIndex((x) => x.id === t.id);
    if (i < 0) return { response: json({ error: '没有这条随笔（可能刚被改过，刷新再试）' }, 404, cors) };
    const old = from.entries[i];
    oldVideoSrc = entryMedia(old.body).video && entryMedia(old.body).video.src;
    const tree = [];
    const media = await uploadEntryPictures(gh, headSha, f, tree);
    const entry = renderEntry(f, media.images, media.poster);
    let to = from;
    if (newMonth === t.month) from.entries[i] = entry;
    else { from.entries.splice(i, 1); to = await readMonth(gh, newPath, headSha); to.entries.push(entry); }
    const fromText = monthText(from), toText = monthText(to);
    tree.push({ path: t.path, mode: '100644', type: 'blob', content: fromText });
    if (to !== from) tree.push({ path: newPath, mode: '100644', type: 'blob', content: toText });
    await dropPics(gh, headSha, entryPictureFiles(old.body), [fromText, toText], tree);
    return {
      tree,
      message: `随笔: 修改 ${old.stamp}${newMonth === t.month ? '' : ' → ' + momentStamp(f.when)}\n\n(edited from /moments/post.html)`,
      result: {
        url: momentUrl(newMonth, momentId(f.when)),
        month: `/moments/${newMonth}.html`,
        file: newPath,
        images: media.images,
        video: f.video ? { src: f.video.src, poster: media.poster } : null,
      },
    };
  });
  if (!(result instanceof Response) && oldVideoSrc && oldVideoSrc !== (f.video && f.video.src)) {
    await dropMomentVideo(env, oldVideoSrc);
  }
  return result instanceof Response ? result : json(result, 200, { ...cors, 'Cache-Control': 'no-store' });
}

// DELETE /moments { month, id }
async function deleteMoment(request, env, cors) {
  if (!env.GITHUB_APP_PRIVATE_KEY || !env.GITHUB_APP_ID) return json({ error: '发布功能未启用（worker 未配置 GitHub App）' }, 501, cors);
  const author = await requireAuthor(request, env, cors, '删除随笔');
  if (author.error) return author.error;
  const t = momentTarget(await request.json().catch(() => null));
  if (t.error) return json({ error: t.error }, 400, cors);

  let oldVideoSrc = null;
  const result = await momentCommit(env, author.user, async (gh, headSha) => {
    const parsed = await readMonth(gh, t.path, headSha);
    const i = parsed.entries.findIndex((x) => x.id === t.id);
    if (i < 0) return { response: json({ error: '没有这条随笔（可能已经删了）' }, 404, cors) };
    const [old] = parsed.entries.splice(i, 1);
    oldVideoSrc = entryMedia(old.body).video && entryMedia(old.body).video.src;
    const text = monthText(parsed);
    const tree = [{ path: t.path, mode: '100644', type: 'blob', content: text }];
    await dropPics(gh, headSha, entryPictureFiles(old.body), [text], tree);
    return { tree, message: `随笔: 删除 ${old.stamp}\n\n(deleted from /moments/)`, result: { month: `/moments/${t.month}.html`, file: t.path } };
  });
  if (!(result instanceof Response) && oldVideoSrc) await dropMomentVideo(env, oldVideoSrc);
  return result instanceof Response ? result : json(result, 200, { ...cors, 'Cache-Control': 'no-store' });
}

// ---- GitHub App authentication --------------------------------------------

const cachedInstallation = {}; // permissions key -> { token, expiresAt } — per isolate, so a warm worker reuses it

// `permissions` is the subset the token should carry (default: issues write);
// a permission the App does not have makes GitHub answer 422, so /moments asks
// for contents only when it runs.
async function installationToken(env, permissions = { issues: 'write' }) {
  const key = JSON.stringify(permissions);
  const hit = cachedInstallation[key];
  if (hit && hit.expiresAt - Date.now() > 60_000) return hit.token;
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
    body: JSON.stringify({ permissions }),
  });
  const data = await tr.json();
  if (!tr.ok) throw new Error(`installation token (${key}): ${data.message || tr.status}${tr.status === 422 ? ' — GitHub App 缺少该权限？' : ''}`);
  cachedInstallation[key] = { token: data.token, expiresAt: Date.parse(data.expires_at) };
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
