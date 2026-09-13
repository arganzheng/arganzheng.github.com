#!/usr/bin/env node
/**
 * feedback-queue.cjs — turn reader feedback into a work queue.
 *
 * Runs weekly from .github/workflows/feedback-queue.yml (or by hand). For every
 * post it builds the same 修订简报 the dashboard shows (js/feedback-brief.js,
 * shared) from:
 *   - worker GET /feedback            (D1: 存疑 + 原因 + 章节, chapter 有用/没看懂, views…)
 *   - GitHub GraphQL                  (the Comments category: every discussion with comments)
 *   - GitHub REST                     (划线评论 issues and their state; open 待修订 issues)
 *   - the live page                   (does each quote still anchor? which chapter?)
 * and, when the brief's score reaches THRESHOLD, opens — or updates — one issue
 * labelled 待修订 per post, body = the brief. Posts whose score dropped back
 * under the threshold (everything resolved) get their queue issue closed. A
 * coding agent can pick the issue up as-is; `Fixes #N` in the commit closes it.
 *
 * Env: GITHUB_TOKEN (issues: write), GITHUB_REPOSITORY (owner/name; falls back
 * to _config.yml giscus.repo), SITE / API / CATEGORY_ID (default: _config.yml),
 * THRESHOLD (default 3), QUEUE_LABEL (default 待修订), DRY_RUN=1 (print, no writes),
 * ONLY=/slug.html (one post).
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { parseHTML } = require('linkedom');
const FB = require('../js/feedback-brief.js');

const ROOT = path.join(__dirname, '..');
const cfgText = fs.readFileSync(path.join(ROOT, '_config.yml'), 'utf8');
const cfgGet = (re) => { const m = re.exec(cfgText); return m ? m[1].trim().replace(/^["']|["']$/g, '') : ''; };

const TOKEN = process.env.GITHUB_TOKEN;
const REPO = process.env.GITHUB_REPOSITORY || cfgGet(/^\s+repo:\s*(\S+)/m);
const SITE = (process.env.SITE || cfgGet(/^url:\s*(\S+)/m)).replace(/\/$/, '');
const API = (process.env.API || cfgGet(/^\s+api:\s*(\S+)/m)).replace(/\/$/, '');
const CATEGORY_ID = process.env.CATEGORY_ID || cfgGet(/^\s+category_id:\s*(\S+)/m);
const THRESHOLD = Number(process.env.THRESHOLD || 3);
const QUEUE_LABEL = process.env.QUEUE_LABEL || '待修订';
const ERRATA_LABEL = process.env.ISSUE_LABEL || '划线评论';
const DRY = /^(1|true|yes)$/i.test(process.env.DRY_RUN || '');
const ONLY = process.env.ONLY || '';
const MARK = (p) => `<!-- feedback-queue: ${p} -->`;

if (!TOKEN) { console.error('GITHUB_TOKEN is required'); process.exit(2); }
if (!REPO || !SITE || !API || !CATEGORY_ID) { console.error('missing REPO / SITE / API / CATEGORY_ID', { REPO, SITE, API, CATEGORY_ID }); process.exit(2); }

const DOM = { parse: (html) => parseHTML(/<body[\s>]/i.test(html) ? html : `<!doctype html><html><body>${html}</body></html>`).document };
const gh = {
  headers: { Authorization: `Bearer ${TOKEN}`, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'feedback-queue' },
  async rest(method, url, body) {
    const r = await fetch(url.startsWith('http') ? url : `https://api.github.com${url}`, { method, headers: { ...this.headers, 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    const text = await r.text();
    if (!r.ok) throw new Error(`${method} ${url}: HTTP ${r.status} ${text.slice(0, 300)}`);
    return text ? JSON.parse(text) : null;
  },
  async graphql(query, variables) {
    const d = await this.rest('POST', 'https://api.github.com/graphql', { query, variables });
    if (d.errors) throw new Error('GraphQL: ' + d.errors.map((e) => e.message).join('; '));
    return d.data;
  },
  async list(url) { // paginate a REST list
    const out = [];
    for (let page = 1; page < 20; page++) {
      const rows = await this.rest('GET', `${url}${url.includes('?') ? '&' : '?'}per_page=100&page=${page}`);
      out.push(...rows);
      if (rows.length < 100) break;
    }
    return out;
  },
};

const DISCUSSIONS_Q = `query($owner: String!, $name: String!, $cat: ID!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 25, categoryId: $cat, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { title url comments(first: 100) { totalCount nodes {
        url createdAt deletedAt bodyHTML authorAssociation author { login }
        reactionGroups { content reactors { totalCount } }
        replies(first: 50) { nodes { url createdAt bodyHTML authorAssociation author { login } } }
      } } }
    }
  }
}`;

async function discussions() {
  const [owner, name] = REPO.split('/');
  const byPath = {};
  let after = null;
  for (let i = 0; i < 40; i++) {
    const d = await gh.graphql(DISCUSSIONS_Q, { owner, name, cat: CATEGORY_ID, after });
    const page = d.repository.discussions;
    for (const n of page.nodes) {
      if (!n.title.startsWith('/') || !n.comments.totalCount) continue;
      byPath[n.title] = { url: n.url, comments: n.comments.nodes.map((c) => ({ ...c, replies: c.replies.nodes })) };
    }
    if (!page.pageInfo.hasNextPage) break;
    after = page.pageInfo.endCursor;
  }
  return byPath;
}

async function fetchText(url) {
  const r = await fetch(url, { headers: { 'User-Agent': 'feedback-queue' } });
  if (!r.ok) throw new Error(`GET ${url}: HTTP ${r.status}`);
  return r.text();
}

async function main() {
  const today = new Date(Date.now() + 8 * 3600e3).toISOString().slice(0, 10);
  const [fbAll, discs, errata, queue] = await Promise.all([
    fetch(`${API}/feedback`, { headers: { Origin: SITE } }).then((r) => r.json()),
    discussions(),
    gh.list(`/repos/${REPO}/issues?state=all&labels=${encodeURIComponent(ERRATA_LABEL)}`),
    gh.list(`/repos/${REPO}/issues?state=open&labels=${encodeURIComponent(QUEUE_LABEL)}`),
  ]);
  const posts = fbAll.posts || {};
  const queueByPath = {};
  for (const is of queue) { const m = /<!-- feedback-queue: (\S+) -->/.exec(is.body || ''); if (m) queueByPath[m[1]] = is; }

  let paths = new Set([...Object.keys(discs), ...Object.keys(posts).filter((p) => posts[p].reactions.length), ...Object.keys(queueByPath)]);
  if (ONLY) paths = new Set([ONLY]);
  const summary = [];
  for (const p of [...paths].sort()) {
    const fb = posts[p] || { reactions: [], views: 0, up: 0, shares: 0 };
    const disc = discs[p] || null;
    const issues = errata.filter((is) => !is.pull_request && (is.body || '').includes(p));
    let article = null;
    try { article = FB.articleFromHtml(DOM, await fetchText(SITE + p), p); } catch (e) { console.warn(`  ${p}: article not loaded (${e.message})`); }
    const existing = queueByPath[p];
    const { analysis, markdown } = FB.buildBrief({ dom: DOM, path: p, fb, disc, issues, article, site: SITE, date: today, queueIssue: existing ? existing.number : 0 });
    if (DRY && ONLY) console.log(markdown + '\n');
    const line = `${p}  score ${analysis.score}  待处理 ${analysis.todo.length} · 普通评论 ${analysis.plain.length} · 未定位 ${analysis.lost.length} · 章节 ${analysis.chapters.length}`;
    if (analysis.score >= THRESHOLD) {
      const title = `待修订：《${analysis.title}》`;
      const body = `${MARK(p)}\n_由 feedback-queue 每周自动生成；修完请关闭本 Issue（或在 commit message 里写 \`Fixes #N\`）。反馈变化时正文会被更新。_\n\n${markdown}`;
      if (!existing) {
        summary.push(`+ ${line}  → 新建 Issue`);
        if (!DRY) { await ensureLabel(); const is = await gh.rest('POST', `/repos/${REPO}/issues`, { title, body, labels: [QUEUE_LABEL] }); summary[summary.length - 1] += ` #${is.number}`; }
      } else if (strip(existing.body) !== strip(body) || existing.title !== title) {
        summary.push(`~ ${line}  → 更新 #${existing.number}`);
        if (!DRY) await gh.rest('PATCH', `/repos/${REPO}/issues/${existing.number}`, { title, body });
      } else summary.push(`= ${line}  (#${existing.number} 无变化)`);
    } else if (existing) {
      summary.push(`- ${line}  → 关闭 #${existing.number}（反馈已处理或低于阈值 ${THRESHOLD}）`);
      if (!DRY) {
        await gh.rest('POST', `/repos/${REPO}/issues/${existing.number}/comments`, { body: `反馈已处理（简报分数 ${analysis.score} < ${THRESHOLD}），自动关闭。\n\n${markdown}` });
        await gh.rest('PATCH', `/repos/${REPO}/issues/${existing.number}`, { state: 'closed', state_reason: 'completed' });
      }
    } else summary.push(`  ${line}`);
  }
  const out = `feedback-queue ${today}${DRY ? ' (dry run)' : ''} · threshold ${THRESHOLD} · ${paths.size} posts\n\n${summary.join('\n')}\n`;
  console.log(out);
  if (process.env.GITHUB_STEP_SUMMARY) fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, '```\n' + out + '```\n');
}
// the 生成于 date changes every run; compare without it
function strip(s) { return String(s || '').replace(/生成于 \d{4}-\d{2}-\d{2}/, '').replace(/\r\n/g, '\n').trim(); }
async function ensureLabel() {
  try { await gh.rest('POST', `/repos/${REPO}/labels`, { name: QUEUE_LABEL, color: 'fbca04', description: '读者反馈汇总，等待修订（feedback-queue 自动维护）' }); }
  catch (e) { if (!/HTTP 422/.test(e.message)) throw e; }
}

main().catch((err) => { console.error(err); process.exit(1); });
