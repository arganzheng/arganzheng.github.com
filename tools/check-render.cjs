#!/usr/bin/env node
// Usage: node check-render.cjs <post-slug> [more slugs...]
// Opens each page in its own Chrome tab (CDP :9222), waits for Mermaid and images, reports, closes the tab.
// Env: SITE (default http://localhost:4000), CDP (default http://localhost:9222).
// Needs the `ws` module: a local ./node_modules/ws (CI: `npm i --no-save ws`) or the browser skill's copy.
const http = require('http');
let WebSocket;
try { WebSocket = require('ws'); } catch { WebSocket = require(require('os').homedir() + '/.claude/skills/browser/node_modules/ws'); }
const SITE = (process.env.SITE || 'http://localhost:4000').replace(/\/$/, '');
const CDP = (process.env.CDP || 'http://localhost:9222').replace(/\/$/, '');

const getJSON = (url, method = 'GET') => new Promise((res, rej) => {
  const req = http.request(url, { method }, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => { try { res(JSON.parse(d)); } catch { res(d); } }); });
  req.on('error', rej); req.end();
});

// Content images are loading="lazy": force them so brokenImgs/pending mean something below the fold.
const CHECK = `(() => {
  [...document.images].forEach(i=>{ if(i.loading==='lazy') i.loading='eager'; });
  const m=[...document.querySelectorAll(".mermaid")];
  const errs=[...document.querySelectorAll(".mermaid-error")];
  return JSON.stringify({
    href: location.href, mermaid: m.length,
    ok: m.filter(e=>e.querySelector("svg")&&!e.classList.contains("mermaid-error")).length,
    errs: errs.length,
    errTexts: errs.map(e=>e.textContent.trim().split("\\n").slice(0,3).join(" | ").slice(0,200)),
    sizes: m.map(e=>{const s=e.querySelector("svg"); return s? [Math.round(s.getBoundingClientRect().width), Math.round(s.getBoundingClientRect().height)] : null}),
    brokenImgs: [...document.images].filter(i=>i.complete&&i.naturalWidth===0).map(i=>i.src),
    imgs: [...document.images].filter(i=>i.src.includes("/img/in-post/")).map(i=>[i.src.split("/").pop(), i.naturalWidth, i.naturalHeight]),
    widePre: [...document.querySelectorAll("pre")].filter(p=>p.scrollWidth>p.clientWidth+2).map(p=>p.textContent.trim().split("\\n")[0].slice(0,60)),
    pending: m.filter(e=>!e.querySelector("svg")&&!e.classList.contains("mermaid-error")).length + [...document.images].filter(i=>!i.complete).length
  });
})()`;

async function check(slug) {
  const url = `${SITE}/${slug}.html`;
  const t = await getJSON(`${CDP}/json/new?${encodeURIComponent(url)}`, 'PUT');
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  let id = 0; const pending = {};
  const send = (method, params = {}) => new Promise(r => { pending[++id] = r; ws.send(JSON.stringify({ id, method, params })); });
  await new Promise(r => ws.on('open', r));
  ws.on('message', d => { const m = JSON.parse(d); if (m.id && pending[m.id]) { pending[m.id](m.result); delete pending[m.id]; } });
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  let out;
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 1500));
    const r = await send('Runtime.evaluate', { expression: CHECK, returnByValue: true });
    out = JSON.parse(r.result.value);
    if (out.pending === 0 && !out.href.includes('about:blank')) break;
  }
  ws.close();
  await getJSON(`${CDP}/json/close/${t.id}`);
  const status = out.href.endsWith(`/${slug}.html`) && !out.errs && out.ok === out.mermaid && !out.brokenImgs.length ? 'PASS' : 'FAIL';
  console.log(`\n== ${slug} [${status}]`);
  console.log(`mermaid=${out.mermaid} ok=${out.ok} errs=${out.errs} pending=${out.pending} brokenImgs=${out.brokenImgs.length}`);
  if (out.errTexts.length) console.log('errTexts:', out.errTexts);
  console.log('sizes:', JSON.stringify(out.sizes));
  if (out.imgs.length) console.log('imgs:', JSON.stringify(out.imgs));
  if (out.widePre.length) console.log(`widePre(${out.widePre.length}):`, JSON.stringify(out.widePre));
  return status === 'PASS';
}

(async () => {
  let all = true;
  for (const s of process.argv.slice(2)) all = (await check(s)) && all;
  process.exit(all ? 0 : 1);
})().catch(e => { console.error(e); process.exit(2); });
