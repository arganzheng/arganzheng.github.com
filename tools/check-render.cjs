#!/usr/bin/env node
// Usage: node check-render.cjs <post-slug> [more slugs...]
// Opens each page in its own Chrome tab (CDP :9222), waits for Mermaid and images, reports, closes the tab.
// Env: SITE (default http://localhost:4000), CDP (default http://localhost:9222).
// Needs the `ws` module: a local ./node_modules/ws (CI: `npm i --no-save ws`) or the browser skill's copy.
const http = require('http');
const fs = require('fs');
const { spawnSync } = require('child_process');
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
  // reveal.js decks keep every slide but the current one display:none — show them all while measuring
  const secs=[...document.querySelectorAll(".reveal .slides section")]; const saved=secs.map(s=>s.style.display); secs.forEach(s=>{ s.style.display="block"; });
  // hidden fragments would be invisible to the block checks — show them like a fully-stepped slide
  document.querySelectorAll(".fragment").forEach(f=>f.classList.add("visible"));
  const m=[...document.querySelectorAll(".mermaid")];
  const errs=[...document.querySelectorAll(".mermaid-error")];
  const out={
    href: location.href, ready: document.readyState, populated: !!(document.body && document.body.children.length), mermaid: m.length,
    ok: m.filter(e=>e.querySelector("svg")&&!e.classList.contains("mermaid-error")).length,
    errs: errs.length,
    errTexts: errs.map(e=>e.textContent.trim().split("\\n").slice(0,3).join(" | ").slice(0,200)),
    sizes: m.map(e=>{const s=e.querySelector("svg"); return s? [Math.round(s.getBoundingClientRect().width), Math.round(s.getBoundingClientRect().height)] : null}),
    brokenImgs: [...document.images].filter(i=>i.complete&&i.naturalWidth===0&&new URL(i.src).origin===location.origin).map(i=>i.src),
    brokenExternalImgs: [...document.images].filter(i=>i.complete&&i.naturalWidth===0&&new URL(i.src).origin!==location.origin).map(i=>i.src),
    imgs: [...document.images].filter(i=>i.src.includes("/img/in-post/")).map(i=>[i.src.split("/").pop(), i.naturalWidth, i.naturalHeight]),
    katexErrors: [...document.querySelectorAll(".katex-error")].map(e=>e.textContent.slice(0,80)),
    zeroImgs: [...document.images].filter(i=>i.complete&&i.naturalWidth>0&&i.getBoundingClientRect().width<2&&i.src.includes("/img/in-post/")).map(i=>i.src.split("/").pop()),
    rawMd: [...document.querySelectorAll("td, p, li")].filter(e=>{
      if(e.closest("pre")) return false;
      const clone=e.cloneNode(true);
      clone.querySelectorAll("code").forEach(c=>c.remove());
      return /\\]\\(\\/img\\/in-post\\//.test(clone.textContent);
    }).map(e=>e.textContent.trim().slice(0,80)),
    widePre: [...document.querySelectorAll("pre")].filter(p=>p.scrollWidth>p.clientWidth+2).map(p=>p.textContent.trim().split("\\n")[0].slice(0,60)),
    hoverflow: secs.filter(s=>!s.querySelector("section")).map((s,i)=>{const sr=s.getBoundingClientRect();
      return [...s.querySelectorAll('*')].filter(e=>{
        if(e.closest('.katex-mathml'))return false;               // hidden MathML annotation
        if(e.tagName.toLowerCase()!=='svg'&&e.closest('svg'))return false; // measure the svg, not its children
        const cs=getComputedStyle(e);
        if(cs.opacity==='0'||cs.visibility==='hidden')return false;
        const b=e.getBoundingClientRect();return b.width>0&&(b.right>sr.right+2||b.left<sr.left-2);
      }).map(e=>{const b=e.getBoundingClientRect();
        return "#"+(i+1)+" "+e.tagName+" "+e.textContent.trim().slice(0,30)+" (right "+Math.round(b.right-sr.right)+"px over)";});}).flat(),
    overflow: secs.filter(s=>!s.querySelector("section")).map((s,i)=>[i+1,s.scrollHeight,(s.querySelector("h1,h2,h3")||s).textContent.trim().slice(0,40)])
      .filter(x=>x[1]>720).map(x=>"#"+x[0]+" "+x[1]+"px "+x[2]),
    // visible leaf blocks that physically collide (table row over bullet, figure over heading):
    // intersect >2 px both ways. Descendants, not just direct children — a bullet's <li> and a
    // table's <tr> never appear at section level. Tables are compared by row rects.
    overlap: secs.filter(s=>!s.querySelector("section")).map((s,i)=>{
      const BS='p,ul,ol,table,tr,img,.mermaid,pre,blockquote,h1,h2,h3,figure,figcaption,li';
      const vis=e=>{
        if(e.closest('aside.notes')||e.matches('aside.notes,style,script,template')) return false;
        if(e.closest('svg')) return false;                       // measure the svg via its .mermaid host
        const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden'||cs.opacity==='0') return false;
        const b=e.getBoundingClientRect(); return b.width>0&&b.height>0;
      };
      const leafs=[...s.querySelectorAll(BS)].filter(e=>vis(e)&&!e.querySelector(BS));
      const tag=e=>e.tagName.toLowerCase()+'.'+(e.className&&e.className.baseVal===undefined?(''+e.className).split(' ')[0]:'');
      const hits=[];
      for(let a=0;a<leafs.length;a++)for(let c=a+1;c<leafs.length;c++){
        const A=leafs[a].getBoundingClientRect(),B=leafs[c].getBoundingClientRect();
        const ox=Math.min(A.right,B.right)-Math.max(A.left,B.left), oy=Math.min(A.bottom,B.bottom)-Math.max(A.top,B.top);
        if(ox>2&&oy>2) hits.push("#"+(i+1)+" "+tag(leafs[a])+"/"+tag(leafs[c])+" ("+Math.round(ox)+"x"+Math.round(oy)+"px)");
      }
      return hits;
    }).flat(),
    // mermaid / inline svg text rendered below ~14 px is unreadable at 1280x720.
    // getBoundingClientRect() on foreignObject children lies when the svg is height-bound
    // (max-height + preserveAspectRatio=meet): it reports the label's local-space layout
    // height, not the downscaled render. Use computed font-size x the svg's effective
    // content scale = min(renderedW/vbW, renderedH/vbH) instead — both dims can bind.
    smalltext: secs.filter(s=>!s.querySelector("section")).map((s,i)=>{
      return [...s.querySelectorAll('.mermaid svg, section > svg')].map(sv=>{
        const vb=(sv.getAttribute('viewBox')||'').split(/[ \t]+/).map(Number);
        if(vb.length<4||!vb[2]||!vb[3]) return null;
        const r=sv.getBoundingClientRect();
        const sc=Math.min(r.width/vb[2], r.height/vb[3]);
        if(!(sc>0)) return null;
        const hs=[...sv.querySelectorAll('text,tspan,foreignObject div,foreignObject span,foreignObject p')]
          .map(e=>{const h=e.getBoundingClientRect().height; if(h<=0) return null;
            const fs=parseFloat(getComputedStyle(e).fontSize)||16; return fs*sc;})
          .filter(x=>x!=null);
        const min=Math.min(...hs,999);
        return (min<14&&min<999)?"#"+(i+1)+" min "+min.toFixed(1)+"px":null;
      }).filter(Boolean);
    }).flat(),
    // consecutive visible block children crammed together: <8 px of air between a
    // block's bottom edge and the next block's top edge means the slide reads as
    // one wall (e.g. table's last row touching the first bullet)
    gap: secs.filter(s=>!s.querySelector("section")).map((s,i)=>{
      const D='p,ul,ol,table,img,.mermaid,pre,blockquote,h1,h2,h3,figure';
      const kids=[...s.children].filter(e=>e.matches(D)&&!(e.closest('aside.notes')||e.matches('aside.notes')))
        .filter(e=>{const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden'||cs.opacity==='0')return false;
          const b=e.getBoundingClientRect(); return b.width>0&&b.height>0;})
        .sort((a,b)=>a.getBoundingClientRect().top-b.getBoundingClientRect().top);
      const hits=[];
      for(let k=0;k+1<kids.length;k++){
        const A=kids[k].getBoundingClientRect(),B=kids[k+1].getBoundingClientRect();
        const g=B.top-A.bottom;
        if(g>-2&&g<8) hits.push("#"+(i+1)+" "+kids[k].tagName.toLowerCase()+"/"+kids[k+1].tagName.toLowerCase()+" "+g.toFixed(0)+"px");
      }
      return hits;
    }).flat(),
    // <img> svg squeezed below 55% of natural width: its text shrinks proportionally
    shrunk: secs.filter(s=>!s.querySelector("section")).map((s,i)=>{
      return [...s.querySelectorAll('img')].filter(im=>im.src.endsWith('.svg')&&im.naturalWidth>0)
        .map(im=>{const r=im.getBoundingClientRect().width/im.naturalWidth;
          return r<0.55?"#"+(i+1)+" "+im.src.split('/').pop()+" "+(r*100).toFixed(0)+"%":null;}).filter(Boolean);
    }).flat(),
    pending: m.filter(e=>!e.querySelector("svg")&&!e.classList.contains("mermaid-error")).length + [...document.images].filter(i=>!i.complete).length
      + document.querySelectorAll("pre code.language-mermaid, div.language-mermaid pre code").length
  };
  secs.forEach((s,i)=>{ s.style.display=saved[i]; });
  return JSON.stringify(out);
})()`;

// dark pass: flip the deck to dark theme and measure WCAG contrast of every
// visible text element against its effective background. Mermaid is excluded —
// its light backing card is intentional (nodes stay dark-on-light). Inline code,
// table cells, links and blockquote text are the usual casualties.
const DARK = `(() => {
  const secs=[...document.querySelectorAll(".reveal .slides section")]; const saved=secs.map(s=>s.style.display); secs.forEach(s=>{ s.style.display="block"; });
  document.querySelectorAll(".fragment").forEach(f=>f.classList.add("visible"));
  const lum=c=>{const f=v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4)};return 0.2126*f(c[0])+0.7152*f(c[1])+0.0722*f(c[2]);};
  const rgba=cs=>{const m=cs.match(/rgba?\\((\\d+)[, ]+(\\d+)[, ]+(\\d+)(?:[, /]+([\\d.]+))?\\)/);return m?{c:[+m[1],+m[2],+m[3]],a:m[4]==null?1:+m[4]}:null;};
  const bg=e=>{let n=e;while(n&&n!==document.documentElement){const r=rgba(getComputedStyle(n).backgroundColor);if(r&&r.a>0.5)return r.c;n=n.parentElement;}return [255,255,255];};
  const out=secs.filter(s=>!s.querySelector("section")).map((s,i)=>{
    return [...s.querySelectorAll('p,li,td,th,h1,h2,h3,code,span,a,blockquote,strong,b,em,dt,dd,figcaption')].map(e=>{
      if(e.closest('aside.notes,style,script,template,.mermaid,svg')||e.closest('svg'))return null;
      const cs=getComputedStyle(e); if(cs.display==='none'||cs.visibility==='hidden'||cs.opacity==='0')return null;
      const b=e.getBoundingClientRect(); if(b.width<=0||b.height<=0)return null;
      if(!(e.textContent||'').trim())return null;
      const fg=rgba(cs.color); if(!fg)return null;
      const L1=lum(fg.c),L2=lum(bg(e)); const ratio=(Math.max(L1,L2)+0.05)/(Math.min(L1,L2)+0.05);
      return ratio<4.5?"#"+(i+1)+" "+e.tagName.toLowerCase()+" '"+(e.textContent||'').trim().slice(0,24)+"' "+ratio.toFixed(1)+":1":null;
    }).filter(Boolean);
  }).flat();
  secs.forEach((s,i)=>{ s.style.display=saved[i]; });
  return JSON.stringify(out);
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
  // never judge a stale cached svg / blog.min.js (the tab was opened before this ran, so reload once uncached)
  await send('Network.enable'); await send('Network.setCacheDisabled', { cacheDisabled: true });
  await send('Page.reload', { ignoreCache: true });
  let out;
  // up to 90 s: mermaid.min.js (2.4 MB) comes from a CDN and the cache is off
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 1500));
    const r = await send('Runtime.evaluate', { expression: CHECK, returnByValue: true });
    out = JSON.parse(r.result.value);
    // a snapshot taken while the reload is still in flight has no images and no diagrams — keep waiting
    if (out.pending === 0 && out.ready === 'complete' && out.populated && !out.href.includes('about:blank')) break;
  }
  // dark pass: contrast < 4.5 on visible text -> FAIL (decks only; posts keep their own theme handling)
  let contrast = [];
  if (slug.startsWith('slides/')) {
    await send('Runtime.evaluate', { expression: `const __s=document.createElement('style');__s.id='cr-notrans';__s.textContent='*{transition:none!important}';document.head.appendChild(__s);document.documentElement.setAttribute('data-theme','dark');document.body&&document.body.setAttribute('data-theme','dark');document.querySelectorAll('link[rel=stylesheet]').forEach(l=>{if(/theme\\/(white|black)\\.css/.test(l.href)) l.href=l.href.replace(/theme\\/(white|black)/,'theme/black')})` });
    // the swapped-in black.css needs a beat to load and apply
    for (let i = 0; i < 20; i++) {
      await new Promise(r => setTimeout(r, 400));
      const r = await send('Runtime.evaluate', { expression: `getComputedStyle(document.body).backgroundColor`, returnByValue: true });
      if (r.result && r.result.value && !/255/.test(r.result.value)) break;
    }
    const r = await send('Runtime.evaluate', { expression: DARK, returnByValue: true });
    try { contrast = JSON.parse(r.result.value) || []; } catch { contrast = []; }
    await send('Runtime.evaluate', { expression: `document.getElementById('cr-notrans')&&document.getElementById('cr-notrans').remove();document.documentElement.setAttribute('data-theme','light');document.body&&document.body.setAttribute('data-theme','light');document.querySelectorAll('link[rel=stylesheet]').forEach(l=>{if(/theme\\/(white|black)\\.css/.test(l.href)) l.href=l.href.replace(/theme\\/(white|black)/,'theme/white')})` });
  }
  ws.close();
  await getJSON(`${CDP}/json/close/${t.id}`);
  // static mermaid lint for slide decks: slides/<key>/play -> slides/*-<key>.md
  let mlint = [];
  if (slug.startsWith('slides/')) {
    const key = slug.split('/')[1];
    const files = fs.readdirSync('slides').filter(f => f.endsWith(`-${key}.md`));
    if (files.length === 1) {
      try {
        const r = spawnSync('python3', ['tools/mermaid-lint.py', `slides/${files[0]}`], { encoding: 'utf8' });
        mlint = (r.stdout || '').split('\n').filter(l => l.includes(':mermaid#'));
        const r2 = spawnSync('python3', ['tools/deck-lint.py', `slides/${files[0]}`], { encoding: 'utf8' });
        mlint = mlint.concat((r2.stdout || '').split('\n').filter(l => l.includes(': concl-')));
      } catch { /* lint unavailable — skip */ }
    }
  }
  const status = out.href.endsWith(`/${slug}.html`) && !out.errs && !out.pending && out.ok === out.mermaid && !out.brokenImgs.length && !out.zeroImgs.length && !out.katexErrors.length && !out.rawMd.length && !out.overflow.length && !out.hoverflow.length && !out.overlap.length && !out.smalltext.length && !(out.gap||[]).length && !contrast.length && !mlint.length ? 'PASS' : 'FAIL';
  console.log(`\n== ${slug} [${status}]`);
  console.log(`mermaid=${out.mermaid} ok=${out.ok} errs=${out.errs} pending=${out.pending} brokenImgs=${out.brokenImgs.length} zeroImgs=${out.zeroImgs.length} katexErrors=${out.katexErrors.length}`);
  // reveal.js decks (slides/<name>/play): a slide taller than the 1280×720 canvas is cut off when presenting
  if (out.hoverflow.length) console.log('hoverflow (element wider than its slide — shorten the line or wrap it):', JSON.stringify(out.hoverflow));
  if (out.overflow.length) console.log('overflow (slide content taller than 720 px — split the slide or move detail to a `<!-- v -->` sub-slide):', JSON.stringify(out.overflow));
  if (out.overlap.length) console.log('overlap (two sibling blocks collide on the slide — give the slide one block or re-layout):', JSON.stringify(out.overlap));
  if (out.smalltext.length) console.log('smalltext (mermaid/svg text renders < 14 px — too wide a diagram; make it TB, split it, or use a post figure):', JSON.stringify(out.smalltext));
  if ((out.gap||[]).length) console.log('gap (consecutive blocks < 8 px apart — the slide reads as one wall; split or move a block to a `<!-- v -->` sub-slide):', JSON.stringify(out.gap));
  if (contrast.length) console.log('contrast (dark-mode text under WCAG 4.5:1 — fix the dark theme rule in _layouts/deck.html):', JSON.stringify(contrast.slice(0,12))+(contrast.length>12?` … ${contrast.length} total`:''));
  if (mlint.length) console.log('mlint (mermaid source problem — bare-id node label, isolated node, or dead-end decision):', JSON.stringify(mlint));
  if (out.shrunk.length) console.log('WARN shrunk (svg <img> under 55% of natural width — text inside it shrinks the same amount; not failing):', JSON.stringify(out.shrunk));
  if (out.pending) console.log(`pending: ${out.pending} diagram(s) / image(s) never finished loading (readyState ${out.ready}) — CDN slow or blocked; rerun before trusting anything else on this page`);
  if (out.katexErrors.length) console.log('katexErrors:', JSON.stringify(out.katexErrors));
  if (out.rawMd.length) console.log('rawMd (an image line swallowed into a table cell — a `|` in the alt text; use 「」or 竖线 instead):', JSON.stringify(out.rawMd));
  // loaded but laid out at 0 px wide: an SVG with viewBox but no width/height inside the shrink-wrapped .fig-media
  if (out.zeroImgs.length) console.log('zeroImgs (loaded, rendered 0 px wide — svg root needs width/height):', JSON.stringify(out.zeroImgs));
  if (out.errTexts.length) console.log('errTexts:', out.errTexts);
  if (out.brokenExternalImgs.length) console.log('WARN broken external images (not failing; hotlinks rot):', JSON.stringify(out.brokenExternalImgs));
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
