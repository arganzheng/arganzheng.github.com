#!/usr/bin/env node
// Usage: node svg-collide.cjs <file.svg> [more...]
// Loads each SVG in Chrome (CDP :9222), getBBox() every <text>, fails when a text
//  (a) leaves its enclosing <rect> (smallest rect containing the text bbox),
//  (b) intersects another <text> bbox,
//  (c) is crossed by a <line>/<path> segment (sampled points),
//  (d) leaves the viewBox.
// Env: CDP (default http://localhost:9222).
const http = require('http');
const fs = require('fs');
const path = require('path');
let WebSocket;
try { WebSocket = require('ws'); } catch { WebSocket = require(require('os').homedir() + '/.claude/skills/browser/node_modules/ws'); }
const CDP = (process.env.CDP || 'http://localhost:9222').replace(/\/$/, '');
const getJSON = (url, method = 'GET') => new Promise((res, rej) => {
  const req = http.request(url, { method }, r => { let d = ''; r.on('data', c => d += c); r.on('end', () => { try { res(JSON.parse(d)); } catch { res(d); } }); });
  req.on('error', rej); req.end();
});
const CHECK = `(() => {
  const svg=document.querySelector('svg'); if(!svg) return JSON.stringify({err:'no svg'});
  const vb=svg.viewBox.baseVal;
  const bb=e=>{try{const b=e.getBBox();return{x0:b.x,y0:b.y,x1:b.x+b.width,y1:b.y+b.height};}catch{return null;}};
  const hit=(a,b,p=2)=>a.x0<b.x1+p&&a.x1>b.x0-p&&a.y0<b.y1+p&&a.y1>b.y0-p;
  const texts=[...svg.querySelectorAll('text')].map(t=>({t:t.textContent.trim().replace(/\\s+/g,' ').slice(0,40),b:bb(t),cls:t.getAttribute('class')||''})).filter(t=>t.t&&t.b);
  const rects=[...svg.querySelectorAll('rect')].map(r=>bb(r)).filter(Boolean);
  const errs=[];
  for(const t of texts){
    if(t.b.x0<vb.x-1||t.b.y0<vb.y-1||t.b.x1>vb.x+vb.width+1||t.b.y1>vb.y+vb.height+1) errs.push('viewBox 越界: '+t.t);
    // enclosing rect = smallest rect containing the text's centre (not the whole
    // bbox — text often overflows the box it labels, which is exactly what we catch)
    const cx=(t.b.x0+t.b.x1)/2, cy=(t.b.y0+t.b.y1)/2;
    const host=rects.filter(r=>r.x0<=cx&&cx<=r.x1&&r.y0<=cy&&cy<=r.y1)
      .sort((a,b)=>(a.x1-a.x0)*(a.y1-a.y0)-(b.x1-b.x0)*(b.y1-b.y0))[0];
    if(host){
      const full=rects.find(r=>r===host)||host;
      const isbg=full.x0<=vb.x+2&&full.y0<=vb.y+2&&full.x1>=vb.x+vb.width-2&&full.y1>=vb.y+vb.height-2;
      if(!isbg&&(t.b.x0<host.x0+3||t.b.x1>host.x1-3||t.b.y0<host.y0+3||t.b.y1>host.y1-3)) errs.push('文字出框: '+t.t);
    }
  }
  for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++)
    if(hit(texts[i].b,texts[j].b)) errs.push('文字相交: '+texts[i].t+' × '+texts[j].t);
  // text overlapping a rect that is NOT its host (e.g. activation bars): depth >3px on any side
  for(const t of texts){
    const cx=(t.b.x0+t.b.x1)/2, cy=(t.b.y0+t.b.y1)/2;
    for(const r of rects){
      const inside=cx>=r.x0&&cx<=r.x1&&cy>=r.y0&&cy<=r.y1;
      if(inside) continue;
      const dx=Math.min(t.b.x1,r.x1)-Math.max(t.b.x0,r.x0), dy=Math.min(t.b.y1,r.y1)-Math.max(t.b.y0,r.y0);
      if(dx>3&&dy>3) { errs.push('文字压非宿主框: '+t.t); break; }
    }
  }
  const segs=[];
  [...svg.querySelectorAll('line')].forEach(l=>{if(l.getAttribute('stroke-dasharray')||/lifeline/.test(l.getAttribute('class')||'')) return; const x1=+l.getAttribute('x1'),y1=+l.getAttribute('y1'),x2=+l.getAttribute('x2'),y2=+l.getAttribute('y2');segs.push([[x1,y1],[x2,y2]]);});
  [...svg.querySelectorAll('path')].forEach(p=>{
    if(p.closest('defs')||p.closest('marker')) return;
    const L=p.getTotalLength(); if(!L) return;
    const n=Math.max(2,Math.ceil(L/20)); let prev=null;
    for(let k=0;k<=n;k++){const pt=p.getPointAtLength(L*k/n); const cur=[pt.x,pt.y]; if(prev) segs.push([prev,cur]); prev=cur;}
  });
  const cls=(a,b,c,d)=>{ // segment a-b vs rect c-d (Liang-Barsky)
    const dx=b[0]-a[0], dy=b[1]-a[1]; let t0=0,t1=1;
    const p=[-dx,dx,-dy,dy], q=[a[0]-c.x0,c.x1-a[0],a[1]-c.y0,c.y1-a[1]];
    for(let i=0;i<4;i++){ if(p[i]===0){ if(q[i]<0) return false; } else { const r=q[i]/p[i]; if(p[i]<0){ if(r>t1) return false; if(r>t0) t0=r; } else { if(r<t0) return false; if(r<t1) t1=r; } } }
    return true;
  };
  for(const t of texts){
    const pad={x0:t.b.x0+2,y0:t.b.y0+2,x1:t.b.x1-2,y1:t.b.y1-2};
    for(const [a,b] of segs){
      const insideA=a[0]>=pad.x0&&a[0]<=pad.x1&&a[1]>=pad.y0&&a[1]<=pad.y1;
      const insideB=b[0]>=pad.x0&&b[0]<=pad.x1&&b[1]>=pad.y0&&b[1]<=pad.y1;
      if((insideA||insideB)&&!(insideA&&insideB)) continue; // endpoint labels are fine when arrow starts/ends at text edge? treat crossing only
      if(cls(a,b,pad)) errs.push('线段穿文字: '+t.t);
    }
  }
  return JSON.stringify({errs:[...new Set(errs)],ntexts:texts.length});
})()`;
async function check(file) {
  const url = 'file://' + path.resolve(file);
  const t = await getJSON(`${CDP}/json/new?${encodeURIComponent(url)}`, 'PUT');
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  let id = 0; const pending = {};
  const send = (m, p = {}) => new Promise(r => { pending[++id] = r; ws.send(JSON.stringify({ id, method: m, params: p })); });
  await new Promise(r => ws.on('open', r));
  ws.on('message', d => { const m = JSON.parse(d); if (m.id && pending[m.id]) { pending[m.id](m.result); delete pending[m.id]; } });
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 1400, deviceScaleFactor: 1, mobile: false });
  let out;
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 400));
    const r = await send('Runtime.evaluate', { expression: CHECK, returnByValue: true });
    out = JSON.parse(r.result.value);
    if (out.ntexts || out.err) break;
  }
  ws.close();
  await getJSON(`${CDP}/json/close/${t.id}`);
  if (out.err) { console.log(`== ${file} [FAIL] ${out.err}`); return false; }
  console.log(`== ${file} [${out.errs.length ? 'FAIL' : 'PASS'}] texts=${out.ntexts}`);
  out.errs.forEach(e => console.log('   ' + e));
  return !out.errs.length;
}
(async () => {
  const files = process.argv.slice(2);
  let ok = true;
  for (const f of files) ok = (await check(f)) && ok;
  process.exit(ok ? 0 : 1);
})();
