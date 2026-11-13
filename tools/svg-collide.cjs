#!/usr/bin/env node
// Usage: node svg-collide.cjs <file.svg> [more...]
// Loads each SVG in Chrome (CDP :9222), getBBox() every <text>, fails when a text
//  (a) leaves its enclosing <rect> (smallest rect containing the text bbox),
//  (b) intersects another <text> bbox,
//  (c) is crossed by a <line>/<path> segment (sampled points),
//  (d) leaves the viewBox,
//  (e) ends an arrow 4–60px short of every box edge (dangling arrowhead);
//      axis / direction arrows that end in open space carry data-axis.
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
  const inv=svg.getScreenCTM().inverse();
  const mm=e=>{try{return inv.multiply(e.getScreenCTM())}catch{return null}};
  const xf=(m,px,py)=>[px*m.a+py*m.c+m.e,px*m.b+py*m.d+m.f];
  const bb=e=>{try{
    const b=e.getBBox(),m=mm(e);
    if(!m)return{x0:b.x,y0:b.y,x1:b.x+b.width,y1:b.y+b.height};
    const xs=[],ys=[];
    for(const[px,py]of[[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]]){
      const t=xf(m,px,py);xs.push(t[0]);ys.push(t[1]);
    }
    return{x0:Math.min(...xs),y0:Math.min(...ys),x1:Math.max(...xs),y1:Math.max(...ys)};
  }catch{return null;}};
  const hit=(a,b,p=2)=>a.x0<b.x1+p&&a.x1>b.x0-p&&a.y0<b.y1+p&&a.y1>b.y0-p;
  const texts=[
    ...[...svg.querySelectorAll('text')].filter(t=>!t.closest('g.mathtext')).map(t=>({t:t.textContent.trim().replace(/\\s+/g,' ').slice(0,40),b:bb(t),cls:t.getAttribute('class')||''})),
    ...[...svg.querySelectorAll('g.mathtext')].map(g=>({t:(g.getAttribute('aria-label')||g.getAttribute('data-tex')||'?').replace(/\\s+/g,' ').slice(0,40),b:bb(g),cls:'mathtext'}))
  ].filter(t=>t.t&&t.b);
  const rectEls=[...svg.querySelectorAll('rect')].filter(r=>!r.hasAttribute('data-band')&&!r.closest('g.mathtext'));
  const rects=rectEls.map(r=>bb(r)); rects.forEach((b,i)=>{if(!b) rectEls[i]=null}); const rects2=rects.map((b,i)=>({b,el:rectEls[i]})).filter(o=>o.b);
  const rectList=rects2.map(o=>o.b);
  // boxes an arrow may land on: visible rects that are not the page background
  const boxRects=rects2.filter(o=>{const f=(o.el.getAttribute('fill')||getComputedStyle(o.el).fill||'');
    if(/^(transparent|none)$/.test(f)||/rgba\(0, 0, 0, 0\)/.test(f)) return false;
    return !((o.b.x1-o.b.x0)>=vb.width*0.95&&(o.b.y1-o.b.y0)>=vb.height*0.9);}).map(o=>o.b);
  // closed, filled <path> shapes (rounded cards drawn as paths by generators) are boxes too
  [...svg.querySelectorAll('path')].filter(p=>!p.closest('defs,marker,g.mathtext')&&/[Zz]\\s*$/.test(p.getAttribute('d')||'')
    &&!/^(none|transparent)$/.test(getComputedStyle(p).fill)&&!/rgba\\(0, 0, 0, 0\\)/.test(getComputedStyle(p).fill))
    .forEach(p=>{const b=bb(p);if(b&&(b.x1-b.x0)>12&&(b.y1-b.y0)>12&&!((b.x1-b.x0)>=vb.width*0.95&&(b.y1-b.y0)>=vb.height*0.9))boxRects.push(b);});
  const errs=[];
  for(const t of texts){
    if(t.b.x0<vb.x-1||t.b.y0<vb.y-1||t.b.x1>vb.x+vb.width+1||t.b.y1>vb.y+vb.height+1) errs.push('viewBox 越界: '+t.t);
    // enclosing rect = smallest rect containing the text's centre (not the whole
    // bbox — text often overflows the box it labels, which is exactly what we catch)
    const cx=(t.b.x0+t.b.x1)/2, cy=(t.b.y0+t.b.y1)/2;
    const host=rects2.filter(o=>o.b.x0<=cx&&cx<=o.b.x1&&o.b.y0<=cy&&cy<=o.b.y1)
      .sort((a,b)=>(a.b.x1-a.b.x0)*(a.b.y1-a.b.y0)-(b.b.x1-b.b.x0)*(b.b.y1-b.b.y0))[0];
    if(host){
      const r=host.b;
      const isbg=r.x0<=vb.x+2&&r.y0<=vb.y+2&&r.x1>=vb.x+vb.width-2&&r.y1>=vb.y+vb.height-2;
      if(!isbg&&(t.b.x0<r.x0+3||t.b.x1>r.x1-3||t.b.y0<r.y0+3||t.b.y1>r.y1-3)) errs.push('文字出框: '+t.t);
      // label vertically centered in its box (text bbox centre vs rect centre ≤1.5px);
      // a container rect holding several independent labels opts out with data-no-center
      if(!isbg&&!host.el.hasAttribute('data-no-center')&&Math.abs(cy-(r.y0+r.y1)/2)>1.5&&(r.y1-r.y0)>=(t.b.y1-t.b.y0)+4)
        errs.push('文字未垂直居中: '+t.t);
    }
  }
  for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++)
    if(hit(texts[i].b,texts[j].b)) errs.push('文字相交: '+texts[i].t+' × '+texts[j].t);
  // text overlapping a rect that is NOT its host (e.g. activation bars): depth >3px on any side
  for(const t of texts){
    const cx=(t.b.x0+t.b.x1)/2, cy=(t.b.y0+t.b.y1)/2;
    for(const r of rectList){
      const inside=cx>=r.x0&&cx<=r.x1&&cy>=r.y0&&cy<=r.y1;
      if(inside) continue;
      const dx=Math.min(t.b.x1,r.x1)-Math.max(t.b.x0,r.x0), dy=Math.min(t.b.y1,r.y1)-Math.max(t.b.y0,r.y0);
      if(dx>3&&dy>3) { errs.push('文字压非宿主框: '+t.t); break; }
    }
  }
  const segs=[];
  const dashed=e=>getComputedStyle(e).strokeDasharray!=='none';
  [...svg.querySelectorAll('line')].forEach(l=>{if(dashed(l)||/lifeline/.test(l.getAttribute('class')||'')) return; const m=mm(l); if(!m) return; const a=xf(m,+l.getAttribute('x1'),+l.getAttribute('y1')),b=xf(m,+l.getAttribute('x2'),+l.getAttribute('y2'));segs.push([a,b]);});
  [...svg.querySelectorAll('path')].forEach(p=>{
    if(p.closest('defs')||p.closest('marker')||p.closest('g.mathtext')||dashed(p)) return;
    const L=p.getTotalLength(); if(!L) return;
    const m=mm(p); const n=Math.max(2,Math.ceil(L/20)); let prev=null;
    for(let k=0;k<=n;k++){const pt=p.getPointAtLength(L*k/n); const cur=m?xf(m,pt.x,pt.y):[pt.x,pt.y]; if(prev) segs.push([prev,cur]); prev=cur;}
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
  // marker-less <line>s (lifelines, axes, bus bars) an arrow may end on
  const plainLines=[...svg.querySelectorAll('line')].filter(l=>!l.closest('defs,marker,g.mathtext')&&!l.getAttribute('marker-end')&&getComputedStyle(l).markerEnd==='none').map(l=>{const m=mm(l);if(!m)return null;
    const a=xf(m,+l.getAttribute('x1'),+l.getAttribute('y1')),b=xf(m,+l.getAttribute('x2'),+l.getAttribute('y2'));return [a[0],a[1],b[0],b[1]];}).filter(Boolean);
  // arrows: visible shaft (drawn length minus head) ≥16px
  [...svg.querySelectorAll('line,path')].forEach(p=>{
    if(p.closest('defs')||p.closest('marker')||p.closest('g.mathtext')) return;
    const cs=getComputedStyle(p), cssRef=v=>(v&&v!=='none')?v:null;
    const mend=p.getAttribute('marker-end')||cssRef(cs.markerEnd), mstart=p.getAttribute('marker-start')||cssRef(cs.markerStart);
    if(!mend&&!mstart&&!/url\\(#/.test(p.getAttribute('style')||'')) return;
    let L=0; try{L=p.getTotalLength()}catch{return;}
    let head=0;
    for(const mref of[mend,mstart]){
      if(!mref) continue;
      const id=(mref.match(/#([^')"]+)/)||[])[1]; if(!id) continue;
      const mk=svg.querySelector('marker#'+CSS.escape(id)); if(!mk) continue;
      const w=+(mk.getAttribute('markerWidth')||3), mh=+(mk.getAttribute('markerHeight')||3);
      // drawn head length: markerWidth in user units, × stroke-width unless markerUnits=userSpaceOnUse
      const user=(mk.getAttribute('markerUnits')||'strokeWidth')==='userSpaceOnUse';
      head=Math.max(head,Math.max(w,mh)*(user?1:(parseFloat(cs.strokeWidth)||1)));
    }
    if(L-head<16) {const a=p.getPointAtLength(0),z=p.getPointAtLength(L);errs.push('箭头杆过短('+Math.round(L-head)+'px): ('+Math.round(a.x)+','+Math.round(a.y)+')→('+Math.round(z.x)+','+Math.round(z.y)+')');}
    // (e) dangling arrow (2026-11-12, 「箭头没有连接上方框」): the arrowhead end
    // must touch a box edge — within 4px of a visible rect's border — or a
    // label it points at. 4–60px from the nearest box means a gap was left.
    // axis / direction arrows (a time axis, 「→ 分片组」 over a row) end in open
    // space on purpose; they opt out with data-axis
    const m=mm(p); if(!m||p.hasAttribute('data-axis')) return;
    for(const [mref,at] of [[mend,L],[mstart,0]]){
      if(!mref) continue;
      const q=p.getPointAtLength(at); const [px,py]=xf(m,q.x,q.y);
      let d=1e9;
      for(const b of boxRects){
        const ox=Math.max(b.x0-px,0,px-b.x1), oy=Math.max(b.y0-py,0,py-b.y1);
        const inside=!ox&&!oy;
        d=Math.min(d, inside?Math.min(px-b.x0,b.x1-px,py-b.y0,b.y1-py):Math.hypot(ox,oy));
      }
      const nearText=texts.some(t=>px>t.b.x0-8&&px<t.b.x1+8&&py>t.b.y0-8&&py<t.b.y1+8);
      // sequence-style figures: an arrow may end on another line (a lifeline)
      const onLine=plainLines.some(([x1,y1,x2,y2])=>{const dx=x2-x1,dy=y2-y1,l2=dx*dx+dy*dy||1;
        const t=Math.max(0,Math.min(1,((px-x1)*dx+(py-y1)*dy)/l2));return Math.hypot(px-x1-t*dx,py-y1-t*dy)<=4;});
      if(onLine) continue;
      // the head's tip overhangs the path end by (tip − refX) marker units
      let ext=0;try{const id=(mref.match(/#([^')"]+)/)||[])[1];const mk=svg.querySelector('marker#'+CSS.escape(id));
        const mw=+(mk.getAttribute('markerWidth')||3),rx=+(mk.getAttribute('refX')||0);const vbm=(mk.getAttribute('viewBox')||'').split(/[ ,]+/).map(Number);
        const sw=parseFloat(cs.strokeWidth)||1;const shape=mk.querySelector('path,polygon');let tip=vbm.length===4?vbm[0]+vbm[2]:mw;
        if(shape){const b2=shape.getBBox();tip=at?b2.x+b2.width:b2.x;}
        ext=Math.abs(tip-rx)*(vbm.length===4?mw/vbm[2]:1)*((mk.getAttribute('markerUnits')||'strokeWidth')==='userSpaceOnUse'?1:sw);}catch{}
      d-=ext;
      if(d>4&&d<60&&!nearText) errs.push('箭头悬空（端点离最近方框 '+Math.round(d)+'px）: 端点 ('+Math.round(q.x)+','+Math.round(q.y)+')');
    }
  });
  return JSON.stringify({errs:[...new Set(errs)],ntexts:texts.length});
})()`;
async function check(file) {
  const url = 'file://' + path.resolve(file);
  let t;
  try { t = await getJSON(`${CDP}/json/new?${encodeURIComponent(url)}`, 'PUT'); }
  catch (e) {
    console.log(`== ${file} [FAIL] cannot reach Chrome at ${CDP} — start Chrome with --remote-debugging-port or set CDP=<url>`);
    return false;
  }
  if (!t || !t.webSocketDebuggerUrl) {
    console.log(`== ${file} [FAIL] no debuggable tab from ${CDP} — set CDP=<url> (and SITE=<url> if the SVG is served, not a file)`);
    return false;
  }
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
    if (r.exceptionDetails) {
      const t = r.exceptionDetails.exception && r.exceptionDetails.exception.description;
      out = { err: 'eval: ' + (t || r.exceptionDetails.text || 'error') };
      continue;
    }
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
