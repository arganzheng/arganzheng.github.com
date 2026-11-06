// debug: print flagged text-text overlaps AND text-line crossings with coords
const http=require('http'),path=require('path');
let WebSocket; try{WebSocket=require('ws');}catch{WebSocket=require(require('os').homedir()+'/.claude/skills/browser/node_modules/ws');}
const CDP=(process.env.CDP||'http://localhost:9222').replace(/\/$/,'');
const getJSON=(u,m='GET')=>new Promise((res,rej)=>{const r=http.request(u,{method:m},x=>{let d='';x.on('data',c=>d+=c);x.on('end',()=>{try{res(JSON.parse(d))}catch{res(d)}})});r.on('error',rej);r.end()});
const CHECK=`(()=>{
 const svg=document.querySelector('svg');const vb=svg.viewBox.baseVal;
 const bb=e=>{try{const b=e.getBBox();return{x0:b.x,y0:b.y,x1:b.x+b.width,y1:b.y+b.height}}catch{return null}};
 const F=v=>Math.round(v*10)/10;
 const texts=[...svg.querySelectorAll('text')].map(t=>({t:t.textContent.trim().replace(/\\s+/g,' ').slice(0,60),b:bb(t)})).filter(t=>t.t&&t.b);
 const out=[];
 for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
   const a=texts[i].b,b=texts[j].b;
   if(a.x0<b.x1+2&&a.x1>b.x0-2&&a.y0<b.y1+2&&a.y1>b.y0-2)
     out.push('X '+texts[i].t+' @('+F(a.x0)+','+F(a.y0)+'-'+F(a.x1)+','+F(a.y1)+') ~ '+texts[j].t+' @('+F(b.x0)+','+F(b.y0)+'-'+F(b.x1)+','+F(b.y1)+')');
 }
 const segs=[];
 [...svg.querySelectorAll('line')].forEach(l=>{if(l.getAttribute('stroke-dasharray')||/lifeline/.test(l.getAttribute('class')||''))return;segs.push([[+l.getAttribute('x1'),+l.getAttribute('y1')],[+l.getAttribute('x2'),+l.getAttribute('y2')],'line '+(l.getAttribute('stroke')||'')]);});
 [...svg.querySelectorAll('path')].forEach((p,pi)=>{
   if(p.closest('defs')||p.closest('marker'))return;
   const inmath=p.closest('g.mathtext')?'MATH':'';
   const L=p.getTotalLength();if(!L)return;
   const n=Math.max(2,Math.ceil(L/20));let prev=null;
   for(let k=0;k<=n;k++){const pt=p.getPointAtLength(L*k/n);const cur=[pt.x,pt.y];if(prev)segs.push([prev,cur,'path'+(inmath?'-math':'')+'#'+pi+' '+(p.getAttribute('stroke')||p.getAttribute('fill')||'')]);prev=cur;}
 });
 const cls=(a,b,c,d)=>{const dx=b[0]-a[0],dy=b[1]-a[1];let t0=0,t1=1;
   const p=[-dx,dx,-dy,dy],q=[a[0]-c.x0,c.x1-a[0],a[1]-c.y0,c.y1-a[1]];
   for(let i=0;i<4;i++){if(p[i]===0){if(q[i]<0)return false}else{const r=q[i]/p[i];if(p[i]<0){if(r>t1)return false;if(r>t0)t0=r}else{if(r<t0)return false;if(r<t1)t1=r}}}
   return true};
 for(const t of texts){
   const pad={x0:t.b.x0+2,y0:t.b.y0+2,x1:t.b.x1-2,y1:t.b.y1-2};
   for(const [a,b,src] of segs){
     const iA=a[0]>=pad.x0&&a[0]<=pad.x1&&a[1]>=pad.y0&&a[1]<=pad.y1;
     const iB=b[0]>=pad.x0&&b[0]<=pad.x1&&b[1]>=pad.y0&&b[1]<=pad.y1;
     if((iA||iB)&&!(iA&&iB))continue;
     if(cls(a,b,pad))out.push('CROSS '+t.t+' @('+F(t.b.x0)+','+F(t.b.y0)+'-'+F(t.b.x1)+','+F(t.b.y1)+') by '+src+' seg('+F(a[0])+','+F(a[1])+'->'+F(b[0])+','+F(b[1])+')');
   }
 }
 return JSON.stringify(out);})()`;
(async()=>{
 const url='file://'+path.resolve(process.argv[2]);
 const t=await getJSON(`${CDP}/json/new?${encodeURIComponent(url)}`,'PUT');
 const ws=new WebSocket(t.webSocketDebuggerUrl);let id=0;const pend={};
 const send=(m,p={})=>new Promise(r=>{pend[++id]=r;ws.send(JSON.stringify({id,method:m,params:p}))});
 await new Promise(r=>ws.on('open',r));
 ws.on('message',d=>{const m=JSON.parse(d);if(m.id&&pend[m.id]){pend[m.id](m.result);delete pend[m.id]}});
 let out;
 for(let i=0;i<20;i++){await new Promise(r=>setTimeout(r,300));
   const r=await send('Runtime.evaluate',{expression:CHECK,returnByValue:true});
   out=JSON.parse(r.result.value); if(out.length||i>6)break;}
 console.log(out.join('\n'));
 ws.close();await getJSON(`${CDP}/json/close/${t.id}`);
})();
