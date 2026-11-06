// for each <text>, find host rect (smallest rect containing centre) and report deficit
const http=require('http'),path=require('path');
let WebSocket; try{WebSocket=require('ws');}catch{WebSocket=require(require('os').homedir()+'/.claude/skills/browser/node_modules/ws');}
const CDP=(process.env.CDP||'http://localhost:9222').replace(/\/$/,'');
const getJSON=(u,m='GET')=>new Promise((res,rej)=>{const r=http.request(u,{method:m},x=>{let d='';x.on('data',c=>d+=c);x.on('end',()=>{try{res(JSON.parse(d))}catch{res(d)}})});r.on('error',rej);r.end()});
const CHECK=`(()=>{
 const svg=document.querySelector('svg');const vb=svg.viewBox.baseVal;
 const bb=e=>{try{const b=e.getBBox();return{x0:b.x,y0:b.y,x1:b.x+b.width,y1:b.y+b.height}}catch{return null}};
 const texts=[...svg.querySelectorAll('text')].map(t=>({t:t.textContent.trim().replace(/\\s+/g,' ').slice(0,40),b:bb(t)})).filter(t=>t.t&&t.b);
 const raws=[...svg.querySelectorAll('rect')];
 const rects=raws.map(r=>bb(r)).filter(Boolean);
 const out=[];
 for(const t of texts){
   const cx=(t.b.x0+t.b.x1)/2, cy=(t.b.y0+t.b.y1)/2;
   const cand=rects.map((r,i)=>({r,i})).filter(o=>o.r.x0<=cx&&cx<=o.r.x1&&o.r.y0<=cy&&cy<=o.r.y1);
   cand.sort((a,b)=>((a.r.x1-a.r.x0)*(a.r.y1-a.r.y0))-((b.r.x1-b.r.x0)*(b.r.y1-b.r.y0)));
   if(!cand.length)continue;
   const host=cand[0];
   const full=rects[host.i];
   const isbg=full.x0<=vb.x+2&&full.y0<=vb.y+2&&full.x1>=vb.x+vb.width-2&&full.y1>=vb.y+vb.height-2;
   if(isbg)continue;
   const dl=host.r.x0+3-t.b.x0, dr=t.b.x1-(host.r.x1-3), dt=host.r.y0+3-t.b.y0, db=t.b.y1-(host.r.y1-3);
   if(dl>0||dr>0||dt>0||db>0){
     const el=raws[host.i];
     out.push(t.t+' | deficit L'+Math.max(0,dl).toFixed(1)+' R'+Math.max(0,dr).toFixed(1)+' T'+Math.max(0,dt).toFixed(1)+' B'+Math.max(0,db).toFixed(1)+' | rect '+el.getAttribute('x')+','+el.getAttribute('y')+','+el.getAttribute('width')+','+el.getAttribute('height'));
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
