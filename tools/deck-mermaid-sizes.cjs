const http=require('http');const WebSocket=require(require('os').homedir()+'/.claude/skills/browser/node_modules/ws');
const getJSON=(url,method='GET')=>new Promise((res,rej)=>{const req=http.request(url,{method},r=>{let d='';r.on('data',c=>d+=c);r.on('end',()=>{try{res(JSON.parse(d))}catch(e){res(d)}})});req.on('error',rej);req.end();});
(async()=>{for(const slug of process.argv.slice(2)){
 const url='http://localhost:4000/'+slug+'.html';
 const t=await getJSON('http://localhost:9222/json/new?'+encodeURIComponent(url),'PUT');
 const ws=new WebSocket(t.webSocketDebuggerUrl);let id=0;const p={};
 const send=(m,pa={})=>new Promise(r=>{p[++id]=r;ws.send(JSON.stringify({id,method:m,params:pa}))});
 await new Promise(r=>ws.on('open',r));ws.on('message',d=>{const m=JSON.parse(d);if(m.id&&p[m.id]){p[m.id](m.result);delete p[m.id]}});
 let out;for(let i=0;i<60;i++){await new Promise(r=>setTimeout(r,1500));
  const r=await send('Runtime.evaluate',{expression:`JSON.stringify((()=>{const secs=[...document.querySelectorAll(".reveal .slides section")];secs.forEach(s=>s.style.display="block");const pend=document.querySelectorAll("pre code.language-mermaid").length;const m=[...document.querySelectorAll(".mermaid svg")].map(s=>{const vb=(s.getAttribute("viewBox")||"0 0 0 0").split(/\\s+/).map(Number);const sec=s.closest("section");const h=sec&&sec.querySelector("h1,h2,h3");return {t:h?h.textContent.trim().slice(0,30):"",w:Math.round(vb[2]),h:Math.round(vb[3])}});return {pend,ready:document.readyState,m}})())`,returnByValue:true});
  out=JSON.parse(r.result.value); if(out.pend===0&&out.ready==='complete'&&out.m.length) break;}
 console.log('== '+slug); for(const m of out.m) console.log((m.h/m.w).toFixed(2).padStart(5),String(m.w).padStart(5),String(m.h).padStart(5),' ',m.t);
 ws.close();await getJSON('http://localhost:9222/json/close/'+t.id);}})();
