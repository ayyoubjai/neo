// Offline graph overview. Layout includes every exported claim, including isolates.
const connectionColors = {
 EQUIVALENT_TO:'#c4b5fd', REFINES:'#60a5fa', SUPPORTS:'#4ade80',
 POTENTIALLY_CONTRADICTS:'#fb7185', DEPENDS_ON:'#fb923c', RELATED_TO:'#94a3b8',
 EXEMPLIFIES:'#facc15', EXPLAINS:'#22d3ee', CLARIFIES:'#e879f9', ELABORATES:'#2dd4bf',
 ABOUT:'#a78bfa', MENTIONS:'#f9a8d4', 'CONTAINS CLAIM':'#d6a46a',
 'EXPRESSES PROPOSITION':'#d4d879', ASSESSES:'#86efac'
};
function connectionColor(kind) { return connectionColors[kind] || '#cbd5e1'; }
function buildOverview(records, options = {}) {
 const nodes = new Map(), edges = new Map(); let omitted = 0;
 const key = (kind, id) => JSON.stringify([kind, id]);
 const add = (kind, id, label, claimId) => {
  const k = key(kind, id);
  if (!nodes.has(k)) nodes.set(k, {id:k, kind, label, claimIds:[]});
  if (claimId && !nodes.get(k).claimIds.includes(claimId)) nodes.get(k).claimIds.push(claimId);
  return k;
 };
 const link = (a,b,kind,id) => {
  if(options.connectionTypes && !options.connectionTypes.includes(kind))return;
  edges.set(id || JSON.stringify([a,b,kind]), {a,b,kind});
 };
 for (const r of records) {
  const id=add('claim',r.claim.id,r.claim.text,r.claim.id);
  nodes.get(id).searchText=[r.claim.text,r.claim.speaker || '',...(r.claim.concepts || [])].join(' ');
 }
 for (const r of records) {
  const c = key('claim',r.claim.id);
  for (const relation of r.relationships || []) {
   const a=key('claim',relation.source_id), b=key('claim',relation.target_id);
   if (!nodes.has(a) || !nodes.has(b)) {omitted++; continue;}
   link(a,b,relation.relationship.kind,relation.relationship.id || JSON.stringify([a,b,relation.relationship.kind]));
  }
  if(options.concepts) for(const name of r.claim.concepts || []) link(c,add('concept',name,name,r.claim.id),'ABOUT');
  if(options.sources) for(const s of r.sources || []) {
   // Content hash distinguishes different books/versions with the same title.
   const id=s.document_hash || JSON.stringify([s.document,s.paths]);
   link(add('source',id,s.document,r.claim.id),c,'CONTAINS CLAIM');
  }
  if(options.entities) for(let i=0;i<(r.claim.entities || []).length;i++) {
   const name=r.claim.entities[i],kind=(r.claim.entity_kinds || [])[i] || 'entity';
   link(c,add('entity',JSON.stringify([kind,name]),name+' ('+kind+')',r.claim.id),'MENTIONS');
  }
  if(options.propositions) for(const p of r.propositions || []) link(c,add('proposition',p.id,p.text,r.claim.id),'EXPRESSES PROPOSITION');
  if(options.assessments) for(const item of r.assessments || []) {
   const a=item.assessment;
   link(add('assessment',a.id,a.position+': '+(a.reasoning || ''),r.claim.id),c,'ASSESSES');
  }
 }
 return {nodes:[...nodes.values()],edges:[...edges.values()],omitted};
}
function layoutOverview(graph) {
 const byId=new Map(graph.nodes.map(n=>[n.id,n])),adj=new Map(graph.nodes.map(n=>[n.id,[]]));
 for(const e of graph.edges){adj.get(e.a).push(e.b);adj.get(e.b).push(e.a);}
 const seen=new Set(),components=[];
 for(const n of graph.nodes){
  if(seen.has(n.id))continue;
  const pending=[n.id],component=[];seen.add(n.id);
  while(pending.length){const id=pending.pop();component.push(byId.get(id));for(const other of adj.get(id))if(!seen.has(other)){seen.add(other);pending.push(other);}}
  components.push(component);
 }
 components.sort((a,b)=>b.length-a.length);
 const boxes=components.map(nodes=>({nodes,size:Math.max(110,Math.ceil(Math.sqrt(nodes.length))*76)}));
 const target=Math.max(600,Math.sqrt(boxes.reduce((sum,b)=>sum+b.size*b.size,0))*1.4);
 let x=0,y=0,row=0,width=0;
 for(const box of boxes){
  if(x && x+box.size>target){x=0;y+=row+35;row=0;}
  const sorted=box.nodes.sort((a,b)=>adj.get(b.id).length-adj.get(a.id).length || a.id.localeCompare(b.id));
  sorted[0].x=x+box.size/2;sorted[0].y=y+box.size/2;
  let offset=1,ring=1;
  while(offset<sorted.length){
   const count=Math.min(ring*8,sorted.length-offset),radius=ring*34;
   for(let j=0;j<count;j++){const angle=2*Math.PI*j/count;const n=sorted[offset+j];n.x=x+box.size/2+radius*Math.cos(angle);n.y=y+box.size/2+radius*Math.sin(angle);}
   offset+=count;ring++;
  }
  x+=box.size+35;row=Math.max(row,box.size);width=Math.max(width,x);
 }
 return {...graph,byId,components:components.length,width:Math.max(110,width),height:Math.max(110,y+row)};
}
function initOverview(records, selectClaim) {
 const panel=document.getElementById('overview'),canvas=document.getElementById('graph-canvas'),ctx=canvas.getContext('2d');
 const colors={claim:'#74c9ed',concept:'#b69af5',source:'#f4be72',assessment:'#8ddaac',entity:'#ee92ba',proposition:'#f0dc72'};
 let graph,scale=1,tx=0,ty=0,width=800,height=520,selected=null,query='',drag=null;
 const tooltip=document.getElementById('graph-tooltip');
 const connectionControls=document.getElementById('graph-connections');
 const allEdges=buildOverview(records,{concepts:true,sources:true,entities:true,propositions:true,assessments:true}).edges;
 const connectionInputs=[];
 for(const kind of [...new Set(allEdges.map(e=>e.kind))].sort()) {
  const label=document.createElement('label'),input=document.createElement('input'),swatch=document.createElement('span');
  input.type='checkbox';input.checked=true;input.value=kind;
  swatch.className='connection-swatch';swatch.style.backgroundColor=connectionColor(kind);
  label.append(input,swatch,document.createTextNode(kind.replaceAll('_',' ').toLowerCase()));
  connectionControls.appendChild(label);connectionInputs.push(input);input.addEventListener('change',rebuild);
 }
 function draw(){
  if(!graph)return;
  ctx.clearRect(0,0,width,height);ctx.save();ctx.translate(tx,ty);ctx.scale(scale,scale);
  for(const e of graph.edges){
   const a=graph.byId.get(e.a),b=graph.byId.get(e.b);ctx.strokeStyle=connectionColor(e.kind);ctx.lineWidth=1.4/scale;
   ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
   if(scale>.65){const angle=Math.atan2(b.y-a.y,b.x-a.x),px=b.x-10*Math.cos(angle),py=b.y-10*Math.sin(angle);ctx.fillStyle=connectionColor(e.kind);ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px-5*Math.cos(angle-.5),py-5*Math.sin(angle-.5));ctx.lineTo(px-5*Math.cos(angle+.5),py-5*Math.sin(angle+.5));ctx.fill();}
  }
  for(const n of graph.nodes){
   const match=query && (n.searchText || n.label).toLowerCase().includes(query);
   ctx.globalAlpha=query && !match ? .25:1;ctx.fillStyle=colors[n.kind];ctx.beginPath();ctx.arc(n.x,n.y,n.kind==='claim'?8:6,0,Math.PI*2);ctx.fill();
   if(n.id===selected || match){ctx.strokeStyle='#fff';ctx.lineWidth=2/scale;ctx.stroke();}
   if(document.getElementById('graph-labels').checked || n.id===selected || match){ctx.font='12px system-ui';ctx.fillStyle='#edf5fc';ctx.fillText(n.label.slice(0,48)+(n.label.length>48?'…':''),n.x+11,n.y+4);}
  }
  ctx.restore();ctx.globalAlpha=1;
 }
 function fit(){scale=Math.min(width/(graph.width+60),height/(graph.height+60));tx=(width-graph.width*scale)/2;ty=(height-graph.height*scale)/2;draw();}
 function resize(){if(panel.hidden)return;width=canvas.clientWidth;const ratio=window.devicePixelRatio || 1;canvas.width=width*ratio;canvas.height=height*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);fit();}
 function rebuild(){
  graph=layoutOverview(buildOverview(records,{connectionTypes:connectionInputs.filter(input=>input.checked).map(input=>input.value),concepts:document.getElementById('graph-concepts').checked,sources:document.getElementById('graph-sources').checked,assessments:document.getElementById('graph-assessments').checked,entities:document.getElementById('graph-entities').checked,propositions:document.getElementById('graph-propositions').checked}));
  const counts=Object.keys(colors).map(k=>{const count=graph.nodes.filter(n=>n.kind===k).length;return count+' '+(k==='entity'?(count===1?'entity':'entities'):(k==='source'?'source document':k)+(count===1?'':'s'));}).join(' · ');
  document.getElementById('graph-stats').textContent=counts+' · '+graph.edges.length+' connections · '+graph.components+' connected components'+(graph.omitted?' · Some connections lead outside this export.':'');resize();
 }
 function zoom(factor,x=width/2,y=height/2){const next=Math.max(.015,Math.min(12,scale*factor));tx=x-(x-tx)*next/scale;ty=y-(y-ty)*next/scale;scale=next;draw();}
 function point(event){const r=canvas.getBoundingClientRect();return {x:event.clientX-r.left,y:event.clientY-r.top};}
 function hit(p){const x=(p.x-tx)/scale,y=(p.y-ty)/scale;let best=null,distance=Infinity;for(const n of graph.nodes){const d=Math.hypot(n.x-x,n.y-y);if(d<Math.max(10,8/scale)&&d<distance){best=n;distance=d;}}return best;}
 canvas.addEventListener('wheel',e=>{e.preventDefault();const p=point(e);zoom(Math.exp(-e.deltaY*.0015),p.x,p.y);},{passive:false});
 canvas.addEventListener('pointerdown',e=>{drag={...point(e),tx,ty,moved:false};canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{const p=point(e);if(drag){const dx=p.x-drag.x,dy=p.y-drag.y;drag.moved ||= Math.hypot(dx,dy)>4;tx=drag.tx+dx;ty=drag.ty+dy;draw();}else{const n=hit(p);tooltip.textContent=n?n.kind+': '+n.label:'Drag to pan · Scroll to zoom · Select a node for details';canvas.style.cursor=n?'pointer':'grab';}});
 canvas.addEventListener('pointerup',e=>{if(drag&&!drag.moved){const n=hit(point(e));if(n){selected=n.id;const info=document.getElementById('graph-selection');info.replaceChildren();const title=document.createElement('p');title.textContent=n.kind+': '+n.label;info.appendChild(title);for(const id of n.claimIds){const b=document.createElement('button');b.textContent=records.find(r=>r.claim.id===id).claim.text;b.addEventListener('click',()=>selectClaim(id));info.appendChild(b);}if(n.kind==='claim')selectClaim(n.claimIds[0]);draw();}}drag=null;});
 canvas.addEventListener('pointercancel',()=>{drag=null;});
 for(const id of ['graph-concepts','graph-sources','graph-assessments','graph-entities','graph-propositions'])document.getElementById(id).addEventListener('change',rebuild);
 document.getElementById('graph-labels').addEventListener('change',draw);
 document.getElementById('graph-fit').addEventListener('click',fit);
 document.getElementById('graph-in').addEventListener('click',()=>zoom(1.4));document.getElementById('graph-out').addEventListener('click',()=>zoom(1/1.4));
 window.addEventListener('resize',resize);
 rebuild();return {show(){panel.hidden=false;resize();},hide(){panel.hidden=true;},search(value){query=value.toLowerCase();draw();}};
}
