const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const dir=path.resolve(__dirname,'../demo');
const sandbox={window:{}};vm.runInNewContext(fs.readFileSync(path.join(dir,'data.js'),'utf8'),sandbox);
const data=sandbox.window.PROTEIN_DATA,api=require(path.join(dir,'app.js'));
assert.equal(data.records.length,2304);assert.equal(Object.keys(data.groups).length,144);
assert.equal(Object.values(data.groups).reduce((a,g)=>a+g.rawCalls.length,0),94);
for(const seed of data.seeds)for(const method of Object.keys(data.methods))for(const round of [1,2,3]){
 const rows=api.recommendations(data,seed,method,round);assert.equal(rows.length,16);
 assert.equal(new Set(rows.map(r=>r.variant)).size,16);
 for(const r of rows){const tokens=api.sequenceTokens(data,r.variant);assert.equal(tokens.length,56);assert.equal(tokens.map(t=>t.aa).join(''),r.sequence)}
 assert.equal(data.groups[`${seed}:${method}:${round}`].observedN,512+(round-1)*16);
}
assert(api.esc('<script>"test"</script>').includes('&lt;script&gt;'));
assert.equal(api.mutationLabel('40:D>W'),'第 40 位：D → W');
assert.equal(api.actionLabel('40:D>W@IDGV/LDGV'),'第 40 位 D → W；两个检验背景：IDGV、LDGV');
assert.equal(api.evidenceLabel('C:GDGV>GWGV'),'已测序列对照 GDGV → GWGV');
assert.equal(api.actionLabel('UNKNOWN_FORMAT'),'UNKNOWN_FORMAT');
for(const group of Object.values(data.groups)){
 for(const action of group.actions)assert(!api.actionLabel(action.id).includes('@'));
 for(const id of group.nextActions)assert(!api.actionLabel(id).includes('@'));
}
const contrast=data.groups['45001:tool_llm:2'].actions.flatMap(a=>a.feedback).filter(f=>['IFGM','LFGM'].includes(f.child));
assert.equal(contrast.length,2);assert(contrast.find(f=>f.child==='IFGM').actualDelta>0);assert(contrast.find(f=>f.child==='LFGM').actualDelta<0);
for(const [v,r,y] of [['FWAA',2,8.7620],['KWAA',3,0.0093]])assert(Math.abs(api.recommendations(data,45005,'tool_llm',r).find(x=>x.variant===v).actual-y)<0.0001);
assert.equal((api.chartSVG(data,45001,'mean_fitness','tool_llm').match(/<circle/g)||[]).length,18);
assert.equal(api.toCSV(api.recommendations(data,45001,'tool_llm',1,5)).split('\r\n').length,6);

// Exercise actual page initialization and handlers in a DOM test double. This is
// interaction logic verification, not a substitute for browser visual inspection.
class Element{
 constructor(tag='div',id=''){this.tag=tag;this.id=id;this.value='';this.dataset={};this.listeners={};this.children=[];this._html='';this.textContent='';this.classList={toggle(){}}}
 set innerHTML(v){this._html=v;this.children=[]}get innerHTML(){return this._html}
 addEventListener(t,f){this.listeners[t]=f}append(...items){this.children.push(...items)}
 querySelectorAll(selector){if(selector.includes('data-variant'))return [...this._html.matchAll(/data-variant="([A-Z]+)"/g)].map(m=>{let b=new Element('button');b.dataset.variant=m[1];return b});return []}
 click(){this.listeners.click?.({target:this})}scrollIntoView(){}
}
const html=fs.readFileSync(path.join(dir,'index.html'),'utf8'),elements={};
for(const m of html.matchAll(/id="([\w-]+)"/g))elements[m[1]]=new Element('div',m[1]);
const roundButtons=[1,2,3].map(v=>{let e=new Element('button');e.dataset.round=String(v);return e});
const caseButtons=[2,3].map((v,i)=>{let e=new Element('button');e.dataset.caseRound=String(v);e.dataset.caseVariant=i?'KWAA':'FWAA';return e});
const document={getElementById(id){assert(elements[id],id);return elements[id]},createElement(tag){return new Element(tag)},querySelectorAll(s){return s==='[data-round]'?roundButtons:s==='[data-case-round]'?caseButtons:[]}};
const ctx={window:{PROTEIN_DATA:data},document,Blob,URL,setTimeout(){}};
vm.runInNewContext(fs.readFileSync(path.join(dir,'app.js'),'utf8'),ctx);
assert.equal(ctx.window.PROTEIN_VIEW.state.seed,45001);assert(elements['chart'].innerHTML.includes('<svg'));
assert(!elements.steps.innerHTML.replace(/<details>[\s\S]*?<\/details>/g,'').includes('@'));
assert(elements.steps.innerHTML.includes('原始记录编号'));
caseButtons[1].click();assert.equal(ctx.window.PROTEIN_VIEW.state.variant,'KWAA');assert.equal(ctx.window.PROTEIN_VIEW.state.round,3);
assert(elements['variant-badge'].textContent.includes('0.0093'));
elements['contrast-case'].click();assert.equal(ctx.window.PROTEIN_VIEW.state.seed,45001);assert.equal(ctx.window.PROTEIN_VIEW.state.round,2);
assert(elements.steps.innerHTML.includes('LFGM'));assert(elements.steps.innerHTML.includes('−')||elements.steps.innerHTML.includes('-1.1634'));
elements.method.value='random';elements.method.listeners.change({target:elements.method});assert(elements['raw-calls'].children[0].textContent.includes('没有真实'));
elements.reset.click();assert.equal(ctx.window.PROTEIN_VIEW.state.method,'tool_llm');
console.log(JSON.stringify({records:2304,states:144,raw_LLM_calls:94,sequence_consistency:true,filter_and_case_handlers:true,SVG_points:18,CSV_export:true,actual_browser_render_verified:false}));
