(function(root){
'use strict';
const colors={random:'#9a9c96',point_greedy:'#a58155',full_greedy:'#285b94',tool_rule:'#9c79ac',tool_llm:'#db7d43',knowledge_tool_llm:'#24816c'};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=4)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(n);
const delta=v=>(v>=0?'+':'')+fmt(v);
function recommendations(data,seed,method,round,k=16){
 const rows=data.records.filter(r=>r.seed===Number(seed)&&r.method===method&&r.round===Number(round));
 rows.sort((a,b)=>a.predicted==null||b.predicted==null?a.rank-b.rank:b.predicted-a.predicted||a.variant.localeCompare(b.variant));
 return rows.slice(0,k);
}
function sequenceTokens(data,variant){
 if(!/^[ACDEFGHIKLMNPQRSTVWY]{4}$/.test(variant))throw new Error('Invalid GB1 variant');
 const residues=data.WTSequence.split('');data.positions.forEach((p,i)=>{residues[p-1]=variant[i]});
 return residues.map((aa,i)=>({aa,position:i+1,mutable:data.positions.includes(i+1),changed:aa!==data.WTSequence[i]}));
}
function chartSVG(data,seed,metric,selected){
 const W=800,H=268,L=43,R=20,T=15,B=40;
 const rows=data.metrics.filter(r=>r.panel_seed===Number(seed));
 const vals=rows.map(r=>r[metric]);let lo=metric==='cumulative_best'?Math.max(0,Math.floor(Math.min(...vals)-.5)):0;
 let hi=Math.ceil(Math.max(...vals)*1.08*2)/2;if(hi<=lo)hi=lo+1;
 const X=t=>L+(t-1)*(W-L-R)/2,Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg"><title>初始化${seed}三轮${metric==='mean_fitness'?'批次均值':'累计最佳'}；无误差线</title>`;
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=Y(v);s+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="#e2e9e0"/><text x="${L-10}" y="${y+4}" text-anchor="end" fill="#617268" font-size="12">${v.toFixed(1)}</text>`}
 [1,2,3].forEach(t=>s+=`<text x="${X(t)}" y="${H-15}" text-anchor="middle" font-size="12" fill="#536a5c">第 ${t} 轮</text>`);
 Object.keys(data.methods).forEach(m=>{const points=rows.filter(r=>r.method===m).sort((a,b)=>a.round-b.round);const active=m===selected;
 s+=`<polyline points="${points.map(r=>`${X(r.round)},${Y(r[metric])}`).join(' ')}" fill="none" stroke="${colors[m]}" stroke-width="${active?3:1.6}" opacity="${active?1:.55}"/>`;
 points.forEach(r=>s+=`<circle cx="${X(r.round)}" cy="${Y(r[metric])}" r="${active?4.5:3}" fill="${colors[m]}" stroke="white" stroke-width="1"><title>${esc(data.methods[m])} 第${r.round}轮：${fmt(r[metric])}</title></circle>`);
 });return s+'</svg>';
}
function toCSV(rows){const keys=['seed','method','round','variant','sequence','mutations','predicted','actual','role','parent','reason'];
 const cell=v=>'"'+String(v??'').replace(/"/g,'""')+'"';return '\ufeff'+[keys.join(','),...rows.map(r=>keys.map(k=>cell(r[k])).join(','))].join('\r\n');}
const api={recommendations,sequenceTokens,chartSVG,toCSV,esc,fmt};
if(typeof module!=='undefined'&&module.exports){module.exports=api;return;}
const data=root.PROTEIN_DATA,$=id=>document.getElementById(id);
const state={seed:45001,method:'tool_llm',round:1,topk:5,metric:'mean_fitness',variant:null};
function badge(r){return r.role==='hypothesis_test'?(state.method==='tool_rule'?'规则检验名额':'LLM 检验名额'):r.role==='random'?'随机抽样':'模型利用'}
function displaySequence(id,variant){$(id).innerHTML=sequenceTokens(data,variant).map(t=>`<span class="${t.changed?'changed':t.mutable?'mutable':''}" title="第 ${t.position} 位：${t.aa}">${t.aa}</span>`).join('')}
function selectVariant(variant){state.variant=variant;renderSequence();renderRows();}
function currentRows(k=state.topk){return recommendations(data,state.seed,state.method,state.round,k)}
function renderSequence(){const r=data.records.find(r=>r.seed===state.seed&&r.method===state.method&&r.round===state.round&&r.variant===state.variant)||currentRows(1)[0];state.variant=r.variant;
 displaySequence('wt-sequence',data.WT);displaySequence('mutant-sequence',r.variant);
 $('variant-badge').textContent=`${r.variant} · 实测 ${fmt(r.actual)}`;$('selected-title').textContent='推荐变体 '+r.variant;
 $('selected-mutations').textContent=r.mutations.replace(/;/g,' · ')||'相对 WT 无替换';
 $('parent-info').textContent=r.parent?'实际检验亲本：'+r.parent:'唯一生成亲本未记录';
 const source=r.role==='hypothesis_test'?'LLM 当时保存的理由':'Python 策略当时保存的选择理由';
 $('selected-reason').innerHTML=`<b>${esc(r.variant)} · ${source}</b><br>${esc(r.reason)}${r.role==='hypothesis_test'?'<br><small>该文本不自动证明解释正确；数值预测由 P2 工具完成。</small>':''}`;
}
function renderRows(){const rows=currentRows();$('recommendation-body').innerHTML=rows.map((r,i)=>`<tr class="${r.variant===state.variant?'active':''}"><td>${i+1}</td><td><button class="variant-button" data-variant="${r.variant}">${r.variant}</button><small>${esc(r.mutations.replace(/;/g,' '))}</small></td><td>${fmt(r.predicted)}</td><td>${fmt(r.actual)}</td><td class="role">${badge(r)}</td></tr>`).join('');
 $('recommendation-body').querySelectorAll('button[data-variant]').forEach(b=>b.addEventListener('click',()=>selectVariant(b.dataset.variant)));
}
function renderSteps(){const g=data.groups[`${state.seed}:${state.method}:${state.round}`];const llm=state.method.includes('llm');
 const actionHtml=g.actions.map(a=>`<article class="action"><h4>${esc(a.substitution)} · 双背景检验</h4><code>${esc(a.id)}</code><p>观测证据：${esc(a.intent.evidence_id)}；可见支持背景 ${a.positiveContexts} / 反例背景 ${a.negativeContexts}。关联不等于因果。</p><blockquote>${esc(a.intent.reason)}</blockquote><div class="table-scroll"><table class="feedback"><thead><tr><th>亲本 → 子代</th><th>亲本实测</th><th>子代预测</th><th>子代实测</th><th>实测增益</th></tr></thead><tbody>${a.feedback.map(f=>`<tr><td>${f.parent} → ${f.child}</td><td>${fmt(f.parentActual)}</td><td>${fmt(f.childPrediction)}</td><td>${fmt(f.childActual)}</td><td class="${f.actualDelta>=0?'up':'down'}">${delta(f.actualDelta)}</td></tr>`).join('')}</tbody></table></div><p>为什么组合：此动作把相同目标替换放入仅相差一个背景位点的两个已测亲本中，形成可比较的双背景检验。结构由 Python 构造；${llm?'LLM 选择该动作':'该方法由规则选择动作'}，不代表 LLM 自由生成全体序列。</p></article>`).join('');
 const entries=[['读取当前实验',`已有 <b>${g.observedN}</b> 条观测，当前最佳 ${fmt(g.bestKnown)}。已知 top 变体：${g.topObserved.map(r=>`${r.variant} (${fmt(r.fitness)})`).join('、')}。`],
 ['证据与突变假设',g.actions.length?`本轮保存 ${g.actions.length} 个双背景动作。${llm?'下面展示原始选择和理由。':'本方法没有调用 LLM，下面是规则的动作记录。'}`:'本方法没有双背景动作；随机或模型排名完成选择，不把它描述为 LLM 推理。'],
 ['调用预测工具',`使用当轮观测训练的 P2 预测器。${state.method==='random'?'随机对照未保存预测值。':'三成员均值提供候选评分；单 P2 对照使用点模型。'} P2 负责数值评分，程序负责排序、预算和合法性。`],
 ['选择 16 条并揭示反馈',`当轮完整 16 条推荐均有历史实测记录，先选择再揭值。${g.actions.length?'检验子代预测与实测对照如下：'+actionHtml:'本轮没有检验子代，推荐表给出每条入选序列的实测反馈。'}`],
 ['更新观测并进入下一轮',`观测 ${g.observedN} → <b>${g.nextObservedN}</b>。${g.nextActions.length?'下一轮所选动作：'+g.nextActions.map(esc).join('；')+'。':state.round===3?'已完成三轮迭代。':'下一轮没有保存双背景动作。'} 反馈进入下一轮模型训练；动作变化本身不证明 LLM 明确修正了失败假设。`]];
 $('steps').innerHTML=entries.map(([h,t],i)=>`<div class="step"><strong>${i+1}</strong><div><h3>${h}</h3><div class="step-body">${t}</div></div></div>`).join('');
 $('raw-calls').innerHTML='';if(!g.rawCalls.length){const p=document.createElement('p');p.textContent='该状态没有真实 LLM 调用记录；规则与直接推荐不伪装为模型推理。';$('raw-calls').append(p)}
 g.rawCalls.forEach((c,i)=>{const details=document.createElement('details'),sum=document.createElement('summary');sum.textContent=`真实调用 ${i+1} · ${c.callId} · finish=${c.finish}`;details.append(sum);['输入','原始输出'].forEach((title,j)=>{const label=document.createElement('p'),pre=document.createElement('pre');label.textContent=title;pre.textContent=j===0?JSON.stringify(c.input,null,2):c.rawOutput;details.append(label,pre)});$('raw-calls').append(details)});
 $('site-bars').innerHTML=g.siteStatistics.map(s=>`<div class="site"><b>${s.site}</b><span> 位</span><p>非 WT 替换 ${s.nonWT}/16</p><div class="bar"><span style="width:${s.nonWT/16*100}%"></span></div><p>${Object.entries(s.residues).sort((a,b)=>b[1]-a[1]).map(([aa,n])=>`${aa}: ${n}`).join(' · ')}</p></div>`).join('');
}
function renderChart(){ $('chart').innerHTML=chartSVG(data,state.seed,state.metric,state.method);
 $('chart-legend').innerHTML=Object.entries(data.methods).map(([m,l])=>`<span><i style="background:${colors[m]}"></i>${l}</span>`).join('');
 $('chart-caption').textContent=`当前为 ${state.seed} 组的单组轨迹，无误差线；高亮 ${data.methods[state.method]}。${state.metric==='cumulative_best'?'累计最佳包含初始观测，天然不下降，不能据此推断策略学会了优化。':'新增批次均值可能回落，不保证逐轮提升。'} 全局随机与精英邻域候选的生成分布不同。`;
}
function render(){const all=currentRows(16),metric=data.metrics.find(r=>r.panel_seed===state.seed&&r.method===state.method&&r.round===state.round),g=data.groups[`${state.seed}:${state.method}:${state.round}`];
 if(!all.some(r=>r.variant===state.variant))state.variant=all[0].variant;
 $('seed').value=state.seed;$('method').value=state.method;$('round').value=state.round;$('topk').value=state.topk;$('metric').value=state.metric;$('topk-value').textContent=state.topk;
 document.querySelectorAll('[data-round]').forEach(b=>b.classList.toggle('active',Number(b.dataset.round)===state.round));
 $('recommendations-heading').textContent=`第 ${state.round} 轮推荐 · ${data.methods[state.method]}`;
 $('batch-info').innerHTML=`<span>当前观测<b>${g.observedN}</b></span><span>新增均值<b>${fmt(metric.mean_fitness)}</b></span><span>累计最佳<b>${fmt(metric.cumulative_best)}</b></span><span>已选<b>16</b></span>`;
 renderSequence();renderRows();renderSteps();renderChart();
}
$('seed').innerHTML=data.seeds.map(s=>`<option>${s}</option>`).join('');$('method').innerHTML=Object.entries(data.methods).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');
['seed','method','round','topk','metric'].forEach(id=>$(id).addEventListener(id==='topk'?'input':'change',e=>{state[id]=['seed','round','topk'].includes(id)?Number(e.target.value):e.target.value;render()}));
document.querySelectorAll('[data-round]').forEach(b=>b.addEventListener('click',()=>{state.round=Number(b.dataset.round);render()}));
$('reset').addEventListener('click',()=>{Object.assign(state,{seed:45001,method:'tool_llm',round:1,topk:5,metric:'mean_fitness',variant:null});render()});
$('download').addEventListener('click',()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([toCSV(currentRows())],{type:'text/csv;charset=utf-8'}));a.href=url;a.download=`GB1_${state.seed}_${state.method}_round${state.round}_top${state.topk}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
document.querySelectorAll('[data-case-round]').forEach(b=>b.addEventListener('click',()=>{Object.assign(state,{seed:45005,method:'tool_llm',round:Number(b.dataset.caseRound),topk:16,variant:b.dataset.caseVariant});render();$('sequences').scrollIntoView({behavior:'smooth'})}));
$('contrast-case').addEventListener('click',()=>{Object.assign(state,{seed:45001,method:'tool_llm',round:2,topk:5,variant:null});render();$('reasoning').scrollIntoView({behavior:'smooth'})});
$('summary-body').innerHTML=data.summary.map(r=>`<tr><td>${esc(data.methods[r.method])}</td><td>${fmt(r.mean)} ± ${fmt(r.std)}</td><td>${fmt(r.best_gain)}</td><td>${r.hits_ge5}</td></tr>`).join('');
render();root.PROTEIN_VIEW={state,render,selectVariant};
})(typeof window!=='undefined'?window:globalThis);
