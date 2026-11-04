"""Pre-score mutation design, falsifiable claims and measured-context feedback.

No predictor or oracle is accepted by the designer. Python expands a declared
mutation into a bounded library; it does not rank designs before the LLM call.
"""
import json
import time
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import AMINO_ACIDS, POSITIONS
from .contextual_science import observed_lookup, one_edit_evidence
from .contextual_llm import format_prompt
from .study_support import write_json
from .workbench_v2.llm import ScientificLLM
from .workbench.run import digest
from .knowledge import AA_GROUP

SYSTEM=(
    'Design the NEXT virtual GB1 binding experiment at positions39,40,41,54. '
    'This call occurs BEFORE candidate generation and predictor scoring. No candidate model scores or unknown fitness are available. '
    'Propose one source-parent/position/target-amino-acid hypothesis using permitted_edits. '
    'Parent fitness and observed contrasts are measured. Effects can reverse in different sequence backgrounds. '
    'Your design directs up to200 of800 candidates; the rest are shared baseline candidates. '
    'Choose test_slots from the permitted budget options:0,1,2,4. These slots reserve single-edit children in nearby measured-parent backgrounds. '
    'Remaining slots in a16-sequence batch follow later model ranking. There are only3 rounds: consider remaining learning opportunity and binding performance. '
    'Choose expected_effect increase,decrease,or uncertain. An increase claim is refuted by a measured delta<=0.05; a decrease claim by delta>=-0.05. '
    'An uncertain claim is exploratory, not automatically supported. Claims apply only in backgrounds near the declared parent. '
    'C:parent>child identifiers are observed contrasts. A cited different substitution is context evidence only, not direct proof for your target. '
    'Use NO_DIRECT_EVIDENCE when appropriate. Refer to supplied feedback if present; a counterexample does not justify a global residue ban. '
    'Return design(parent,position,to),test_slots,expected_effect,evidence_id,response_to,revision,reason. '
    'revision is new,retain,or revise; reason is1..120 characters. No binding-energy, stability, numerical child-fitness, or significance claims.'
)


def direct_pairs(design, known, available):
    parent=design['parent'];i=POSITIONS.index(design['position']);before=parent[i];after=design['to']
    pairs=[]
    for p,y in known.items():
        distance=sum(p[j]!=parent[j] for j in range(4) if j!=i)
        child=p[:i]+after+p[i+1:]
        if p[i]==before and distance<=1 and child in available:
            pairs.append((distance,-float(y),p,child))
    return [(p,ch) for _,_,p,ch in sorted(pairs)]


def design_context(observed,available,history,turn,*,explicit_feedback=False):
    known=observed_lookup(observed)
    if set(known)&set(available):raise ValueError('Measured candidates cannot remain available')
    parents=observed.sort_values(['fitness','variant'],ascending=[False,True]).head(8).variant.tolist()
    evidence=one_edit_evidence(observed);facts=[]
    for site in POSITIONS:
        rows=[r for key,rs in evidence.items() if key.startswith(str(site)+':') for r in rs]
        # One strongest positive and negative at each site, then nearby evidence.
        for order in (sorted(rows,key=lambda r:(-r['delta'],r['evidence_id'])),sorted(rows,key=lambda r:(r['delta'],r['evidence_id']))):
            if order and order[0]['evidence_id'] not in {x['id'] for x in facts}:
                r=order[0];facts.append(dict(id=r['evidence_id'],substitution=r['substitution'],parent=r['parent'],child=r['child'],delta=r['delta'],background=r['background']))
    options=[];domain={}
    for parent in parents:
        for i,site in enumerate(POSITIONS):
            groups=defaultdict(list)
            for aa in AMINO_ACIDS:
                if aa==parent[i] or parent[:i]+aa+parent[i+1:] not in available:continue
                d=dict(parent=parent,position=site,to=aa)
                pairs=direct_pairs(d,known,available)
                budgets=tuple(k for k in (0,1,2,4) if k<=len(pairs))
                domain[(parent,site,aa)]={'pairs':pairs,'budgets':budgets}
                groups[budgets].append(aa)
            for budgets,targets in sorted(groups.items()):
                options.append(dict(parent=parent,position=site,targets=targets,test_budget_options=list(budgets)))
    if not options:raise ValueError('No legal unmeasured mutation design available')
    feedback=history[-2:] if explicit_feedback else []
    context=dict(round=turn,rounds_remaining=4-turn,observed_n=len(known),
        best_known=max(known.values()),top_observed=[dict(variant=p,fitness=known[p]) for p in parents],
        amino_acid_properties=AA_GROUP,observed_contrasts=facts,permitted_edits=options,
        effect_threshold=.05,explicit_hypothesis_feedback=feedback,
        phase='before_candidate_scoring',candidate_model_predictions_available=False)
    return context,domain,evidence


def schema(context):
    refs=[r['id'] for r in context['observed_contrasts']]+['NO_DIRECT_EVIDENCE']
    previous=context['explicit_hypothesis_feedback'][-1]['hypothesis_id'] if context['explicit_hypothesis_feedback'] else 'NONE'
    branches=[]
    for option in context['permitted_edits']:
        fields=dict(design={'type':'object','properties':{
            'parent':{'const':option['parent']},'position':{'const':option['position']},
            'to':{'enum':option['targets']}},'required':['parent','position','to'],'additionalProperties':False},
            test_slots={'enum':option['test_budget_options']},expected_effect={'enum':['increase','decrease','uncertain']},
            evidence_id={'enum':refs},response_to={'const':previous},
            revision={'enum':['new','retain','revise'] if previous!='NONE' else ['new']},
            reason={'type':'string','minLength':1,'maxLength':120})
        branches.append({'type':'object','properties':fields,'required':list(fields),'additionalProperties':False})
    return {'oneOf':branches}


def validate_plan(raw,context,domain,turn):
    if set(raw)!={'design','test_slots','expected_effect','evidence_id','response_to','revision','reason'}:raise ValueError('Unexpected designer fields')
    d=raw['design']
    if set(d)!={'parent','position','to'}:raise ValueError('Invalid mutation design')
    key=(d['parent'],d['position'],d['to'])
    if key not in domain or type(raw['test_slots']) is not int or raw['test_slots'] not in domain[key]['budgets']:raise ValueError('Unavailable mutation or unsupported test budget')
    refs={r['id']:r for r in context['observed_contrasts']}
    if raw['evidence_id'] not in set(refs)|{'NO_DIRECT_EVIDENCE'}:raise ValueError('Unobserved evidence reference')
    previous=context['explicit_hypothesis_feedback'][-1]['hypothesis_id'] if context['explicit_hypothesis_feedback'] else 'NONE'
    if raw['response_to']!=previous or raw['revision'] not in (['new','retain','revise'] if previous!='NONE' else ['new']):raise ValueError('Invalid feedback reference')
    if raw['expected_effect'] not in {'increase','decrease','uncertain'} or not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=120:raise ValueError('Invalid claim or reason')
    i=POSITIONS.index(d['position']);sub=f'{d["position"]}:{d["parent"][i]}>{d["to"]}'
    matches=raw['evidence_id'] in refs and refs[raw['evidence_id']]['substitution']==sub
    return dict(raw,hypothesis_id=f'H{turn}:{d["parent"]}:{sub}',substitution=sub,
        citation_matches_target_substitution=bool(matches),
        falsification_condition={'increase':'measured_delta<=0.05','decrease':'measured_delta>=-0.05','uncertain':'exploratory_no_direction_claim'}[raw['expected_effect']],
        predictor_scores_used_for_design=False)


class Designer(ScientificLLM):
    def propose(self,context,domain,turn,trace_path):
        from llama_cpp import LlamaGrammar
        e=self.effective;client=self.client();prompt=format_prompt(SYSTEM,context)
        tokens=client.tokenize(prompt.encode(),add_bos=False,special=True)
        if len(tokens)>e.input:raise ValueError('Pre-score context exceeds frozen token budget')
        grammar_schema=schema(context)
        trace=dict(status='prepared',inference_performed=False,model=self.settings.check_model(),
            prompt=prompt,context=context,schema=grammar_schema,effective_settings=e.dictionary(),prompt_tokens=len(tokens),
            phase='before_candidate_scoring',trace_id=uuid.uuid4().hex)
        write_json(trace_path,trace);start=time.perf_counter()
        try:
            response=client.create_completion(prompt=tokens,grammar=LlamaGrammar.from_json_schema(json.dumps(grammar_schema),verbose=False),
                max_tokens=e.output_per_call,temperature=e.temperature,seed=e.seed,
                stop=['<|eot_id|>','<|end_of_text|>'],echo=False)
            trace.update(raw=response['choices'][0]['text'],finish=response['choices'][0]['finish_reason'],
                usage=response.get('usage',{}),seconds=time.perf_counter()-start,inference_performed=True)
            write_json(trace_path,trace)
            if trace['finish']!='stop':raise ValueError('Truncated design; no silent repair or retry')
            plan=validate_plan(json.loads(trace['raw']),context,domain,turn)
            trace.update(status='complete',plan=plan);write_json(trace_path,trace)
            return plan
        except Exception as exc:
            trace.update(status='failed',error=repr(exc));write_json(trace_path,trace);raise


def rule_plan(context,domain,evidence,turn):
    known={r['variant']:r['fitness'] for r in context['top_observed']}
    candidates=[]
    for (p,site,aa),info in domain.items():
        i=POSITIONS.index(site);key=f'{site}:{p[i]}>{aa}';rows=evidence.get(key,[])
        local=[r for r in rows if sum(r['parent'][j]!=p[j] for j in range(4) if j!=i)<=1]
        delta=float(np.median([r['delta'] for r in local])) if local else -1e9
        candidates.append((delta,len(local),known[p],p,site,aa))
    delta,_,_,p,site,aa=max(candidates)
    budget=0  # Pure greedy evaluation isolates the rule-generated library.
    raw=dict(design=dict(parent=p,position=site,to=aa),test_slots=budget,
        expected_effect='increase' if delta>.05 else 'uncertain',evidence_id='NO_DIRECT_EVIDENCE',
        response_to='NONE',revision='new',reason='Nearby observed single-edit median effect; no predictor used to generate hypothesis')
    return validate_plan(raw,context,domain,turn)


def assemble_library(standard,observed,available,plan,domain,seed,budget=800,guided_budget=200):
    known=observed_lookup(observed)
    if len(standard)!=budget or not set(standard)<=set(available):raise ValueError('Invalid standard proposal budget')
    d=plan['design'];i=POSITIONS.index(d['position']);before=d['parent'][i]
    tests=domain[(d['parent'],d['position'],d['to'])]['pairs'][:plan['test_slots']]
    required={ch:p for p,ch in tests}
    generated={}
    for parent in sorted(known):
        if parent[i]!=before or sum(parent[j]!=d['parent'][j] for j in range(4) if j!=i)>1:continue
        core=parent[:i]+d['to']+parent[i+1:]
        candidates=[core]
        for j in range(4):
            if j==i:continue
            candidates.extend(core[:j]+aa+core[j+1:] for aa in AMINO_ACIDS if aa!=core[j])
        for candidate in candidates:
            if candidate in available:generated.setdefault(candidate,parent)
    if not set(required)<=set(generated):raise ValueError('Requested direct tests missing from designed family')
    rng=np.random.default_rng(seed)
    remaining=sorted(set(generated)-set(required))
    extra=list(map(str,rng.choice(np.asarray(remaining,dtype=str),min(guided_budget-len(required),len(remaining)),replace=False))) if remaining else []
    guided=set(required)|set(extra)
    # Uniformly select the common 600, rather than taking a lexical prefix.
    common=set(map(str,np.random.default_rng(seed+1).choice(np.asarray(standard,dtype=str),budget-guided_budget,replace=False)))
    pool=guided|common
    for variant in standard:
        if len(pool)>=budget:break
        pool.add(variant)
    assert len(pool)==budget and len(required)<=4
    provenance={v:dict(source='guided' if v in guided else 'standard',
        generator_parent=generated.get(v),novel_vs_standard=v not in standard,
        hypothesis_id=plan['hypothesis_id'] if v in guided else None) for v in sorted(pool)}
    return sorted(pool),required,dict(provenance=provenance,guided_available=len(generated),
        guided_in_pool=len(guided),novel_vs_standard=sum(v not in standard for v in pool),
        standard_pool_sha256=digest(standard),requested_test_slots=plan['test_slots'])


def hypothesis_feedback(plan,observed,measured,turn):
    known=observed_lookup(observed);truth=observed_lookup(measured)
    if set(known)&set(truth):raise ValueError('Feedback requires selected new observations')
    d=plan['design'];i=POSITIONS.index(d['position']);pairs=[]
    for parent,pvalue in sorted(known.items()):
        if parent[i]!=d['parent'][i] or sum(parent[j]!=d['parent'][j] for j in range(4) if j!=i)>1:continue
        child=parent[:i]+d['to']+parent[i+1:]
        if child in truth:
            delta=truth[child]-pvalue
            direction='increase' if delta>.05 else 'decrease' if delta<-.05 else 'near_zero'
            pairs.append(dict(parent=parent,child=child,delta=delta,direction=direction,child_fitness=truth[child],parent_fitness=pvalue))
    status='untested' if not pairs else 'exploratory_observed' if plan['expected_effect']=='uncertain' else 'supported' if all(r['direction']==plan['expected_effect'] for r in pairs) else 'refuted'
    return dict(hypothesis_id=plan['hypothesis_id'],design=d,expected_effect=plan['expected_effect'],
        requested_test_slots=plan['test_slots'],observed_round=turn,status=status,contrasts=pairs,
        counterexamples=[r for r in pairs if plan['expected_effect']!='uncertain' and r['direction']!=plan['expected_effect']],
        scope='Only tested nearby sequence backgrounds; no global residue ban',
        revision_reference_is_schema_constrained=True)
