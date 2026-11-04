"""P39: source-to-target evidence and matched self/tool review before scoring."""
import json
import time
import uuid

from . import design_agent as base
from .config import POSITIONS
from .contextual_llm import format_prompt
from .study_support import write_json

SYSTEM=base.SYSTEM+(
    ' Evidence must keep SOURCE and TARGET residues distinct. A positive G>V observation is not support for V>D. '
    'Before asserting increase/decrease, cite a matching substitution with that direction in a related observed background. '
    'If direct evidence is missing or contradictory, choose uncertain rather than fabricating support. '
    'You will review the draft before execution; final unsupported directional claims will be rejected by the same guard in all LLM conditions.'
)


def design_context(observed,available,history,turn,*,explicit_feedback=False):
    context,domain,evidence=base.design_context(observed,available,history,turn,explicit_feedback=explicit_feedback)
    relevant={}
    for (parent,site,to),_ in domain.items():
        i=POSITIONS.index(site);key=f'{site}:{parent[i]}>{to}'
        for row in evidence.get(key,[]):
            distance=sum(parent[j]!=row['parent'][j] for j in range(4) if j!=i)
            if distance<=1:
                current=relevant.get(row['evidence_id'])
                if current is None or distance<current[0]:relevant[row['evidence_id']]=(distance,row)
    facts=[]
    for site in POSITIONS:
        rows=[(distance,row) for distance,row in relevant.values() if row['substitution'].startswith(str(site)+':')]
        # Four query-relevant contrasts per site, keeping both signs when present.
        chosen=[]
        for sign in (1,-1):
            signed=[x for x in rows if x[1]['delta']*sign>.05]
            for distance,row in sorted(signed,key=lambda x:(x[0],-abs(x[1]['delta']),x[1]['evidence_id'])):
                if row['substitution'] not in {r['substitution'] for _,r in chosen}:
                    chosen.append((distance,row))
                if sum(r['delta']*sign>.05 for _,r in chosen)>=2:break
        for distance,row in sorted(rows,key=lambda x:(x[0],-abs(x[1]['delta']),x[1]['evidence_id'])):
            if len(chosen)>=4:break
            if row['evidence_id'] not in {r['evidence_id'] for _,r in chosen}:chosen.append((distance,row))
        for distance,row in chosen[:4]:
            facts.append(dict(id=row['evidence_id'],substitution=row['substitution'],parent=row['parent'],child=row['child'],
                delta=row['delta'],background=row['background'],nearest_design_background_distance=distance))
    context['observed_contrasts']=facts
    context['evidence_interpretation']='Source-to-target matching and exact backgrounds matter; related measured contrasts are not guarantees for an unmeasured child.'
    return context,domain,evidence


def critique(raw,context):
    d=raw['design'];i=POSITIONS.index(d['position']);key=f'{d["position"]}:{d["parent"][i]}>{d["to"]}'
    refs={r['id']:r for r in context['observed_contrasts']}
    cited=refs.get(raw['evidence_id']);matching=[r for r in refs.values() if r['substitution']==key]
    issues=[];direction=raw['expected_effect']
    if direction!='uncertain':
        if cited is None:issues.append('Directional claim has no directly cited observed contrast; use matching evidence or mark uncertain.')
        elif cited['substitution']!=key:issues.append(f'Cited {cited["substitution"]} does not support target {key}; source or target residue differs.')
        else:
            distance=sum(cited['parent'][j]!=d['parent'][j] for j in range(4) if j!=i)
            correct_sign=cited['delta']>.05 if direction=='increase' else cited['delta']<-.05
            if distance>1:issues.append('Cited background is outside the declared local transfer radius.')
            if not correct_sign:issues.append('Cited measured effect does not have the asserted direction at threshold0.05.')
    return dict(target_substitution=key,issues=issues,semantic_accepted=not issues,
        relevant_observed_contrasts=matching,
        scope='Checks evidence consistency only; accepted transfer is not guaranteed biological benefit.')


def validate_plan(raw,context,domain,turn):
    plan=base.validate_plan(raw,context,domain,turn)
    review=critique(raw,context)
    return dict(plan,semantic_accepted=review['semantic_accepted'],semantic_issues=review['issues'])


def rule_plan(context,domain,evidence,turn):
    plan=base.rule_plan(context,domain,evidence,turn)
    return dict(plan,semantic_accepted=True,semantic_issues=[],semantic_author='deterministic_nearby_observation_rule')


class Designer(base.Designer):
    tool_critic=False

    def propose(self,context,domain,turn,trace_path):
        from llama_cpp import LlamaGrammar
        client=self.client();e=self.effective;grammar_schema=base.schema(context)
        grammar=LlamaGrammar.from_json_schema(json.dumps(grammar_schema),verbose=False)
        trace=dict(status='prepared',calls=[],context=context,schema=grammar_schema,model=self.settings.check_model(),
            effective_settings=e.dictionary(),phase='before_candidate_scoring',trace_id=uuid.uuid4().hex,
            tool_critic_enabled=bool(self.tool_critic),inference_performed=False)
        write_json(trace_path,trace)
        raw=None
        try:
            for step in (1,2):
                payload=dict(context)
                if step==2:
                    review=critique(raw,context)
                    payload.update(draft_plan=raw,review_instruction='Finalize this round. Review evidence and past measured feedback; response_to still refers to the previous round, not this draft.')
                    if self.tool_critic:payload['deterministic_critic']=review
                    else:payload['self_review']='Check source and target residues, effect direction and background transfer yourself; revise or retain the draft.'
                prompt=format_prompt(SYSTEM,payload);tokens=client.tokenize(prompt.encode(),add_bos=False,special=True)
                if len(tokens)>e.input:raise ValueError('Review context exceeds frozen input budget')
                call=dict(status='prepared',step=step,prompt=prompt,context=payload,prompt_tokens=len(tokens));trace['calls'].append(call);write_json(trace_path,trace)
                start=time.perf_counter()
                response=client.create_completion(prompt=tokens,grammar=grammar,max_tokens=e.output_per_call,temperature=e.temperature,
                    seed=e.seed,stop=['<|eot_id|>','<|end_of_text|>'],echo=False)
                call.update(raw=response['choices'][0]['text'],finish=response['choices'][0]['finish_reason'],
                    usage=response.get('usage',{}),seconds=time.perf_counter()-start,inference_performed=True)
                write_json(trace_path,trace)
                if call['finish']!='stop':raise ValueError('Truncated raw design review; no silent repair or retry')
                raw=json.loads(call['raw']);base.validate_plan(raw,context,domain,turn)
                call['status']='complete';write_json(trace_path,trace)
            plan=validate_plan(raw,context,domain,turn)
            trace.update(status='complete',raw=trace['calls'][-1]['raw'],plan=plan,inference_performed=True,
                real_inference_calls=2,prompt_tokens=trace['calls'][-1]['prompt_tokens'],
                finish='stop',usage=trace['calls'][-1]['usage'])
            write_json(trace_path,trace);return plan
        except Exception as exc:
            trace.update(status='failed',error=repr(exc));write_json(trace_path,trace);raise


assemble_library=base.assemble_library
hypothesis_feedback=base.hypothesis_feedback
