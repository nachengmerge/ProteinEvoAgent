"""Evidence-bound transfer avoids independent copying of source/target residues.

Identity consistency is guaranteed by parameterization, not counted as LLM
scientific reasoning. Transfer effects in a new background remain unknown.
"""
import json
import time
import uuid
from collections import defaultdict

from . import design_agent as base
from . import grounded_designer as previous
from .config import POSITIONS
from .contextual_llm import format_prompt
from .study_support import write_json

SYSTEM=(
    'Design virtual GB1 binding experiments at sites39,40,41,54 BEFORE candidate generation or predictor scoring. '
    'Choose an observed evidence_id and a NEW measured parent from permitted_transfers. '
    'Python applies the EXACT source-to-target substitution in that evidence to your chosen parent; you do not copy or invent residue edits. '
    'The evidence endpoints are measured but the new child is UNKNOWN. Identity matching does not guarantee the mutation works in a new sequence background. '
    'Choose expected_effect increase,decrease,or uncertain in the NEW background; delta threshold is0.05. '
    'Choose0/1/2/4 test_slots only as permitted. The batch has16 slots; remaining slots use later ensemble ranking. '
    'Your chosen transfer expands up to200 of800 candidates. Consider binding improvement, background counterexamples and remaining rounds. '
    'For a directional claim, the cited observed direction and related background must be consistent; if evidence is weak, mark uncertain. '
    'Use your measured hypothesis feedback, restrict conclusions to tested contexts, and never turn one refutation into a universal residue ban. '
    'Two calls are allowed: propose, then review and finalize. response_to refers to the previous round, not the draft. '
    'Return JSON evidence_id,parent,test_slots,expected_effect,response_to,revision,reason. reason1..120 characters. '
    'No child fitness, binding-energy, stability, or statistical-significance claims.'
)


def design_context(observed,available,history,turn,*,explicit_feedback=False):
    context,domain,evidence=previous.design_context(observed,available,history,turn,explicit_feedback=explicit_feedback)
    transfers=[]
    facts=[]
    for fact in context['observed_contrasts']:
        site=int(fact['substitution'].split(':')[0]);i=POSITIONS.index(site)
        before,after=fact['parent'][i],fact['child'][i]
        groups=defaultdict(list)
        for (parent,position,to),info in domain.items():
            if position==site and parent[i]==before and to==after and sum(parent[j]!=fact['parent'][j] for j in range(4) if j!=i)<=1:
                groups[tuple(info['budgets'])].append(parent)
        if not groups:continue
        facts.append(fact)
        for budgets,parents in sorted(groups.items()):
            transfers.append(dict(evidence_id=fact['id'],parents=sorted(parents),test_budget_options=list(budgets)))
    if not transfers:raise ValueError('No observed-evidence transfer to an unmeasured child is available')
    context.pop('permitted_edits')
    context['observed_contrasts']=facts
    context['permitted_transfers']=transfers
    context['parameterization']='Choose evidence and new parent; exact mutation is derived by Python; biological transfer remains a hypothesis.'
    return context,domain,evidence


def expand(raw,context):
    if set(raw)!={'evidence_id','parent','test_slots','expected_effect','response_to','revision','reason'}:raise ValueError('Unexpected transfer fields')
    facts={f['id']:f for f in context['observed_contrasts']}
    if raw['evidence_id'] not in facts:raise ValueError('Unobserved evidence binding')
    fact=facts[raw['evidence_id']];site=int(fact['substitution'].split(':')[0]);i=POSITIONS.index(site)
    allowed=[t for t in context['permitted_transfers'] if t['evidence_id']==raw['evidence_id'] and raw['parent'] in t['parents'] and raw['test_slots'] in t['test_budget_options']]
    if len(allowed)!=1:raise ValueError('Unsupported evidence/parent/test-budget combination')
    return dict(design=dict(parent=raw['parent'],position=site,to=fact['child'][i]),
        **{k:v for k,v in raw.items() if k!='parent'})


def validate_plan(raw,context,domain,turn):
    native=expand(raw,context)
    plan=previous.validate_plan(native,context,domain,turn)
    return dict(plan,parameterization='evidence_bound_transfer',raw_bound_plan=raw)


def schema(context):
    previous_id=context['explicit_hypothesis_feedback'][-1]['hypothesis_id'] if context['explicit_hypothesis_feedback'] else 'NONE'
    branches=[]
    for option in context['permitted_transfers']:
        props=dict(evidence_id={'const':option['evidence_id']},parent={'enum':option['parents']},test_slots={'enum':option['test_budget_options']},
            expected_effect={'enum':['increase','decrease','uncertain']},response_to={'const':previous_id},
            revision={'enum':['new','retain','revise'] if previous_id!='NONE' else ['new']},
            reason={'type':'string','minLength':1,'maxLength':120})
        branches.append(dict(type='object',properties=props,required=list(props),additionalProperties=False))
    return {'oneOf':branches}


rule_plan=previous.rule_plan
assemble_library=base.assemble_library
hypothesis_feedback=base.hypothesis_feedback


class Designer(base.Designer):
    tool_critic=False

    def propose(self,context,domain,turn,trace_path):
        from llama_cpp import LlamaGrammar
        client=self.client();e=self.effective;grammar_schema=schema(context)
        grammar=LlamaGrammar.from_json_schema(json.dumps(grammar_schema),verbose=False)
        trace=dict(status='prepared',calls=[],context=context,schema=grammar_schema,model=self.settings.check_model(),
            effective_settings=e.dictionary(),phase='before_candidate_scoring',trace_id=uuid.uuid4().hex,
            tool_critic_enabled=bool(self.tool_critic),inference_performed=False,parameterization='evidence_bound_transfer')
        write_json(trace_path,trace);raw=None
        try:
            for step in (1,2):
                payload=dict(context)
                if step==2:
                    native=expand(raw,context);review=previous.critique(native,context)
                    payload.update(draft_plan=raw,review_instruction='Review the new-background effect and experiment budget. Keep the previous-round response_to reference.')
                    if self.tool_critic:payload['deterministic_critic']=review
                    else:payload['self_review']='Review the cited direction, new sequence background, prior refutations and remaining rounds yourself.'
                prompt=format_prompt(SYSTEM,payload);tokens=client.tokenize(prompt.encode(),add_bos=False,special=True)
                if len(tokens)>e.input:raise ValueError('Evidence-bound review exceeds frozen input budget')
                call=dict(status='prepared',step=step,prompt=prompt,context=payload,prompt_tokens=len(tokens));trace['calls'].append(call);write_json(trace_path,trace)
                start=time.perf_counter()
                response=client.create_completion(prompt=tokens,grammar=grammar,max_tokens=e.output_per_call,temperature=e.temperature,
                    seed=e.seed,stop=['<|eot_id|>','<|end_of_text|>'],echo=False)
                call.update(raw=response['choices'][0]['text'],finish=response['choices'][0]['finish_reason'],usage=response.get('usage',{}),
                    seconds=time.perf_counter()-start,inference_performed=True);write_json(trace_path,trace)
                if call['finish']!='stop':raise ValueError('Truncated evidence-bound plan; no silent retry')
                raw=json.loads(call['raw']);validate_plan(raw,context,domain,turn)
                call['status']='complete';write_json(trace_path,trace)
            plan=validate_plan(raw,context,domain,turn)
            trace.update(status='complete',raw=trace['calls'][-1]['raw'],plan=plan,inference_performed=True,
                real_inference_calls=2,prompt_tokens=trace['calls'][-1]['prompt_tokens'],finish='stop',usage=trace['calls'][-1]['usage'])
            write_json(trace_path,trace);return plan
        except Exception as exc:
            trace.update(status='failed',error=repr(exc));write_json(trace_path,trace);raise
