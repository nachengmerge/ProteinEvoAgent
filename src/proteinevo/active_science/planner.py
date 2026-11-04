"""LLM chooses a scientific question; a separate tool owns numerical forecasts."""
import copy
import json
import time
import uuid

import numpy as np

from .tools import effect_tool, attach_forecast
from ..scientific_action_view import canonical_action_bank
from ..contextual_llm import format_prompt
from ..study_support import write_json
from ..workbench_v2.llm import ScientificLLM

QUESTIONS=('maximize_binding','contrast_backgrounds','resolve_uncertainty')
SYSTEM=(
    'Plan virtual GB1 binding experiments at sites39,40,41,54. Select ONE allowed action and a scientific question. '
    'Two measured parents differ at exactly one background site; the two single-edit children are UNKNOWN. '
    'A deterministic tool has already calculated the child point forecasts, parent-child differences, '
    'interaction difference and member agreement. These are MODEL predictions, not measured child fitness. '
    'You do not perform arithmetic or output direction/interaction labels. Those quantities are owned by the tool. '
    'Choose question maximize_binding, contrast_backgrounds, or resolve_uncertainty. '
    'Consider predicted value, opportunity cost, background contrast, observed evidence and model disagreement. '
    'Member agreement is not calibrated confidence. Sparse evidence is not proof of benefit. '
    'Related revisions only describe their measured contexts, never all residues. '
    'C:parent>child IDs are already observed contrasts; MODEL_FORECAST means no observed contrast is cited. '
    'Return JSON action_id,question,evidence_id,response_to,reason. Reason at most120 characters. '
    'No measured child-fitness, folding-stability or statistical-significance claims.'
)


def add_tools(bank,members,c):
    active,_=canonical_action_bank(bank)
    for a in active:
        a['scope']['backgrounds']=list(a['parents'])
        matrix=np.column_stack([members[ch] for ch in a['children']])
        a['tool_forecast']=effect_tool(a,matrix,effect_threshold=c['effect_threshold'],
            interaction_threshold=c['interaction_threshold'],vote_fraction=c['tool_vote_fraction'])
    return active


def card(a):
    tool=a['tool_forecast']
    return {'id':a['action_id'],'edit':a['substitution'],'scope':a['scope'],
        'tests':[{'label':label,'parent':p,'parent_fitness':tool['parent_fitness'][i],
                  'unknown_child':ch,'child_prediction':tool['child_mean'][i],
                  'point_delta':tool['point_delta'][i],'point_direction':tool['point_direction'][i],
                  'consensus_direction':tool['expect_'+label],
                  'observed_background_summary':dict(a['background_evidence'][p],evidence_ids=a['background_evidence'][p]['evidence_ids'][:3])}
                 for i,(label,p,ch) in enumerate(zip(('A','B'),a['parents'],a['children']))],
        'tool_interaction':{k:tool[k] for k in ('point_interaction_delta','point_interaction','member_interaction_delta','interaction_agreement','interaction','interaction_threshold')},
        'effect_threshold':tool['effect_threshold'],'members':tool['members'],
        'statistics':a['visible_stats'],'related_revision':a.get('revision'),
        'evidence':[{'id':r['evidence_id'],'delta':r['delta']} for r in a['evidence']],
        'uncertainty_interpretation':tool['uncertainty_interpretation']}


def prepare_view(client,bank,observed,settings,members,c):
    active=add_tools(bank,members,c)
    context={'observed_n':len(observed),'best_known':float(observed.fitness.max()),
        'top_observed':observed.sort_values(['fitness','variant'],ascending=[False,True]).head(4).to_dict('records'),
        'quantitative_author':'deterministic_prediction_tool'}
    while True:
        payload=dict(context,actions=[card(a) for a in active],step=2,prior_selected=[active[0]['action_id']] if active else [])
        count=len(client.tokenize(format_prompt(SYSTEM,payload).encode(),add_bos=False,special=True))
        if count<=settings.input-32:break
        if not active:raise ValueError('Tool context cannot fit input budget')
        active.remove(max(active,key=lambda a:(a['visible_stats']['batch_cost_alone'],a['action_id'])))
    return active,context,{'full_actions':len(bank),'active_actions':len(active),'prepared_tokens':count,
                          'input_limit':settings.input,'normalized_before_inference':True}


def intent_schema(actions):
    branches=[]
    for a in actions:
        props={'action_id':{'const':a['action_id']},'question':{'enum':list(QUESTIONS)},
            'evidence_id':{'enum':[r['evidence_id'] for r in a['evidence']]+['MODEL_FORECAST']},
            'response_to':{'const':a['revision']['revision_id'] if a.get('revision') else 'NEW'},
            'reason':{'type':'string','minLength':1,'maxLength':120}}
        branches.append({'type':'object','properties':props,'required':list(props),'additionalProperties':False})
    return {'oneOf':branches}


def validate_intent(raw,allowed):
    if set(raw)!={'action_id','question','evidence_id','response_to','reason'}:raise ValueError('Intent must not contain numerical forecast labels')
    a=next((x for x in allowed if x['action_id']==raw['action_id']),None)
    if a is None or raw['question'] not in QUESTIONS:raise ValueError('Invalid action/question')
    if raw['evidence_id'] not in {r['evidence_id'] for r in a['evidence']}|{'MODEL_FORECAST'}:raise ValueError('False observed evidence citation')
    if raw['response_to']!=(a['revision']['revision_id'] if a.get('revision') else 'NEW'):raise ValueError('Wrong revision reference')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=120:raise ValueError('Invalid intent explanation')
    return a


class Planner(ScientificLLM):
    def choose(self,selector,context,*,trace_dir=None):
        from llama_cpp import LlamaGrammar
        e=self.effective;trace={'decision_id':uuid.uuid4().hex,'status':'prepared','calls':[],
            'effective_settings':e.dictionary(),'model':self.settings.check_model(),
            'quantitative_author':'deterministic_tool','normalized_before_inference':True}
        self.last_trace=trace
        def save():
            if trace_dir is not None:write_json(trace_dir/(trace['decision_id']+'.json'),trace)
        selected=[]
        try:
            for step in range(1,selector.target+1):
                allowed=selector.allowed(selected);payload=dict(context,actions=[card(a) for a in allowed],step=step,
                    prior_selected=[a['action_id'] for a in selected])
                prompt=format_prompt(SYSTEM,payload);client=self.client()
                tokens=client.tokenize(prompt.encode(),add_bos=False,special=True)
                if len(tokens)>e.input:raise ValueError('Effective tool-planner input limit exceeded')
                schema=intent_schema(allowed);call={'call_id':uuid.uuid4().hex,'status':'prepared','prompt':prompt,
                    'context':payload,'schema':schema,'prompt_tokens':len(tokens),'effective_settings':e.dictionary()}
                trace['calls'].append(call);save();start=time.perf_counter()
                response=client.create_completion(prompt=tokens,grammar=LlamaGrammar.from_json_schema(json.dumps(schema),verbose=False),
                    max_tokens=e.output_per_call,temperature=e.temperature,seed=e.seed,
                    stop=['<|eot_id|>','<|end_of_text|>'],echo=False)
                call.update(raw=response['choices'][0]['text'],finish=response['choices'][0]['finish_reason'],
                    usage=response.get('usage',{}),seconds=time.perf_counter()-start,inference_performed=True);save()
                if call['finish']!='stop':raise ValueError('Truncated raw intent; no silent repair/retry')
                raw=json.loads(call['raw']);a=validate_intent(raw,allowed)
                intent={k:v for k,v in raw.items() if k!='action_id'}
                selected.append(attach_forecast(a,a['tool_forecast'],intent,'LLM'))
                call['status']='complete';save()
            trace['status']='complete';trace['selected_actions']=selected;save();return selected
        except Exception as exc:
            trace.update(status='failed',error=repr(exc));save();raise
