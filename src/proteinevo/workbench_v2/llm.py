"""Canonical scoped claims with all inference settings actually wired."""
import copy
import json
import time
import uuid

from .science import DIRECTIONS, INTERACTIONS, direction
from .settings import LLMSettings
from ..contextual_llm import format_prompt
from ..knowledge import AA_GROUP
from ..local_gguf_client import LocalGGUFHypothesisGenerator, _load
from ..scientific_action_view import canonical_action_bank
from ..study_support import write_json

SYSTEM=(
    'Plan virtual GB1 binding experiments at sites39,40,41,54. Choose one allowed action. '
    'Each action is a complete 2x2 cycle: known parents A/B differ at one background site; children are UNKNOWN. '
    'Predict expect_A and expect_B independently: increase, decrease, near_zero, or uncertain. '
    'These directions use the stated practical effect threshold in raw relative binding fitness, NOT measurement uncertainty. '
    'For threshold t: increase means delta>t; decrease means delta<-t; near_zero means -t<=delta<=t. '
    'For example, delta=+0.01 with t=0.05 is near_zero, and delta=-0.01 is also near_zero. '
    'The supplied prediction_direction is computed from the displayed point estimate using this exact definition. '
    'Do not call a tiny positive delta increase merely because it is positive. A different hypothesis requires explicit justification. '
    'Also predict interaction: different_magnitude if abs(delta_B-delta_A) exceeds the stated interaction threshold, '
    'similar_magnitude otherwise, or unresolved. Both increases may have different magnitudes; opposite directions are not required. '
    'Nearby summaries and conditional knowledge use ONLY known observations, not target outcomes. Sparse or conflicting support permits uncertain. '
    'Related prior contexts are not measured outcomes in the target backgrounds. Cite response_to when supplied, '
    'but do not turn one counterexample into a global residue ban. Predictions are model point estimates, not confidence. '
    'Select valuable experiments with low predicted batch opportunity cost. Knowledge properties are weak priors. '
    'Return JSON fields action_id,expect_A,expect_B,interaction,evidence_id,response_to,reason; reason at most120 characters. '
    'No claims of statistical significance, folding stability or causal mechanism from this assay.'
)


def card(a,knowledge=False):
    result={'id':a['action_id'],'edit':a['substitution'],'scope':a['scope'],
        'effect_scale':a['effect_scale'],'effect_threshold':a['effect_threshold'],
        'interaction_threshold':a['interaction_threshold'],
        'tests':[{'label':label,'parent':p,'parent_fitness':round(f,5),'unknown_child':ch,
                  'predicted_fitness':a['visible_stats']['child_prediction'][i],
                  'prediction_delta':a['visible_stats']['predicted_delta'][i],
                  'prediction_direction':direction(a['visible_stats']['predicted_delta'][i],a['effect_threshold']),
                  'observed_background_summary':dict(a['background_evidence'][p],
                      evidence_ids=a['background_evidence'][p]['evidence_ids'][:3])}
                 for i,(label,p,f,ch) in enumerate(zip(('A','B'),a['parents'],a['parent_fitness'],a['children']))],
        'statistics':a['visible_stats'],'related_revision':a.get('revision'),
        'evidence':[{'id':r['evidence_id'],'delta':round(r['delta'],5)} for r in a['evidence']]}
    if knowledge:
        result['knowledge']={'from_property':AA_GROUP[a['from_residue']],'to_property':AA_GROUP[a['to_residue']],
            'conditional_rules':[dict(r,evidence_ids=r['evidence_ids'][:3]) for r in a['knowledge_rules']],
            'limitation':'Same observations; no extra labels; conditional association, not causal law'}
    return result


def prepare_view(client,bank,observed,settings):
    active,aliases=canonical_action_bank(bank)
    for a in active:a['scope']['backgrounds']=list(a['parents'])
    context={'observed_n':len(observed),'best_known':round(float(observed.fitness.max()),5),
             'top_observed':observed.sort_values(['fitness','variant'],ascending=[False,True]).head(4).to_dict('records')}
    while True:
        payload=dict(context,actions=[card(a,True) for a in active],step=2,prior_selected=[active[0]['action_id']] if active else [])
        count=len(client.tokenize(format_prompt(SYSTEM,payload).encode(),add_bos=False,special=True))
        if count<=settings.input:break
        if not active:raise ValueError('Context alone exceeds effective token budget')
        counts={p:sum(a['site']==p for a in active) for p in {a['site'] for a in active}}
        active.remove(max(active,key=lambda a:(counts[a['site']],a['visible_stats']['batch_cost_alone'],a['action_id'])))
    return active,context,{'full_actions':len(bank),'active_actions':len(active),'knowledge_budget_shared':True,
                          'worst_knowledge_input_tokens':count,'input_limit':settings.input,'aliases':aliases}


def claim_schema(actions):
    branches=[]
    for a in actions:
        props={'action_id':{'const':a['action_id']},'expect_A':{'enum':list(DIRECTIONS)},
            'expect_B':{'enum':list(DIRECTIONS)},'interaction':{'enum':list(INTERACTIONS)},
            'evidence_id':{'enum':[r['evidence_id'] for r in a['evidence']]+['EXPLORATION']},
            'response_to':{'const':a['revision']['revision_id'] if a.get('revision') else 'NEW'},
            'reason':{'type':'string','minLength':1,'maxLength':120}}
        branches.append({'type':'object','properties':props,'required':list(props),'additionalProperties':False})
    return {'oneOf':branches}


def validate_claim(raw,allowed):
    expected={'action_id','expect_A','expect_B','interaction','evidence_id','response_to','reason'}
    if set(raw)!=expected:raise ValueError('Invalid scoped claim fields')
    a=next((a for a in allowed if a['action_id']==raw['action_id']),None)
    if a is None:raise ValueError('Unknown or overlapping action')
    if raw['expect_A'] not in DIRECTIONS or raw['expect_B'] not in DIRECTIONS or raw['interaction'] not in INTERACTIONS:raise ValueError('Invalid direction/interaction')
    revision=a['revision']['revision_id'] if a.get('revision') else 'NEW'
    if raw['response_to']!=revision:raise ValueError('Wrong contextual revision reference')
    if raw['evidence_id'] not in {r['evidence_id'] for r in a['evidence']}|{'EXPLORATION'}:raise ValueError('False evidence citation')
    if not isinstance(raw['reason'],str) or not 1<=len(raw['reason'])<=120:raise ValueError('Invalid reason')
    if len(a['scope']['differing_positions'])!=1:raise ValueError('Interaction claim requires complete two-site cycle')
    chosen=copy.deepcopy(a);chosen['claim']={k:raw[k] for k in raw if k!='action_id'}
    return chosen


class ScientificLLM:
    def __init__(self,settings):
        self.effective=LLMSettings.parse(settings) if isinstance(settings,dict) else settings
        self.settings=LocalGGUFHypothesisGenerator(seed=self.effective.seed)
        self.settings.n_ctx=self.effective.context;self.settings.threads=self.effective.threads
        self.settings.gpu_layers=self.effective.gpu_layers
        self.last_trace={}

    def client(self):
        self.settings.check_model()
        e=self.effective
        return _load(str(self.settings.model_path),e.context,e.threads,e.seed,e.gpu_layers)

    def choose(self,selector,context,*,knowledge=False,trace_dir=None):
        from llama_cpp import LlamaGrammar
        e=self.effective
        trace={'decision_id':uuid.uuid4().hex,'calls':[],'status':'prepared','knowledge':knowledge,
               'effective_settings':e.dictionary(),'model':self.settings.check_model(),'normalized_before_inference':True}
        self.last_trace=trace
        def save():
            if trace_dir is not None:write_json(trace_dir/(trace['decision_id']+'.json'),trace)
        selected=[]
        try:
            for step in range(1,selector.target+1):
                allowed=selector.allowed(selected)
                payload=dict(context,actions=[card(a,knowledge) for a in allowed],step=step,
                             prior_selected=[a['action_id'] for a in selected])
                prompt=format_prompt(SYSTEM,payload);client=self.client()
                tokens=client.tokenize(prompt.encode(),add_bos=False,special=True)
                if len(tokens)>e.input:raise ValueError('Effective input limit exceeded')
                schema=claim_schema(allowed)
                call={'call_id':uuid.uuid4().hex,'status':'prepared','prompt':prompt,'context':payload,
                      'schema':schema,'prompt_tokens':len(tokens),'effective_settings':e.dictionary()}
                trace['calls'].append(call);save()
                start=time.perf_counter()
                response=client.create_completion(prompt=tokens,grammar=LlamaGrammar.from_json_schema(json.dumps(schema),verbose=False),
                    max_tokens=e.output_per_call,temperature=e.temperature,seed=e.seed,
                    stop=['<|eot_id|>','<|end_of_text|>'],echo=False)
                call.update(raw=response['choices'][0]['text'],finish=response['choices'][0]['finish_reason'],
                            usage=response.get('usage',{}),seconds=time.perf_counter()-start,inference_performed=True)
                save()
                if call['finish']!='stop':raise ValueError('Truncated claim; no silent retry or repair')
                selected.append(validate_claim(json.loads(call['raw']),allowed))
                call['status']='complete';save()
            trace['status']='complete';trace['selected_actions']=selected;save()
            return selected
        except Exception as exc:
            trace.update(status='failed',error=repr(exc));save();raise
