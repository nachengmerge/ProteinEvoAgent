"""Matched ordinary/knowledge Llama views on the same full-data prediction tools."""
from .active_science.planner import SYSTEM,add_tools,card
from .active_science.tools import attach_forecast
from .contextual_llm import format_prompt
from .knowledge import AA_GROUP
from .workbench_v2.science import Scoring,SharedSelector,build_bank


def contexts(bank,observed):
    plain={'observed_n':len(observed),'best_known':float(observed.fitness.max()),
        'top_observed':observed.sort_values(['fitness','variant'],ascending=[False,True]).head(4).to_dict('records'),
        'quantitative_author':'deterministic_prediction_tool'}
    knowledge=dict(plain,knowledge={
        'instructions':'Use properties and observed conditional contrasts as weak priors for action selection; they do not guarantee benefit or override measured evidence.',
        'scope':'Same current observations only; no extra assays, external fitness, or structural evidence.',
        'properties_are_coarse':'Charge depends on conditions; these groups do not predict binding or stability by themselves.',
        'actions':[{'id':a['action_id'],'from_property':AA_GROUP[a['from_residue']],'to_property':AA_GROUP[a['to_residue']],
            'conditional_rules':[dict(r,evidence_ids=r['evidence_ids'][:3]) for r in a['knowledge_rules']]} for a in bank]})
    return plain,knowledge


def prepare_shared(client,full,observed,settings):
    active=list(full)
    while True:
        plain,knowledge=contexts(active,observed);counts=[]
        for context in (plain,knowledge):
            payload=dict(context,actions=[card(a) for a in active],step=2,prior_selected=[active[0]['action_id']] if active else [])
            counts.append(len(client.tokenize(format_prompt(SYSTEM,payload).encode(),add_bos=False,special=True)))
        if max(counts)<=settings.input-64:break
        if not active:raise ValueError('Shared contexts do not fit the fixed input budget')
        active.remove(max(active,key=lambda a:(a['visible_stats']['batch_cost_alone'],a['action_id'])))
    return active,plain,knowledge,{'full_actions':len(full),'active_actions':len(active),
        'plain_tokens':counts[0],'knowledge_tokens':counts[1],'input_limit':settings.input,
        'same_action_cards_and_budget_for_both_conditions':True,'context_pruned_for_larger_condition':True}


def decide(observed,pool,members,c,registry,method,planner,trace_dir):
    if method not in {'tool_rule','tool_llm','knowledge_tool_llm'}:raise ValueError('Unknown Llama comparison condition')
    scoring=Scoring(pool[['variant','predicted_fitness']],c['batch_size'])
    full=add_tools(build_bank(observed,scoring,c,registry),members,c)
    bank,plain,knowledge,view=prepare_shared(planner.client(),full,observed,planner.effective)
    selector=SharedSelector(bank,scoring,c);context=knowledge if method=='knowledge_tool_llm' else plain
    if method=='tool_rule':
        actions=[]
        for a in selector.choose_non_llm('adaptive_rule',None):
            forecast=a['tool_forecast'];intent={'question':'resolve_uncertainty' if forecast['interaction']=='unresolved' else 'contrast_backgrounds' if forecast['point_interaction']=='different_magnitude' else 'maximize_binding',
                'evidence_id':a['evidence'][0]['evidence_id'] if a['evidence'] else 'MODEL_FORECAST',
                'response_to':a['revision']['revision_id'] if a.get('revision') else 'NEW',
                'reason':'Observed-evidence rule chooses action; numerical forecast belongs to deterministic tool'}
            actions.append(attach_forecast(a,forecast,intent,'rule'))
    else:actions=planner.choose(selector,context,trace_dir=trace_dir)
    return scoring.choose(actions),actions,{'bank':bank,'full_bank':full,'view':view,'context':context,
        'selector':selector.metadata(),'knowledge_enabled':method=='knowledge_tool_llm',
        'LLM_ref':planner.last_trace.get('decision_id') if method!='tool_rule' and selector.target else None}
