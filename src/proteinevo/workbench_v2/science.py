"""Separate per-background direction, interaction, and context-indexed observations."""
import copy
from collections import Counter

import numpy as np

from ..config import POSITIONS
from ..contextual_science import one_edit_evidence, observed_lookup, validate_action, background
from ..workbench.science import Scoring, Selector, build_bank as original_bank

DIRECTIONS=('increase','decrease','near_zero','uncertain')
INTERACTIONS=('different_magnitude','similar_magnitude','unresolved')


def direction(value,threshold):
    if value is None:return 'uncertain'
    return 'increase' if value>threshold else 'decrease' if value<-threshold else 'near_zero'


def summarize(parent,index,rows,c):
    distances=[sum(parent[j]!=r['parent'][j] for j in range(4) if j!=index) for r in rows]
    minimum=min(distances) if distances else None
    nearest=[r for r,d in zip(rows,distances) if d==minimum] if minimum is not None else []
    counts=Counter(direction(r['delta'],c['effect_threshold']) for r in nearest)
    median=float(np.median([r['delta'] for r in nearest])) if nearest else None
    guess=direction(median,c['effect_threshold'])
    n=len({background(r['parent'],index) for r in nearest})
    supported=(n>=c['min_contexts'] and minimum<=c['max_distance'] and counts[guess]/n>=.8)
    return {'nearest_distance':minimum,'distinct_backgrounds':n,'median_delta':median,
            'direction_counts':dict(sorted(counts.items())),'direction':guess if supported else 'uncertain',
            'evidence_ids':[r['evidence_id'] for r in nearest],
            'limitation':'Observed nearby backgrounds; heuristic support, not calibrated confidence'}


def retrieve(action,registry,c):
    entry=registry.get(action['substitution'])
    if not entry:return None
    i=POSITIONS.index(action['site']);related=[]
    for record in entry['contexts'].values():
        distance=min(sum(p[j]!=record['parent'][j] for j in range(4) if j!=i) for p in action['parents'])
        if distance<=c['max_distance']:related.append(dict(record,distance_to_query=distance))
    if not related:return None
    related.sort(key=lambda r:(r['distance_to_query'],r['parent'],r['child']))
    return {'revision_id':entry['revision_id'],'query_backgrounds':list(action['parents']),
            'related_contexts':related[:4],'related_context_count':len(related),
            'interpretation':'Related measured contexts only; target children remain unknown; no global directional revision'}


def rules(action,rows,c):
    result=[];i=POSITIONS.index(action['site'])
    for j,site in enumerate(POSITIONS):
        if j==i:continue
        for aa in sorted({p[j] for p in action['parents']}):
            subset=[r for r in rows if r['parent'][j]==aa]
            if len(subset)<c['min_contexts']:continue
            med=float(np.median([r['delta'] for r in subset]))
            counts=Counter(direction(r['delta'],c['effect_threshold']) for r in subset)
            result.append({'rule_id':f"K:{action['substitution']}|{site}{aa}",'when':{str(site):aa},
                'distinct_backgrounds':len({background(r['parent'],i) for r in subset}),
                'median_delta':med,'directions':dict(sorted(counts.items())),
                'evidence_ids':[r['evidence_id'] for r in subset],
                'interpretation':'Conditional association; shared labels are not independent measurements'})
    return sorted(result,key=lambda r:(-r['distinct_backgrounds'],r['rule_id']))[:2]


def build_bank(observed,scoring,c,registry):
    evidence=one_edit_evidence(observed)
    bank=original_bank(observed,scoring,limit=c['action_limit'],registry={})
    strict=[]
    for a in bank:
        if len(a['scope']['differing_positions'])!=1:continue
        i=POSITIONS.index(a['site']);rows=evidence.get(a['substitution'],[])
        a['scope']['fixed_background']={str(p):a['parents'][0][j] for j,p in enumerate(POSITIONS)
            if j!=i and p!=a['scope']['single_background_site']}
        a['scope']['design']='complete_2x2_cycle_two_known_parents_two_unknown_children'
        a['background_evidence']={p:summarize(p,i,rows,c) for p in a['parents']}
        a['revision']=retrieve(a,registry,c)
        a['knowledge_rules']=rules(a,rows,c)
        a['effect_scale']='relative_binding_fitness'
        a['effect_threshold']=c['effect_threshold'];a['interaction_threshold']=c['interaction_threshold']
        strict.append(a)
    return strict


def rule_claim(a):
    return {'expect_A':a['background_evidence'][a['parents'][0]]['direction'],
            'expect_B':a['background_evidence'][a['parents'][1]]['direction'],
            'interaction':'unresolved','response_to':a['revision']['revision_id'] if a.get('revision') else 'NEW',
            'evidence_id':a['evidence'][0]['evidence_id'] if a['evidence'] else 'EXPLORATION',
            'reason':'Nearby observed contrasts with minimum background support; otherwise abstain'}


class SharedSelector(Selector):
    """Matched cost gate, with explicitly declared limits; no information-gain claim."""
    def __init__(self,bank,scoring,c):
        self.bank,self.scoring,self.allocation=bank,scoring,'shared_cost_gate'
        self.cost_limit=c['cost_limit']
        self.eligible=[a for a in bank if a['visible_stats']['unresolved'] and
            max(a['visible_stats']['predicted_delta'])>=0 and scoring.cost([a])<=self.cost_limit+1e-9]
        from itertools import combinations
        self.pairs=[(a,b) for a,b in combinations(self.eligible,2) if not set(a['children'])&set(b['children']) and
            scoring.cost([a,b])<=self.cost_limit+1e-9]
        self.target=2 if self.pairs else 1 if self.eligible else 0

    def choose_non_llm(self,method,rng):
        selected=[]
        for _ in range(self.target):
            allowed=self.allowed(selected)
            if method=='adaptive_random':a=allowed[int(rng.integers(len(allowed)))]
            else:a=max(allowed,key=lambda a:(a['visible_stats']['median_delta'] if a['visible_stats']['median_delta'] is not None else -float('inf'),-a['visible_stats']['batch_cost_alone'],a['action_id']))
            a=copy.deepcopy(a);a['claim']=rule_claim(a);selected.append(a)
        return selected

    def metadata(self):
        result=super().metadata();result['cost_limit']=self.cost_limit
        result['budget_controller']='Maximum feasible action count; not marginal information optimization'
        return result


def ledger_update(actions,observed,measured,turn,registry,selected_ids,c):
    known,truth=observed_lookup(observed),observed_lookup(measured)
    if set(truth)!=set(selected_ids) or set(known)&set(truth):raise ValueError('Only selected new observations may update evidence')
    output=[]
    for a in actions:
        claim=a['claim'];pairs=[]
        for label,p,ch in zip(('A','B'),a['parents'],a['children']):
            if ch not in truth:continue
            delta=truth[ch]-known[p]
            pairs.append({'parent':p,'child':ch,'parent_fitness':known[p],'child_fitness':truth[ch],
                'delta':delta,'direction':direction(delta,c['effect_threshold']),'expected':claim['expect_'+label],
                'observed_round':turn,'evidence_id':f'C:{p}>{ch}'})
        if len(pairs)!=2:
            output.append({'action_id':a['action_id'],'round':turn,'status':'untested' if not pairs else 'incomplete','pairs':pairs});continue
        assertions=[p for p in pairs if p['expected']!='uncertain']
        ds=('exploration_observed' if not assertions else 'supported' if all(p['direction']==p['expected'] for p in assertions) else 'refuted')
        interaction=pairs[1]['delta']-pairs[0]['delta']
        seen='different_magnitude' if abs(interaction)>c['interaction_threshold'] else 'similar_magnitude'
        ins='exploration_observed' if claim['interaction']=='unresolved' else 'supported' if seen==claim['interaction'] else 'refuted'
        status='refuted' if 'refuted' in (ds,ins) else 'supported' if 'supported' in (ds,ins) else 'exploration_observed'
        entry=registry.setdefault(a['substitution'],{'revision_id':'R:'+a['substitution'],'contexts':{},'tests':[]})
        for p in pairs:
            i=POSITIONS.index(a['site']);key=background(p['parent'],i)
            if key in entry['contexts']:raise ValueError('A measured context cannot be silently overwritten')
            entry['contexts'][key]=dict(p,effect_scale=a['effect_scale'],effect_threshold=c['effect_threshold'])
        record={'action_id':a['action_id'],'round':turn,'claim':claim,'scope':a['scope'],'pairs':pairs,
            'direction_status':ds,'interaction_status':ins,'status':status,'asserted_directions':len(assertions),
            'interaction_observed':seen,'interaction_delta':interaction,'effect_scale':a['effect_scale']}
        entry['tests'].append(copy.deepcopy(record));output.append(dict(record,substitution=a['substitution']))
    return output
