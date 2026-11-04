"""Compute quantitative forecasts from known parents and model outputs only."""
import math

import numpy as np


def classify_delta(value,threshold):
    if not np.isfinite(value) or not np.isfinite(threshold) or threshold<=0:raise ValueError('Finite effect and positive threshold required')
    if math.isclose(abs(float(value)),threshold,rel_tol=1e-9,abs_tol=1e-12):return 'near_zero'
    return 'increase' if value>threshold else 'decrease' if value<-threshold else 'near_zero'


def classify_interaction(value,threshold):
    if not np.isfinite(value) or not np.isfinite(threshold) or threshold<=0:raise ValueError('Invalid interaction input')
    return 'different_magnitude' if abs(value)>threshold and not math.isclose(abs(float(value)),threshold,rel_tol=1e-9,abs_tol=1e-12) else 'similar_magnitude'


def effect_tool(action,members,*,effect_threshold,interaction_threshold,vote_fraction=1.):
    """Members are M x 2 child predictions, aligned to canonical parents A/B."""
    if {'experimental_fitness','child_fitness','measured_children'}&set(action):raise ValueError('Unmeasured child outcomes cannot enter forecast tool')
    if len(action['scope']['differing_positions'])!=1:raise ValueError('Complete 2x2 cycle required')
    y=np.asarray(members,dtype=float);parents=np.asarray(action['parent_fitness'],dtype=float)
    if y.ndim!=2 or y.shape[1]!=2 or len(y)<1 or not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid child forecasts')
    if parents.shape!=(2,) or not np.isfinite(parents).all() or (parents<0).any() or not .5<vote_fraction<=1:raise ValueError('Invalid parent/support input')
    deltas=y-parents[None,:];interactions=deltas[:,1]-deltas[:,0]
    point=deltas.mean(axis=0);epsilon=float(interactions.mean())
    directions=[classify_delta(float(x),effect_threshold) for x in point]
    interaction=classify_interaction(epsilon,interaction_threshold)
    votes=[[classify_delta(float(v),effect_threshold) for v in deltas[:,i]] for i in range(2)]
    fractions=[sum(v==label for v in vote)/len(vote) for vote,label in zip(votes,directions)]
    ifractions=sum(classify_interaction(float(v),interaction_threshold)==interaction for v in interactions)/len(interactions)
    return {'tool':'binding_effect_arithmetic_v1','effect_scale':'relative_binding_fitness',
        'parents':list(action['parents']),'children':list(action['children']),
        'parent_fitness':parents.tolist(),'child_mean':y.mean(axis=0).tolist(),
        'point_delta':point.tolist(),'point_direction':directions,'point_interaction_delta':epsilon,
        'point_interaction':interaction,'member_deltas':deltas.tolist(),'member_interaction_delta':interactions.tolist(),
        'direction_agreement':fractions,'interaction_agreement':ifractions,'members':len(y),
        'effect_threshold':effect_threshold,'interaction_threshold':interaction_threshold,'required_vote_fraction':vote_fraction,
        'expect_A':directions[0] if fractions[0]>=vote_fraction else 'uncertain',
        'expect_B':directions[1] if fractions[1]>=vote_fraction else 'uncertain',
        'interaction':interaction if ifractions>=vote_fraction else 'unresolved',
        'uncertainty_interpretation':'Empirical model disagreement; not calibrated confidence or measurement noise',
        'target_fitness_used':False,'quantitative_author':'deterministic_tool_from_model_predictions'}


def attach_forecast(action,tool,intent,planner):
    import copy
    a=copy.deepcopy(action)
    a['intent']=dict(intent);a['tool_forecast']=tool
    a['claim']={k:tool[k] for k in ('expect_A','expect_B','interaction')}
    a['claim'].update(evidence_id=intent['evidence_id'],response_to=intent['response_to'],reason=intent['reason'],
                      quantitative_author='deterministic_tool',planner=planner,question=intent['question'])
    return a


def prediction_table(variants,members,beta):
    """Fixed log-scale UCB-style heuristic; not a statistical upper bound."""
    import pandas as pd
    values=np.asarray(members,dtype=float)
    if values.ndim!=2 or values.shape[1]!=len(variants) or len(values)<2 or not np.isfinite(values).all() or (values<0).any():raise ValueError('Need >=2 finite member prediction vectors')
    if not np.isfinite(beta) or beta<0:raise ValueError('Invalid frozen exploration coefficient')
    mean=values.mean(axis=0);spread=values.std(axis=0,ddof=1);logspread=np.log1p(values).std(axis=0,ddof=1)
    return pd.DataFrame({'variant':variants,'predicted_fitness':mean,'disagreement_raw':spread,
        'disagreement_log':logspread,'ucb_score':np.log1p(mean)+beta*logspread})


def choose_batch(pool,method,batch_size,exploration_slots,rng):
    if {'fitness','experimental_fitness','log1p_fitness'}&set(pool.columns):raise ValueError('Hidden outcomes cannot enter acquisition')
    if pool.variant.duplicated().any() or len(pool)<batch_size or not 0<=exploration_slots<batch_size:raise ValueError('Invalid selection budget')
    ordered=pool.sort_values(['predicted_fitness','variant'],ascending=[False,True],kind='stable')
    if method in ('point_greedy','ensemble_greedy'):return ordered.head(batch_size).variant.tolist()
    if method not in ('ensemble_random','ensemble_ucb'):raise ValueError('Unknown acquisition')
    exploit=ordered.head(batch_size-exploration_slots).variant.tolist()
    remaining=pool.loc[~pool.variant.isin(exploit)].sort_values('variant')
    if method=='ensemble_random':explore=list(map(str,rng.choice(remaining.variant.to_numpy(),size=exploration_slots,replace=False)))
    else:explore=remaining.sort_values(['ucb_score','variant'],ascending=[False,True],kind='stable').head(exploration_slots).variant.tolist()
    assert len(set(exploit+explore))==batch_size
    return exploit+explore
