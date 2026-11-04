"""Logged bootstrap/full-data ensembles with verifiable point-member reuse."""
import numpy as np
import pandas as pd

from ..models import FitnessModel
from ..contextual_science import observed_lookup
from ..workbench.run import state_digest, digest
from .tools import prediction_table


def observations(variants,fitness):
    frame=pd.DataFrame({'variant':list(variants),'fitness':np.asarray(fitness,dtype=float)})
    observed_lookup(frame)
    return frame.sort_values('variant').reset_index(drop=True)


def fit_point(kind,seed,variants,fitness):
    frame=observations(variants,fitness)
    model=FitnessModel(kind,seed).fit(frame.variant.tolist(),frame.fitness.to_numpy())
    model.known_state_sha256=state_digest(frame)
    return model


class LoggedEnsemble:
    """Full-data means all current observations, not all discovery labels."""
    def __init__(self,kind,seed,n_members=3,mode='full'):
        if mode not in {'full','bootstrap'} or type(n_members) is not int or n_members<2:raise ValueError('Invalid ensemble recipe')
        self.kind,self.seed,self.n_members,self.mode=kind,seed,n_members,mode
        self.models=[];self.provenance=[];self.new_fit_calls=0

    def fit(self,variants,fitness,first_model=None):
        frame=observations(variants,fitness);variants=frame.variant.tolist();y=frame.fitness.to_numpy();sha=state_digest(frame)
        if first_model is not None:
            if self.mode!='full' or first_model.kind!=self.kind or first_model.seed!=self.seed or first_model.estimator is None or getattr(first_model,'known_state_sha256',None)!=sha:
                raise ValueError('Reused member must match data, predictor and initialization exactly')
        rng=np.random.default_rng(self.seed);self.models=[];self.provenance=[];self.new_fit_calls=0
        for member in range(self.n_members):
            indices=rng.choice(len(frame),size=len(frame),replace=True) if self.mode=='bootstrap' else np.arange(len(frame))
            reused=member==0 and first_model is not None
            if reused:model=first_model
            elif self.mode=='full':model=fit_point(self.kind,self.seed+member,variants,y);self.new_fit_calls+=1
            else:
                model=FitnessModel(self.kind,self.seed+member).fit([variants[j] for j in indices],y[indices]);self.new_fit_calls+=1
            self.models.append(model)
            payload=[[variants[j],float(y[j])] for j in indices]
            self.provenance.append({'member':member,'model_seed':self.seed+member,'mode':self.mode,
                'observed_state_sha256':sha,'training_rows':len(indices),'unique_training_variants':len(set(x[0] for x in payload)),
                'sampled_indices':[int(x) for x in indices],'training_rows_sha256':digest(payload),'reused_point_model':reused})
        return self

    def predict_members(self,variants):
        if len(self.models)!=self.n_members:raise RuntimeError('Ensemble not fitted')
        return np.stack([m.predict(variants).astype(float) for m in self.models])


def comparison_table(variants,members,beta,batch_size,exploration_slots,shuffle_seed=None):
    table=prediction_table(variants,members,beta)
    table['selection_disagreement_log']=table.disagreement_log
    table['disagreement_source_variant']=table.variant
    if shuffle_seed is not None:
        protected=set(table.sort_values(['predicted_fitness','variant'],ascending=[False,True]).head(batch_size-exploration_slots).variant)
        remaining=table.loc[~table.variant.isin(protected)].sort_values('variant')
        order=np.random.default_rng(shuffle_seed).permutation(len(remaining))
        table.loc[remaining.index,'selection_disagreement_log']=remaining.disagreement_log.to_numpy()[order]
        table.loc[remaining.index,'disagreement_source_variant']=remaining.variant.to_numpy()[order]
    table['selection_score']=np.log1p(table.predicted_fitness)+beta*table.selection_disagreement_log
    return table


def choose_comparison(pool,method,batch_size,exploration_slots):
    if {'fitness','experimental_fitness','log1p_fitness'}&set(pool.columns):raise ValueError('Unknown outcomes cannot enter acquisition')
    if pool.variant.duplicated().any() or len(pool)<batch_size or not 0<=exploration_slots<batch_size:raise ValueError('Invalid budget')
    ranked=pool.sort_values(['predicted_fitness','variant'],ascending=[False,True])
    if method.endswith('_greedy'):return ranked.head(batch_size).variant.tolist()
    if method not in {'bootstrap_ucb','full_ucb','full_shuffled_ucb'}:raise ValueError('Unknown comparison arm')
    exploit=ranked.head(batch_size-exploration_slots).variant.tolist()
    rest=pool.loc[~pool.variant.isin(exploit)].sort_values(['selection_score','variant'],ascending=[False,True])
    return exploit+rest.head(exploration_slots).variant.tolist()
