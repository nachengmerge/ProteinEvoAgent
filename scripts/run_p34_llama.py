"""Real local Llama baseline, matched knowledge view, and explicit CPU reproduction."""
import argparse
import hashlib
import json
import os
import time
import zipfile
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

from proteinevo.config import ROOT,SPLITS,full_sequence,mutation_names
from proteinevo.candidates import generate_proposals,sample_uniform
from proteinevo.oracle import PanelRetrospectiveOracle
from proteinevo.study_support import StudyAudit,write_json,digest_ids,discovery_for_orchestrator
from proteinevo.workbench.run import state_digest,digest,panel_ids as new_panels
from proteinevo.p1_protocol import make_panel_ids
from proteinevo.workbench_v2.settings import data_hashes,environment
from proteinevo.active_science.settings import validate_config as active_validate
from proteinevo.active_science.planner import Planner
from proteinevo.active_science.tools import prediction_table
from proteinevo.active_science.deep_ensemble import fit_point,LoggedEnsemble
from proteinevo.workbench_v2.science import ledger_update
from proteinevo.llama_baseline import decide
from proteinevo.local_gguf_client import DEFAULT_MODEL_PATH,EXPECTED_SHA256,file_sha256

VERSION='p34-local-llama-full-data-v1'
METHODS={'random','point_greedy','full_greedy','tool_rule','tool_llm','knowledge_tool_llm'}
TOOLS={'tool_rule','tool_llm','knowledge_tool_llm'}


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(c):
    if not c['methods'] or not set(c['methods'])<=METHODS or len(set(c['methods']))!=len(c['methods']):raise ValueError('Unknown/duplicate P34 methods')
    if any(a not in c['methods'] or b not in c['methods'] for a,b in c['contrast_pairs']):raise ValueError('Missing contrast arm')
    base=dict(c,methods=['point_greedy'],contrast_pairs=[],exploration_slots=0,ucb_beta=0.)
    active_validate(base);return c


class Audit(StudyAudit):
    def __init__(self,output,resume=False):
        if not resume:super().__init__(output);return
        self.output=Path(output);saved=load(self.output/'checkpoint.json');self.design=load(self.output/'preregistered_design.json')
        if saved['sha256']!=digest(saved['payload']) or saved['manifest_sha256']!=digest(self.design):raise ValueError('Checkpoint/manifest bytes differ')
        prior=load(self.output/'execution_state.json')
        if prior['status']=='complete':raise ValueError('Completed run cannot resume')
        if prior['status']=='failed':raise ValueError('Failed run needs explicit diagnosis; no automatic model retry')
        for name,value in saved['payload']['artifact_sha256'].items():
            if sha(self.output/name)!=value:raise ValueError('Completed artifact changed: '+name)
        self.started=time.perf_counter()-prior['elapsed_seconds'];self.reads=load(self.output/'read_access_log.json')
        self.completed=saved['payload']['completed'];self.original_read=pd.read_csv;pd.read_csv=self._read

    def freeze(self,design):
        files=sorted((ROOT/'src/proteinevo').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'configs/llama_baseline_v1.json',ROOT/'docs/p34_llama_protocol_cn.md',ROOT/'docs/model_baseline_selection_cn.md']
        design=dict(design,protocol=VERSION,data_sha256=data_hashes(),environment=environment(),
            code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
        self.output.mkdir(parents=True);write_json(self.output/'preregistered_design.json',design)
        with zipfile.ZipFile(self.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        self.design=design

    def check_sources(self):
        super().check_sources()
        if data_hashes()!=self.design['data_sha256']:raise RuntimeError('Discovery input bytes changed')


def panels(c,audit,mode,manifest):
    if mode=='new':return new_panels(c,audit,'new')
    if mode!='reproduce' or manifest.get('protocol')!=VERSION or manifest['config_sha256']!=digest(c):raise ValueError('Incompatible P34 reproduction')
    for name,value in manifest['code_sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('Source changed: '+name)
    if manifest['environment']!=environment() or manifest['data_sha256']!=data_hashes():raise ValueError('Runtime/data differs')
    part=pd.read_csv(SPLITS/'gb1_partition.csv.gz',usecols=['variant','mutation_count','split'])
    ids={s:make_panel_ids(part,s,(512,))[512] for s in c['seeds']};hashes={str(s):digest_ids(v) for s,v in ids.items()}
    if hashes!=manifest['panel_ids_sha256']:raise ValueError('Initial panel differs')
    return ids,hashes


def predict(observed,ids,method,c,seed,cache,fits,member_logs,attempt=0):
    state=state_digest(observed);models=cache.setdefault(state,{})
    if 'point' not in models:
        start=time.perf_counter();models['point']=fit_point(c['predictor'],seed,observed.variant,observed.fitness)
        fits.append({'panel_seed':seed,'state_sha256':state,'kind':'point','labels':len(observed),'model_fits':1,'attempt':attempt,'seconds':time.perf_counter()-start})
    if method=='point_greedy':return pd.DataFrame({'variant':ids,'predicted_fitness':models['point'].predict(ids).astype(float)}),None
    if 'full' not in models:
        start=time.perf_counter();ensemble=LoggedEnsemble(c['predictor'],seed,c['ensemble_members'],'full').fit(observed.variant,observed.fitness,models['point'])
        models['full']=ensemble;fits.append({'panel_seed':seed,'state_sha256':state,'kind':'full','labels':len(observed),'model_fits':ensemble.new_fit_calls,'attempt':attempt,'seconds':time.perf_counter()-start})
        member_logs.extend(dict(x,panel_seed=seed,attempt=attempt) for x in ensemble.provenance)
    matrix=models['full'].predict_members(ids);return prediction_table(ids,matrix,0.),matrix


def acquire_lock(output):
    import psutil
    path=Path(output)/'process_lock.json'
    if path.exists():
        prior=load(path)
        try:
            process=psutil.Process(prior['pid'])
            if abs(process.create_time()-prior['create_time'])<.01 and process.is_running():raise RuntimeError('Run process is still alive; do not start a duplicate')
        except psutil.NoSuchProcess:pass
        path.unlink()
    with path.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'create_time':psutil.Process().create_time()},stream)
    return path


def save_checkpoint(audit,records,metrics,fits,member_logs,ledgers):
    frame=pd.concat(records,ignore_index=True) if records else pd.DataFrame()
    values=frame.astype(object).where(pd.notna(frame),None).to_dict('records')
    artifacts={p.relative_to(audit.output).as_posix():sha(p) for pattern in ['panel_*/initial_observations.csv','panel_*/*/decision_*.json','panel_*/*/belief_state_*.json'] for p in audit.output.glob(pattern)}
    payload={'completed':list(audit.completed),'recommendations':values,'metrics':metrics,'fits':fits,'member_logs':member_logs,'ledgers':ledgers,'artifact_sha256':artifacts}
    write_json(audit.output/'checkpoint.json',{'manifest_sha256':digest(audit.design),'sha256':digest(payload),'payload':payload})


def run(c,output,mode='new',manifest=None,stop_after=None):
    validate(c);os.environ['PROTEINEVO_TORCH_DEVICE']=c['device'];needs_planner=bool(TOOLS&set(c['methods']));has_llm=bool({'tool_llm','knowledge_tool_llm'}&set(c['methods']))
    if needs_planner and file_sha256(DEFAULT_MODEL_PATH)!=EXPECTED_SHA256:raise ValueError('Llama GGUF bytes differ')
    if mode=='resume':manifest=load(Path(output)/'preregistered_design.json')
    audit=Audit(Path(output),resume=mode=='resume');label='freeze';lock=None;validated=False
    try:
        if mode=='resume':lock=acquire_lock(audit.output)
        ids,hashes=panels(c,audit,'reproduce' if mode=='resume' else mode,manifest)
        validated=True
        if mode!='resume':audit.freeze({'config':c,'config_sha256':digest(c),'mode':mode,'panel_ids_sha256':hashes,
            'reproduces_manifest_sha256':digest(manifest) if manifest else None,'initial_labels':512,
            'assay_events_expected':len(ids)*len(c['methods'])*c['rounds']*c['batch_size'],
            'LLM_calls_max':len(ids)*c['rounds']*2*len({'tool_llm','knowledge_tool_llm'}&set(c['methods'])),
            'model_sha256':EXPECTED_SHA256 if needs_planner else None,'model':'Meta Llama3.1 8B Instruct Q4_K_M' if needs_planner else None,
            'new_final_holdout_reads':False,'primary':['All 48 acquired mean fitness','Final best gain over initial'],
            'random_is_uniform_global':True,'shared_knowledge_input_budget':True,'UCB_acquisition_used':False,'uncertainty_is_calibrated':False})
        if mode!='resume':lock=acquire_lock(audit.output)
        if mode=='new':
            index=ROOT/'results/workbench_run_index.json';prior=load(index) if index.exists() else []
            prior.append({'manifest':str((audit.output/'preregistered_design.json').resolve()),'panel_ids_sha256':hashes,'config_sha256':digest(c)});write_json(index,prior)
        secret=discovery_for_orchestrator();planner=Planner(c['LLM']) if needs_planner else None
        records=[];metrics=[];fits=[];member_logs=[];ledgers=[];attempt=0
        if mode=='resume':
            saved=load(audit.output/'checkpoint.json')['payload'];records=[pd.DataFrame(saved['recommendations'])] if saved['recommendations'] else []
            metrics,fits,member_logs,ledgers=[saved[k] for k in ['metrics','fits','member_logs','ledgers']]
            events=load(audit.output/'resume_events.json') if (audit.output/'resume_events.json').exists() else []
            events.append({'at_utc':datetime.now(timezone.utc).isoformat(),'completed_batches_reused':len(audit.completed),'model_cache_lost':True,
                'uncommitted_inflight_work_may_be_recomputed':True});write_json(audit.output/'resume_events.json',events);attempt=len(events)
        def flush():
            if records:pd.concat(records,ignore_index=True).to_csv(audit.output/'recommendations.csv',index=False)
            pd.DataFrame(metrics).to_csv(audit.output/'round_metrics.csv',index=False)
            pd.DataFrame(fits).to_csv(audit.output/'training_cost.csv',index=False)
            write_json(audit.output/'member_training.json',member_logs);write_json(audit.output/'hypothesis_ledger.json',ledgers)
        for seed in c['seeds']:
            directory=audit.output/f'panel_{seed}';directory.mkdir(exist_ok=mode=='resume');initial=secret.loc[sorted(ids[seed])].reset_index()
            if mode=='resume' and (directory/'initial_observations.csv').exists():
                prior=pd.read_csv(directory/'initial_observations.csv',float_precision='round_trip')
                if state_digest(prior)!=state_digest(initial):raise ValueError('Initial observation values changed')
                initial=prior
            else:initial.to_csv(directory/'initial_observations.csv',index=False)
            states={m:initial.copy() for m in c['methods']};registries={m:{} for m in c['methods']};cache={}
            if mode=='resume' and records:
                old=pd.concat(records,ignore_index=True)
                for method in c['methods']:
                    for turn in range(1,c['rounds']+1):
                        if f'{seed}:{turn}:{method}' not in audit.completed:continue
                        block=old.loc[(old.panel_seed==seed)&(old.method==method)&(old['round']==turn)].sort_values('rank')
                        if len(block)!=c['batch_size']:raise ValueError('Incomplete committed batch')
                        states[method]=pd.concat([states[method],block[['variant','experimental_fitness']].rename(columns={'experimental_fitness':'fitness'})],ignore_index=True)
                        registries[method]=load(directory/method/f'belief_state_{turn}.json')
            gates={m:PanelRetrospectiveOracle(states[m].variant.tolist()) for m in c['methods']}
            for turn in range(1,c['rounds']+1):
                first_ids=[];first_full=[];first_banks=[];point=None;member0=None
                for method in c['methods']:
                    label=f'{seed}:{turn}:{method}'
                    if label in audit.completed:continue
                    audit.state('running',label);print('START '+label,flush=True)
                    observed=states[method].sort_values('variant').reset_index(drop=True);state=state_digest(observed);folder=directory/method;folder.mkdir(exist_ok=True)
                    available=gates[method].available_variants();actions=[];meta={};members=None
                    if method=='random':
                        proposals=sample_uniform(available,c['candidate_budget'],np.random.default_rng(seed+1000*turn))
                        chosen=list(map(str,np.random.default_rng(seed+2000*turn).choice(np.array(proposals),c['batch_size'],replace=False)))
                        pool=pd.DataFrame({'variant':proposals,'predicted_fitness':np.nan});reference=[];cost=None;extra=0
                        meta={'proposal_source':'uniform_global','reference_greedy_ids':[],'predicted_mean_cost':None,'additional_candidates':0,'test_slots':0}
                    else:
                        proposals,_=generate_proposals(available,observed,[],c['candidate_budget'],np.random.default_rng(seed+1000*turn))
                        pool,members=predict(observed,proposals,method,c,seed,cache,fits,member_logs,attempt)
                        if members is not None:np.savez_compressed(folder/f'members_{turn}.npz',variant=np.array(proposals,dtype=str),predictions=members)
                        reference=pool.sort_values(['predicted_fitness','variant'],ascending=[False,True]).head(c['batch_size']).variant.tolist()
                        if method in TOOLS:
                            lookup=dict(zip(proposals,members.T));chosen,actions,meta=decide(observed,pool,lookup,c,registries[method],method,planner,folder/'llm_calls')
                        else:chosen=reference
                        cost=max(float(pool.set_index('variant').loc[reference,'predicted_fitness'].mean()-pool.set_index('variant').loc[chosen,'predicted_fitness'].mean()),0.)
                        extra=len(set(chosen)-set(reference));tests={v for a in actions for v in a['children']}
                        meta.update(proposal_source='same_elite_neighborhood_generator',reference_greedy_ids=reference,predicted_mean_cost=cost,
                            additional_candidates=extra,test_slots=len(tests),tests_already_in_greedy=len(tests&set(reference)))
                        if turn==1:
                            first_ids.append(digest_ids(proposals))
                            if members is not None:first_full.append(members.tobytes());member0=members[0]
                            else:point=pool.predicted_fitness.to_numpy()
                            if method in TOOLS:first_banks.append(digest({'bank':meta['bank'],'selector':meta['selector']}))
                    selected=pool.set_index('variant').loc[chosen].reset_index().copy();tests={ch:(a,p) for a in actions for p,ch in zip(a['parents'],a['children'])}
                    selected['role']=['hypothesis_test' if v in tests else 'random' if method=='random' else 'exploitation' for v in chosen]
                    selected['action_id']=[tests[v][0]['action_id'] if v in tests else None for v in chosen]
                    selected['test_parent']=[tests[v][1] if v in tests else None for v in chosen]
                    selected['reason']=[tests[v][0]['intent']['reason'] if v in tests else 'Uniform random reference' if method=='random' else 'Highest predicted fitness after reserved tests' for v in chosen]
                    pool['selected']=pool.variant.isin(chosen);pool.to_csv(folder/f'candidates_{turn}.csv.gz',index=False)
                    write_json(folder/f'decision_{turn}.json',{'observed_sha256':state,'chosen':chosen,'actions':actions,'metadata':meta})
                    measured=gates[method].measure(chosen);feedback=ledger_update(actions,observed,measured,turn,registries[method],chosen,c)
                    write_json(folder/f'belief_state_{turn}.json',registries[method]);ledgers.extend(dict(x,panel_seed=seed,method=method) for x in feedback)
                    selected['experimental_fitness']=selected.variant.map(measured.set_index('variant').fitness)
                    selected['panel_seed'],selected['method'],selected['round']=seed,method,turn;selected['rank']=range(1,len(chosen)+1)
                    selected['full_sequence']=selected.variant.map(full_sequence);selected['mutations']=selected.variant.map(lambda v:';'.join(mutation_names(v)));records.append(selected)
                    best=max(float(observed.fitness.max()),float(measured.fitness.max()))
                    metrics.append({'panel_seed':seed,'method':method,'round':turn,'observed_n':len(observed),'mean_fitness':float(measured.fitness.mean()),
                        'batch_best':float(measured.fitness.max()),'initial_best':float(initial.fitness.max()),'cumulative_best':best,'gain_over_initial':best-float(initial.fitness.max()),
                        'hits_ge5':int((measured.fitness>=5).sum()),'additional_candidates':extra,'test_slots':len(tests),'predicted_mean_cost':cost})
                    states[method]=pd.concat([observed,measured],ignore_index=True);assert len(states[method])==512+turn*c['batch_size']
                    audit.completed.append(label);save_checkpoint(audit,records,metrics,fits,member_logs,ledgers);flush()
                    print(f'DONE {label} mean={measured.fitness.mean():.4f} tests={len(tests)}',flush=True)
                    if stop_after is not None and len(audit.completed)>=stop_after:
                        audit.state('interrupted_for_validation',label);return
                assert len(set(first_ids))<=1 and len(set(first_full))<=1 and len(set(first_banks))<=1
                if point is not None and member0 is not None:assert np.array_equal(point,member0)
        panel=pd.concat(records).groupby(['panel_seed','method']).experimental_fitness.agg(['mean','max']).reset_index();panel.to_csv(audit.output/'panel_summary.csv',index=False)
        panel.groupby('method')['mean'].agg(['mean','std','count']).to_csv(audit.output/'method_summary.csv')
        calls=[a for p in audit.output.glob('panel_*/*/llm_calls/*.json') for a in load(p)['calls']]
        write_json(audit.output/'audit_summary.json',{'assay_events':sum(len(r) for r in records),'LLM_calls':len(calls),'individual_P2_fits':sum(r['model_fits'] for r in fits),
            'fit_states':len(fits),'member_training_records':len(member_logs),'point_members_reused':sum(r['reused_point_model'] for r in member_logs),
            'input_tokens':sum(r['usage']['prompt_tokens'] for r in calls),'output_tokens':sum(r['usage']['completion_tokens'] for r in calls),
            'tool_actions':len(ledgers),'new_final_holdout_reads':False,'ESM_used':False})
        audit.check_sources();audit.state('complete');print('COMPLETE '+str(audit.output),flush=True)
    except Exception as exc:
        if audit.output.exists() and validated:audit.state('failed',label,repr(exc))
        raise
    finally:
        audit.close()
        if lock is not None and lock.exists() and load(lock)['pid']==os.getpid():lock.unlink()


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    x=s.add_parser('run');x.add_argument('--config',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
    x=s.add_parser('reproduce');x.add_argument('--manifest',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
    x=s.add_parser('resume');x.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='resume':return run(load(a.output/'preregistered_design.json')['config'],a.output,'resume')
    if a.command=='reproduce':
        m=load(a.manifest);return run(m['config'],a.output,'reproduce',m)
    return run(load(a.config),a.output)


if __name__=='__main__':main()
