"""Report-scoped entry points: data, predictors, recommendation and small evolution."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,sys
from pathlib import Path
from ..config import ROOT,RAW,SPLITS

def doctor(verify_model=False):
    from ..local_gguf_client import DEFAULT_MODEL_PATH,EXPECTED_SIZE,EXPECTED_SHA256
    packages={}
    for n in ('numpy','pandas','scipy','scikit-learn','torch','llama-cpp-python'):
        try:packages[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:packages[n]=None
    result={'python':sys.version.split()[0],'packages':packages,'project':str(ROOT),
            'initial_observations_present':(SPLITS/'gb1_initial_labeled.csv').is_file(),
            'llama_model_exists':DEFAULT_MODEL_PATH.exists(),'inference_performed':False,
            'llama_model_hash_checked':verify_model,'ready_for_cpu_demo':all(packages[n] for n in ('numpy','pandas','scipy','scikit-learn'))}
    if verify_model:
        digest=hashlib.sha256()
        if DEFAULT_MODEL_PATH.is_file():
            with DEFAULT_MODEL_PATH.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        result['llama_model_verified']=DEFAULT_MODEL_PATH.is_file() and DEFAULT_MODEL_PATH.stat().st_size==EXPECTED_SIZE and digest.hexdigest()==EXPECTED_SHA256
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    x=s.add_parser('doctor');x.add_argument('--verify-model',action='store_true')
    x=s.add_parser('prepare');x.add_argument('--source',type=Path,default=RAW);x.add_argument('--output',type=Path,required=True)
    x=s.add_parser('train');x.add_argument('--observations',type=Path,default=SPLITS/'gb1_initial_labeled.csv');x.add_argument('--predictor',choices=['ridge','blend_resmlp_pairwise'],default='ridge');x.add_argument('--device',choices=['cpu','cuda'],default='cpu');x.add_argument('--output',type=Path,required=True)
    x=s.add_parser('evaluate');x.add_argument('--model',type=Path,required=True);x.add_argument('--labels',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
    x=s.add_parser('recommend');x.add_argument('--observations',type=Path,default=SPLITS/'gb1_initial_labeled.csv');x.add_argument('--model',type=Path,required=True);x.add_argument('--method',choices=['random','full_greedy','tool_rule','tool_llm','knowledge_tool_llm'],default='full_greedy');x.add_argument('--candidates',type=Path);x.add_argument('--candidate-budget',type=int,default=800);x.add_argument('--batch-size',type=int,default=16);x.add_argument('--output',type=Path,required=True)
    s.add_parser('results')
    args=p.parse_args()
    try:
        if args.command=='doctor':result=doctor(args.verify_model)
        elif args.command=='results':
            import pandas as pd
            result=pd.read_csv(ROOT/'results/evolution/method_summary.csv').to_dict('records')
        else:
            import pandas as pd,joblib
            from . import workflow as w
            if args.command=='prepare':
                from ..dataset import prepare,make_splits
                dest=w.fresh_directory(args.output);measured=dest/'processed/gb1_measured.csv.gz'
                result={'source':prepare(args.source,measured),'splits':make_splits(measured,dest/'splits')}
            elif args.command=='train':result=w.train(pd.read_csv(args.observations),args.output,args.predictor,device=args.device)
            elif args.command=='evaluate':result=w.evaluate_model(joblib.load(args.model),pd.read_csv(args.labels),args.output)
            else:
                known=w.observations(pd.read_csv(args.observations))
                available=w.candidate_ids(pd.read_csv(args.candidates),known.variant) if args.candidates else None
                rec,_,_=w.recommend(known,joblib.load(args.model),args.output,args.method,available,args.candidate_budget,args.batch_size)
                result={'recommendations':len(rec),'measured_new_fitness':False}
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except (ValueError,RuntimeError,FileExistsError,FileNotFoundError,ImportError) as e:
        print(str(e),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
