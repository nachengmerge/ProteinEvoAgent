"""Build browser data from the included archived results; performs no inference."""
from pathlib import Path
import argparse,csv,gzip,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1]
METHODS={'random':'全局随机','point_greedy':'单 P2 直推','full_greedy':'集成直接推荐','tool_rule':'工具规则','tool_llm':'LLM Agent','knowledge_tool_llm':'知识增强 Agent'}
def number(v):
    try:x=float(v);return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None
def read(name):
    with (ROOT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def build(output):
    rows=read('results/evolution/recommendations.csv')
    records=[{'seed':int(r['panel_seed']),'method':r['method'],'round':int(r['round']),'rank':int(r['rank']),
              'variant':r['variant'],'sequence':r['full_sequence'],'mutations':r['mutations'],'predicted':number(r['predicted_fitness']),
              'actual':number(r['experimental_fitness']),'role':r['role'],'parent':r['test_parent'] or None,
              'reason':r['reason'],'action':r['action_id'] or None} for r in rows]
    with gzip.open(ROOT/'results/agent/decision_states.json.gz','rt',encoding='utf-8') as f:groups=json.load(f)
    source=json.loads((ROOT/'docs/report_source.json').read_text(encoding='utf-8'))
    data={'title':'ProteinEvoAgent','methods':METHODS,'seeds':list(range(45001,45009)),
          'positions':[39,40,41,54],'WT':'VDGV','WTSequence':'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
          'records':records,'groups':groups,
          'summary':[{k:number(v) if k!='method' else v for k,v in r.items()} for r in read('results/evolution/method_summary.csv')],
          'metrics':[{k:v if k=='method' else number(v) for k,v in r.items()} for r in read('results/evolution/round_metrics.csv')],
          'provenance':{'scope':'P34 measured historical feedback; no new inference','assayEvents':2304,'realLLMCalls':94,
                        'reportSHA256':source['report_sha256'],
                        'recommendationsSourceSHA256':source['recommendations_original_gzip_sha256']}}
    assert len(records)==2304 and len(groups)==144 and sum(len(g['rawCalls']) for g in groups.values())==94
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('window.PROTEIN_DATA = '+json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+';\n',encoding='utf-8')
    return {'records':len(records),'states':len(groups),'raw_calls':94,'new_inference':False,'output':str(output)}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'demo/data.js')
    print(json.dumps(build(p.parse_args().output),ensure_ascii=False))
