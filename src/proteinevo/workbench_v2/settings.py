"""Strict, effective settings and byte-level inputs for reproducible studies."""
import hashlib
import importlib.metadata
import platform
from dataclasses import asdict, dataclass

from ..config import SPLITS


@dataclass(frozen=True)
class LLMSettings:
    context: int
    input: int
    output_per_call: int
    temperature: float
    seed: int
    gpu_layers: int
    threads: int

    @classmethod
    def parse(cls, value):
        if set(value) != set(cls.__dataclass_fields__): raise ValueError('Unknown/missing LLM setting')
        obj=cls(**value)
        for key in ('context','input','output_per_call','seed','gpu_layers','threads'):
            if type(getattr(obj,key)) is not int: raise ValueError('Integer LLM setting required: '+key)
        if obj.context<512 or obj.input<128 or obj.output_per_call<32 or obj.threads<1: raise ValueError('Invalid LLM budget')
        if obj.input+obj.output_per_call+32>obj.context or not 0<=obj.temperature<=2: raise ValueError('LLM budgets do not fit context')
        return obj

    def dictionary(self): return asdict(self)


def data_hashes():
    # Hash bytes, without parsing extra fitness or opening final validation/test files.
    return {name:hashlib.sha256((SPLITS/name).read_bytes()).hexdigest() for name in
            ('gb1_partition.csv.gz','gb1_initial_labeled.csv','gb1_oracle.csv.gz')}


def environment():
    versions={}
    for package in ('numpy','pandas','scipy','scikit-learn','torch','llama-cpp-python'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    return {'python':platform.python_version(),'packages':versions,'system':platform.system(),
            'machine':platform.machine()}


def validate_config(c):
    expected={'seeds','methods','rounds','candidate_budget','batch_size','predictor','device','action_limit',
              'cost_limit','effect_threshold','interaction_threshold','min_contexts','max_distance','LLM',
              'contrast_pairs','purpose'}
    if set(c)!=expected: raise ValueError('Unknown/missing config fields: '+str(set(c)^expected))
    if not c['seeds'] or not all(type(x) is int for x in c['seeds']) or len(set(c['seeds']))!=len(c['seeds']):raise ValueError('Invalid seeds')
    if not c['methods'] or len(set(c['methods']))!=len(c['methods']) or not set(c['methods'])<={'greedy','adaptive_rule','adaptive_random','adaptive_llm','adaptive_knowledge_llm'}:raise ValueError('Invalid methods')
    if not 1<=c['rounds']<=3 or not 4<=c['batch_size']<=c['candidate_budget'] or not 4<=c['action_limit']<=20:raise ValueError('Invalid study budgets')
    if c['predictor'] not in {'ridge','blend_resmlp_pairwise'} or c['device'] not in {'cpu','cuda'}:raise ValueError('Invalid predictor/device')
    if c['effect_threshold']<=0 or c['interaction_threshold']<=0 or c['cost_limit']<0:raise ValueError('Explicit positive practical-effect thresholds required')
    if c['min_contexts']<2 or c['max_distance'] not in (0,1):raise ValueError('Unsupported evidence transfer radius')
    if any(a not in c['methods'] or b not in c['methods'] for a,b in c['contrast_pairs']):raise ValueError('Contrast method absent')
    LLMSettings.parse(c['LLM'])
    return c
