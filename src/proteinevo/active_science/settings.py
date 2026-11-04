"""Keep the scoped scientific conventions and explicitly declare new acquisition settings."""
from ..workbench_v2.settings import LLMSettings, data_hashes, environment, validate_config as scoped_validate

METHODS={'point_greedy','ensemble_greedy','ensemble_random','ensemble_ucb','tool_rule','tool_llm'}
NEW={'ensemble_members','exploration_slots','ucb_beta','tool_vote_fraction'}


def validate_config(c):
    base={k:v for k,v in c.items() if k not in NEW}
    base['methods']=['greedy'];base['contrast_pairs']=[]
    scoped_validate(base)
    if set(c)-set(base)!=NEW:raise ValueError('Missing active-learning settings')
    if not c['methods'] or len(set(c['methods']))!=len(c['methods']) or not set(c['methods'])<=METHODS:raise ValueError('Unknown/duplicate active methods')
    if type(c['ensemble_members']) is not int or not 2<=c['ensemble_members']<=8:raise ValueError('Invalid ensemble size')
    if type(c['exploration_slots']) is not int or not 0<=c['exploration_slots']<c['batch_size']:raise ValueError('Invalid exploration fraction')
    if not 0<=c['ucb_beta']<=5 or not .5<c['tool_vote_fraction']<=1:raise ValueError('Invalid fixed acquisition/support settings')
    if any(a not in c['methods'] or b not in c['methods'] for a,b in c['contrast_pairs']):raise ValueError('Missing contrast arm')
    return c
