import time

from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

params_for_grid_search = {
    # 'attention_type': ['position', 'content'],
    'attention_type': ['position'],
    # ('num_layer', 'num_transformer_layers'): [(1, 1), (4, 4)],
    # 'pyramid_size': 10,
    'adj_stacks': [ListArgument([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
    'num_layer': 1,
    'num_transformer_layers': 1,
    'emb_dim': 100,
    'transformer_ff_dim': 4 * 100,
    'num_heads': [1],
    'seed': [1],
    'use_ffn_for_attention_weights': True,
    'attention_weights_ffn_hidden_multiplier': [2, 4, 8]
}

params = {
    'drop_ratio': 0.,
    'patience': 200,
    'exp_name': 'pyramid-10-baselines',
    'gnn': 'gin',
}

os.chdir('..')
job_name = '''main_color_pyramid.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, p, slurm=True, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
while True:
    running = os.popen("squeue |grep glick | awk '{print $1}' | xargs").read()
    not_running = [x for x in ids if str(x) not in running]
    len_running = len(ids) - len(not_running)
    print(f'running {len_running}/{len(ids)} jobs. {not_running} stopped.')
    time.sleep(10)
