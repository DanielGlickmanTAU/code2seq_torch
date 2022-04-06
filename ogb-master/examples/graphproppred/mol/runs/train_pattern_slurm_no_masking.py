import time

from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

from runs.run_configs import get_pattern_gating_hyperparam_search_space

params_for_grid_search = {
    'attention_type': 'position',
    'gating': False,
    'use_batch_norm_in_transformer_mlp': [False],
    'adj_stacks': ListArgument([0, 1, 2, 3, 4]),
    'num_layer': 4,
    'num_transformer_layers': 4,
    'emb_dim': 60,
    'transformer_ff_dim': 4 * 60,
    'num_heads': [1]
}

params = {
    'drop_ratio': 0.,
    'patience': 60,
    'exp_name': 'test-no-proper-masking',
    'mask_far_away_nodes': False,
    'gnn': 'gin',
    'graph_pooling': 'sum',
}

graph_benchmark_search_params = {
    'residual': [True],
    'learning_rate': [1e-4, 1e-5],
    'gin_conv_mlp_hidden_breath': 1.,
    'lr_schedule_patience': 5,
    'lr_reduce_factor': 0.5,
    'seed': [0]
}

assert not set(graph_benchmark_search_params.keys()).intersection(
    params_for_grid_search.keys()), 'defined hyper params twice'
params_for_grid_search.update(graph_benchmark_search_params)

os.chdir('..')
job_name = '''main_pattern.py'''
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
