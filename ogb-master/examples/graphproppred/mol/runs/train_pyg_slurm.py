import time

from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm
import os

params_for_grid_search = {
    'num_layer': [1, 2, 4],
    # 'num_transformer_layers': [6],
    # 'transformer_ff_dim': [2400],
    # 'residual': [True],
    # 'distance_bias': [True],
    # 'num_heads': [75, 150, 300]
}

params = {
    'exp_name': 'graph-filter-network-distance',
    'gnn': 'gin',
    'graph_pooling': 'attention',
}

graph_benchmark_search_params = {
    'learning_rate': [1e-3, 1e-4],
    'gin_conv_mlp_hidden_breath': 1.,
    'lr_schedule_patience': 5,
    'lr_reduce_factor': 0.5,

    # 'seed':list(range(4))
}

assert not set(graph_benchmark_search_params.keys()).intersection(
    params_for_grid_search.keys()), 'defined hyper params twice'
params_for_grid_search.update(graph_benchmark_search_params)

os.chdir('..')
job_name = '''main_pyg.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, p, slurm=True, sleep=True)
    ids.append(ids)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
while True:
    running = os.popen("squeue |grep glick | awk '{print $1}' | xargs").read()
    not_running = [x for x in ids if str(x) not in running]
    len_running = len(ids) - len(not_running)
    print(f'running {len_running}/{len(ids)} jobs. {not_running} stopped.')
    time.sleep(10)
