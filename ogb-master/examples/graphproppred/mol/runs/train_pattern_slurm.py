import time

from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm
import os

from runs.run_configs import get_benchmarking_gnns_hyperparam_search_space, get_params_for_position_transformer_search, \
    get_params_for_content_transformer_search

# params_for_grid_search = get_params_for_position_transformer_search()
params_for_grid_search = get_params_for_content_transformer_search()

params = {
    'drop_ratio': 0.,
    'patience': 40,
    'exp_name': 'gin-pattern',
    'gnn': 'gin',
    'graph_pooling': 'sum',
}

graph_benchmark_search_params = get_benchmarking_gnns_hyperparam_search_space(True)

assert not set(graph_benchmark_search_params.keys()).intersection(
    params_for_grid_search.keys()), 'defined hyper params twice'
params_for_grid_search.update(graph_benchmark_search_params)

os.chdir('..')
job_name = '''main_pattern.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, p, slurm=True, sleep=True)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
while True:
    running = os.popen("squeue |grep glick | awk '{print $1}' | xargs").read()
    not_running = [x for x in ids if str(x) not in running]
    len_running = len(ids) - len(not_running)
    print(f'running {len_running}/{len(ids)} jobs. {not_running} stopped.')
    time.sleep(10)
