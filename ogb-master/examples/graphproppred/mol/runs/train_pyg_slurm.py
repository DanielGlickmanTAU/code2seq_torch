from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm
import os

params_for_grid_search = {
    'num_layer': [6],
    'num_transformer_layers': [2, 4],
    # 'num_transformer_layers': [0],
    'transformer_ff_dim': [1200],
    'residual': [True],
    'distance_bias': [True],
    'num_heads': [10, 30]
    # 'receptive_fields': ['1 2 4 8', '1 1 40 40', '1 4 6 40', '1 2 40 40']
}

params = {
    'exp_name': 'graph-filter-network-distance',
    'gnn': 'gin',
    'graph_pooling': 'attention',
    'max_graph_dist': 20
}
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
