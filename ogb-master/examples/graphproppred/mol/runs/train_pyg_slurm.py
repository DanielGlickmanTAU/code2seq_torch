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
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True, sleep=True)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
