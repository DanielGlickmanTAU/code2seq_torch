from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm
import os

params_for_grid_search = {
    'num_layer': [6],
    'num_transformer_layers': [1, 2, 3],
    # 'num_transformer_layers': [0],
    'transformer_ff_dim': [1200, 2400],
    'residual': [True],
    'distance_bias': [False]
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
