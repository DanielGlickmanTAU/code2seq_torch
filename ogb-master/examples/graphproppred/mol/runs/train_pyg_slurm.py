from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm
import os

params_for_grid_search = {
    'num_layer': [6, 4],
    'num_transformer_layers': [1, 2],
    'transformer_ff_dim': [600, 1200]
}

params = {

    'gnn': 'gin',
    'graph_pooling': 'attention'
}
os.chdir('..')
job_name = '''main_pyg.py'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
