from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm
import os
params_for_grid_search = {
    'num_layer': [5, 6],
    'gnn': ['gin','gcn']
}
params = {

}
os.chdir('..')
job_name = '''main_pyg.py'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
