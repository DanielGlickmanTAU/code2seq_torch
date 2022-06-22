import time

from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

# params_for_grid_search = {
#     # 'coloring_mode': ['both', 'instance', 'global'],
#     'coloring_mode': ['both', 'instance'],
#     # 'atoms_set':[1,2,3]
#     # 'atoms_set': [2, 3, 4, 5, 6, 7],
#     # 'atoms_set': [4, 5, 6, 7],
#     'atoms_set': [4],
#     'num_colors': [2, 10],
#     'edge_p': [1.]
#     # 'edge_p': [0.5]
# }

params_for_grid_search = {
    'coloring_mode': ['rows'],
    # 'atoms_set': [0, 1, 6],
    'atoms_set': [6],
    # 'num_colors': [2],
    'num_colors': [20],
    # 'num_layer': [12, 24],
    'num_layer': [ 60],
    # 'only_color': [True]
    'only_color': [True],
    # 'unique_atoms_per_example': [True]
    'unique_colors_per_example': [True],
    # 'row_size': [3, 4]
    'row_size': [4]
}

params = {
    'exp_name': 'row_coloring',
    # 'exp_name': 'row_coloring_should_overfit',
}

os.chdir('..')
job_name = '''rows_coloring_main.py'''
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
