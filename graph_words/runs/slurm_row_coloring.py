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
    'atoms_set': [1, 6],
    # 'num_colors': [2],
    'num_colors': [2],
    'edge_p': [1.],
    # 'num_layer': [12, 24],
    'num_layer': [24],
    # 'only_color': [True]
    'only_color': [False],
    'unique_atoms_per_example': [True]
}

params = {
    'exp_name': 'row_coloring',
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
