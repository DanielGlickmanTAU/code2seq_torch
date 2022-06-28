import time

from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

params_for_grid_search = {
    'optim.base_lr': [0.001, 0.0005],
    # 'gnn.layer_type': ['gcnconv', 'GINE']
    'gnn.layer_type': ['ginconv']

}

params = {
    # '--cfg': 'tests/configs/graph/row-coloring.yaml'
    # '--cfg': 'tests/configs/graph/row-coloring-laplace.yaml'
    # '--cfg': 'tests/configs/graph/row-coloring-restore.yaml'
}

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, slurm=True, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
while True:
    running = os.popen("squeue |grep glick | awk '{print $1}' | xargs").read()
    not_running = [x for x in ids if str(x) not in running]
    len_running = len(ids) - len(not_running)
    print(f'running {len_running}/{len(ids)} jobs. {not_running} stopped.')
    time.sleep(10)
