import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

params_for_exp = {
    'dataset.only_color': [False],
    'wandb.project': 'coloring-rows-shapes',
}
params = {
    '--cfg': 'tests/configs/graph/row-coloring-laplace.yaml'
}

params_for_grid_search = [
    baseline_config.get_gps_laplace_transformer_config(),
    baseline_config.get_gnn_transformer_laplace_transformer_config(),
    baseline_config.get_signnet_transformer_config()
]

for p in params_for_grid_search:
    p.update(params_for_exp)

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
