import time

from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

params_for_grid_search = {
    'optim.base_lr': [0.0001, 0.0003],
    # # 'gnn.layer_type': ['gcnconv', 'GINE']
    'gt.layers': [6],
    # 'gt.n_layers_gnn_only': [2, 4],
    'gt.n_layers_gnn_only': [1, 3],
    'posenc_LapPE.model': ['Transformer'],
    # 'posenc_LapPE.layers': [3, 6],

    # 'posenc_LapPE.enable': [False]

    # 'gnn.layer_type': ['ginconv']
    # 'wandb.project': 'molpcba',
    # 'gt.layers': [10],
    # 'gt.n_layers_gnn_only': [5,1],
    'wandb.project': 'gps_vs_graphtrans_molpcba',
    # 'optim.base_lr': [0.0005, 0.0002]
    # 'gt.layers': [3],
    # 'gt.n_layers_gnn_only': [3, 1],
    # 'wandb.project': 'gps_vs_graphtrans_ppa',
    # 'optim.base_lr': [0.0003, 0.0002]
    # 'optim.base_lr': [0.0003]

}

params = {
    # '--cfg': 'tests/configs/graph/row-coloring.yaml'
    '--cfg': 'tests/configs/graph/row-coloring-laplace.yaml'
    # '--cfg': 'tests/configs/graph/row-coloring-restore.yaml'
    # '--cfg': 'configs/GPS/ogbg-ppa-GPS.yaml'
    # '--cfg': 'configs/GPS/ogbg-molpcba-GPS+RWSE.yaml'
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
