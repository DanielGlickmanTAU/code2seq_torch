import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os
import sys

batch_acc = 4

params_for_exp = {
    '--cfg': ['configs/GPS/ogbg-code2-GPS-ablation.yaml', 'configs/GPS/ogbg-code2-GraphTrans-ablation.yaml',
              'configs/GPS/ogbg-code2-Transformer-ablation.yaml'],
    'optim.early_stop_patience': 9999,
    # '--max_examples': 100_000,

    'train.batch_size': int(32 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'seed': [3],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16, 17, 18, 19,20]',
    'nagasaki.edge_model_type': ['res-mlp'],
    'nagasaki.edge_reduction': ['linear'],

    'nagasaki.ffn_layers': [2],
    'nagasaki.add_cls': [False],
    'nagasaki.symmetric_edge_reduce': [False],
}

baseline = {}
diffuser = {
    'gt.layer_type': 'CustomGatedGCN+Nagasaki',

    'nagasaki.learn_edges_weight': [True],
    'nagasaki.project_diagonal': [True, False]
}

diffuser_not_learned = {
    'gt.layer_type': 'CustomGatedGCN+Nagasaki',

    'nagasaki.learn_edges_weight': [False],
    'nagasaki.project_diagonal': [True]
}

params_for_grid_search = [
    baseline, diffuser, diffuser_not_learned

]

for p in params_for_grid_search:
    p.update(params_for_exp)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params_for_exp, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params_for_exp, params_for_grid_search))} jobs')
