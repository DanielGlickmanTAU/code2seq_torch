import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 2
params = {
    '--cfg': 'configs/GPS/ogbg-ppa-GPS.yaml',
    '--ogb_eval': True,
    'optim.early_stop_patience': 9999

}

params_for_exp = {
    'train.batch_size': int(32 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'seed': [1, 2],
    'nagasaki.learn_edges_weight': [False],
    'nagasaki.type': 'gin',
    'nagasaki.merge_attention': None,
    ('nagasaki.gating', 'nagasaki.edge_reduction'): [('relu', 'linear-bn'), ('softmax', 'linear')],

    'optim.base_lr': [0.001, 0.0004, 0.0002, 0.0006],
    'gt.attn_dropout': [0.0, 0.2, 0.5],

    'gt.layer_type': ['None+Nagasaki', 'CustomGatedGCN+Nagasaki'],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10]',
    # 'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10]',
    'nagasaki.edge_model_type': ['res-mlp'],
    # 'nagasaki.edge_reduction': ['bn-mlp', 'linear'],
    'nagasaki.edge_reduction': ['linear'],

    # 'nagasaki.ffn_layers': [1, 2],
    'nagasaki.ffn_layers': [2],
    'nagasaki.add_cls': [False],
    'nagasaki.symmetric_edge_reduce': [False],

    'nagasaki.project_diagonal': [True],

}

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
