import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os
import sys

batch_acc = 4
params = {
    '--cfg': 'configs/GPS/ogbg-code2-GPS.yaml',
}

params_for_exp = {
    'train.batch_size': int(32 / batch_acc),
    'optim.batch_accumulation': batch_acc,

    'nagasaki.learn_edges_weight': [True],

    # 'gt.dropout': [0.2, 0.5],
    # 'gt.attn_dropout': [0.2, 0.5],
    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]',
    'nagasaki.edge_model_type': ['bn-mlp'],
    # 'nagasaki.edge_reduction': ['bn-mlp', 'linear'],
    'nagasaki.edge_reduction': ['bn-mlp', 'linear'],

    # DO NOT SET TO EXP...
    # 'nagasaki.kernel': ['sigmoid', 'softmax'],
    'nagasaki.kernel': ['sigmoid'],
    # DO NOT SET TO 1... 2 is better
    'nagasaki.ffn_layers': [1, 2],
    # 'nagasaki.merge_attention': ['plus', 'gate'],
    'nagasaki.merge_attention': ['plus'],
    'dataset.node_encoder_name': 'ASTNode+RWSE'
}

params_for_grid_search = [
    baseline_config.get_nagasaki_config(total_layers=6, gnn_layers=3, rwse=True),
]

for p in params_for_grid_search:
    p.update(params_for_exp)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
