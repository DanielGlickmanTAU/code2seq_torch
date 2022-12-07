import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 4
params = {
    '--cfg': 'configs/GPS/ogbg-ppa-GPS.yaml',
    '--ogb_eval': True,
    'optim.early_stop_patience': 9999

}

params_for_exp = {
    'train.batch_size': int(32 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    # 'seed': [5, 6, 7, 8, 9, 10],
    'seed': [5],

    'nagasaki.learn_edges_weight': [True],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10]',
    # 'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10]',
    'nagasaki.edge_model_type': ['res-mlp'],
    # 'nagasaki.edge_reduction': ['bn-mlp', 'linear'],
    'nagasaki.edge_reduction': ['linear'],

    # 'nagasaki.ffn_layers': [1, 2],
    'nagasaki.ffn_layers': [2],
    'nagasaki.add_cls': [False],
    'nagasaki.symmetric_edge_reduce': [False],

    ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],
    'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    'nagasaki.project_diagonal': [True],

    'nagasaki.content_attention_only': [False, True],
    'nagasaki.type': 'vid',
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    'gnn.residual': [False],
    'gt.n_layers_gnn_only': [3, 4],
    'gt.layers': [8],

}

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
