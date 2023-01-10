import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 8
params = {
    '--cfg': 'configs/GPS/ogbg-molpcba-GPS+RWSE.yaml',
    '--ogb_eval': True,
    'optim.early_stop_patience': 9999

}

params_for_exp = {
    'train.batch_size': int(512 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'seed': [2, 3],

    'nagasaki.learn_edges_weight': [True],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16]',
    # 'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 16,32]',
    # 'nagasaki.steps': '[1, 2, 3, 4, 8, 16,32]',
    # 'nagasaki.steps': '[1, 2, 3, 4, 8,16]',
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_model_type': ['res-net'],
    # 'nagasaki.edge_model_type': ['res-mlp', 'res-net'],
    'nagasaki.edge_reduction': ['linear'],
    # 'nagasaki.edge_model_type': ['None'],
    # 'nagasaki.edge_reduction': ['softmax-linear'],
    # ('nagasaki.kernel', 'nagasaki.merge_attention',): [('softmax', 'plus')],

    ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],
    # 'dataset.node_encoder_name': 'Atom',
    # 'posenc_RWSE.enable': False,
    'posenc_RWSE.enable': True,
    'nagasaki.content_attention_only': False,

    # 'nagasaki.ffn_layers': [3, 4],
    'nagasaki.ffn_layers': [6],
    'nagasaki.add_cls': [False],
    'nagasaki.symmetric_edge_reduce': [False],

    # HANDLE THIS:

    'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    'nagasaki.project_diagonal': [True],
    'optim.early_stop_patience': 15,
    ('gnn.dim_inner', 'gt.dim_hidden'): [(400, 400)],
    # ('gnn.dim_inner', 'gt.dim_hidden'): [(440, 440)],
    # 'gt.attn_dropout': [0.4, 0.5],
    'gt.attn_dropout': [0.5],
    'optim.base_lr': [0.0003, 0.0005, 0.0008]

}

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=5)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
