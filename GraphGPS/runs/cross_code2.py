import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os
import sys

batch_acc = 4
params = {
    # '--cfg': 'configs/GPS/ogbg-code2-sat.yaml',
    '--cfg': 'configs/GPS/ogbg-code2-GPS.yaml',
    'optim.max_epoch': 60
}

params_for_exp = {
    'train.batch_size': int(32 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'seed': [7],
    'gt.ffn_multiplier': 4,
    # 'nagasaki.ffn_hidden_multiplier': 1,

    'nagasaki.learn_edges_weight': [True],
    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 16,32,64,128,256]',
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_model_type': ['None'],
    'nagasaki.edge_reduction': ['softmax-linear'],

    'dataset.node_encoder_name': 'ASTNode',
    'posenc_RWSE.enable': False,

    # ('nagasaki.kernel', 'nagasaki.merge_attention',): [('sigmoid', 'gate')],

    ('nagasaki.kernel', 'nagasaki.merge_attention',): [('softmax', 'plus')],
    # 'nagasaki.scale_attention': [True],
    # 'nagasaki.ffn_hidden_multiplier': [1],

    'nagasaki.ffn_layers': [2],
    'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    'nagasaki.type': 'cross',
    'gt.n_layers_gnn_only': 3,
    'gt.layers': 6,

    # 'nagasaki.add_cls': [True, False],
    'nagasaki.add_cls': [False],
    # 'nagasaki.add_cls': [True],
    # 'nagasaki.skip_cls_pooling': [True],
    'nagasaki.project_diagonal': [True],
    'nagasaki.symmetric_edge_reduce': [False],
}

params_for_grid_search = [
    # baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=3),
    # baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=2),
    # baseline_config.get_nagasaki_basic_config(total_layers=7, gnn_layers=4),
]
#
for p in params_for_grid_search:
    p.update(params)
os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
