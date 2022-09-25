import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os
import sys

batch_acc = 4
params = {
    # '--cfg': 'configs/GPS/ogbg-code2-sat.yaml',
    '--cfg': 'configs/GPS/ogbg-code2-ablation.yaml',
    'optim.early_stop_patience': 10,
}

params_for_exp = {
    'train.batch_size': int(32 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'seed': 2,
    'gt.ffn_multiplier': 2,
    # 'nagasaki.ffn_hidden_multiplier': 1,

    'nagasaki.learn_edges_weight': [True],
    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20]',
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_model_type': ['res-mlp'],
    'nagasaki.edge_reduction': ['linear'],

    'dataset.node_encoder_name': 'ASTNode',
    'posenc_RWSE.enable': False,

    # ('nagasaki.kernel', 'nagasaki.merge_attention',): [('sigmoid', 'gate')],

    # ('nagasaki.kernel', 'nagasaki.merge_attention',): [('softmax', 'plus')],
    # 'nagasaki.scale_attention': [True],
    # 'nagasaki.ffn_hidden_multiplier': [1],

    'nagasaki.ffn_layers': [2],
    'gt.layer_type': ['CustomGatedGCN+Nagasaki', 'CustomGatedGCN+Transformer'],
    # 'nagasaki.add_cls': [True, False],
    'nagasaki.add_cls': [False],
    # 'nagasaki.add_cls': [True],
    # 'nagasaki.skip_cls_pooling': [True],
    'nagasaki.project_diagonal': [True],
    'nagasaki.symmetric_edge_reduce': [False],
}

no_change = {}
params_for_grid_search = [
    no_change,
    # gnn->trans
    # baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=3, far_away=True),
    # transformer only
    # baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=0, far_away=True),
    # baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=2),
    # baseline_config.get_nagasaki_basic_config(total_layers=7, gnn_layers=4),
]

# params.update(baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=3, far_away=True))
params.update(baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=0, far_away=True))

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
