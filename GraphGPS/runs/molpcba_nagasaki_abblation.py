import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 8

params = {
    '--cfg': ['configs/GPS/ogbg-molpcba-GPS-ablation.yaml', 'configs/GPS/ogbg-molpcba-GraphTrans-ablation.yaml',
              'configs/GPS/ogbg-molpcba-Transformer-ablation.yaml'],
    'train.batch_size': int(512 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'optim.early_stop_patience': 10,
    'seed': [3, 4],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10,11,12,13,14,15]',
    'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_reduction': ['bn-mlp'],

    'nagasaki.ffn_layers': [1],

    # we do not use regular rwpe, only learned.
    'dataset.node_encoder_name': 'Atom',
    'posenc_RWSE.enable': False,
    'nagasaki.kernel': ['exp-norm'],
    'nagasaki.add_cls': [False],

    'nagasaki.merge_attention': ['plus'],
}

baseline = {}
diffuser = {
    'gt.layer_type': 'CustomGatedGCN+Nagasaki',

    'nagasaki.learn_edges_weight': [True, False],
    'nagasaki.project_diagonal': [True, False]
}

params_for_grid_search = [
    baseline, diffuser

]

# params_for_grid_search = [
#     # baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=3),
#     # baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=2),
#     baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=4),
#     baseline_config.get_nagasaki_basic_config(total_layers=8, gnn_layers=4),
# ]
#
for p in params_for_grid_search:
    p.update(params)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=True)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
