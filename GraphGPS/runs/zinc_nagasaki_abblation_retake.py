import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 1

params = {
    '--cfg': ['configs/GPS/zinc-GPS-ablation.yaml', 'configs/GPS/zinc-GraphTrans-ablation.yaml',
              'configs/GPS/zinc-Transformer-ablation.yaml'],
    'train.batch_size': int(32 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'optim.early_stop_patience': 100,
    'seed': [3],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10,11,12,13,14,15]',
    'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_reduction': ['linear'],

    'nagasaki.ffn_layers': [2],

    # we do not use regular rwpe, only learned.
    'dataset.node_encoder_name': 'TypeDictNode',
    'posenc_RWSE.enable': False,
    'nagasaki.add_cls': [False],

    'nagasaki.kernel': ['sigmoid'],
    'nagasaki.merge_attention': ['gate'],
    #
    # 'nagasaki.kernel': ['softmax'],
    # 'nagasaki.merge_attention': ['plus'],
    # 'nagasaki.scale_attention': [True],

    'nagasaki.symmetric_edge_reduce': [True]

}

diffuser = {
    'gt.layer_type': 'GINE+Nagasaki',

    'nagasaki.learn_edges_weight': [True],
    # 'nagasaki.project_diagonal': [True, False]
    # todo:temp just to run faster right now..
    'nagasaki.project_diagonal': [True]
}

params_for_grid_search = [
    diffuser

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
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
