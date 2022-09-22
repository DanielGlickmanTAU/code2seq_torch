import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 1
params = {
    '--cfg': 'configs/GPS/mnist-GPS.yaml',
    # '--cfg':'configs/GPS/cifar10-GPS.yaml',
    # '--cfg':'configs/GPS/pattern-GPS.yaml',
    # '--cfg':'configs/GPS/cluster-GPS.yaml',

    '--ogb_eval': True,
    'optim.early_stop_patience': 9999

}

params_for_exp = {
    'seed': [0],
    # seed 5, 6, 7, 8, 9, 10

    'nagasaki.learn_edges_weight': [True],
    # 'nagasaki.learn_edges_weight': [False],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10]',

    'nagasaki.edge_model_type': ['res-mlp'],
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_reduction': ['linear'],

    ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],

    # ('nagasaki.kernel', 'nagasaki.merge_attention'): [('softmax', 'plus')],
    # 'nagasaki.scale_attention': True,

    'nagasaki.ffn_layers': [2],
    # 'nagasaki.add_cls': [True],
    # 'nagasaki.skip_cls_pooling': [True],
    'nagasaki.add_cls': [False],
    # 'nagasaki.add_cls': [False],
    'nagasaki.symmetric_edge_reduce': [False],

    'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    'nagasaki.project_diagonal': [True],

    'dataset.node_encoder_name': 'LinearNode',
    # 'dataset.node_encoder': False,

    'posenc_RWSE.enable': False,
    'posenc_LapPE.enable': False,
}

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
