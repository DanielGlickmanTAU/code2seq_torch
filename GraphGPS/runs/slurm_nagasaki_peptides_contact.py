import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 4
params = {
    '--cfg': 'configs/GPS/pcqm-contact-Transformer.yaml',
    '--ogb_eval': True,
    'optim.early_stop_patience': 9999

}

params_for_exp = {
    'train.batch_size': int(128 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'seed': [1, 2],
    # seed 5, 6, 7, 8, 9, 10

    'nagasaki.learn_edges_weight': [True],
    # 'nagasaki.learn_edges_weight': [False],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16]',

    # 'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_model_type': ['res-mlp'],
    'nagasaki.edge_reduction': ['linear'],

    # 'nagasaki.kernel': ['sigmoid'],
    # 'nagasaki.merge_attention': ['gate'],
    # ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],
    ('nagasaki.kernel', 'nagasaki.merge_attention'): [('softmax', 'plus')],
    'nagasaki.scale_attention': True,

    # 'nagasaki.kernel': ['softmax'],
    # 'nagasaki.merge_attention': ['plus'],
    # 'nagasaki.scale_attention': True,

    'nagasaki.ffn_layers': [2],
    # only False works, it has different prediction head.
    'nagasaki.add_cls': [False],
    'nagasaki.symmetric_edge_reduce': [False],

    'dataset.node_encoder_name': 'Atom',
    'posenc_RWSE.enable': False,
    'posenc_LapPE.enable': False,

    'gt.layer_type': 'None+Nagasaki',
    'nagasaki.project_diagonal': [True],

}

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
