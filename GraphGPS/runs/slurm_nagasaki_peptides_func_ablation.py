import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 8

params_for_exp = {
    '--cfg': [
        'configs/GPS/peptides-func-Transformer.yaml',
        'configs/GPS/peptides-func-GraphTrans.yaml',
        'configs/GPS/peptides-func-GPS.yaml',
    ],
    # '--cfg': 'configs/GPS/peptides-func-GPS.yaml',
    '--ogb_eval': True,
    'optim.early_stop_patience': 9999,

    'train.batch_size': int(128 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    # 'seed': [1],
    'seed': [1, 2],
    # seed 5, 6, 7, 8, 9, 10

    # 'nagasaki.learn_edges_weight': [False],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16]',

    'nagasaki.edge_model_type': ['res-mlp'],
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_reduction': ['linear'],

    ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],

    'nagasaki.ffn_layers': [2],
    # 'nagasaki.add_cls': [True],
    # 'nagasaki.skip_cls_pooling': [True],
    'nagasaki.add_cls': [False],
    # 'nagasaki.add_cls': [False],
    'nagasaki.symmetric_edge_reduce': [False],

    'dataset.node_encoder_name': 'Atom',
    'posenc_RWSE.enable': False,
    'posenc_LapPE.enable': False,

    'gt.layer_type': 'None+Nagasaki',

}

baseline = {}
diffuser = {
    'gt.layer_type': 'CustomGatedGCN+Nagasaki',

    'nagasaki.learn_edges_weight': [True, False],
    'nagasaki.project_diagonal': [True]
}

params_for_grid_search = [
    baseline,
    diffuser

]

for p in params_for_grid_search:
    p.update(params_for_exp)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params_for_exp, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=True)
    ids.append(id)
print(f'submited {len(gridsearch(params_for_exp, params_for_grid_search))} jobs')
