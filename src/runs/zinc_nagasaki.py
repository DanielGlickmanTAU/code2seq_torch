import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

batch_acc = 2

params = {
    '--cfg': 'configs/GPS/zinc-GPS+RWSE.yaml',
    'train.batch_size': int(32 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'optim.early_stop_patience': 9999
}

params_for_grid_search = {'nagasaki.learn_edges_weight': [True],
                          # 'seed': [3, 4, 5, 6, 7, 8, 9, 10],
                          'seed': [1, 2, 3, 4],
                          'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]',
                          'nagasaki.edge_model_type': ['res-mlp'],
                          'nagasaki.edge_reduction': ['linear'],

                          ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],
                          'dataset.node_encoder_name': 'TypeDictNode',
                          'posenc_RWSE.enable': False,
                          'nagasaki.ffn_layers': [2],
                          'gt.layer_type': 'GINE+Nagasaki',
                          # 'nagasaki.add_cls': [True, False],
                          'nagasaki.add_cls': [False],
                          'nagasaki.project_diagonal': [True],
                          'nagasaki.symmetric_edge_reduce': [False]

                          }

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
