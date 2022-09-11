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

                          'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10]',
                          'nagasaki.edge_model_type': ['bn-mlp'],
                          'nagasaki.edge_reduction': ['linear'],
                          # 'nagasaki.edge_reduction': ['bn-mlp'],

                          'nagasaki.kernel': ['sigmoid'],

                          'nagasaki.ffn_layers': [2],
                          'nagasaki.merge_attention': ['plus'],
                          'gt.layer_type': 'GINE+Nagasaki',
                          'nagasaki.add_cls': [True, False]

                          }

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
