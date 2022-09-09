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
}

params_for_exp = {
    'train.batch_size': int(512 / batch_acc),
    'optim.batch_accumulation': batch_acc,

    'nagasaki.learn_edges_weight': [True],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]',
    'nagasaki.edge_model_type': ['bn-mlp'],
    # 'nagasaki.edge_reduction': ['bn-mlp', 'linear'],
    'nagasaki.edge_reduction': ['bn-mlp'],

    # 'nagasaki.kernel': ['sigmoid'],
    'nagasaki.kernel': ['exp', 'softmax', 'exp-norm'],
    # 'nagasaki.ffn_layers': [1, 2],
    'nagasaki.ffn_layers': [1],
    'nagasaki.merge_attention': ['plus', 'gate'],

}

params_for_grid_search = [
    baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=3),
    baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=2),
    baseline_config.get_nagasaki_basic_config(total_layers=6, gnn_layers=4),
]

for p in params_for_grid_search:
    p.update(params_for_exp)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')