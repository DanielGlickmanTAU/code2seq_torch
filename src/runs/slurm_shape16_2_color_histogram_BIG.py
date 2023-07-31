import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os
import sys

params_for_exp = {
    'optim.base_lr': [0.0004],
    'posenc_RWSE.kernel.times_func': "range(1, 21)",
    'posenc_RWSE.model': "Linear",
    'posenc_RWSE.dim_pe': 24,
    'posenc_RWSE.raw_norm_type': "BatchNorm",

    'dataset.only_color': False,

    'nagasaki.ffn_hidden_multiplier': [1],
    'nagasaki.ffn_layers': [1],

    'nagasaki.learn_edges_weight': [True],

    # 'gt.dropout': [0.2],
    'gt.attn_dropout': [0.2],
    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20]',

    'nagasaki.edge_reduction': ['linear'],

    'optim.early_stop_patience': 250,

    'nagasaki.edge_model_type': ['bn-mlp'],
    ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],
    'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    'gnn.residual': [False],
    'nagasaki.type': ['vid', 'transformer'],
    'nagasaki.content_attention_only': [False, True],

}

params = {
    '--cfg': 'tests/configs/graph/color-histogram.yaml',

    '--num_rows': 10,
    '--words_per_row': 10,
    '--atom_set': 16,
    '--num_unique_atoms': 3,
    '--num_unique_colors': 2,

    '--row_color_mode': 'histogram',

}

params_for_grid_search = [
    baseline_config.get_nagasaki_config(total_layers=5, gnn_layers=3),
    baseline_config.get_nagasaki_config(total_layers=3),
    # baseline_config.get_nagasaki_config(total_layers=4, gnn_layers=2),
    # baseline_config.get_RWSE_GNN_config(layers=22)
]

assert len(params_for_grid_search) > 0

for p in params_for_grid_search:
    p.update(params_for_exp)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    # id = run_on_slurm(job_name, params={}, no_flag_param=p, slurm=True, sleep=False)
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
# while True:
#     running = os.popen("squeue |grep glick | awk '{print $1}' | xargs").read()
#     not_running = [x for x in ids if str(x) not in running]
#     len_running = len(ids) - len(not_running)
#     print(f'running {len_running}/{len(ids)} jobs. {not_running} stopped.')
#     time.sleep(10)
