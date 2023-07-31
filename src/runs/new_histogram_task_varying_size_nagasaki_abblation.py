import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os
import sys

params_for_exp = {
    'optim.base_lr': [0.0008],
    # 'seed': [1, 2, 3, 4, 5],
    'seed': [4,5,6,7,8],
    # 'optim.base_lr': [0.0004],
    'posenc_RWSE.kernel.times_func': "range(1, 21)",
    'posenc_RWSE.model': "Linear",
    'posenc_RWSE.dim_pe': 24,
    'posenc_RWSE.raw_norm_type': "BatchNorm",

    'dataset.only_color': False,

    'gt.dim_hidden': 32,
    'gnn.dim_inner': 32,

    # 'gt.dropout': [0.0, 0.2],
    'gt.dropout': [0.0],
    'gt.attn_dropout': [0.0],

    # 'nagasaki.nhead': 2,
    'optim.early_stop_patience': 100,

    'nagasaki.learn_edges_weight': [True],

    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]',
    'nagasaki.edge_model_type': ['bn-mlp'],
    'nagasaki.edge_reduction': ['linear'],
    # 'nagasaki.edge_reduction': ['bn-mlp'],

    'nagasaki.kernel': ['softmax'],

    'nagasaki.ffn_layers': [1],
    # 'nagasaki.merge_attention': ['plus', 'gate'],
    'nagasaki.merge_attention': ['plus', None],
    'nagasaki.add_cls': [True, False]
    # 'nagasaki.add_cls': [True]

}

params = {
    '--cfg': 'tests/configs/graph/color-histogram.yaml',

    '--num_rows': 10,
    '--row_sizes': '"[10,11,12,13]"',
    # '--num_rows': 12,
    '--words_per_row': 10,
    # '--words_per_row': 2,
    '--atom_set': 8,
    '--num_unique_atoms': 1,
    '--num_unique_colors': 20,

    '--row_color_mode': 'histogram',

}

params_for_grid_search = [
    # baseline_config.get_RWSE_gps_config(6),
    # baseline_config.get_RWSE_GNN_config(12),
    # baseline_config.get_laplace_transformer_config(),
    # baseline_config.get_rwse_transformer_config(),
    # baseline_config.get_gnn_transformer_config(),

    baseline_config.get_nagasaki_config(total_layers=4, gnn_layers=0),

]

assert len(params_for_grid_search) > 0

for p in params_for_grid_search:
    p.update(params_for_exp)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    # id = run_on_slurm(job_name, params={}, no_flag_param=p, slurm=True, sleep=False)
    id = run_on_slurm(job_name, params={}, no_flag_param=p)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
# while True:
#     running = os.popen("squeue |grep glick | awk '{print $1}' | xargs").read()
#     not_running = [x for x in ids if str(x) not in running]
#     len_running = len(ids) - len(not_running)
#     print(f'running {len_running}/{len(ids)} jobs. {not_running} stopped.')
#     time.sleep(10)
