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

    'gt.dim_hidden': 32,
    'gnn.dim_inner': 32,

    'nagasaki.edge_model_type': ['bn-mlp'],


    'nagasaki.learn_edges_weight': [True],

    'gt.dropout': 0.,
    'gt.attn_dropout': 0.,
    'nagasaki.steps': '[1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]',


    'nagasaki.edge_reduction': ['bn-mlp', 'mlp','linear'],


    # DO NOT SET TO EXP...
    # 'nagasaki.kernel': ['sigmoid', 'softmax'],
    'nagasaki.kernel': ['sigmoid'],
    # DO NOT SET TO 1... 2 is better
    'nagasaki.ffn_layers': [2, 3],
    'nagasaki.ffn_hidden_multiplier': [2],

    # 'nagasaki.nhead': 2,
    'optim.early_stop_patience': 50

}

params = {
    '--cfg': 'tests/configs/graph/color-histogram.yaml',

    '--num_rows': 10,
    # '--num_rows': 12,
    '--words_per_row': 10,
    # '--words_per_row': 2,
    '--atom_set': 8,
    '--num_unique_atoms': 1,
    '--num_unique_colors': 20,

    '--row_color_mode': 'histogram',

}

params_for_grid_search = [
    # baseline_config.get_gps_laplace_transformer_config(),
    # baseline_config.get_RWSE_GNN_config(layers=12),
    # baseline_config.get_gps_signnet_deepset_config(n_layers=6),
    # baseline_config.get_gnn_transformer_laplace_transformer_config(),
    # baseline_config.get_RWSE_gps_config(6),
    # baseline_config.get_gnn_transformer_signnet_deepset_config(),
    # baseline_config.get_RSWE_gnn_transformer_signnet_deepset_config(),
    # baseline_config.get_STRONG_RSWE_gnn_transformer_signnet_AFTERGNN_deepset_config(),
    # baseline_config.get_nagasaki_config(total_layers=7, gnn_layers=5, far_away=True),
    # baseline_config.get_nagasaki_config(total_layers=7, gnn_layers=5),
    # baseline_config.get_content_transformer_config(total_layers=10, gnn_layers=5),
    # baseline_config.get_nagasaki_config(total_layers=8, gnn_layers=5, far_away=True),
    # baseline_config.get_nagasaki_config(total_layers=8, gnn_layers=3)
    # baseline_config.get_nagasaki_config(total_layers=3, gnn_layers=1),
    # baseline_config.get_nagasaki_config(total_layers=4, gnn_layers=1),
    baseline_config.get_nagasaki_config(total_layers=3, gnn_layers=0),
    # baseline_config.get_nagasaki_config(total_layers=6, gnn_layers=1),

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
