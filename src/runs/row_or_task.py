import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os
import sys

params_for_exp = {
    'optim.base_lr': [0.0004],
    # 'gt.layers': [5, 10],
    # 'gt.n_layers_gnn_only': 1,
    'posenc_LapPE.enable': False,
    'dataset.node_encoder_name': "TypeDictNode+RWSE",
    # 'dataset.node_encoder_name': "TypeDictNode",
    'posenc_SignNet.enable': False,
    'posenc_RWSE.enable': True,
    'posenc_RWSE.kernel.times_func': "range(1, 21)",
    'posenc_RWSE.model': "Linear",
    'posenc_RWSE.dim_pe': 24,
    'posenc_RWSE.raw_norm_type': "BatchNorm",
    'dataset.only_color': False,
    # 'dataset.transformer_node_encoder_name': 'SignNet',
    'gt.dim_hidden': 32,
    'gnn.dim_inner': 32,
    # 'gt.layers': [10],
    # 'gt.n_layers_gnn_only': [6],
    'optim.early_stop_patience': 20,
    # 'nagasaki.edge_model_type': ['bn-mlp', 'mlp'],
    'nagasaki.edge_model_type': ['bn-mlp'],
    # 'nagasaki.edge_model_type': ['mlp'],
    # 'nagasaki.ffn_layers': [1, 2],
    # 'nagasaki.ffn_layers': [1, 2],
    # 'nagasaki.ffn_layers': [2],
    'nagasaki.ffn_layers': [1],
    'nagasaki.learn_edges_weight': [True],
    # 'nagasaki.two_diffusion': [True, False],

    'nagasaki.two_diffusion': [True],
    'gt.dropout': 0.1,
    'gt.attn_dropout': 0.1,
    'nagasaki.steps': '[1, 2, 3, 4, 5]',

    'nagasaki.kernel': ['sigmoid', 'exp', 'softmax'],
    'nagasaki.nhead': 2

}

params = {
    '--cfg': 'tests/configs/graph/row-coloring-laplace.yaml',

    '--num_rows': 4,
    # '--num_rows': 12,
    '--words_per_row': 4,
    # '--words_per_row': 2,
    '--atom_set': 11,
    '--num_unique_atoms': 4,
    '--num_unique_colors': 10,

    '--row_color_mode': 'or',
    # '--max_examples':100

}

params_for_grid_search = [
    # baseline_config.get_gps_laplace_transformer_config(),
    # baseline_config.get_RWSE_GNN_config(layers=12),
    # baseline_config.get_RWSE_GNN_config(layers=6),
    # baseline_config.get_gps_signnet_deepset_config(),
    # baseline_config.get_gnn_transformer_laplace_transformer_config(),
    # baseline_config.get_RWSE_gps_config(7),
    # baseline_config.get_gnn_transformer_signnet_deepset_config(),
    # baseline_config.get_RSWE_gnn_transformer_signnet_deepset_config(),
    # baseline_config.get_STRONG_RSWE_gnn_transformer_signnet_AFTERGNN_deepset_config(),
    # baseline_config.get_nagasaki_config(total_layers=7, gnn_layers=5, far_away=True),
    baseline_config.get_nagasaki_config(total_layers=7, gnn_layers=5),
    # baseline_config.get_nagasaki_config(total_layers=8, gnn_layers=5, far_away=True),
    # baseline_config.get_nagasaki_config(total_layers=8, gnn_layers=3)
    # baseline_config.get_nagasaki_config(total_layers=2, gnn_layers=1),

]

for p in params_for_grid_search:
    p.update(params_for_exp)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    # id = run_on_slurm(job_name, params={}, no_flag_param=p, slurm=True, sleep=False)
    id = run_on_slurm(job_name, params={}, no_flag_param=p, slurm=False, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
# while True:
#     running = os.popen("squeue |grep glick | awk '{print $1}' | xargs").read()
#     not_running = [x for x in ids if str(x) not in running]
#     len_running = len(ids) - len(not_running)
#     print(f'running {len_running}/{len(ids)} jobs. {not_running} stopped.')
#     time.sleep(10)
