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
    'optim.early_stop_patience': 10,
    'nagasaki.edge_model_type': ['bn-mlp'],
    # 'nagasaki.edge_model_type': ['mlp'],
    # 'nagasaki.ffn_layers': [1, 2],
    'nagasaki.ffn_layers': [2],
    'nagasaki.learn_edges_weight': [True],
    # 'nagasaki.edge_model_type': ['linear']
    # 'nagasaki.edge_model_type': ['bn-linear']
    # 'gt.dropout': 0.,
    # 'gt.attn_dropout': 0.

}

params = {
    '--cfg': 'tests/configs/graph/row-coloring-laplace.yaml',
    # '--words_per_row': 4,
    '--num_rows': 4,
    '--words_per_row': 4,
    # '--atom_set': 9,
    '--atom_set': 10,
    # '--num_unique_atoms': 1,
    '--num_unique_colors': 2,
    # '--make_prob_of_row_half': True
    '--make_prob_of_row_half': False,
    '--shape_per_row': True
}

params_for_grid_search = [
    # baseline_config.get_gps_laplace_transformer_config(),
    # baseline_config.get_RWSE_GNN_config(layers=12),
    # baseline_config.get_RWSE_GNN_config(layers=6),
    # baseline_config.get_gps_signnet_deepset_config(),
    # baseline_config.get_gnn_transformer_laplace_transformer_config(),
    # baseline_config.get_gnn_transformer_signnet_deepset_config(),
    # baseline_config.get_RSWE_gnn_transformer_signnet_deepset_config(),
    # baseline_config.get_STRONG_RSWE_gnn_transformer_signnet_AFTERGNN_deepset_config(),
    baseline_config.get_nagasaki_config(total_layers=7, gnn_layers=5),
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
