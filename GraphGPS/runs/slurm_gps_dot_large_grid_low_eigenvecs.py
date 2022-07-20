import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

params_for_exp = {
    'wandb.project': 'dot_large_grid_low_freq',
    'optim.base_lr': [0.0001, 0.0002, 0.0004],
    'gt.layers': [5, 10],
    'gt.n_layers_gnn_only': 1,
    'posenc_LapPE.enable': False,
    'posenc_LapPE.layers': 0,
    'dataset.node_encoder_name': "TypeDictNode+RWSE",
    # 'dataset.node_encoder_name': "TypeDictNode",
    'posenc_SignNet.enable': True,
    'posenc_SignNet.model': "DeepSet",
    'posenc_SignNet.dim_pe': 16,
    'posenc_SignNet.layers': 3,
    'posenc_SignNet.post_layers': 2,
    'posenc_RWSE.enable': True,
    'posenc_RWSE.kernel.times_func': "range(1, 21)",
    'posenc_RWSE.model': "Linear",
    'posenc_RWSE.dim_pe': 24,
    'posenc_RWSE.raw_norm_type': "BatchNorm",
    'dataset.only_color': False,
    'dataset.transformer_node_encoder_name': 'SignNet',
    'gt.dim_hidden': 32,
    'gnn.dim_inner': 32,
    'posenc_SignNet.eigen.max_freqs': [3],
}

params = {
    '--cfg': 'tests/configs/graph/row-coloring-laplace.yaml',
    '--atom_set': 8,
    '--words_per_row': 5,
    '--num_rows': 8,
    '--num_unique_atoms': 1,
    '--num_unique_colors': 2,
    '--make_prob_of_row_half': True
}

params_for_grid_search = [
    # baseline_config.get_gps_laplace_transformer_config(),
    # baseline_config.get_RWSE_GNN_config(layers=15),
    # baseline_config.get_gps_signnet_deepset_config(),
    # baseline_config.get_gnn_transformer_laplace_transformer_config(),
    # baseline_config.get_gnn_transformer_signnet_deepset_config(),
    # baseline_config.get_RSWE_gnn_transformer_signnet_deepset_config(),
    baseline_config.get_STRONG_RSWE_gnn_transformer_signnet_AFTERGNN_deepset_config(),

]

for p in params_for_grid_search:
    p.update(params_for_exp)

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_grid_search):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, slurm=True, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
while True:
    running = os.popen("squeue |grep glick | awk '{print $1}' | xargs").read()
    not_running = [x for x in ids if str(x) not in running]
    len_running = len(ids) - len(not_running)
    print(f'running {len_running}/{len(ids)} jobs. {not_running} stopped.')
    time.sleep(10)
