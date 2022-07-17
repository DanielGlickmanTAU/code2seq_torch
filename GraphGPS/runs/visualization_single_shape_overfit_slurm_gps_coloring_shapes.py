import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

params_for_exp = {
    'dataset.only_color': [False],
    # 'wandb.project': 'coloring-rows-shapes-laplace-transformer',
    # 'wandb.project': 'coloring-rows-shapes-gps-signnet-vs-gnn-t-signet',
    'wandb.project': 'single-shape-coloring-rows-shapes-visualization-no-decay',
    'optim.early_stop_patience': 9999,
    'optim.base_lr': [0.00010, 0.0002, 0.0004],
    'train.eval_period': 50,
    'gt.attn_dropout': 0.,
    'gt.layers': [11],
    'gt.n_heads': 1

}
params = {
    '--cfg': 'tests/configs/graph/row-coloring-laplace.yaml',
    '--atom_set': 9,
    '--words_per_row': 3,
    '--num_rows': 5,
    '--num_unique_atoms': 1,
    '--num_unique_colors': 4
}

params_for_grid_search = [
    # baseline_config.get_gps_laplace_transformer_config(),
    baseline_config.get_RWSE_GNN_config(layers=15),
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