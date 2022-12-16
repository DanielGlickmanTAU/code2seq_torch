import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

# batch_acc = 8
batch_acc = 16
params = {
    '--cfg': 'configs/GPS/ogbg-molpcba-GPS+RWSE.yaml',
    '--ogb_eval': True,
    'optim.early_stop_patience': 10

}

params_for_exp = {
    'train.batch_size': int(512 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    # 'optim.base_lr': [0.0002],
    # 'optim.base_lr': [0.0005],
    # 'optim.base_lr': [0.0003],
    'seed': [2],
    'gt.layers': [8],
    'gt.n_layers_gnn_only': [4],

    # GNN -> Transformer + positional
    'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    # 'gnn.residual': [False, True],

    # GNN -> Transformer + positional
    # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8]',
    # 'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    # 'nagasaki.learn_edges_weight': [True],
    # 'nagasaki.edge_model_type': ['bn-mlp', 'res-net'],

    # GNN -> Transformer + positional + JK
    # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8]',
    # 'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    # 'nagasaki.learn_edges_weight': [True],
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    # 'nagasaki.type': 'jk',
    # 'gnn.residual': [True, False],

    # vid, content attention only
    # 'nagasaki.edge_model_type': ['bn-mlp', 'res-net'],
    # 'nagasaki.type': 'vid',
    # 'nagasaki.content_attention_only': True,
    # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8]',
    # 'gt.layer_type': 'CustomGatedGCN+Nagasaki',
    # 'nagasaki.learn_edges_weight': [True],
    # # 'nagasaki.edge_model_type': ['bn-mlp'],
    # 'gnn.residual': [False],

    # Vid
    'nagasaki.edge_model_type': ['bn-mlp'],
    ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],
    # ('nagasaki.kernel', 'nagasaki.merge_attention'): [('softmax', 'plus')],
    # 'nagasaki.scale_attention': [True],
    'nagasaki.type': 'vid',
    'nagasaki.content_attention_only': False,
    'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8,16,32,64]',
    'nagasaki.learn_edges_weight': [True],
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    'gnn.residual': [False],
    'nagasaki.ffn_layers': [1, 2],
    'nagasaki.project_diagonal': [True, False],

    # Cross Attn
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    # ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],
    # 'nagasaki.content_attention_only': [False],
    # # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8]',
    # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8,16,32,64]',
    # 'nagasaki.learn_edges_weight': [True],
    # 'gnn.residual': [False],
    # 'nagasaki.type': 'cross',
    # 'nagasaki.ffn_layers': [2],

    # Cross Attn alternate cross and self attn
    # 'nagasaki.edge_model_type': ['bn-mlp'],
    # ('nagasaki.kernel', 'nagasaki.merge_attention'): [('sigmoid', 'gate')],
    # 'nagasaki.content_attention_only': [False],
    # # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8]',
    # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8,16,32,64]',
    # 'nagasaki.learn_edges_weight': [True],
    # 'gnn.residual': [False],
    # 'nagasaki.type': 'cross',
    # 'nagasaki.interleave_self_cross_attn': True,
    # 'nagasaki.ffn_layers': [1],

    # Cross attn with affine edge projection
    # 'nagasaki.edge_model_type': ['None'],
    # ('nagasaki.kernel', 'nagasaki.merge_attention'): [('softmax', 'plus')],
    # 'nagasaki.edge_reduction': ['softmax-linear'],
    # # 'nagasaki.ffn_layers': [2],
    # 'nagasaki.content_attention_only': [False],
    # # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8]',
    # 'nagasaki.steps': '[1, 2, 3, 4, 5, 6, 7, 8,16,32,64]',
    # 'nagasaki.learn_edges_weight': [True],
    # 'gnn.residual': [False],
    # 'nagasaki.type': 'cross',
    # 'nagasaki.interleave_self_cross_attn': [True, False]

}

os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
