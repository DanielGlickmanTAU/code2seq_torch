import time

import GraphGPS.runs.gps_baselines_config as baseline_config
from GraphGPS.runs.gps_molpcba_config import get_hiroshima_config_molpcba
from code2seq.utils.gridsearch import gridsearch, ListArgument
from code2seq.utils.slurm import run_on_slurm
import os

params = {
    '--eval-every': 15,
    # '--hyper-batch-size': 64,
    '--hyper-batch-size': 8,
    # '--hyper-batch-size': 16,
    # '--num-nodes': 4000,
    '--num-nodes': 500,
    # '--num-nodes': 100,
    # '--embedding_type': 'attention'
    # '--normalization': 'softmax',
    # '--normalization': 'sigmoid',
    # '--normalization': run above then comment above
    '--n-hidden': 0,
    '--decode_parts': True,
    # '--project_per_layer': True
    # '--normalization': 'norm'
}

params_for_exp = {

}

os.chdir('pfedhn')
job_name = '''trainer.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False, slurm=True, wandb=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
