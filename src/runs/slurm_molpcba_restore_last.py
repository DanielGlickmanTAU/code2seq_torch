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
    'optim.early_stop_patience': 9999

}

params_for_exp = {
    'train.batch_size': int(512 / batch_acc),
    'optim.batch_accumulation': batch_acc,
    'seed': [4],
}


os.chdir('..')
job_name = '''main.py'''
ids = []
for p in gridsearch(params, params_for_exp):
    id = run_on_slurm(job_name, params={}, no_flag_param=p, sleep=False)
    ids.append(id)
print(f'submited {len(gridsearch(params, params_for_exp))} jobs')
