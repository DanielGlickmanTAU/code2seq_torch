from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm

params_for_grid_search = {
    # 'data_folder': ['../data/out_python/compressed_20_1', '../data/out_python/compressed_10_2']
    'data_folder': ['../data/out_python/uncompressed'],
    # 'embedding_size': [128, 256],

}
model_name = 'facebook/bart-base'
params = {
    'c': '../config/code2seq-py150k.yaml',

}

job_name = '''code2seq_wrapper'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True, no_flag_param="train")
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
