from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm

params_for_grid_search = {
    'max_word_joins': [1, 2],
    'vocab_size': [10, 50, 100]
}
params = {
    'limit': 20_000
}

job_name = '''compress_script'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True, gpu=False, sleep=False)
    print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
