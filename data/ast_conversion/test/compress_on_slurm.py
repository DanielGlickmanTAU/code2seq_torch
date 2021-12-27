from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm

params_for_grid_search = {
    'max_word_joins': [2, 3, 5],
    'vocab_size': [100, 1_000, 10_000]
}
params = {
}

job_name = '''compress_script'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True, gpu=False,sleep=False)
    print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
