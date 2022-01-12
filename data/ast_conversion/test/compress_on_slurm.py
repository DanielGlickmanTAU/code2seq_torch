from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm

params_for_grid_search = {
    # 'max_word_joins': [1, 2],
    'max_word_joins': [3],
    # 'vocab_size': [10, 50, 100],
    'vocab_size': [1000],
    'merging_value_nodes': [False,True]

    # 'max_word_joins': [1],
    # 'vocab_size': [0],
    # 'merging_value_nodes': [True]
}
params = {
    # 'limit': 100
}

job_name = '''compress_script'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True, gpu=False, sleep=10)
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
