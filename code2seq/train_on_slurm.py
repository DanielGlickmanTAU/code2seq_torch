from code2seq.utils.gridsearch import gridsearch
from code2seq.utils.slurm import run_on_slurm

params_for_grid_search = {
    'data_folder': [
        # '../data/out_python/compressed_0_1_VmergeTrue',
        # '../data/out_python/compressed_50_2_VmergeTrue'
        '../data/out_python/compressed_1000_3_VmergeTrue'
        '../data/out_python/compressed_1000_3_VmergeFalse'

    ],
    # 'data_folder': ['../data/out_python/uncompressed'],
    'embedding_size': [128],
}
params = {
    'config': '../config/code2seq-py150k.yaml',

}

job_name = '''code2seq_wrapper'''
for p in gridsearch(params, params_for_grid_search):
    run_on_slurm(job_name, p, slurm=True, no_flag_param="train")
print(f'submited {len(gridsearch(params, params_for_grid_search))} jobs')
