import os

import time
import random

print(os.path)
python = os.sys.executable

slurm_file = 'my_slurm.slurm'


def get_partition_and_time_limit():
    num_jobs_in_student_batch = os.popen('squeue | grep glick | grep studentba | wc -l').read()
    num_jobs_in_student_batch = int(num_jobs_in_student_batch) if num_jobs_in_student_batch else 0
    # if 'studentb' in os.popen('squeue | grep glickman').read():
    num_jobs_that_can_run_on_studentbatch_at_one_time = 6
    if num_jobs_in_student_batch > num_jobs_that_can_run_on_studentbatch_at_one_time:
        return 'studentkillable', 'infinite'

    return 'studentbatch', '7-00:00:00'


def run_on_slurm(job_name, params, no_flag_param='', slurm=True, gpu=True, sleep=True):
    partition, time_limit = get_partition_and_time_limit()
    python_file = job_name
    python_file = python_file.replace('.py', '')
    job_name = job_name + str(time.time())
    # need to for gps main stuff
    if isinstance(no_flag_param, dict):
        no_flag_param = ' '.join([f'{key} {value}' for key, value in no_flag_param.items()])
    if slurm:
        slurm_script = f'''#! /bin/sh
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH -p {partition}
## SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus={'1' if gpu else '0'}
{python} {python_file}.py ''' + ' '.join([f'--{key} {value}' for key, value in params.items()]) + ' ' + no_flag_param
        with open(slurm_file, 'w') as f:
            f.write(slurm_script)

        job_id = os.popen(f'sbatch {slurm_file}').read()[-6:].strip()
        print(f'executing {job_name} with job id {job_id}')
        open(f'./slurm_id_{job_id}_outfile_{job_name}', 'w').write(slurm_script)
    else:
        f = f'{python} {python_file}.py ' + ' '.join([f'--{key} {value}' for key, value in params.items()])
        os.system(f"nohup sh -c ' {f} > res.txt '&")
    # os.system('chmod 700 slurm.py')

    if sleep:
        if isinstance(sleep, int):
            time.sleep(random.randint(0, sleep))
        else:
            time.sleep(random.randint(0, 15))
    else:
        time.sleep(1)
    return job_id
