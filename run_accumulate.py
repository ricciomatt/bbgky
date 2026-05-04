import sys
from core.core import run_job, unpack_script_args, run_job_cums

if __name__ == '__main__':
    kwargs = dict(n = 2, N = 100, Nsteps = 5000, Nproc = 5, tot_proc = 100, renorm = False, max_cores = 32, pycmd='python3')
    print(sys.argv)
    kwargs= unpack_script_args(sys.argv, **kwargs)
    print(kwargs)
    run_job_cums(**kwargs)
    print('Done')