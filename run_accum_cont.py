import sys
from core.run import run_job_cum_from_start as run_jobs
from core import unpack_script_args
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.core import fit_r0

# Have to finish this logic for the parallel calls
if __name__ == '__main__':
    kwargs = dict(n = 2, N = 100, Nsteps = 5000, Nproc = 5, tgt_u0 = 5, u0 = 10_000*1e2, tot_proc = 100, renorm = False, max_cores = 32, pycmd='python3')
    print(sys.argv)
    kwargs= unpack_script_args(sys.argv, **kwargs)
    kwargs['r0_Rv'] = fit_r0(tgtu0=kwargs['tgt_u0'],u0=kwargs['u0'], N = kwargs['N'])
    print(kwargs)
    del kwargs['tgt_u0']
    print('init_jobs')
    run_jobs(**kwargs)
    print('Done')