import sys
from core.core import run_job, unpack_script_args, run_job_cums
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.core import fit_r0
if __name__ == '__main__':
    kwargs = dict(n = 2, N = 100, Nsteps = 5000, Nproc = 5, tgt_u0 = 5, u0 = 10_000*1e2, tot_proc = 100, renorm = False, max_cores = 32, pycmd='python3')
    print(sys.argv)
    kwargs= unpack_script_args(sys.argv, **kwargs)
    kwargs['r0_Rv'] = fit_r0(tgtu0=kwargs['tgt_u0'],u0=kwargs['u0'], N = kwargs['N'])
    print(kwargs)
    del kwargs['tgt_u0']
    print('init_jobs')
    run_job_cums(**kwargs)
    print('Done')