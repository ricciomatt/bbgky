import sys
from core.core import run_job
if __name__ == '__main__':
    kwargs = dict(n = 2, N = 100, Nsteps = 5000, sl_time= 60, Nproc = 5, tot_proc = 100)
    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        kwargs[key] = int(val)
    run_job(**kwargs)
    print('Done')