import qunum.numerical as qn
from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.bbgky import BBGKYDecoupling as BBGKY
import os, torch,sys
import time
import tqdm
from plotly import graph_objects as go 
import plotly.io as pio
import numpy as np
import polars as pl



def mk_obj(N:int, n:int, angular_dependence = 'random_uniform', **kwargs)->tuple[BBGKY, int]:
    a = dict(Nsteps= int(4500), u0 = 10_000*1e2, tgt_u0 = 5, debug=False, load_data=False, dp = .1, p0 = torch.tensor(1e10), method = 'fullrk', adaptive_epsilon=5e-3, dt  = 1e-2, adaptive_time_step=True, r0_Rv = 7.5, p_start_f = .1)
    a.update(kwargs)
    i = 0
    while True:
        path = os.path.join('BBGKYSims', f'BBGKY_{N}-{n}_{i}_{angular_dependence}')
        if not (os.path.exists(path)):
            break
        i+=1
    print(path)
    return BBGKY(
        N=N, n=n, 
        data_storage_loc=path,angular_dependence=angular_dependence,
        timeit=True,
        **a
    ), i 
if __name__ == '__main__':
    print(sys.argv, 'Running')
    print(os.cpu_count())
    try:
        os.environ["POLARS_MAX_THREADS"] = str((min(os.cpu_count(), int(sys.argv[-1]))))
    except:
        os.environ["POLARS_MAX_THREADS"] = str((min(os.cpu_count(), 8)))
    print(pl.thread_pool_size())
    try:
        n = int(sys.argv[1])
    except:
        n = 2
    try:
        N = int(sys.argv[2])
    except: 
        N = 100
    
    tc = time.time()
    O, i = mk_obj(N, n, angular_dependence = 'random_uniform')
    t0 = time.time()
    print('Created object in time: {dt}'.format(dt = t0 - tc))
    for o in tqdm.tqdm(O):
        pass
    dt = time.time() - t0
    with open(os.path.join(O.data_loc, f'log_{i}.txt'), 'w') as file:
        file.write("Time for {BB}: Time to compute Index Contractions: {dtc:.2e}, Time to Solve: {dt:.2e} s\n".format(BB=str(O), dt = dt, dtc = tc - t0))
    try:
        with open(os.path.join(O.data_loc, 'times.dat', 'w')) as file:
            np.array(O.delta_t, dtype = np.float64).tofile(file)
    except:
        np.array(O.delta_t, dtype=np.float64).tofile(os.path.join(O.data_loc, 'times.dat'))
    

