import qunum.numerical as qn
from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.bbgky import BBGKYDecoupling as BBGKY
import os, torch
import time
import tqdm
from plotly import graph_objects as go 
import plotly.io as pio
import numpy as np


def mk_obj(N:int, n:int,**kwargs)->BBGKY:
    a = dict(Nsteps= int(4500), u0 = 10_000*1e2, tgt_u0 = 5, debug=False, load_data=False, dp = .1, p0 = torch.tensor(1e10), method = 'fullrk', adaptive_epsilon=5e-3, dt  = 1e-2, adaptive_time_step=True, r0_Rv = 7.5, p_start_f = .1)
    a.update(kwargs)
    return BBGKY(
        N=N, n=n, 
        data_storage_loc=os.path.join('BBGKYMuliAngleSims', f'BBGKY_{N}-{n}'), angular_dependence='random_uniform',
        timeit=True,
        **a
    )
if __name__ == '__main__':
    n = int(input('n: '))
    N = int(input('N: '))
    tc = time.time()
    O = mk_obj(N, n)
    t0 = time.time()
    for o in tqdm.tqdm(O):
        pass
    dt = time.time() - t0
    with open(os.path.join(O.data_loc, 'log.txt'), 'w') as file:
        file.write("Time for {BB}: Time to compute Index Contractions: {dtc:.2e}, Time to Solve: {dt:.2e} s\n".format(BB=str(O), dt = dt, dtc = tc - t0))
    try:
        with open(os.path.join(O.data_loc, 'times.dat', 'w')) as file:
            np.array(O.delta_t, dtype = np.float64).tofile(file)
    except:
        np.array(O.delta_t, dtype=np.float64).tofile(os.path.join(O.data_loc, 'times.dat'))
    

