import qunum.numerical as qn
from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.bbgky import BBGKYDecoupling as BBGKY
import os, torch
import time
import tqdm
from plotly import graph_objects as go 
import plotly.io as pio


def mk_obj(N:int, n:int,**kwargs)->BBGKY:
    a = dict(Nsteps= int(4500), u0 = 10_000*1e2, tgt_u0 = 5, debug=False, load_data=False, dp = .1, p0 = torch.tensor(1e10), method = 'fullrk', adaptive_epsilon=5e-3, dt  = 1e-2, adaptive_time_step=True, r0_Rv = 7.5, p_start_f = .1)
    a.update(kwargs)
    return BBGKY(
        N=N, n=n, 
        data_storage_loc=os.path.join('BBGKYSims', f'BBGKY_{N}-{n}'),
        **a
    )
if __name__ == '__main__':
    D = {2:range(100, 120, 5), 3:range(35,75,5), 4:range(10,20,5)}
    for n, rng in zip(D.keys(), D.values()):
        for N in rng:
            tc = time.time()
            O = mk_obj(N, n)
            t0 = time.time()
            for o in tqdm.tqdm(O):
                pass
            dt = time.time() - t0
            with open(os.path.join(os.getcwd(), 'log.txt'), 'a') as file:
                file.write("Time for {BB}: Time to compute Index Contractions: {dtc:.2e}, Time to Solve: {dt:.2e} s\n".format(BB=str(O), dt = dt, dtc = tc - t0))
    pass