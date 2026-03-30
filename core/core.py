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
import subprocess
def mk_obj(N:int = 100, n:int = 100, I:int = 0, r0_Rv:float = 7.5, angular_dependence = 'random_uniform', map_loc:str|None = None, load_map:bool = True, **kwargs)->BBGKY:
    a = dict(
        Nsteps= int(4500), 
        u0 = 10_000*1e2, tgt_u0 = 5, 
        debug=False, load_data=False, 
        dp = .1, p0 = torch.tensor(1e10), method = 'fullrk', 
        adaptive_epsilon=5e-3, dt  = 1e-2, adaptive_time_step=True, p_start_f = .1
    )
    a.update(kwargs)
    path = os.path.join(f"BBGKY{N}-{n}-{int(r0_Rv)}/Data", f'{I}_{angular_dependence}')
    if not (os.path.exists(path)):
        os.makedirs(path)
    print(path)
    return BBGKY(
        N=N, n=n, 
        data_storage_loc = path,
        maps_loc= map_loc, load_maps=load_map,
        angular_dependence=angular_dependence,
        r0_Rv=r0_Rv,
        timeit=True,
        **a
    )

def run_job(n:int = 2, N:int = 100, Nsteps:int = 5000, sl_time:float = 60, Nproc:int = 5, tot_proc:int = 100, **kwargs):
    procs = dict()
    i = 0 
    path = os.path.join(os.getcwd(), f'logs_{N}-{n}')
    if not (os.path.exists(path)):
        os.mkdir(path)
    while i < tot_proc:
        for j in range(len(procs), Nproc):
            with open(f'{path}/log{i}.log', 'w') as file:
                temp = subprocess.Popen(['python3', 'multangle.py', f'n={n}', f'N={N}', f'Nsteps={Nsteps}', f'I={i}'], stdout=file, stderr=file, text=True)
            procs[temp.pid] = dict(proc = temp, i=i, t = time.time())
            print(f'Initialized {i}')
            i+=1
        time.sleep(sl_time)
        del_ = set()
        for pid, proc in procs.items():
            if(proc['proc'].poll() is not None):
                del_.add(pid)
                print(f'Finished {proc["i"]} in time ~ {time.time()-proc["t"]}')
        for d in del_:
            del procs[d]
    return 