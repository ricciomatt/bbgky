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
import argparse


def mk_obj(N:int = 100, n:int = 100, I:int = 0, angular_dependence = 'random_uniform', map_loc:str|None = None, load_map:bool = True, **kwargs)->BBGKY:
    a = dict(
        Nsteps= int(4500), 
        u0 = 10_000*1e2, tgt_u0 = 5, 
        debug=False, load_data=False, 
        dp = .1, p0 = torch.tensor(1e10), method = 'fullrk', 
        adaptive_epsilon=5e-3, dt  = 1e-2, adaptive_time_step=True, r0_Rv = 7.5, p_start_f = .1
    )
    a.update(kwargs)
    path = os.path.join(f'BBGKY{N}-{n}/Data', f'{I}_{angular_dependence}')
    if not (os.path.exists(path)):
        os.makedirs(path)
    print(path)
    return BBGKY(
        N=N, n=n, 
        data_storage_loc = path,
        maps_loc= map_loc, load_maps=load_map,
        angular_dependence=angular_dependence,
        timeit=True,
        **a
    )