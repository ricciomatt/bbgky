import qunum.numerical as qn
from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.bbgky import BBGKYDecoupling as BBGKY
import os, torch, time, numpy as np, polars as pl, subprocess, re, copy
from typing import Any
def mk_obj(N:int = 100, n:int = 100, I:int = 0, r0_Rv:float = 7.5, angular_dependence = 'random_uniform', map_loc:str|None = None, load_map:bool = True, **kwargs)->BBGKY:
    a = dict(
        Nsteps= int(4500), 
        u0 = 10_000*1e2, tgt_u0 = 5, 
        debug=False, load_data=False, 
        dp = .1, p0 = torch.tensor(1e10), method = 'fullrk', 
        adaptive_epsilon=5e-3, dt  = 1e-2, adaptive_time_step=True, p_start_f = .1
    )
    for key, val in kwargs.items():
        a[key] = val
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
    done_= 0
    print('\rProgress [{A}]'.format(A = ''.join(map(lambda x: ['.','#'][int(bool(x in done_))], range(tot_proc)))))
    t0 = time.time()
    completed_procs = {}
    Nt = tot_proc/20
    while i < tot_proc:
        print('\rTotal Compute Time:{B:.2e}s\tProgress [{A}'.format(B = time.time()-t0, A = ''.join(map(lambda x: ['.','#'][int(len(done_)>=x*Nt-1)], range(1,21)))), end = "]") 
        for j in range(len(procs), Nproc):
            with open(f'{path}/log{i}.log', 'w') as file:
                temp = subprocess.Popen(['python3', 'multangle.py', f'n={n}', f'N={N}', f'Nsteps={Nsteps}', f'I={i}'], stdout=file, stderr=file, text=True)
            procs[i] = dict(proc = temp, pid=temp.pid, i=i, t = time.time())
            print(f'Initialized {i}')
            i+=1
        time.sleep(sl_time)
        del_ = set()
        for pid, proc in procs.items():
            if(proc['proc'].poll() is not None):
                del_.add(i)
                done+=i
        for d in del_:
            completed_procs[i] = copy.copy(proc[d])
            del procs[d]
    print('\n\n Completed iterations')
    return 

def unpack_script_args(inp_args:list[str], **kwargs:dict[str:Any])->dict[str:Any]:
    for key, val in map(lambda x: x.split('='), inp_args[1:]):
        if(key in kwargs):
            kwargs[key] = type(kwargs[key])(val)
        else:
            pats = {int:r"^[+-]?(0|[1-9]\d*)$", float:r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"}
            mt = True
            for tp, pat in pats.items():
                if re.match(pat, val):
                    kwargs[key] = tp(val)
                    mt = False
                    break 
            if(mt):
                kwargs[key] = val
    return kwargs