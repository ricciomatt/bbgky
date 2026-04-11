import qunum.numerical as qn
from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.bbgky import BBGKYDecoupling as BBGKY
import os, torch, time, numpy as np, polars as pl, subprocess, re, copy, glob, datetime , tqdm
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
def load_obj(
        N:int = 100, 
        n:int = 100, 
        I:int = 0,  
        r0_Rv:int = 4,
        take_input:bool = True,
        angular_dependence = 'random_uniform', 
        map_loc:str|None = None, 
        load_map:bool = True, 
        **kwargs
    )->BBGKY:
    a = dict(
        Nsteps= int(4500), 
        u0 = 10_000*1e2, tgt_u0 = 5, 
        debug=False, 
        dp = .1, p0 = torch.tensor(1e10), method = 'fullrk', 
        adaptive_epsilon=5e-3, dt  = 1e-2, adaptive_time_step=True, p_start_f = .1
    )
    import glob

    path = glob.glob(f'./BBGKY{N}-{n}*')
    if(take_input):
        print('\n'.join(map(lambda x: f'{x[0]}) {x[1]}', enumerate(path))))
        ix = input('Which dir? ') 
        ix = 0 if ix == '' else int(ix)
        print()
        path = glob.glob(os.path.join(path[ix], 'Data', f'*_{angular_dependence}'))
        print('\n'.join(map(lambda x: f'{x[0]}) {x[1]}', enumerate(path))))
        ix = input('Which dir? ') 
        ix = 0 if ix == '' else int(ix)
        path = path[ix]
    else:
        path = os.path.join(f"BBGKY{N}-{n}-{int(r0_Rv)}/Data", f'{I}_{angular_dependence}')
    assert os.path.exists(path), FileNotFoundError(f'No File at {path}')
    for key, val in kwargs.items():
        a[key] = val
    if not (os.path.exists(path)):
        os.makedirs(path)
    return BBGKY(
        N=N, 
        n=n, 
        load_data= True,
        data_storage_loc = path,
        maps_loc= map_loc, 
        load_maps=load_map,
        angular_dependence=angular_dependence,
        r0_Rv=r0_Rv,
        timeit=True,
        **a
    )


def run_job(n:int = 2, N:int = 100, Nsteps:int = 5000, sl_time:float = 60, Nproc:int = 5, tot_proc:int = 100, **kwargs):
    print(N,n,sl_time)
    procs = dict()
    i = 0 
    path = os.path.join(os.getcwd(), 'logs')
    if not (os.path.exists(path)):
        os.mkdir(path)
    done = 0
    t0 = time.time()
    completed_procs = {}
    def init_proc(i:int = i):
        with open(f'{path}/log_{i}_{time.time()}.log', 'w') as file:
            temp = subprocess.Popen(['python3', 'multangle.py', f'n={n}', f'N={N}', f'Nsteps={Nsteps}', f'I={i}', *(f"{key}={val}" for key,val in kwargs.items())], stdout=file, stderr=file, text=True)
        return temp 
    def monitor(procs:dict[dict], done:int, completed_procs:dict[dict], C:int)->tuple[dict[dict], int, dict[dict], float]:
        del_ = set()
        for pid, proc in procs.items():
            if(proc['proc'].poll() is not None):
                del_.add(pid)
                done += 1
                pbar.update(1)
            else:
                procs[pid]['cycles']+=1
        for d in del_:
            completed_procs[i] = copy.copy(procs[d])
            completed_procs[i]['tf'] = time.time()
            del procs[d]
        if(len(del_) != 0):
            C = sum(comp['cycles'] for comp in completed_procs.values())/done
        pbar.set_description(f'Spawned {i}/100 Avg Num Cycles = {C}')
        return procs, done, completed_procs, C
    done = 0
    C = 0
    with tqdm.tqdm(total = tot_proc, desc = f'Spawned {i}/100 Avg Num Cycles = {C}', ncols = 80 ) as pbar:
        while i < tot_proc:
            for j in range(len(procs), Nproc):
                temp = init_proc(i=i)
                procs[i] = dict(proc = temp, pid=temp.pid, i=i, t = time.time(), cycles= 0)
                i+=1
                pbar.set_description(f'Spawned {i}/100 Avg Num Cycles = {C}')
            time.sleep(sl_time)
            procs,done,completed_procs, C= monitor(procs=procs, done = done, completed_procs=completed_procs, C = C)
        while len(procs) != 0:
            time.sleep(sl_time)
            procs,done,completed_procs, C = monitor(procs=procs, done = done, completed_procs = completed_procs, C = C)
            

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