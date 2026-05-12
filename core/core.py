import qunum.numerical as qn
from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.bbgky import BBGKYDecoupling as BBGKY
import os, torch, time, numpy as np, polars as pl, subprocess, re, copy, glob, datetime , tqdm, sys
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch import Tensor
import pickle, gc, shutil
from threading import BoundedSemaphore

def mk_obj(tgt_path:str|None = None, N:int = 100, n:int = 100, I:int = 0, r0_Rv:float = 7.5, angular_dependence = 'random_uniform', map_loc:str|None = None, load_map:bool = True, **kwargs)->BBGKY:
    if(tgt_path is None):
        tgt_path = os.getcwd()
    a = dict(
        Nsteps= int(4500), 
        u0 = 10_000*1e2, tgt_u0 = 5, 
        debug=False, load_data=False, 
        dp = .1, p0 = torch.tensor(1e10), method = 'fullrk', 
        adaptive_epsilon=5e-3, dt  = 1e-2, adaptive_time_step=True, p_start_f = .1
    )
    for key, val in kwargs.items():
        a[key] = val
    path = os.path.join(tgt_path, f"BBGKY{N}-{n}/Data", f'{I}_{angular_dependence}')
    if not (os.path.exists(path)):
        os.makedirs(path)
    return BBGKY(
        N=N, n=n, 
        data_storage_loc = path,
        maps_loc = map_loc, load_maps=load_map,
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
        path = os.path.join(f"BBGKY{N}-{n}/Data", f'{I}_{angular_dependence}')
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
    
    def pat_match(val, key):
        pats = {int:r"^[+-]?(0|[1-9]\d*)$", float:r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"}
        if(val =='None'):
            return None
        print(key, type(val), val)
        for tp, pat in pats.items():
            if re.match(pat, val):
                try:
                    val = tp(val)
                except:
                    pass
                break 
        print(key,type(val), val)
        return val
    for key, val in map(lambda x: x.split('='), inp_args[1:]):
        if(key in kwargs):
            match kwargs[key]:
                case None:
                    if(str(val) == 'None'):
                        kwargs[key] = None
                    else:    
                        kwargs[key] = pat_match(val,key)
                case Tensor():

                    if(val == 'None'):
                        kwargs[key] = None
                    else:
                        kwargs[key] = torch.tensor(float(val))
                case _:
                    if(val == 'None'):
                        kwargs[key] = None
                    else:
                        kwargs[key] = type(kwargs[key])(val)
            
        else:
            kwargs[key] = pat_match(val, key)
    return kwargs


def run_single_job(args:tuple[int, int, int, int, str, str, dict]):
    n, N, Nsteps, i, max_cores, path, tgt_path, kwargs = args
    log_file = f"{path}/log_{i}_{int(time.time())}.log"
    # Construct the command
    cmd = [sys.executable, 'multangle.py', f'n={n}', f'N={N}', f'Nsteps={Nsteps}', f'I={i}', f"tgt_path={tgt_path}", *(f"{key}={val}" for key,val in dict(kwargs).items())]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = f"{max_cores}" 
    env["MKL_NUM_THREADS"] = f"{max_cores}"
    with open(log_file, 'w') as f:
        # We use run() here because the Executor handles the parallelism
        result = subprocess.run(cmd, stdout=f, stderr=f, text=True, env=env)
    return i, result.returncode

def run_job_par(tgt_path:None|str=None, n:int=2, N:int=100, Nsteps:int=5000, Nproc:int=5, tot_proc:int=100, I0:int = 0, max_cores:int|None = None, **kwargs:dict):
    if(tgt_path is None):
        tgt_path = os.getcwd()
    else:
        if(not os.path.exists(tgt_path)):
            print(tgt_path)
            os.makedirs(tgt_path)

    path = os.path.join(tgt_path, 'logs')
    if not os.path.exists(path):
        os.mkdir(path)
    # Prepare task arguments
    max_cores =  min(max_cores if max_cores is not None else os.cpu_count(), os.cpu_count())
    tasks = [(n, N, Nsteps, i+I0, max_cores, path, tgt_path, kwargs) for i in range(tot_proc)]
    print(f"Starting {tot_proc} jobs using {Nproc} workers with {max_cores} cores per worker...")
    with tqdm.tqdm(total=tot_proc, desc="Computing", unit="job", ncols=100) as pbar:
        with ProcessPoolExecutor(max_workers=Nproc) as executor:
            future_to_job = {executor.submit(run_single_job, task): task for task in tasks}
            for future in as_completed(future_to_job):
                job_id, return_code = future.result()
                pbar.update(1)
                if return_code != 0:
                    print(f"Job {job_id} failed with code {return_code}")
    print('\nCompleted all iterations.')

def run_job_cums(tgt_path:None|str=None, log_path:str|None = None, n:int=2, N:int=100, Nsteps:int=5000, Nproc:int=5, tot_proc:int=100, I0:int = 0, max_cores:int|None = None, **kwargs:dict):
    if(tgt_path is None):
        tgt_path = os.getcwd()
        tmp_path = os.path.join(tgt_path,'temp')
    else:
        os.path.join(tgt_path,'temp')
        if(not os.path.exists(tgt_path)):
            print(tgt_path)
            os.makedirs(tgt_path)
        tmp_path = os.path.join(tgt_path,'temp')
        if(not os.path.exists(tmp_path)):
            print(tmp_path)
            os.makedirs(tmp_path)
    if(log_path is None):
        log_path = os.getcwd()
    else:
        os.path.join(log_path,'temp')
        if(not os.path.exists(tgt_path)):
            print(tgt_path)
            os.makedirs(tgt_path)
    path = os.path.join(log_path, 'logs')
    if not os.path.exists(path):
        os.makedirs(path)
    # Prepare task arguments
    max_cores =  min(max_cores if max_cores is not None else os.cpu_count(), os.cpu_count())
    tasks = [(n, N, Nsteps, i%Nproc, max_cores, path, tmp_path, i, kwargs) for i in range(tot_proc)]
    data_path = os.path.join(tmp_path, 'Data')
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        PhiBar = np.memmap(os.path.join(data_path, 'PhiBar.dat'), dtype = np.float64, shape = (Nsteps, N, n**2-1) , mode = 'w+')
        D = N*(pow(n,2)-1)
        GammaBar = np.memmap(os.path.join(data_path, 'GammaBar.dat'), shape = int(pow(D,2)*(1-1/N)/2), dtype = np.float64, mode='w+')
        PhiBar[...] = 0
        GammaBar[...] = 0 
    else:
        PhiBar = np.memmap(os.path.join(data_path, 'PhiBar.dat'), shape = (Nsteps, N, n**2-1), dtype = np.float64, mode = 'w+')
        D = N*(pow(n,2)-1)
        GammaBar = np.memmap(os.path.join(data_path, 'GammaBar.dat'), shape =(Nsteps, int(pow(D,2)*(1-1/N)/2)), dtype = np.float64, mode='w+')
    print(f"Starting {tot_proc} jobs using {Nproc} workers with {max_cores} cores per worker...")

    n_complete = 0
    print(Nsteps, n, GammaBar.shape, PhiBar.shape)
    with tqdm.tqdm(total=tot_proc, desc=f"Computed 0/{Nproc}", unit="job", ncols=75,) as pbar:
        with ProcessPoolExecutor(max_workers=Nproc) as executor:
            while n_complete<tot_proc:
                jobs = []
                completed = []
                for task in tasks[n_complete:n_complete+Nproc]:
                    future = executor.submit(run_single_job_get_data, task)
                    jobs.append(future)
                i = 0
                for future in as_completed(jobs):
                    job_id, return_code,= future.result() 
                    completed.append(job_id)
                    pbar.set_description(f"Computed {len(completed)}/{Nproc}")
                    
                    if return_code != 0:
                        print(f"Job {job_id} failed with code {return_code}")                
                    n_complete+=1
                pbar.set_description(f"Aggregating Processes")
                for i in completed:
                    Phi = np.memmap(os.path.join(tmp_path,f'BBGKY{N}-{n}/Data/{i}_random_uniform/Phi(t).npy'), dtype = np.float64, mode='r+', shape = (Nsteps, N, pow(n,2)-1)).copy()
                    D = N*(pow(n,2)-1)
                    Gamma = np.memmap(os.path.join(tmp_path,f'BBGKY{N}-{n}/Data/{i}_random_uniform/Gamma(t).npy'), dtype = np.float64, mode = 'r+', shape = (Nsteps,int(pow(D,2)*(1-1/N)/2))).copy()
                    GammaBar+=Gamma
                    PhiBar+=Phi
                    del Phi; del Gamma
                    pbar.update(1)
                    gc.collect()
                pbar.set_description(f"Computed {0}/{Nproc}")
    print('\nCompleted all iterations. Storing Data')
    out_path = os.path.join(tgt_path, 'data')
    print(PhiBar, GammaBar)
    if(os.path.exists(out_path)):
        os.makedirs(out_path, exist_ok= True)
        with open(os.path.join(out_path,'Nshots.pkl'), 'rb') as file:
            T = pickle.load(file)
        PhiBar = torch.from_numpy(PhiBar.copy())
        GammaBar = torch.from_numpy(GammaBar.copy())
        print('Processing')
        with open(os.path.join(out_path,'PhiBar.dat'), 'rb') as file:
            PhiBar += torch.load(file)*T
        with open(os.path.join(out_path,'GammaBar.dat'), 'rb') as file:
            GammaBar += torch.load(file)*T
        tot_proc+=T
        PhiBar/=tot_proc
        GammaBar/=tot_proc
    else:
        os.makedirs(out_path)
        PhiBar = torch.from_numpy(PhiBar.copy())/tot_proc
        GammaBar = torch.from_numpy(GammaBar.copy())/tot_proc
    print(PhiBar, GammaBar)
    with open(os.path.join(out_path,'Nshots.pkl'), 'wb') as file :
        pickle.dump(tot_proc, file)
    with open(os.path.join(out_path, 'PhiBar.dat'), 'wb') as file: 
        torch.save(PhiBar,file) 
    with open(os.path.join(out_path, 'GammaBar.dat'), 'wb') as file: 
        torch.save(PhiBar,file)
    print(f'Data saved at {out_path}')
    print(f'Removing data at {tmp_path}')
    shutil.rmtree(tmp_path)
    return

def run_single_job_get_data(args:tuple[int, int, int, int,str, str, int, dict])->tuple[int,int, Tensor, Tensor]:
    n, N, Nsteps, i, max_cores, path, tgt_path, I, kwargs = args
    log_file = f"{path}/log_{I}_{int(time.time())}.log"
    # Construct the command
    cmd = [sys.executable, 'multangle.py', f'n={n}', f'N={N}', f'tgt_u0=None', f"tgt_path={tgt_path}", f'Nsteps={Nsteps}', f'I={i}', *(f"{key}={val}" for key,val in dict(kwargs).items())]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = f"{max_cores}" 
    env["MKL_NUM_THREADS"] = f"{max_cores}"
    with open(log_file, 'w') as f:
        # We use run() here because the Executor handles the parallelism
        result = subprocess.run(cmd, stdout=f, stderr=f, text=True, env=env)
    if(result.returncode != 0):
        with open(log_file, 'r') as f:
            print(f.read())
    return i, result.returncode