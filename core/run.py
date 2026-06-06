import qunum.numerical as qn
from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.bbgky import BBGKYDecoupling as BBGKY
import os, torch, time, numpy as np, polars as pl, subprocess, re, copy, glob, datetime , tqdm, sys
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch import Tensor
import pickle, gc, shutil
from threading import BoundedSemaphore


def run_job_cum_from_start(
        source_path:str,
        N:int , n:int, Nsteps:int,
        tgt_path:None|str=None, 
        log_path:str|None = None,
        NewSteps:int=5000, 
        Nproc:int=5, 
        tot_proc:int=100, 
        I0:int = 0, 
        max_cores:int|None = None, 
        **kwargs:dict
    ):
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
    log_path = os.path.join(log_path, 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    # Prepare task arguments
    max_cores =  min(max_cores if max_cores is not None else os.cpu_count(), os.cpu_count())
    paths = enumerate(glob.glob(f"{source_path}/*"))
    tasks = [(max_cores, tmp_path, i%Nproc, i, NewSteps, log_path, src) for i, src in paths]
    data_path = os.path.join(tmp_path, 'Data')
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        PhiBar = np.memmap(os.path.join(data_path, 'PhiBar.dat'), dtype = np.float64, shape = (Nsteps+NewSteps, N, n**2-1) , mode = 'w+')
        D = N*(pow(n,2)-1)
        GammaBar = np.memmap(os.path.join(data_path, 'GammaBar.dat'), shape = (Nsteps+NewSteps, int(pow(D,2)*(1-1/N)/2)), dtype = np.float64, mode='w+')
        PhiBar[...] = 0
        GammaBar[...] = 0 
    else:
        PhiBar = np.memmap(os.path.join(data_path, 'PhiBar.dat'), shape = (Nsteps+NewSteps, N, n**2-1), dtype = np.float64, mode = 'w+')
        D = N*(pow(n,2)-1)
        GammaBar = np.memmap(os.path.join(data_path, 'GammaBar.dat'), shape =(Nsteps+NewSteps, int(pow(D,2)*(1-1/N)/2)), dtype = np.float64, mode='w+')
    print(f"Starting {tot_proc} jobs using {Nproc} workers with {max_cores} cores per worker...")
    out_path = os.path.join(tgt_path, 'data')
    os.makedirs(out_path, exist_ok= True)
    try: 
        os.mkdir(os.path.join(out_path, f'configs{N}-{n}'))
    except:
        pass
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
                    path = os.path.join(tmp_path, f'{i}_run')
                    Phi = np.memmap(os.path.join(path,'Phi(t).npy'), dtype = np.float64, mode='r+', shape = (Nsteps+NewSteps, N, pow(n,2)-1)).copy()
                    D = N*(pow(n,2)-1)
                    Gamma = np.memmap(os.path.join(path,'Gamma(t).npy'), dtype = np.float64, mode = 'r+', shape = (Nsteps+NewSteps,int(pow(D,2)*(1-1/N)/2))).copy()
                    GammaBar+=Gamma
                    PhiBar+=Phi
                    del Phi; del Gamma
                    shutil.copyfile(f'{path}/configs.pkl', os.path.join(out_path, f'configs{N}-{n}', f'{n_complete-(Nproc-i)}_configs.pkl'))
                    shutil.rmtree(os.path.join(path))
                    pbar.update(1)
                    gc.collect()
                pbar.set_description(f"Computed {0}/{Nproc}")
    print('\nCompleted all iterations. Storing Data')
    out_path = os.path.join(tgt_path, 'data')
    if(os.path.exists(out_path)):
        #os.makedirs(out_path, exist_ok= True)
        file_tree = glob.glob(f'{out_path}/*')
        if(os.path.join(out_path,f'Nshots{N}-{n}.pkl') in file_tree):
            with open(os.path.join(out_path,f'Nshots{N}-{n}.pkl'), 'rb') as file:
                try:
                    T = pickle.load(file)
                    rd = True
                except:
                    T = 0
                    rd = False
        else:
            T = 0 
            rd = False
        tot_proc += T
        PhiBar = torch.from_numpy(PhiBar.copy())
        GammaBar = torch.from_numpy(GammaBar.copy())
        if(rd and os.path.join(out_path,f'PhiBar{N}-{n}.dat') in file_tree ):
            with open(os.path.join(out_path,f'PhiBar{N}-{n}.dat'), 'rb') as file:
                try:
                    tempP:Tensor = torch.load(file)*T
                except:
                    tempP = None
            if(tempP is not None):
                print(tempP.shape)
                PhiBar += tempP*T
                PhiBar /= tot_proc
        if(rd and os.path.join(out_path,f'PhiBar{N}-{n}.dat') in file_tree ):
            with open(os.path.join(out_path,f'GammaBar{N}-{n}.dat'), 'rb') as file:
                try:
                    tempG:Tensor = torch.load(file)*T
                except:
                    tempG = None
            if(tempP is not None and tempG is not None):
                print(tempG.shape)
                GammaBar += tempG * T
                GammaBar /= tot_proc
        del tempP; del tempG; del T
        gc.collect()
    else:
        os.makedirs(out_path, exist_ok=True)
        PhiBar = torch.from_numpy(PhiBar.copy())/tot_proc
        GammaBar = torch.from_numpy(GammaBar.copy())/tot_proc
    with open(os.path.join(out_path,f'Nshots{N}-{n}.pkl'), 'wb') as file:
        pickle.dump(tot_proc, file)
    with open(os.path.join(out_path, f'PhiBar{N}-{n}.dat'), 'wb') as file: 
        torch.save(PhiBar,file) 
    with open(os.path.join(out_path, f'GammaBar{N}-{n}.dat'), 'wb') as file: 
        torch.save(GammaBar,file)
    print(f'Data saved at {out_path}')
    print(f'Removing data at {tmp_path}')
    shutil.rmtree(tmp_path)
    return


def run_single_job_get_data(args:tuple[int, int, int, int,str, str, int, dict])->tuple[int,int, Tensor, Tensor]:
    max_cores, tmp_path, i, I, NewSteps, log_path, src = args
    log_file = f"{log_path}/log_{I}_{int(time.time())}.log"
    # Construct the command
    tmp_path = os.path.join(tmp_path, f'{i}_run')
    if not (os.path.exists(tmp_path)):
        os.makedirs(tmp_path, exist_ok= True)
    shutil.copytree(src, tmp_path)
    cmd = [sys.executable, 'multangle_cont.py', f"tgt_path={tmp_path}", f'NewSteps={NewSteps}',]
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