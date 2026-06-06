import sys, os, tqdm,time, numpy as np
from copy import deepcopy as dc
from core import load__obj as load_obj, unpack_script_args
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core import CMap
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation import BBGKYDecoupling
import pickle
import torch
import gc
if __name__ == '__main__':
    kwargs = dict(tgt_path = None, N=100, n=2, NewSteps = 5_000, dt = None)
    print(kwargs)
    kwargs = unpack_script_args(sys.argv, **kwargs)
    assert kwargs['tgt_path'] is not None, 'Must have tgt_path'
    
    NewSteps = kwargs['NewSteps']
    O = load_obj(**kwargs)
    O.dt.dt = torch.cat((O.dt.dt, torch.zeros(NewSteps,)))
    O.dt.t = torch.cat((O.dt.t, torch.zeros(NewSteps,)))
    for t in O.dt:
        pass
    O.dt.Nsteps+=NewSteps
    if(kwargs['dt'] is not None):
        O.dt.dt0 = kwargs['dt']
    O.n=O.dt.n
    O.Nsteps+=NewSteps
    Phi_t = O.Phi_t.copy()
    Gamma_t = O.Gamma_t.copy()
    file = (os.path.join(O.data_loc, 'Phi(t).npy'))
    mode = 'r+' if os.path.exists(file) else 'w+'
    O.Phi_t = np.memmap(file, mode = mode, shape=(O.Nsteps, *Phi_t.shape[1:]), dtype = Phi_t.dtype)
    file = (os.path.join(O.data_loc, 'Gamma(t).npy'))
    mode = 'r+' if os.path.exists(file) else 'w+'
    O.Gamma_t = np.memmap(file, mode = mode, shape=(O.Nsteps, *Gamma_t.shape[1:]), dtype = Gamma_t.dtype)
    O.Phi_t[:O.n] = Phi_t.copy()
    O.Gamma_t[:O.n] = Gamma_t.copy()
    del Phi_t; del Gamma_t
    gc.collect()
    
    for i, o in tqdm.tqdm(enumerate(O)):
        if(o.isnan().any()):
            print(i, o)
            status = 'Failed'
            break
        pass
    print(O.Phi_t.shape)
    print(O.Phi_t.__pow__(2).sum(-1))