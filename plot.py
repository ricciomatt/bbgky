from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical import SUConnection, AdaptiveLinspace
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.core import rebuild_timeDep
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core import *
from qunum.numerical.physics.constants import h_barEvS as hbar
import torch, pickle, numpy as np, polars as pl, os, sys
from core.core import unpack_script_args
from core.plot import plot_non_local, plot_ptrans, plot_flav
if __name__ == '__main__':
    kwargs = dict(N = 100,n = 2, path_base = None, data_path = None)
    kwargs = unpack_script_args(sys.argv, **kwargs)
    n = kwargs['n']; N = kwargs['N']; path_base = kwargs['path_base']; data_path = kwargs['data_path']
    if(path_base is None):
        path_base = os.getcwd()
    if(data_path is None):
        data_path = os.getcwd()
    assert (os.path.exists(data_path,f'Data{N}-{n}')), FileNotFoundError('Could Not Find Data')
    plot_path = os.path.join(path_base,f'PaperImages{N}-{n}')
    os.makedirs(plot_path, exist_ok=True)
    PhiBar = torch.load(os.path.join(path_base,f'PhiBar.dat'))
    GammaBar = torch.load(os.path.join(path_base,f'GammaBar.dat'))
    configs = pickle.load(open(os.path.join(path_base,f'configs.pkl'), 'rb'))
    pMag = configs['pMag']
    sun = SUConnection(configs['n'])
    if('Nsteps' in kwargs):
        configs['Nsteps'] = kwargs['Nsteps']
    B, u, w0 = rebuild_timeDep(sun=sun, **configs)
    Adt = AdaptiveLinspace(dt = configs['dt'], adaptive_epsilon=configs['adaptive_epsilon'], Nsteps=PhiBar.shape[0], t0 = 0, adaptive_function=u)
    for a in Adt:
        pass 
    t = Adt.t
    dt = Adt.dt
    tfact = hbar/w0
    #lf, *_ = computeBaseMaps(N, n, )
    lfs = computeBaseMaps(N, n, ret_dict=True)
    lf:pl.LazyFrame = lfs['lf']
    print('Data Loaded')
    plt = PlotIt()
    plot_non_local(PhiBar,GammaBar,t, pMag, plot_path, sun, lfs, plt)
    plot_ptrans(PhiBar, tfact, t, plot_path, sun, B, plt, u, N)
    plot_flav(PhiBar, sun, plt, t , path_base, N)

