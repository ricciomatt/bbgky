from qunum.jupyter_tools.plotting import PlotIt
from qunum.numerical import SUConnection, AdaptiveLinspace
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.core import rebuild_timeDep
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core import *
from qunum.numerical.physics.constants import h_barEvS as hbar
import torch, pickle, numpy as np, polars as pl, os, sys
from core.core import unpack_script_args
from core.plot import plot_non_local, plot_ptrans, plot_flav
if __name__ == '__main__':
    kwargs = dict(N = 100, n = 2, path_base = None, data_path = None)
    kwargs = unpack_script_args(sys.argv, **kwargs)
    n = kwargs['n']; N = kwargs['N']; path_base = kwargs['path_base']; data_path = kwargs['data_path']
    