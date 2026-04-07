import sys, os, tqdm,time, numpy as np
from core import mk_obj, unpack_script_args
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core import CMap
import pickle
if __name__ == '__main__':
    kwargs = dict(N = 100, n = 2, Nsteps = 5_000, adaptive_epsilon = 1e-3, dt = 5e-3, renorm = False, I = 0, conserve_Jz = True)
    d = pickle.load(open('configs.pkl','rb'))
    kwargs['r0_Rv'] = d['r0_Rv']
    kwargs['tgt_u0'] = None
    print(kwargs)
    kwargs = unpack_script_args(sys.argv, **kwargs)
    print(kwargs)
    tc = time.time()
    N = kwargs["N"]
    n = kwargs['n']
    r = int(kwargs['r0_Rv'])
    if not os.path.exists(os.path.join(os.getcwd(), f"BBGKY{N}-{n}-{r}/indexmap")):
        CMap(N=N, n= n, maps_loc=f"BBGKY{N}-{n}-{r}/indexmap")
    O = mk_obj(map_loc = f'BBGKY{N}-{n}-{r}/indexmap', **kwargs)
    t0 = time.time()
    print('Creation time {dt:.2e}s'.format(dt =  t0-tc))
    status = 'Completed'
    print(O.u(0))
    for i, o in tqdm.tqdm(enumerate(O)):
        if(o.isnan().any()):
            print(i, o)
            status = 'Failed'
            break
        pass
    dt = time.time() - t0
    with open(os.path.join(O.data_loc, f'log_{kwargs["I"]}.txt'), 'w') as file:
        file.write("Status:{status}\nNsteps Taken = {n} computed in {dt:.2e}\nFOR {BB}\nCreation Time {dtc}".format(status = status, n = O.n, BB=str(O), dt = dt, dtc = tc - t0))
    try:
        with open(os.path.join(O.data_loc, 'times.dat', 'w')) as file:
            np.array(O.delta_t, dtype = np.float64).tofile(file)
    except:
        np.array(O.delta_t, dtype=np.float64).tofile(os.path.join(O.data_loc, 'times.dat'))

    
