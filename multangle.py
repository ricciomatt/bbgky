import sys, os, tqdm,time, numpy as np
from core import mk_obj
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core import CMap

if __name__ == '__main__':
    kwargs = dict(N = 100, n = 100, Nsteps = 4500, adaptive_epsilon = 1e-3, renorm = False, I = 0)
    print(sys.argv)
    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        kwargs[key] = int(val)
    tc = time.time()
    if not os.path.exists(os.path.join(os.getcwd(), f'BBGKY{kwargs['N']}-{kwargs['n']}/indexmap')):
        CMap(kwargs['N'], kwargs['n'], maps_loc=f'BBGKY{kwargs['N']}-{kwargs['n']}/indexmap')
    O = mk_obj(map_loc = f'BBGKY{kwargs['N']}-{kwargs['n']}/indexmap', **kwargs)
    t0 = time.time()
    print('Creation time {dt:.2e}s'.format(dt =  t0-tc))
    for i, o in tqdm.tqdm(enumerate(O)):
        if(o.isnan().any()):
            print(i, o)
            break
        pass
    dt = time.time() - t0
    with open(os.path.join(O.data_loc, f'log_{i}.txt'), 'w') as file:
        file.write("Nsteps {n} computed in {dt:.2e}\nFOR {BB}\nCreation Time {dtc}".format(n = O.n, BB=str(O), dt = dt, dtc = tc - t0))
    try:
        with open(os.path.join(O.data_loc, 'times.dat', 'w')) as file:
            np.array(O.delta_t, dtype = np.float64).tofile(file)
    except:
        np.array(O.delta_t, dtype=np.float64).tofile(os.path.join(O.data_loc, 'times.dat'))

    
