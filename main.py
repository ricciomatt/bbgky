import qunum.numerical as qn
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.bbgky import BBGKYDecoupling as BBGKY
import os, torch
import time
def mk_obj(N:int, n:int,**kwargs)->BBGKY:
    a  = dict(
        adaptive_time_step= True, 
        tgt_u0 = 5,
        u0 = 10_000*1e2, 
        tgt_u0 = 5,
        debug=False, load_data=False, dp = .1, 
        p0 = torch.tensor(1e10), 
        method = 'fullrk', 
        adaptive_epsilon=5e-3, 
        dt  = 1e-2, 
        adaptive_time_step=True, 
        r0_Rv = 7.5, 
        p_start_f = .1,
        angular_dependnce = False
    )
    a.update(kwargs)
    return BBGKY(
        N, n, 
        data_storage_loc=os.path.join(os.getcwd(), 'BBGKYSims', f'BBGKY_{N}-{n}'),
        **a
    )
if __name__ == '__main__':
    
    D = {2:range(50, 105, 5), 3:range(25,75,5), 4:range(10,50,5)}
    
    for n in D:
        for N in D:
            Obj = mk_obj(N, n)
            t0 = time.time()
            for o in Obj:
                pass
            dt = time.time() - t0
            with open(os.path.join(os.getcwd(), 'log.txt'), 'a') as file:
                file.write("Time for (N = {N},n = {n}): {dt:.2e} s\n".format(N = str(N), n = str(n), dt = dt))
    pass