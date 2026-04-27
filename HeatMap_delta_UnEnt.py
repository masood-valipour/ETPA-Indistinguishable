from multiprocessing import Pool
import numpy as np
from time import time
import pandas as pd
from util import optimize, get_unique_filename
import os
import mpmath as mp
import pickle
# os.makedirs('DataSets',exist_ok=True)
# os.chdir('DataSets')


"""
This script generates datasets for plotting the optimized maximum probability (P_max) and other parameters 
with respect to atomic detuning (delta_a) and field detuning (delta_f) in a two-photon absorption process 
specifically for unentangled photons. The code uses high-precision arithmetic (via mpmath) and parallel processing to 
efficiently optimize and compute results for a grid of detuning values.

Key Features:
- Computes P_max and other optimized parameters (OmegaA, OmegaB, t_max, Mu) for each (delta_a, delta_f) pair (for 'Delay' case).
- For 'No-Delay', only (OmegaA, OmegaB, t_max) are optimized (Mu is set to 0).
- delta_a: Atomic detuning parameter.
- delta_f: Field detuning parameter.
- Uses multiprocessing to parallelize the optimization tasks.
- Results are saved to a CSV file for further analysis and plotting.
- Designed to handle both 'Delay' and 'No-Delay' scenarios in the unentangled photon field setup.
- Reads problematic points from a CSV file and recomputes their optimized values.

This script is intended for use in generating heatmaps or other visualizations of P_max and related 
quantities as functions of atomic and field detuning, specifically for unentangled photon absorption processes.
"""




########### High precision float64 #####################################################################3
def P_UnEnt( Gamma_e, Gamma_f= 1, delta_a=0):
    def inner(params):
        if status == 'Delay':
            t, OmegaA, OmegaB, delta_f, Mu =  map(mp.mpf, params)
        elif status == 'No-Delay':
            t, OmegaA, OmegaB, delta_f =  map(mp.mpf, params)
            Mu = 0
        MuA = 0
        MuB = Mu
        w_fe = mp.mpf(0.5)*delta_a
        w_eg = -mp.mpf(0.5)*delta_a
        w_a = -mp.mpf(0.5)*delta_f
        w_b = mp.mpf(0.5)*delta_f

        N = (MuB - MuA)**2*OmegaA**2*OmegaB**2 + 4*(w_b - w_a)**2
        N /= -2*(OmegaA**2 + OmegaB**2)
        N = mp.exp(N) * 2* OmegaA*OmegaB/(OmegaA**2 + OmegaB**2) + 1
        N  *= 4
        A = 1j*w_fe + Gamma_f/2 - Gamma_e/2 
        B = 1j*w_eg +  Gamma_e/2 

        T = OmegaB*(t-MuB)/2 - (A - 1j*w_b)/OmegaB
        p1 = mp.quad(lambda X: mp.exp(-X**2)  * \
                  (mp.mpf(1) +  mp.erf(OmegaA*X/OmegaB + (A - 1j*w_b)*OmegaA/OmegaB**2 + OmegaA*(MuB-MuA)/2 - (B - 1j*w_a)/OmegaA )),\
                  [-mp.inf, T],complex_func=True )
        p1 *= mp.exp((A-1j*w_b)**2/OmegaB**2)
        p1 *= mp.exp((OmegaA**2*MuA + 2*B - 2j*w_a)**2/OmegaA**2/4 - MuA**2*OmegaA**2/4 + (A - 1j*w_b)*MuB)

        
        T = OmegaA*t/2 - OmegaA*MuA/2 - (A-1j*w_a)/OmegaA
        p2 = mp.quad(lambda X: mp.exp(-X**2)  * \
                  (1 +  mp.erf(OmegaB*X/OmegaA + (A - 1j*w_a)*OmegaB/OmegaA**2 - OmegaB*(MuB-MuA)/2 - (B - 1j*w_b)/OmegaB )),\
                  [-mp.inf, T],complex_func=True )
        p2 *= mp.exp((A-1j*w_a)**2/(OmegaA**2))
        p2 *= mp.exp((OmegaB**2*MuB + 2*B - 2j*w_b)**2/OmegaB**2/4 - MuB**2*OmegaB**2/4 + (A - 1j*w_a)*MuA)
                
        p = abs( p1 + p2 )**2
        p *= 8*mp.exp(-Gamma_f*t) *Gamma_e*Gamma_f/(OmegaA*OmegaB*N)
        return -p
    return inner


def task(pairs) -> pd.DataFrame:
    print(f'PID: {os.getpid()}')
    Gamma_e, d_a  =  map(mp.mpf, pairs)
    print(f'Processing Gamma_e={Gamma_e}, delta_a={d_a} ')
    t0 = time()

    if status == 'Delay':
        res = optimize(     P_UnEnt (Gamma_e, delta_a = d_a), 
                        sp_bounds = np.array([[0,2],[0.01,5],[Gamma_e*0.3,max(Gamma_e,1)],[0,25],[0,1]]),
                        p_bounds = ((None,None),(0.00001,None),(0.0001,None),(0,30),(0,None)), 
                        N = 5)
        res['t_max'], res['OmegaA'], res['OmegaB'], res['delta_f'], res['Mu']=  res.x
    elif status == 'No-Delay':
        res = optimize(     P_UnEnt (Gamma_e, delta_a = d_a), 
                        sp_bounds = np.array([[0,2],[0.01,2],[Gamma_e*0.5,max(Gamma_e,1)],[0,25]]),
                        p_bounds = ((None,None),(0.00001,None),(0.0001,None),(0,30)), 
                        N = 5)
        res['t_max'], res['OmegaA'], res['OmegaB'], res['delta_f']=  res.x
        res['Mu'] = 0
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma_e
    res['delta_a'] = d_a
    # res['delta_f'] = d_f
    res['mp.dps'] = mp.mp.dps
    res = pd.DataFrame([res])
    res = res[['P_max', 'OmegaA', 'OmegaB', 't_max', 'Mu', 'Gamma_e', 'delta_a', 'delta_f', 'number_of_fails','mp.dps', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return res

status = 'Delay'
# status = 'No-Delay'
# mp.mp.dps = 60




fileName = get_unique_filename(f'Cluster/HeatMap_deltas_UnEnt_{status}.csv')


Gamma_f = 1



log_points = np.logspace(-2, 2, num=17)
delta_a = np.linspace(0,25,50, endpoint=False)

X, Y = np.meshgrid(log_points, delta_a)
pairs = list(zip(X.ravel(), Y.ravel()))
time0 = time()

p = Pool()
p.map(task,pairs)

df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_e'])
df.to_csv(fileName, index=False)
print(f'All tasks done in {int(time()-time0)}s and saved in "{fileName}"')