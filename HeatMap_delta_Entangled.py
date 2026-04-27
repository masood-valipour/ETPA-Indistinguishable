from multiprocessing import Pool
import numpy as np
from time import time
import pandas as pd
from util import optimize, get_unique_filename
import mpmath as mp
import pickle

"""
This script generates datasets for plotting the optimized maximum probability (P_max) and other parameters 
with respect to atomic detuning (delta_a) and field detuning (delta_f) in a two-photon absorption process 
specifically for entangled photons. The code uses high-precision arithmetic (via mpmath) and parallel processing to 
efficiently optimize and compute results for a grid of detuning values.

Key Features:
- Computes P_max and other optimized parameters (OmegaP, OmegaM, t_max, Mu) for each (delta_a, delta_f) pair.
- delta_a: Atomic detuning parameter.
- delta_f: Field detuning parameter.
- Uses multiprocessing to parallelize the optimization tasks.
- Results are saved to a CSV file for further analysis and plotting.
- Designed to handle both 'Delay' and 'No-Delay' scenarios in the entangled photon field setup.
- Reads problematic points from a CSV file and recomputes their optimized values.

This script is intended for use in generating heatmaps or other visualizations of P_max and related 
quantities as functions of atomic and field detuning, specifically for entangled photon absorption processes.
"""

########### High precision float64 #####################################################################3
def P_Entangled_withDelay ( Gamma_e, Gamma_f= 1, delta_a = 0):
    def inner(params):
        if status == 'Delay':
            t, OmegaP, OmegaM, delta_f, Mu =  map(mp.mpf, params)
        elif status == 'No-Delay':
            t, OmegaP, OmegaM, delta_f =  map(mp.mpf, params)
            Mu = 0
        MuP = Mu
        MuM = Mu
        w_fe = mp.mpf(0.5)*delta_a
        w_eg = -mp.mpf(0.5)*delta_a
        w_01 = -mp.mpf(0.5)*delta_f
        w_02 = mp.mpf(0.5)*delta_f

        N = 1 + mp.exp(-OmegaM**2 * MuM**2/4 - (w_02-w_01)**2/OmegaM**2)
        N *= 8*mp.pi/(OmegaP*OmegaM)
        
        A = 1j*w_fe + Gamma_f/2 - Gamma_e/2       # -1j*w_01
        B = 1j*w_eg +  Gamma_e/2        #- 1j*w_02

        T = -2**0.5/4 * OmegaM*MuM - 2**0.5/(2*OmegaM) * (A-B-1j*(w_02-w_01))
        p1 = mp.quad(lambda y: mp.exp(-y**2) * \
                  (1 +  mp.erf(2**0.5*OmegaP*(2*t - MuM-MuP )/4 - y*OmegaP/OmegaM  \
                               -OmegaP/OmegaM * 2**0.5/OmegaM * (A-B -1j*(w_02-w_01))/2 - 2**0.5/2 * (A+B -1j*(w_02+w_01))/OmegaP)),\
                  [T, +mp.inf])
        p1 *= mp.exp( (A-B-1j*(w_02-w_01))**2/(2*OmegaM**2))
        p1 *= 4/OmegaP/OmegaM * mp.exp((A+B -1j*(w_02 +w_01))*MuP/2 + (A-B -1j*(w_02 - w_01))*MuM/2)
        p1 *= mp.pi**0.5/2 *mp.exp((A+B -1j*(w_02+w_01))**2/(2*OmegaP**2))
        
        T = +2**0.5/4 * OmegaM*MuM - 2**0.5/(2*OmegaM) * (A-B+1j*(w_02-w_01))
        p2 = mp.quad(lambda y: mp.exp(-y**2) * \
                  (1 +  mp.erf(2**0.5*OmegaP*(2*t - MuM-MuP )/4 - y*OmegaP/OmegaM \
                               - 2**0.5*OmegaP/OmegaM**2 * (A-B +1j*(w_02-w_01))/2 - 2**0.5/2 * (A+B -1j*(w_02+w_01))/OmegaP)),\
                  [T, +mp.inf])
        p2 *= mp.exp( (A-B+1j*(w_02-w_01))**2/(2*OmegaM**2))
        p2 *= 4/OmegaP/OmegaM * mp.exp((A+B + 1j*(w_02 +w_01))*MuP/2 - (A-B -1j*(w_02 - w_01))*MuM/2)
        p2 *= mp.pi**0.5/2 *mp.exp((A+B -1j*(w_02+w_01))**2/(2*OmegaP**2))
        
        p = abs( p1 + p2)**2
        p *= 4*mp.exp(-Gamma_f*t) *Gamma_e*Gamma_f/N
        return -p
    return inner


def task(pairs) -> pd.DataFrame:
    Gamma_e, d_a =  map(mp.mpf, pairs)
    t0 = time()
    if status == 'Delay':
        res = optimize(     P_Entangled_withDelay (Gamma_e, delta_a= d_a), 
                        sp_bounds = np.array([[0,2],[0.01,2],[Gamma_e*0.5,max(Gamma_e,1)],[0,30],[0,1]]),
                        p_bounds = ((None,None),(0.00001,None),(0.0001,None),(0,30),(0,None)), 
                        N = 5)
        res['t_max'], res['OmegaP'], res['OmegaM'], res['delta_f'], res['Mu']=  res.x
    elif status == 'No-Delay':
        res = optimize(     P_Entangled_withDelay (Gamma_e,delta_a= d_a), 
                        sp_bounds = np.array([[0,2],[0.01,2],[Gamma_e*0.5,max(Gamma_e,1)],[-d_a,3*d_a]]),
                        p_bounds = ((None,None),(0.00001,None),(0.0001,None),(-d_a,3*d_a)), 
                        N = 3)
        res['t_max'], res['OmegaP'], res['OmegaM'], res['delta_f']=  res.x
        res['Mu'] = 0
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma_e
    res['delta_a'] = d_a
    # res['delta_f'] = d_f
    res['mp.dps'] = mp.mp.dps
    res = pd.DataFrame([res])
    res = res[['P_max', 'OmegaP', 'OmegaM', 't_max', 'Mu', 'Gamma_e', 'delta_a', 'delta_f', 'number_of_fails','mp.dps', 'ComputTime'] ]
    res = res.sort_values(['delta_f', 'delta_a',]).reset_index(drop=True)
    res.to_csv(fileName, mode='a', header=True, index=False)
    return res


mp.mp.dps = 30
Gamma_f = 1




time0 = time()


status = 'No-Delay'
fileName = get_unique_filename(f'Cluster/HeatMap_deltas_Entangled_{status}.csv')

Gamma_e_points = np.logspace(-2, 2, num=17)
delta_a = np.linspace(0,25,50, endpoint=False)
X, Y = np.meshgrid(Gamma_e_points, delta_a)
pairs = list(zip(X.ravel(), Y.ravel()))

p = Pool(processes=18)
p.map(task,pairs)

df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_e'])
df.to_csv(fileName, index=False)
print(f'All tasks done in {int(time()-time0)}s and saved in "{fileName}"')