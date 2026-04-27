from multiprocessing import Pool
from scipy.special import erf
from scipy.integrate import quad
import numpy as np
from time import time
from typing import List, Tuple, Optional
import pandas as pd
from util import optimize, P_negated, get_unique_filename
import os
import mpmath as mp
os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')



"""
This is a general method for integration and optimization. 
It works for any photon profile by simply changing the profile function.
However, because it uses nested integration with NumPy and scipy.integrate.quad, 
it is slower and less accurate compared to the semi-analytical approach.
"""

"""
def Gaussian_Entangled_withDelay(params: Tuple[float, float, float]) -> callable:
    OmegaP, OmegaM, Mu = params
    N = 1 + np.exp(-OmegaM**2 * Mu**2/4)
    N *= 8*np.pi/(OmegaP*OmegaM)
    def PSI(t2: float, t1: float): 
        Psi =  np.exp(-OmegaP**2 *(t1 + t2 - Mu)**2 / 8) *( np.exp( - OmegaM ** 2 * (t2 - t1 - Mu) ** 2 / 8)  + np.exp(-OmegaM**2 * (t1 - t2 - Mu) ** 2 / 8))
        return Psi/N**0.5
    return PSI

def task(Gamma_e: float) -> pd.DataFrame:
    t0 = time()
    res = optimize(	P_negated(Gamma_e, Gamma_f, Gaussian_Entangled_withDelay, tol = 1e-9,limit = 100), 
                    sp_bounds = np.array([[1,10],[0.01,2],[0.01,max(Gamma_e,1)],[0,10]]),
                    p_bounds = ((None,None),(0,None),(0,None),(0,None)), 
                    N = 10)
    res['t_max'], res['OmegaP'], res['OmegaM'], res['Mu']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma_e
    res = pd.DataFrame([res])
    res = res[['P_max', 'OmegaP', 'OmegaM', 't_max', 'Mu', 'Gamma_e',  'number_of_fails', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return res
"""



mp.mp.dps = 25

########### High precision float64 #####################################################################3
def P_Entangled_withDelay ( Gamma_e, Gamma_f= 1, w_fe = 0, w_eg = 0, w_01 = 0, w_02 = 0):
    def inner(params):
        t, OmegaP, OmegaM, Mu =  map(mp.mpf, params)
        MuP = Mu
        MuM = Mu
        N = 1 + mp.exp(-OmegaM**2 * MuM**2/4- (w_02-w_01)**2/OmegaM**2)
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


def task(Gamma_e: float) -> pd.DataFrame:
    t0 = time()
    res = optimize(     P_Entangled_withDelay (Gamma_e), 
                    sp_bounds = np.array([[0,2],[0.01,2],[Gamma_e*0.5,max(Gamma_e,1)],[0,1]]),
                    p_bounds = ((None,None),(0.00001,None),(0.0001,None),(0,None)), 
                    N = 20)
    res['t_max'], res['OmegaP'], res['OmegaM'], res['Mu']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma_e
    res['mp.dps'] = mp.mp.dps
    res = pd.DataFrame([res])
    res = res[['P_max', 'OmegaP', 'OmegaM', 't_max', 'Mu', 'Gamma_e',  'number_of_fails','mp.dps', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return res
fileName = get_unique_filename('P_Gamma_Entangled_withDelay.csv')

Gamma_f = 1

Gamma_e_range = np.logspace(-3,4,28*5,endpoint=False)
Gamma_e_range = Gamma_e_range[6:7]
time0 = time()
p = Pool()
p.map(task,Gamma_e_range)
df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_e'])
df.to_csv(fileName, index=False)
print(f'All tasks done in {int(time()-time0)}s and saved in "{fileName}"')
