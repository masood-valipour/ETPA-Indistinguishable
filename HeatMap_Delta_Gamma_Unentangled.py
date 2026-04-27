from multiprocessing import Pool
import numpy as np
from time import time
from typing import List, Tuple, Optional
import pandas as pd
from util import optimize, P_negated, get_unique_filename
import os
os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')



def Gaussian_Unentangled_withDelay(params: Tuple[float, float, float]) -> callable:
    OmegaA, OmegaB, Mu = params
    N_PHI = np.exp(-(Mu**2 * OmegaA**2 * OmegaB**2)/(2*OmegaA**2 + 2*OmegaB**2))
    N_PHI *= 2 * (OmegaA * OmegaB)/(OmegaA**2 + OmegaB**2)
    N_PHI +=1 
    def PSI(t2: float, t1: float):       
        return (OmegaA*OmegaB)**0.5*np.exp((-OmegaB**2*(t2-Mu)**2 - OmegaA**2*t1**2)/4)/(2*np.pi)**0.5 / N_PHI**0.5
    def PSI_sym(t2: float, t1: float):      
        return (PSI(t2,t1)+PSI(t1,t2))
    return PSI_sym

def task(params: tuple) -> pd.DataFrame:
    Gamma1, Delta = params
    Gamma2 = 1
    Delta1 = Delta
    Delta2 = -Delta
    t0 = time()
    res = optimize(	P_negated(Gamma1, Gamma2, Gaussian_Unentangled_withDelay , Delta1, Delta1, tol = 1e-9, limit = 100), 
                    sp_bounds = np.array([[0.5/Gamma1, 1.5/Gamma1],[1e-6,max(5,Gamma1/5)],[1e-6,max(5,Gamma1/5)],[0.5/Gamma1, 1.5/Gamma1]]),
                    p_bounds = ((None,None),(0,None),(0,None),(0,None)), 
                     N = 5)
    res['t_max'], res['OmegaA'], res['OmegaB'], res['Mu']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma1
    res['Delta1'] = Delta
    res['Delta2'] = -Delta
 
    res = pd.DataFrame([res])
    res = res[['P_max', 'OmegaA', 'OmegaB', 't_max', 'Mu', 'Gamma_e', 'Delta1', 'Delta2',  'number_of_fails', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return

fileName = get_unique_filename('P_heatMap_Gamma_Delta_Unentangled.csv')

t1 = -np.inf
t2 = 0

#Gamma1_range = np.logspace(-3,4,28,endpoint=False)
Gamma1_range = np.logspace(-2,2,21,endpoint=True)

# Delta_range = np.linspace(0,3,31)
Delta_range = np.linspace(0,10,101)
params_range = [(d1, d2) for d1 in Gamma1_range for d2 in Delta_range]

time0 = time()
p = Pool()
p.map(task,params_range)
df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_e'])
df.to_csv(fileName, index=False)
print(f'All tasks done in {int(time()-time0)}s and saved in "{fileName}"')

