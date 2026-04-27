from multiprocessing import Pool
import numpy as np
from time import time
import pandas as pd
from util import optimize, get_unique_filename
import os
from scipy.integrate import complex_ode

os.makedirs('DataSets',exist_ok=True)
os.chdir('DataSets')



def Gaussian(t: float, Omega: float = 1)-> float:
    t = np.array(t)
    return (Omega**2/(2*np.pi))**(1/4) * np.exp(-Omega**2*((t)**2)/4)


def P_coherent(Omega: float, Gamma_e: float = 1, Gamma_f: float = 1 
               , Delta_1: float = 0, Delta_2: float = 0, n_bar: int = 2, nBins: int = 10000) -> float:
    def rhs(t, initial):
        Rho_ff, Rho_ef, Rho_gf, Rho_ee, Rho_ge, Rho_gg = initial 
        alpha = float(Gaussian(t,Omega = Omega))
        alpha =  np.sqrt(n_bar) * alpha
        dRho_ffdt = - Gamma_f * Rho_ff - alpha*np.sqrt(Gamma_f) * (Rho_ef + np.conjugate(Rho_ef)) 
        dRho_efdt = alpha * np.sqrt(Gamma_f) * (Rho_ff - Rho_ee) - alpha * np.sqrt(Gamma_e) * Rho_gf + (1j*Delta_2 - (Gamma_e + Gamma_f)/2)*Rho_ef
        dRho_gfdt = - alpha *np.sqrt(Gamma_f) * Rho_ge + alpha * np.sqrt(Gamma_e) * Rho_ef  + (1j*Delta_1 + 1j*Delta_2 - Gamma_f/2) * Rho_gf
        dRho_gedt = alpha * np.sqrt(Gamma_e) * (Rho_ee - Rho_gg) + (1j*Delta_1 - Gamma_e/2) * Rho_ge + alpha * np.sqrt(Gamma_f) * Rho_gf + np.sqrt(Gamma_e*Gamma_f) * Rho_ef
        dRho_eedt = Gamma_f*Rho_ff - Gamma_e*Rho_ee  - alpha *np.sqrt(Gamma_e)* (Rho_ge + np.conjugate(Rho_ge)) + alpha * np.sqrt(Gamma_f) * (Rho_ef + np.conjugate(Rho_ef)) 
        dRho_ggdt = Gamma_e * Rho_ee + alpha * np.sqrt(Gamma_e) * (Rho_ge + np.conjugate(Rho_ge)) 
        return [dRho_ffdt, dRho_efdt, dRho_gfdt, dRho_eedt, dRho_gedt, dRho_ggdt]
        
    initial_condition = [0 , 0 , 0 , 0 , 0 , 1 ]
    t = np.linspace(-2, 6 , nBins) / np.min([Gamma_e, Gamma_f])
    solver = complex_ode(rhs)
    solver.set_initial_value(initial_condition, t[0])
    solver.set_integrator('vode', method='bdf', rtol=1e-9, atol=1e-12)  
    r = []
    for time in t[1:]:
        r.append(solver.integrate(time))
    r.insert(0, initial_condition)
    r = np.array(r)
    return -r[:,0].max()


def task(PARAMS: tuple)->None:
    Gamma_e, Delta_1 = PARAMS
    Delta_2 = -Delta_1
    t0 = time()        
    def P_negated(params):
        Omega,= params
        return P_coherent(Omega, Gamma_e, Gamma_f, Delta_1, Delta_2, n_bar, nBins)
    res = optimize(P_negated, 
                    sp_bounds = np.array([[1e-6,10]]),
                    p_bounds = [(1e-6,None)], 
                    N = 100)
    try:
        res['Omega'], =  res.x
        res['P_max'] = -res['fun']
        res['ComputTime'] = time()-t0
        res['Gamma_e'] = Gamma_e
        res['Delta_1'] = Delta_1
        res['Delta_2'] = Delta_2
        res = pd.DataFrame([res])
        res = res[['P_max', 'Omega', 'Gamma_e', 'Delta_1', 'Delta_1', 'number_of_fails', 'ComputTime'] ]
        res.to_csv(fileName, mode='a', header=True, index=False)
        return 
    except Exception as e:
        print('\nThere was an error while processing the results:')
        print(f'Error: {e}')   


fileName = get_unique_filename('HeatMap_Coherent_P_Delta.csv')
print(f'Optimization for "{fileName}" ')

Gamma_e_range = np.logspace(-2,3,101,endpoint=True)
# Gamma_e_range = Gamma_e_range[5:8*5]  # To limit the number of tasks for demo
Delta_range = np.linspace(5.1,10,50)
# Delta_range = Delta_range[0:3]  # To limit the number of tasks for demo
params_range = [(d1, d2) for d1 in Gamma_e_range for d2 in Delta_range]


Gamma_f = 1
n_bar = 2
nBins = 10000


time0 = time()
p = Pool()
p.map(task,params_range)
df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_e'])
df.to_csv(fileName, index=False)
print(f'All tasks done in {int(time()-time0)}s and saved in "{fileName}"')
