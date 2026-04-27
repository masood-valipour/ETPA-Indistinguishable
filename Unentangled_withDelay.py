from multiprocessing import Pool
import numpy as np
from time import time
from typing import List, Tuple, Optional
import pandas as pd
from util import optimize, P_negated, get_unique_filename
import os
from scipy.special import erf
from scipy.integrate import quad
import mpmath as mp

os.makedirs('DataSets',exist_ok=True)
# os.chdir('DataSets')
os.chdir('Cluster')


'''
Unentangled Two-Photon Absorption Probability Calculation with Delay
This module provides high-precision computation of the two-photon absorption probability, P(t), for unentangled photons with a temporal delay. The calculation is based on a closed-form formula involving error functions and Gaussian integrals, allowing for efficient and accurate evaluation using arbitrary-precision arithmetic (mpmath).
The main function, `P`, returns a callable suitable for numerical optimization over the relevant physical parameters (pulse widths, delay, etc.). The script is designed to scan a range of decay rates (`Gamma_e`) and optimize the absorption probability for each value, saving results to a CSV file.
If, for any reason, a slower and less confident but more general method is required, the user may uncomment and use the `Gaussian_Unentangled_withDelay` function provided in the code. This alternative approach uses direct numerical integration and is less efficient but more broadly applicable.
Formula Used
------------
The probability of two-photon absorption as a function of time, P(t), is given by:
\begin{equation}
\begin{split}
P(t) = &\; \frac{8 \Gamma_e \Gamma_f}{\Omega_a \Omega_b \, N_{\text{sym}}} \, e^{-\Gamma_f t} \\
&\quad \times \Bigg|\, 
    \exp\left\{ \frac{\left( 2\beta + \Omega_a^2\mu_a - 2i\omega_a \right)^2}{4\Omega_a^2} - \frac{\mu_a^2 \Omega_a^2}{4} +(\alpha -i\omega_b)\mu_b + (\frac{\alpha-i\omega_b}{\Omega_b})^2 \right\} \\
&\qquad \times \int_{-\infty}^{\frac{\Omega_b}{2}(t-\mu_b) - \frac{\alpha-i\omega_b}{\Omega_b}} dx\, \exp\left( -x^2 \right)
       \left[ 1 + \mathrm{erf}\left( \frac{\Omega_a }{\Omega_b}x  + \frac{\Omega_a}{\Omega_b}\frac{\alpha - i\omega_b}{\Omega_b}  + \frac{\Omega_a}{2}(\mu_b - \mu_a) - \frac{\beta - i\omega_a}{\Omega_a} \right) \right] \\
&\qquad + \exp\left\{ \frac{\left( 2\beta + \Omega_b^2\mu_b  - 2i\omega_b \right)^2}{4\Omega_b^2} - \frac{\mu_b^2 \Omega_b^2}{4} +(\alpha - i\omega_a)\mu_a + (\frac{\alpha-i\omega_a}{\Omega_a})^2 \right\} \\
&\qquad \times \int_{-\infty}^{\frac{\Omega_a}{2}(t-\mu_a) - \frac{\alpha-i\omega_a}{\Omega_a}} dx\, \exp\left( -x^2 \right)
       \left[ 1 + \mathrm{erf}\left( \frac{\Omega_b}{\Omega_a}x + \frac{\Omega_b}{\Omega_a}\frac{\alpha - i\omega_a}{\Omega_a}  - \frac{\Omega_b}{2}(\mu_b - \mu_a) - \frac{\beta - i\omega_b}{\Omega_b} \right) \right] 
    \,\Bigg|^2
\end{split}
\end{equation}
where:
    - $\alpha = i\omega_{fe} + (\Gamma_f-\Gamma_e)/2$
    - $\beta = i\omega_{eg} + \Gamma_e/2$
    - $\Omega_a$, $\Omega_b$ are the pulse bandwidths
    - $\mu_a$, $\mu_b$ are the pulse centers (delay)
    - $\omega_a$, $\omega_b$, $\omega_{fe}$, $\omega_{eg}$ are transition frequencies
    - $\Gamma_e$, $\Gamma_f$ are decay rates
    - $N_{\text{sym}}$ is a normalization factor
Usage Notes
-----------
- The main calculation uses high-precision arithmetic (mpmath) for stability and accuracy.
- Results are saved in a uniquely named CSV file for each run.
- For a more general but slower approach, uncomment and use the `Gaussian_Unentangled_withDelay` function.
'''




"""
def Gaussian_Unentangled_withDelay(params: Tuple[float, float, float]) -> callable:
    OmegaA, OmegaB, Mu = params
    N_PHI = np.exp(-(Mu**2 * OmegaA**2 * OmegaB**2)/(2*OmegaA**2 + 2*OmegaB**2))
    N_PHI *= 2 * (OmegaA * OmegaB)/(OmegaA**2 + OmegaB**2)
    N_PHI +=1 
    N_PHI *= 4
    def PSI(t2: float, t1: float):       
        return (OmegaA*OmegaB)**0.5*np.exp((-OmegaB**2*(t2-Mu)**2 - OmegaA**2*t1**2)/4)/(2*np.pi)**0.5 / N_PHI**0.5
    def PSI_sym(t2: float, t1: float):      
        return (PSI(t2,t1)+PSI(t1,t2))
    return PSI_sym

def task(Gamma_e: float) -> pd.DataFrame:
    t0 = time()
    res = optimize(	P_negated(Gamma_e, Gamma_f, Gaussian_Unentangled_withDelay, tol = 1e-15, limit = 100), 
                    sp_bounds = np.array([[0.5/Gamma_e, 1/Gamma_e],[1e-9,0.1],[1e-6,2],[0.5/Gamma_e, 1/Gamma_e]]),
                    p_bounds = ((None,None),(0,None),(0,None),(0,None)), 
                     N = 20)
    res['t_max'], res['OmegaA'], res['OmegaB'], res['Mu']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma_e
    res = pd.DataFrame([res])
    res = res[['P_max', 'OmegaA', 'OmegaB', 't_max', 'Mu', 'Gamma_e',  'number_of_fails', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return
"""

############################# high precision  ########################################
############################# with change of variable according to the furmula of ppaper  ########################################
mp.mp.dps = 20
def P( Gamma_e, Gamma_f= 1, w_fe = 0, w_eg = 0, w_a = 0, w_b = 0):
    def inner(params):
        t, OmegaA, OmegaB, Mu = map(mp.mpf, params)
        MuA = 0
        MuB = Mu
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


def task(Gamma_e: float) -> pd.DataFrame:
    t0 = time()
    res = optimize(     P (Gamma_e), 
                    sp_bounds = np.array([[0, 10],[0,300],[0,300],[0, 10]]),
                    p_bounds = ((None,None),(1e-9,None),(1e-6,None),(0,None)), 
                     N = 10)
    res['t_max'], res['OmegaA'], res['OmegaB'], res['Mu']=  res.x
    res['P_max'] = -res['fun']
    res['ComputTime'] = time()-t0
    res['Gamma_e'] = Gamma_e
    res['mp.dps'] = mp.mp.dps
    res = pd.DataFrame([res])
    res = res[['P_max', 'OmegaA', 'OmegaB', 't_max', 'Mu', 'Gamma_e',  'number_of_fails','mp.dps', 'ComputTime'] ]
    res.to_csv(fileName, mode='a', header=True, index=False)
    return

fileName = get_unique_filename('P_Gamma_Unentangled_withDelay.csv')

Gamma_f = 1

Gamma_e_range = np.logspace(-3,4,28*5,endpoint=False)
Gamma_e_range = Gamma_e_range[130:]
time0 = time()
p = Pool()
p.map(task,Gamma_e_range)
df = pd.read_csv(fileName)
df = df[df['P_max']!='P_max'].astype('float').reset_index(drop=True).sort_values(by=['Gamma_e'])
df.to_csv(fileName, index=False)
print(f'All tasks done in {int(time()-time0)}s and saved in "{fileName}"')

