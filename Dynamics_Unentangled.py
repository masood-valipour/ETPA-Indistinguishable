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
from functools import partial
os.makedirs('DataSets/Dynamics',exist_ok=True)
os.chdir('DataSets/Dynamics')

############################### Unentangled case ##############################################

mp.mp.dps = 60
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
        return p
    return inner


def Marginal(OmegaA, OmegaB, Mu, delta_f=0):
    MuB = Mu
    MuA = 0

    w_gf = 0
    wB = mp.mpf('0.5') * (w_gf + delta_f)
    wA = mp.mpf('0.5') * (w_gf - delta_f)

    def inner(t):
        t = mp.mpf(t)
        PsiA = (OmegaA**2 / (2 * mp.pi))**mp.mpf('0.25')
        PsiA *= mp.exp(-OmegaA**2 * (t - MuA)**2 / 4 - 1j * wA * t)
        PsiB = (OmegaB**2 / (2 * mp.pi))**mp.mpf('0.25')
        PsiB *= mp.exp(-OmegaB**2 * (t - MuB)**2 / 4 - 1j * wB * t)
        N = mp.exp(
            -(
                (MuB - MuA)**2 * OmegaA**2 * OmegaB**2
                + 4 * (wB - wA)**2
            ) / (2 * OmegaA**2 + 2 * OmegaB**2)
        )

        N = 4 * (
            1
            + 2 * OmegaA * OmegaB * N / (OmegaA**2 + OmegaB**2)
        )

        X = (
            OmegaA**2 * OmegaB**2 * (MuB - MuA)**2
            + 4 * (wB - wA)**2
            - 4j * (wB - wA) * (MuA * OmegaA**2 + MuB * OmegaB**2)
        )

        X /= -4 * (OmegaA**2 + OmegaB**2)

        X = mp.exp(X)
        X *= mp.sqrt(
            2 * OmegaA * OmegaB / (OmegaA**2 + OmegaB**2)
        )

        marginal = (
            abs(PsiA)**2
            + abs(PsiB)**2
            + mp.conj(PsiA) * PsiB * X
            + mp.conj(PsiB) * PsiA * mp.conj(X)
        )
        marginal *= 2 / N
        return abs(marginal)

    return inner

def task(t, params):
    delta_a = params['delta_a']
    delta_f = params['delta_f']

    w_fe = mp.mpf(0.5) * delta_a
    w_eg = -mp.mpf(0.5) * delta_a
    w_a = -mp.mpf(0.5) * delta_f
    w_b = mp.mpf(0.5) * delta_f

    P_result = P(
        Gamma_e=params['Gamma_e'],
        Gamma_f=params['Gamma_f'],
        w_fe=w_fe,
        w_eg=w_eg,
        w_a=w_a,
        w_b=w_b
    )([
        t,
        params['OmegaA'],
        params['OmegaB'],
        params['MuB']
    ])

    marginal_result = Marginal(
        OmegaA=mp.mpf(params['OmegaA']),
        OmegaB=mp.mpf(params['OmegaB']),
        Mu=mp.mpf(params['MuB']),
        delta_f=mp.mpf(delta_f)
    )(t)

    return {
        't': float(t),
        'P': float(P_result),
        'Marginal': float(marginal_result)
    }


parameter_sets = [
    {
        'name': 'G0.1',
        'Gamma_e': 0.1,
        'Gamma_f': 1,
        'delta_a': 2.5,
        'delta_f': 2.49,
        'OmegaA': 0.15,
        'OmegaB': 1.5,
        'MuB': 9.99,
        't_min': -10,
        't_max': 20,
        't_points': 5000
    },
    {
        'name': 'G1',
        'Gamma_e': 1,
        'Gamma_f': 1,
        'delta_a': 2.5,
        'delta_f': 2.09,
        'OmegaA': 1.02,
        'OmegaB': 1.78,
        'MuB': 1.05,
        't_min': -10,
        't_max': 20,
        't_points': 5000
    },
    {
        'name': 'G10',
        'Gamma_e': 10,
        'Gamma_f': 1,
        'delta_a': 2.50,
        'delta_f': 4.16,
        'OmegaA': 2.71,
        'OmegaB': 2.79,
        'MuB': 0.04,
        't_min': -10,
        't_max': 20,
        't_points': 5000
    }
]



for params in parameter_sets:
    T = np.linspace(
        params['t_min'],
        params['t_max'],
        params['t_points']
)
    current_task = partial(task, params=params)

    with Pool() as pool:
        results = pool.map(current_task, T)

    df = pd.DataFrame(results)

    file_name = f"Unentangled_dynamics_{params['name']}.csv"
    df.to_csv(file_name, index=False)

    print(f"Saved: {file_name}")