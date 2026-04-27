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

############################### Entangled case ##############################################

mp.mp.dps = 60
def P ( Gamma_e, Gamma_f= 1, w_fe = 0, w_eg = 0, w_01 = 0, w_02 = 0):
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
        return p
    return inner

def Entangled_Marginal(OmegaP, OmegaM, Mu, delta_f=0):
    MuP = Mu
    MuM = Mu
    w_gf = 0
    w02 = mp.mpf('0.5') * (w_gf + delta_f)
    w01 = mp.mpf('0.5') * (w_gf - delta_f)
    dW = w02 - w01

    N = mp.exp(-OmegaM**2 * MuM**2 / 4 - dW**2 / OmegaM**2) + 1
    N *= 8 * mp.pi / (OmegaP * OmegaM)

    def inner(t):
        t = mp.mpf(t)
        A = OmegaP**2 / 4
        B = OmegaM**2 / 4
        marginal = (
            2
            * mp.exp(
                -A * B * (2 * t - MuP)**2 / (A + B)
                - B * MuM**2
                - dW**2 / (4 * A + 4 * B)
            )
            * mp.cos(A * dW * (2 * t - MuP) / (A + B))
        )

        marginal += mp.exp(
            -A * B * (2 * t - MuP + MuM)**2 / (A + B)
        )

        marginal += mp.exp(
            -A * B * (2 * t - MuP - MuM)**2 / (A + B)
        )

        marginal *= mp.sqrt(mp.pi / (A + B)) * 2 / N
        return marginal

    return inner

def task(t, params):
    delta_a = params['delta_a']
    delta_f = params['delta_f']

    w_fe = mp.mpf(0.5) * delta_a
    w_eg = -mp.mpf(0.5) * delta_a
    w_01 = -mp.mpf(0.5) * delta_f
    w_02 = mp.mpf(0.5) * delta_f

    P_result = P(
        Gamma_e=params['Gamma_e'],
        Gamma_f=params['Gamma_f'],
        w_fe=w_fe,
        w_eg=w_eg,
        w_01=w_01,
        w_02=w_02
    )([
        t,
        params['OmegaP'],
        params['OmegaM'],
        params['Mu']
    ])

    marginal_result = Entangled_Marginal(
        OmegaP=mp.mpf(params['OmegaP']),
        OmegaM=mp.mpf(params['OmegaM']),
        Mu=mp.mpf(params['Mu']),
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
        'delta_f': 2.45,
        'OmegaP': 0.42,
        'OmegaM': 0.28,
        'Mu': 0,
        't_min': -8,
        't_max': 12,
        't_points': 5000
    },
    {
        'name': 'G1',
        'Gamma_e': 1,
        'Gamma_f': 1,
        'delta_a': 2.5,
        'delta_f': 1.81,
        'OmegaP': 0.89,
        'OmegaM': 1.21,
        'Mu': 0,
        't_min': -8,
        't_max': 12,
        't_points': 5000
    },
    {
        'name': 'G10',
        'Gamma_e': 10,
        'Gamma_f': 1,
        'delta_a': 2.50,
        'delta_f': 0.002,
        'OmegaP': 1.03,
        'OmegaM': 11.09,
        'Mu': 0.0,
        't_min': -8,
        't_max': 12,
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

    file_name = f"Entangled_dynamics_{params['name']}.csv"
    df.to_csv(file_name, index=False)

    print(f"Saved: {file_name}")