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

############################### Optimal case ##############################################
mp.mp.dps = 20

def P(Gamma_e, w_fe=0, w_eg=0):
    def inner(t):
        t = mp.mpf(t)
        t_star = mp.mpf(0)
        Gamma_f = mp.mpf(1)

        Gamma_e_local = mp.mpf(Gamma_e)

        if Gamma_e_local == Gamma_f:
            Gamma_e_local += mp.mpf('1e-6') * Gamma_e_local

        N_opt = mp.exp(Gamma_f * t_star) / (Gamma_e_local * Gamma_f)

        def Phi_ordered(t1, t2):
            # Theta(t - t2) Theta(t2 - t1) Theta(t1 )
            if not ( t1 <= t2 <= t_star):
                # print(f"Invalid order: t1={t1}, t2={t2}, t={t}")
                return mp.mpf(0)

            phi = mp.exp(
                Gamma_f * t2 / 2
                - Gamma_e_local * (t2 - t1) / 2
                - 1j * w_fe * t2
                - 1j * w_eg * t1
            )

            # phi /= mp.sqrt(N_opt)

            return phi

        def integrand_t2(t2):
            def integrand_t1(t1):
                phi_sym = Phi_ordered(t1, t2) + Phi_ordered(t2, t1)

                prefactor = mp.exp(
                    1j * w_fe * t2
                    - Gamma_f * (t - t2) / 2
                )

                prefactor *= mp.exp(
                    1j * w_eg * t1
                    - Gamma_e_local * (t2 - t1) / 2
                )

                return prefactor * phi_sym

            return mp.quad(integrand_t1, [-np.inf, t2])

        amplitude = mp.quad(integrand_t2, [ -np.inf, t])

        p = Gamma_e_local * Gamma_f  * abs(amplitude)**2  / N_opt
        # print(f"\n t={t}, P={p}, Gamma_e={Gamma_e_local}, Gamma_f={Gamma_f}, N_opt={N_opt}\n")
        return p

    return inner




def Optimal_Marginal(Gamma_e, Gamma_f=1, t_star=0):
    def inner(t):
        t = mp.mpf(t)
        if t > t_star:
            return mp.mpf(0)
        Gamma_e_local = mp.mpf(Gamma_e)
        Gamma_f_local = mp.mpf(Gamma_f)
        t_star_local = mp.mpf(t_star)

        if Gamma_e_local == Gamma_f_local:
            Gamma_e_local += mp.mpf('1e-6') * Gamma_e_local

        marginal = (
            mp.exp((Gamma_f_local - Gamma_e_local) * t_star_local + Gamma_e_local * t)
            - mp.exp(Gamma_f_local * t)
        )

        marginal /= (Gamma_f_local - Gamma_e_local)

        marginal += mp.exp(Gamma_f_local * t) / Gamma_e_local

        marginal *= (
            Gamma_e_local
            * Gamma_f_local
            * mp.exp(-Gamma_f_local * t_star_local)
            / 2
        )

        return marginal

    return inner

def task(t, params):
    delta_a = params['delta_a']
    w_fe = mp.mpf(0.5) * delta_a
    w_eg = -mp.mpf(0.5) * delta_a

    P_result = P(
        Gamma_e=params['Gamma_e'],
        w_fe=w_fe,
        w_eg=w_eg
    )(
        t
    )

    marginal_result = Optimal_Marginal(
        Gamma_e=params['Gamma_e']
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
        'delta_a': 2.5,
        't_min': -6,
        't_max': 5,
        't_points': 500
    },
    {
        'name': 'G1',
        'Gamma_e': 1,
        'delta_a': 2.5,
        't_min': -6,
        't_max': 5,
        't_points': 500
    },
    {
        'name': 'G10',
        'Gamma_e': 10,
        'delta_a': 2.50,
        't_min': -6,
        't_max': 5,
        't_points': 500
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

    file_name = f"Optimal_dynamics_{params['name']}.csv"
    df.to_csv(file_name, index=False)

    print(f"Saved: {file_name}")