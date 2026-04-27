import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import itertools as it
import operator
import os
from typing import List, Tuple, Optional


def optimize(objective: callable, sp_bounds: np.ndarray, p_bounds: np.ndarray, N: int ):
    # default_tol = 1e-4
    # xatol = fatol = tol if tol is not None else default_tol

    s = 0; f = 0
    res={'fun':1e100}
    while s < N:
        new_res = minimize(	objective, 
                        np.random.rand(sp_bounds.shape[0])*np.ndarray.__sub__(*sp_bounds.T) + sp_bounds[:,1],
                        method='Nelder-Mead',
                        bounds = p_bounds,
                        )
        if (new_res is not None) and (new_res.success):
            s += 1
            res = min((res,new_res),key=operator.itemgetter('fun'))
        else:
            f += 1        
    res['number_of_fails'] = f
    return res



def P_negated(Gamma_e: float, Gamma_f: float, psi: callable, Omega_fe: Optional[float]=0, Omega_eg: [float]=0, t0: float = -np.inf, tol: float =1e-12, limit: int = 200) -> callable:
    def inner(params):
        t, *params = params
        intg = quad( lambda s2: np.exp(-Gamma_f/2 * t + (1j*Omega_fe + Gamma_f/2 - Gamma_e/2 )*s2)
                    *quad( lambda s1: np.exp((1j*Omega_eg + Gamma_e/2)*s1)*psi(params)(s2,s1)
                          , t0, s2,epsabs=tol, complex_func=True )[0] ,t0, t,epsabs=tol, limit = limit, complex_func=True)[0]
        return - 4*Gamma_e * Gamma_f * np.abs(intg)**2         # Based on second equation of 15.
    return inner



def get_unique_filename(base_filename:str)->str:
    r"""
    Ensures the output file doesn't overwrite an existing one by adding a numeric suffix (_1, _2, etc.) 
    if a file with the same name exists.
    
    Parameters:
    -----------
    base_filename : str
        Desired file name (including extension).
    
    Returns:
    --------
    str
        A unique filename that doesn't conflict with existing files.
    
    Example:
    --------
    'output.csv' -> 'output_1.csv' if 'output.csv' exists.
    """
    
    if not os.path.exists(base_filename):
        return base_filename
    else:
        counter = 1
        filename, ext = os.path.splitext(base_filename)
        new_filename = f"{filename}_{counter}{ext}"
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{filename}_{counter}{ext}"
        return new_filename
