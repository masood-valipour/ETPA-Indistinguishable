from matplotlib.gridspec import GridSpec
import numpy as np
from util import *
import matplotlib.pylab as plt
from scipy.integrate import complex_ode
import os
# os.makedirs('DataSets',exist_ok=True)
# os.chdir('DataSets')
plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')



def Gaussian(t: float, Omega: float = 1)-> float:
    t = np.array(t)
    return (Omega**2/(2*np.pi))**(1/4) * np.exp(-Omega**2*((t)**2)/4)


def P_coherent(Omega: float, Gamma_e: float = 1, Gamma_f: float = 1 
               , Delta_1: float = 0, Delta_2: float = 0, n_bar: int = 2, nBins: int = 100000) -> float:
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
    # t = np.linspace(-2, 6 , nBins) / np.min([Gamma_e, Gamma_f])
    # t = np.linspace(-2, 6 , nBins)*Omega
    t = np.linspace(-3/Omega, 3.5/Omega*len(str(np.round(10*Omega))) , nBins)
    solver = complex_ode(rhs)
    solver.set_initial_value(initial_condition, t[0])
    solver.set_integrator('vode', method='bdf', rtol=1e-9, atol=1e-12)  
    r = []
    for time in t[1:]:
        r.append(solver.integrate(time))
    r.insert(0, initial_condition)
    r = np.array(r)
    P = np.real(r[:, 0])
    return t, P
    

from matplotlib.gridspec import GridSpec
import numpy as np
from util import *
import matplotlib.pylab as plt
from scipy.integrate import complex_ode
import os

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')


def Gaussian(t: float, Omega: float = 1) -> float:
    t = np.array(t)
    return (Omega**2 / (2 * np.pi))**(1 / 4) * np.exp(-Omega**2 * (t**2) / 4)


def P_coherent(Omega: float, Gamma_e: float = 1, Gamma_f: float = 1,
               Delta_1: float = 0, #Delta_2: float = 0,
               n_bar: int = 2, nBins: int = 10000, t: np.ndarray = None) -> float:
    Delta_2 = - Delta_1
    def rhs(t, initial):
        Rho_ff, Rho_ef, Rho_gf, Rho_ee, Rho_ge, Rho_gg = initial
        alpha = float(Gaussian(t, Omega=Omega))
        alpha = np.sqrt(n_bar) * alpha

        dRho_ffdt = -Gamma_f * Rho_ff - alpha * np.sqrt(Gamma_f) * (Rho_ef + np.conjugate(Rho_ef))
        dRho_efdt = alpha * np.sqrt(Gamma_f) * (Rho_ff - Rho_ee) - alpha * np.sqrt(Gamma_e) * Rho_gf + (1j * Delta_2 - (Gamma_e + Gamma_f) / 2) * Rho_ef
        dRho_gfdt = -alpha * np.sqrt(Gamma_f) * Rho_ge + alpha * np.sqrt(Gamma_e) * Rho_ef + (1j * Delta_1 + 1j * Delta_2 - Gamma_f / 2) * Rho_gf
        dRho_gedt = alpha * np.sqrt(Gamma_e) * (Rho_ee - Rho_gg) + (1j * Delta_1 - Gamma_e / 2) * Rho_ge + alpha * np.sqrt(Gamma_f) * Rho_gf + np.sqrt(Gamma_e * Gamma_f) * Rho_ef
        dRho_eedt = Gamma_f * Rho_ff - Gamma_e * Rho_ee - alpha * np.sqrt(Gamma_e) * (Rho_ge + np.conjugate(Rho_ge)) + alpha * np.sqrt(Gamma_f) * (Rho_ef + np.conjugate(Rho_ef))
        dRho_ggdt = Gamma_e * Rho_ee + alpha * np.sqrt(Gamma_e) * (Rho_ge + np.conjugate(Rho_ge))

        return [dRho_ffdt, dRho_efdt, dRho_gfdt, dRho_eedt, dRho_gedt, dRho_ggdt]

    initial_condition = [0, 0, 0, 0, 0, 1]
   
    # print('inside' , Omega, ':   ',t.min(), t.max(), t[1])
    # t = np.linspace(-3,5 , nBins)



    solver = complex_ode(rhs)
    solver.set_initial_value(initial_condition, t[0])
    solver.set_integrator('vode', method='bdf', rtol=1e-9, atol=1e-12)

    r = []
    for time in t[1:]:
        r.append(solver.integrate(time))

    r.insert(0, initial_condition)
    r = np.array(r)

    P = np.real(r[:, 0])
    return t, P


OmegaS = [0.451444566398905, 1.8204451187677, 6.0465979161366]
t = np.linspace(-5, 8 , 100000)
Ts = [np.linspace(-6, 7 , 100000), np.linspace(-3, 7 , 100000), np.linspace(-2, 7 , 100000)]
# ---- Single panel plot ----
fig, [ax1, ax2] = plt.subplots(2,1,figsize=(3.5, 4.5),)

Gamma_e_value = [0.1, 1, 10]
colors = ['tab:blue', 'tab:orange', 'tab:green']
linestyles = ['-', '--', '-.']

for Gamma_e, color, ls, Omega, T in zip(Gamma_e_value, colors, linestyles, OmegaS, Ts):
    t, P = P_coherent(Omega=Omega, Gamma_e=Gamma_e, Gamma_f=1, n_bar=2, Delta_1 = 0, t= T)
    Gaussian_pulse = np.abs(Gaussian(t, Omega=Omega))**2
    ax1.plot(
        t, P,
        color=color,
        linestyle=ls,
        linewidth=1.5,
        label=rf'$\Gamma_e/\Gamma_f={Gamma_e}$'
    )
    ax2.plot(
        t, Gaussian_pulse,
        color=color,
        linestyle=ls,
        linewidth=1.5,
        label=rf'$\Gamma_e/\Gamma_f={Gamma_e}$'
    )

ax1.set_yticks(np.array([0, 0.1, 0.2, 0.3, 0.4]))

ax1.set_xlim(-3, 7)
ax1.set_ylabel(r'$P_f(t)$', fontsize=10)
ax1.legend(fontsize=10, frameon=False)
ax1.grid(True, alpha=1, color='0.7', linewidth=0.10)
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', color='black', weight='bold')

ax2.set_xlabel(r'$\Gamma_f t$', fontsize=10)
ax2.grid(True, alpha=1, color='0.7', linewidth=0.10)
ax2.legend(fontsize=10, frameon=False)
ax2.set_xlim(-3, 7)
ax2.set_ylabel(r'$|\alpha_0(t)|^2$', fontsize=10)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', color='black', weight='bold')

plt.tight_layout()
plt.savefig('./Plots/P_t_coherent.png')

