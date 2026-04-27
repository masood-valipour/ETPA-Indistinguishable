import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
# os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

N_T, N_W = 200, 200

df_UnEnt = pd.read_csv('./DataSets/HeatMap_deltas_UnEnt_Delay.csv')

# === Unentangled Temporal ===
def Unentangled_temporal(OmegaA, OmegaB, Mu, delta_f=0, marginal=False):
    MuB = Mu
    MuA = 0
    w_gf = 0
    wB = 0.5 * (w_gf + delta_f)
    wA = 0.5 * (w_gf - delta_f)

    T1, T2 = np.meshgrid(T_UnEn , T_UnEn )
    def PsiA(t): return (OmegaA**2/(2*np.pi))**0.25 * np.exp(-OmegaA**2 * (t - MuA)**2 / 4 - 1j * wA * t)
    
    def PsiB(t): return (OmegaB**2/(2*np.pi))**0.25 * np.exp(-OmegaB**2 * (t - MuB)**2 / 4 - 1j * wB * t)

    N = np.exp(-((MuB-MuA)**2 * OmegaA**2 * OmegaB**2 + 4 * (wB - wA)**2) / (2 * OmegaA**2 + 2 * OmegaB**2))
    N = 4 * (1 + 2 * OmegaA * OmegaB * N / (OmegaA**2 + OmegaB**2))
    def PHI(t2, t1): return (PsiA(t1) * PsiB(t2) + PsiA(t2) * PsiB(t1)) / N**0.5

    def analytic_marginal (t):
        X = OmegaA**2*OmegaB**2*(MuB-MuA)**2 + 4*(wB-wA)**2 - 4j*(wB-wA)*(MuA*OmegaA**2 + MuB*OmegaB**2)
        X /= -4*(OmegaA**2 + OmegaB**2)
        X = np.exp(X)
        X *= np.sqrt(2*OmegaA*OmegaB /(OmegaA**2 + OmegaB**2))
        P = np.abs(PsiA(t))**2 + np.abs(PsiB(t))**2 + np.conjugate(PsiA(t))*PsiB(t)*X + np.conjugate(PsiB(t))*PsiA(t)*np.conjugate(X)
        P *= 2/N
        return np.abs(P)
    if marginal:
        return analytic_marginal(T_UnEn)
    else:
        return 2 * np.abs(PHI(T2, T1)**2)

# === Unentangled Frequency ===
def Unentangled_frequency( OmegaA, OmegaB, MuA=0, MuB=0, delta_f=0, marginal=False):
    W1, W2 = np.meshgrid(W_UnEn, W_UnEn)
    w_gf = 0
    wB = 0.5 * (w_gf + delta_f)
    wA = 0.5 * (w_gf - delta_f)
    def PsiA(w): return (2 / (OmegaA**2 * np.pi))**0.25 * np.exp(-(w - wA)**2 / OmegaA**2 + 1j * MuA * (w - wA))
    def PsiB(w): return (2 / (OmegaB**2 * np.pi))**0.25 * np.exp(-(w - wB)**2 / OmegaB**2 + 1j * MuB * (w - wB))

    N = np.exp(-(MuB**2 * OmegaA**2 * OmegaB**2 + 4 * (wB - wA)**2) / (2 * OmegaA**2 + 2 * OmegaB**2))
    N = 4 * (1 + 2 * OmegaA * OmegaB * N / (OmegaA**2 + OmegaB**2))

    def PHI(w1, w2): return (PsiA(w1) * PsiB(w2) + PsiA(w2) * PsiB(w1)) / N**0.5
    def analytic_marginal (w1):
        A = 1j*(MuB-MuA) - 2 *(wA-wB)/OmegaB**2
        A = A**2*OmegaA**2*OmegaB**2
        A/= 4*OmegaA**2+4*OmegaB**2
        A += 1j*MuB*(wA-wB) - (wA-wB)**2/OmegaB**2
        A = np.exp(A)* (2*OmegaA*OmegaB)**0.5/(OmegaA**2+OmegaB**2)**0.5
        A = np.conjugate(A)       
        p = np.conjugate(PsiA(w1))*PsiB(w1)*A
        p += np.conjugate(p)
        p += np.abs(PsiA(w1))**2 + np.abs(PsiB(w1))**2 
        p *=2/N
        return np.abs(p)
    if marginal:
        return analytic_marginal(W_UnEn)
    else:
        return 2 * np.abs(PHI(W2, W1)**2)
    

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(62, 106*5, figure=fig, hspace=0.0, wspace=0.1)

delta_a = 2.5


############################# First variable set #############################
W_UnEn = np.linspace(-3, 3, N_W)
T_UnEn = np.linspace(-15, 20, N_T)
Gamma_e = 0.1

row = df_UnEnt[(df_UnEnt["Gamma_e"] == Gamma_e) & (df_UnEnt["delta_a"] == delta_a)]
OmegaA, OmegaB, Mu, delta_f_UnEnt = float(row['OmegaA']), float(row['OmegaB']), float(row['Mu']), float(row['delta_f'])



ax0 = fig.add_subplot(gs[0:8, 0:27*5+2])
P_freq = Unentangled_frequency(OmegaA, OmegaB, MuA=0, MuB=Mu, delta_f=delta_f_UnEnt, marginal=True)
ax0.plot(W_UnEn, P_freq, 'darkred', linewidth=2)
ax0.set_ylabel(r'$p(\omega)$', fontsize=14)
ax0.text(0.03, 0.97, '(a)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='white', weight='bold')
ax0.tick_params(labelbottom=False)
ax0.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax0.text(0.03, 0.97, '(a)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax0.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.set_yticks(np.array([ 0, 1, 2]))


ax1 = fig.add_subplot(gs[8:28, 0:30*5],sharex=ax0)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_freq_marg = Unentangled_frequency(OmegaA, OmegaB, MuA=0, MuB=Mu, delta_f=delta_f_UnEnt, marginal=False)
im1 = ax1.imshow(np.real(P_freq_marg), origin='lower',
                    extent=[W_UnEn.min(), W_UnEn.max(), W_UnEn.min(), W_UnEn.max()],
                    aspect='auto', cmap='hot_r')

Z = np.real(P_freq_marg)
levels = np.percentile(Z, [ 91])
cs = ax1.contour(W_UnEn, W_UnEn, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [( 1, -1.0)]
ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)
# ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f')

ax1.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
ax1.set_ylabel(r"$(\omega_2 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
# axs[1].set_ylabel("$(\omega_2 - \omega_{eg})/\Gamma_f$", fontsize=14)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.text(0.63, 0.85, r'$\omega = \omega_{fe}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.2, 0.15, r'$\omega = \omega_{eg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.locator_params(axis='both', nbins=5)



ax2 = fig.add_subplot(gs[34:42, 0:27*5+2])
P_t = Unentangled_temporal(OmegaA, OmegaB, Mu, delta_f=delta_f_UnEnt, marginal=True)
ax2.plot(T_UnEn, P_t, 'darkred', linewidth=2)
ax2.set_ylabel(r'$p(t)$', fontsize=12)
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)

ax2.tick_params(labelbottom=False)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax2.set_yticks(np.array([ 0, 0.15, 0.3]))


ax3 = fig.add_subplot(gs[42:62, 0:30*5], sharex=ax2)
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_t_marg = Unentangled_temporal(OmegaA, OmegaB, Mu, delta_f=delta_f_UnEnt, marginal=False)
im1 = ax3.imshow(np.real(P_t_marg), origin='lower',
                    extent=[T_UnEn.min(), T_UnEn.max(), T_UnEn.min(), T_UnEn.max()],
                    aspect='auto', cmap='hot_r')

Z = np.real(P_t_marg)
levels = np.percentile(Z, [ 89])
cs = ax3.contour(T_UnEn, T_UnEn, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    ( -7, 10.0),   # lower
]
ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)

ax3.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
ax3 .set_ylabel(r"$t_2\Gamma_f$", fontsize=14)
plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)

ax3.locator_params(axis='both', nbins=5)
ax3.set_yticks(np.array([ -10, 0,10,]))


############################# Second variable set #############################
W_UnEn = np.linspace(-3.5, 3.5, N_W)
T_UnEn = np.linspace(-3, 3, N_T)

Gamma_e = 1

row = df_UnEnt[(df_UnEnt["Gamma_e"] == Gamma_e) & (df_UnEnt["delta_a"] == delta_a)]
OmegaA, OmegaB, Mu, delta_f_UnEnt = float(row['OmegaA']), float(row['OmegaB']), float(row['Mu']), float(row['delta_f'])



ax0 = fig.add_subplot(gs[0:8, 38*5:65*5+2])
P_freq = Unentangled_frequency(OmegaA, OmegaB, MuA=0, MuB=Mu, delta_f=delta_f_UnEnt, marginal=True)
ax0.plot(W_UnEn, P_freq, 'darkred', linewidth=2)
# ax0.set_ylabel(r'$p(\omega)$', fontsize=14)
ax0.tick_params(labelbottom=False)
ax0.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax0.text(0.03, 0.97, '(c)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax0.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.set_yticks(np.array([ 0, 0.2, 0.4]))

ax1 = fig.add_subplot(gs[8:28, 38*5:68*5],sharex=ax0)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_freq_marg = Unentangled_frequency(OmegaA, OmegaB, MuA=0, MuB=Mu, delta_f=delta_f_UnEnt, marginal=False)
im1 = ax1.imshow(np.real(P_freq_marg), origin='lower',
                    extent=[W_UnEn.min(), W_UnEn.max(), W_UnEn.min(), W_UnEn.max()],
                    aspect='auto', cmap='hot_r')
Z = np.real(P_freq_marg)
levels = np.percentile(Z, [ 70])
cs = ax1.contour(W_UnEn, W_UnEn, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    ( 1.5, 1.0),   # lower
]
ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)

ax1.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
# ax1.set_ylabel(r"$(\omega_2 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
# axs[1].set_ylabel("$(\omega_2 - \omega_{eg})/\Gamma_f$", fontsize=14)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.text(0.70, 0.83, r'$\omega = \omega_{fe}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.2, 0.17, r'$\omega = \omega_{eg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.locator_params(axis='both', nbins=5)
ax1.set_yticks(np.array([ -2, 0,2]))



ax2 = fig.add_subplot(gs[34:42, 38*5:65*5+2])
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_t = Unentangled_temporal(OmegaA, OmegaB, Mu, delta_f=delta_f_UnEnt, marginal=True)
ax2.plot(T_UnEn, P_t, 'darkred', linewidth=2)
# ax2.set_ylabel(r'$p(t)$', fontsize=12)
ax2.tick_params(labelbottom=False)
ax2.text(0.03, 0.97, '(d)', transform=ax2.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax2.set_yticks(np.array([ 0, 0.25, 0.5]))


ax3 = fig.add_subplot(gs[42:62, 38*5:68*5], sharex=ax2)
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_t_marg = Unentangled_temporal(OmegaA, OmegaB, Mu, delta_f=delta_f_UnEnt, marginal=False)
im1 = ax3.imshow(np.real(P_t_marg), origin='lower',
                    extent=[T_UnEn.min(), T_UnEn.max(), T_UnEn.min(), T_UnEn.max()],
                    aspect='auto', cmap='hot_r')
Z = np.real(P_t_marg)
levels = np.percentile(Z, [ 70])
cs = ax3.contour(T_UnEn, T_UnEn, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    ( -2, 1.0),   # lower
]
ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)


ax3.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
# ax3 .set_ylabel(r"$t_2\Gamma_f$", fontsize=14)
plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)

ax3.locator_params(axis='both', nbins=5)
ax3.set_yticks(np.array([ -2,0, 2]))


############################# Third variable set #############################
W_UnEn = np.linspace(-5, 5, N_W)
T_UnEn = np.linspace(-1.5, 1.5, N_T)

Gamma_e = 10

row = df_UnEnt[(df_UnEnt["Gamma_e"] == Gamma_e) & (df_UnEnt["delta_a"] == delta_a)]
OmegaA, OmegaB, Mu, delta_f_UnEnt = float(row['OmegaA']), float(row['OmegaB']), float(row['Mu']), float(row['delta_f'])



ax0 = fig.add_subplot(gs[0:8, 76*5:103*5+2])
P_freq = Unentangled_frequency(OmegaA, OmegaB, MuA=0, MuB=Mu, delta_f=delta_f_UnEnt, marginal=True)
ax0.plot(W_UnEn, P_freq, 'darkred', linewidth=2)
# ax0.set_ylabel(r'$p(\omega)$', fontsize=14)
ax0.tick_params(labelbottom=False)
ax0.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax0.text(0.03, 0.97, '(e)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax0.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.set_yticks(np.array([  0,0.05,0.1, 0.15]))

ax1 = fig.add_subplot(gs[8:28, 76*5:106*5],sharex=ax0)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_freq_marg = Unentangled_frequency(OmegaA, OmegaB, MuA=0, MuB=Mu, delta_f=delta_f_UnEnt, marginal=False)
im1 = ax1.imshow(np.real(P_freq_marg), origin='lower',
                    extent=[W_UnEn.min(), W_UnEn.max(), W_UnEn.min(), W_UnEn.max()],
                    aspect='auto', cmap='hot_r')
Z = np.real(P_freq_marg)
levels = np.percentile(Z, [ 50])
cs = ax1.contour(W_UnEn, W_UnEn, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    ( 3, 1.0),   # lower
]
ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)


ax1.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
cbar0 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar0.set_label(r'$p(\omega_1, \omega_2)$',  labelpad=5, loc='center', fontsize=14)
ax1.plot([W_UnEn.min(), W_UnEn.max()], [W_UnEn.max(), W_UnEn.min()], color='black', linestyle='--', linewidth=1)
ax1.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.text(0.65, 0.83, r'$\omega = \omega_{fe}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.3, 0.2, r'$\omega = \omega_{eg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.35, 0.55, r'$\omega_1 + \omega_2 = \omega_{fg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=-45, color="#CBFFF8", weight='bold')
ax1.locator_params(axis='both', nbins=5)



ax2 = fig.add_subplot(gs[34:42, 76*5:103*5+2])
P_t = Unentangled_temporal(OmegaA, OmegaB, Mu, delta_f=delta_f_UnEnt, marginal=True)
ax2.plot(T_UnEn, P_t, 'darkred', linewidth=2)
# ax2.set_ylabel(r'$p(t)$', fontsize=12)
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax2.tick_params(labelbottom=False)
ax2.text(0.03, 0.97, '(f)', transform=ax2.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax2.set_yticks(np.array([  0,0.5,1]))

ax3 = fig.add_subplot(gs[42:62, 76*5:106*5], sharex=ax2)
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_t_marg = Unentangled_temporal(OmegaA, OmegaB, Mu, delta_f=delta_f_UnEnt, marginal=False)
im1 = ax3.imshow(np.real(P_t_marg), origin='lower',
                    extent=[T_UnEn.min(), T_UnEn.max(), T_UnEn.min(), T_UnEn.max()],
                    aspect='auto', cmap='hot_r')

Z = np.real(P_t_marg)
levels = np.percentile(Z, [ 75])
cs = ax3.contour(T_UnEn, T_UnEn, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    ( -2, 2.0),   # lower
    ( 1, 1.0),   # lower
    ( 2, -2.0),   # lower
]
ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)
ax3.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
# ax3 .set_ylabel(r"$t_2\Gamma_f$", fontsize=14)
cbar1 = plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
cbar1.set_label(r'$p(t_1, t_2)$',  labelpad=5, loc='center', fontsize=14)

ax3.locator_params(axis='both', nbins=5)

plt.tight_layout()
plt.savefig('Plots/UnEnt_PDFinTimeFreq.png', dpi=300)
# plt.show()
