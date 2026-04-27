
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

N_T, N_W = 200, 200

df_Ent = pd.read_csv('./DataSets/HeatMap_deltas_Entangled_No-Delay.csv')


# === Entangled Temporal ===
def Entangled_temporal(OmegaP, OmegaM, t02=0, t01=0,  delta_f=0, marginal=False):
    MuP = t02 + t01
    MuM = t02 - t01
    T1, T2 = np.meshgrid(T_Ent+MuP, T_Ent+MuP)
    w_gf = 0
    w02 = 0.5 * (w_gf + delta_f)
    w01 = 0.5 * (w_gf - delta_f)
    dW = w02-w01
    N = np.exp(-OmegaM**2*MuM**2/4 - dW**2/OmegaM**2)+1
    N *= 8*np.pi/(OmegaP*OmegaM)
    def PHI(t2,t1):
        phi = np.exp(-OmegaM**2*(t2 - t1 - MuM)**2/8 -1j*(w01*t1 + w02*t2)) + np.exp(-OmegaM**2*(t1 - t2 - MuM)**2/8-1j*(w01*t2 + w02*t1))
        phi *= np.exp(-OmegaP**2*(t2 + t1 - MuP)**2/8)
        return phi/N**0.5
    def analytic_marginal(t):
        A = OmegaP**2/4
        B = OmegaM**2/4
        P = 2*np.exp(-A*B * (2*t-MuP)**2/(A+B) -B*MuM**2 - dW**2/(4*A + 4*B))  *  np.cos(A*dW*(2*t-MuP)/(A+B))
        P += np.exp(-A*B * (2*t-MuP + MuM)**2/(A+B))
        P += np.exp(-A*B * (2*t-MuP - MuM)**2/(A+B))
        P *= np.sqrt(np.pi/(A+B)) * 2/N
        return P
    if marginal:
        return analytic_marginal(T_Ent+MuP)
    else:
        return 2*np.abs(PHI(T2,T1)**2)

# === Entangled Frequency ===
def Entangled_frequency(OmegaP, OmegaM, t01=0, t02 = 0, delta_f = 0, w_gf = 0, marginal=False):
    MuP = t02 + t01
    MuM = t02 - t01
    W1, W2 = np.meshgrid(W_Ent, W_Ent)
    N = np.exp(-OmegaM**2*MuM**2/4 - (delta_f)**2/OmegaM**2)+1
    N*= 8*np.pi/(OmegaP*OmegaM)
    def PHI(w2,w1):
        DELTA = w2 - w1
        DELTA_0 = delta_f
        SIGMA = w2 + w1
        SIGMA_0 = w_gf
        phi = np.exp(-(DELTA - DELTA_0)**2/(2*OmegaM**2) + 1j*MuM*(DELTA - DELTA_0)/2)
        phi += np.exp(-(DELTA + DELTA_0)**2/(2*OmegaM**2) - 1j*MuM*(DELTA + DELTA_0)/2)
        phi *= np.exp(-(SIGMA - SIGMA_0)**2/(2*OmegaP**2) + 1j*MuP*(SIGMA - SIGMA_0)/2)
        phi *= 2/(OmegaM*OmegaP)
        return phi/N**0.5
    def analytic_marginal(w):
        MuM =0
        p = np.exp(-(2*w - w_gf + delta_f)**2/(OmegaP**2 + OmegaM**2))
        p += np.exp(-(2*w - w_gf - delta_f)**2/(OmegaP**2 + OmegaM**2))
        p += 2*np.exp(-(2*w - w_gf)**2/(OmegaP**2 + OmegaM**2) - OmegaM**2*OmegaP**2*MuM**2/(4*OmegaP**2+4*OmegaM**2) -delta_f**2/OmegaM**2) * \
             np.cos((2*w-w_gf)*OmegaM**2*MuM/(OmegaP**2 + OmegaM**2))
        p *= 4/(N*OmegaM*OmegaP) 
        p *= np.pi**0.5 /(OmegaP**2 + OmegaM**2)**0.5
        return 2*p
    if marginal:
        return analytic_marginal(W_Ent)
    else:
        return 2 * np.abs(PHI(W2, W1)**2)


fig = plt.figure(figsize=(12, 10))
gs = GridSpec(62, 106*5, figure=fig, hspace=0.0, wspace=0.1)



############################# First variable set #############################
# 1) Frequency joint probability (2D) - Gamma_e=0.1, delta_a=10
W_Ent = np.linspace(-2, 2, N_W)
T_Ent = np.linspace(-10, 10, N_T)
Gamma_e = 0.1
delta_a = 2.5

row = df_Ent[(df_Ent["Gamma_e"] == Gamma_e) & (df_Ent["delta_a"] == delta_a)]
OmegaP, OmegaM, t02, delta_f_Ent = float(row['OmegaP']), float(row['OmegaM']), float(row['Mu']) , float(row['delta_f'])


ax0 = fig.add_subplot(gs[0:8, 0:27*5+2])
P_freq = Entangled_frequency(OmegaP, OmegaM, t02 = t02,delta_f=delta_f_Ent, marginal=True)
ax0.plot(W_Ent, P_freq, 'darkred', linewidth=2)
ax0.set_ylabel(r'$p(\omega)$', fontsize=14)
ax0.text(0.03, 0.97, '(a)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax0.tick_params(labelbottom=False)
ax0.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax0.text(0.03, 0.97, '(a)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax0.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=0.5)
ax0.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=0.5)
ax0.set_yticks(np.array([ 0, 0.5, 1]))


ax1 = fig.add_subplot(gs[8:28, 0:30*5],sharex=ax0)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_freq_marg = Entangled_frequency(OmegaP, OmegaM, t02 = t02,delta_f=delta_f_Ent, marginal=False)
im1 = ax1.imshow(np.real(P_freq_marg), origin='lower',
                    extent=[W_Ent.min(), W_Ent.max(), W_Ent.min(), W_Ent.max()],
                    aspect='auto', cmap='hot_r')
Z = np.real(P_freq_marg)
levels = np.percentile(Z, [ 93])
cs = ax1.contour(W_Ent, W_Ent, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    (1.,  -1.0),   # upper
    # (1.0,  1.0),   # upper
    # ( 1.2, -2.0),   # lower
]
ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)
# ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f')

ax1.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
ax1.set_ylabel(r"$(\omega_2 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.text(0.72, 0.65, r'$\omega = \omega_{fe}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.1, 0.35, r'$\omega = \omega_{eg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.locator_params(axis='both', nbins=5)
ax1.set_yticks(np.array([-2, -1, 0, 1]))


ax2 = fig.add_subplot(gs[34:42, 0:27*5+2])
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_t = Entangled_temporal(OmegaP, OmegaM, t02=t02, delta_f=delta_f_Ent, marginal=True)
ax2.plot(T_Ent, P_t, 'darkred', linewidth=2)
ax2.set_ylabel(r'$p(t)$', fontsize=12)
ax2.tick_params(labelbottom=False)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax2.set_yticks(np.array([ 0, 0.05, 0.1]))


ax3 = fig.add_subplot(gs[42:62, 0:30*5], sharex=ax2)
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_t_marg = Entangled_temporal(OmegaP, OmegaM, t02=t02, delta_f=delta_f_Ent, marginal=False)
im1 = ax3.imshow(np.real(P_t_marg), origin='lower',
                    extent=[T_Ent.min(), T_Ent.max(), T_Ent.min(), T_Ent.max()],
                    aspect='auto', cmap='hot_r')
Z = np.real(P_t_marg)
levels = np.percentile(Z, [ 60])
cs = ax3.contour(T_Ent, T_Ent, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    (7.,  -9.0),   # upper
    # (1.0,  1.0),   # upper
    # ( 1.2, -2.0),   # lower
]
ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)
# ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f')

# axs[3].set_title(f"Temporal marginal ($\Gamma_e$={Gamma_e}, $\delta_a$={delta_a})")
ax3.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
ax3 .set_ylabel(r"$t_2\Gamma_f$", fontsize=14)
plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)

ax3.locator_params(axis='both', nbins=5)
ax3.set_yticks(np.array([-8, -4, 0, 4,8]))





############################# Second variable set #############################
W_Ent = np.linspace(-2.5, 2.5, N_W)
T_Ent = np.linspace(-3, 3, N_T)
Gamma_e = 1
delta_a = 2.5

row = df_Ent[(df_Ent["Gamma_e"] == Gamma_e) & (df_Ent["delta_a"] == delta_a)]
OmegaP, OmegaM, t02, delta_f_Ent = float(row['OmegaP']), float(row['OmegaM']), float(row['Mu']) , float(row['delta_f'])


ax0 = fig.add_subplot(gs[0:8, 38*5:65*5+2])
ax0.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_freq = Entangled_frequency(OmegaP, OmegaM, t02 = t02,delta_f=delta_f_Ent, marginal=True)
ax0.plot(W_Ent, P_freq, 'darkred', linewidth=2)
# ax0.set_ylabel(r'$p(\omega)$', fontsize=14)
ax0.tick_params(labelbottom=False)
ax0.text(0.03, 0.97, '(c)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax0.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.set_yticks(np.array([0,0.2,0.4]))


ax1 = fig.add_subplot(gs[8:28, 38*5:68*5],sharex=ax0)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_freq_marg = Entangled_frequency(OmegaP, OmegaM, t02 = t02,delta_f=delta_f_Ent, marginal=False)
im1 = ax1.imshow(np.real(P_freq_marg), origin='lower',
                    extent=[W_Ent.min(), W_Ent.max(), W_Ent.min(), W_Ent.max()],
                    aspect='auto', cmap='hot_r')

Z = np.real(P_freq_marg)
levels = np.percentile(Z, [ 50])
cs = ax1.contour(W_Ent, W_Ent, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
#     (-1.2,  2.0),   # upper
#     (1.0,  1.0),   # upper
    ( -1, -1),   # lower
]
ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)
# ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f')


ax1.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
# ax1.set_ylabel(r"$(\omega_2 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
# axs[1].set_ylabel("$(\omega_2 - \omega_{eg})/\Gamma_f$", fontsize=14)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.text(0.67, 0.84, r'$\omega = \omega_{fe}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.17, 0.15, r'$\omega = \omega_{eg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.locator_params(axis='both', nbins=5)



ax2 = fig.add_subplot(gs[34:42, 38*5:65*5+2])
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_t = Entangled_temporal(OmegaP, OmegaM, t02=t02, delta_f=delta_f_Ent, marginal=True)
ax2.plot(T_Ent, P_t, 'darkred', linewidth=2)
# ax2.set_ylabel(r'$p(t)$', fontsize=12)
ax2.tick_params(labelbottom=False)
ax2.text(0.03, 0.97, '(d)', transform=ax2.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax2.set_yticks(np.array([ 0, 0.2, 0.4]))


ax3 = fig.add_subplot(gs[42:62, 38*5:68*5], sharex=ax2)
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_t_marg = Entangled_temporal(OmegaP, OmegaM, t02=t02, delta_f=delta_f_Ent, marginal=False)
im1 = ax3.imshow(np.real(P_t_marg), origin='lower',
                    extent=[T_Ent.min(), T_Ent.max(), T_Ent.min(), T_Ent.max()],
                    aspect='auto', cmap='hot_r')

Z = np.real(P_t_marg)
levels = np.percentile(Z, [ 50])
cs = ax3.contour(T_Ent, T_Ent, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    (-1.2,  2.0),   # upper
    (1.0,  1.0),   # middle
    # (1.0,  2.0),   # middle
    ( 1.2, -2.0),   # lower
]
ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)

ax3.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)

ax3.locator_params(axis='both', nbins=5)



############################# Third variable set #############################
W_Ent = np.linspace(-10, 10, N_W)
T_Ent = np.linspace(-2.2, 2.2, N_T)
Gamma_e = 10
delta_a = 2.5

row = df_Ent[(df_Ent["Gamma_e"] == Gamma_e) & (df_Ent["delta_a"] == delta_a)]
OmegaP, OmegaM, t02, delta_f_Ent = float(row['OmegaP']), float(row['OmegaM']), float(row['Mu']) , float(row['delta_f'])


ax0 = fig.add_subplot(gs[0:8, 76*5:103*5+2])
ax0.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_freq = Entangled_frequency(OmegaP, OmegaM, t02 = t02,delta_f=delta_f_Ent, marginal=True)
ax0.plot(W_Ent, P_freq, 'darkred', linewidth=2)
# ax0.set_ylabel(r'$p(\omega)$', fontsize=14)
ax0.tick_params(labelbottom=False)
ax0.grid(True, alpha=0.57)
ax0.text(0.03, 0.97, '(e)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax0.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.set_yticks(np.array([ 0, 0.05, 0.1]))


ax1 = fig.add_subplot(gs[8:28, 76*5:106*5],sharex=ax0)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_freq_marg = Entangled_frequency(OmegaP, OmegaM, t02 = t02,delta_f=delta_f_Ent, marginal=False)
im1 = ax1.imshow(np.real(P_freq_marg), origin='lower',
                    extent=[W_Ent.min(), W_Ent.max(), W_Ent.min(), W_Ent.max()],
                    aspect='auto', cmap='hot_r')

Z = np.real(P_freq_marg)
levels = np.percentile(Z, [ 90])
cs = ax1.contour(W_Ent, W_Ent, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
        # (-1.2,  2.0),   # upper
        # (1.0,  1.0),   # upper
    ( 5, -4.0),   # lower
]
ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)


ax1.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
cbar0 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar0.set_label(r'$p(\omega_1, \omega_2)$',  labelpad=5, loc='center', fontsize=14)
ax1.plot([W_Ent.min(), W_Ent.max()], [W_Ent.max(), W_Ent.min()], color='black', linestyle='--', linewidth=1)
ax1.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.text(0.58, 0.8, r'$\omega = \omega_{fe}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.35, 0.2, r'$\omega = \omega_{eg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.15, 0.8, r'$\omega_1 + \omega_2 = \omega_{fg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=-45, color='black', weight='bold')
ax1.locator_params(axis='both', nbins=5)
ax1.set_yticks(np.array([-8, -4, 0, 4, 8]))



ax2 = fig.add_subplot(gs[34:42, 76*5:103*5+2])
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_t = Entangled_temporal(OmegaP, OmegaM, t02=t02, delta_f=delta_f_Ent, marginal=True)
ax2.plot(T_Ent, P_t, 'darkred', linewidth=2)
# ax2.set_ylabel(r'$p(t)$', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelbottom=False)
ax2.text(0.03, 0.97, '(f)', transform=ax2.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax2.set_yticks(np.array([ 0, 0.25, 0.5]))


ax3 = fig.add_subplot(gs[42:62, 76*5:106*5], sharex=ax2)
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_t_marg = Entangled_temporal(OmegaP, OmegaM, t02=t02, delta_f=delta_f_Ent, marginal=False)
im1 = ax3.imshow(np.real(P_t_marg), origin='lower',
                    extent=[T_Ent.min(), T_Ent.max(), T_Ent.min(), T_Ent.max()],
                    aspect='auto', cmap='hot_r')

Z = np.real(P_t_marg)
levels = np.percentile(Z, [ 89.5])
cs = ax3.contour(T_Ent, T_Ent, Z,
                 levels=levels,
                 colors='#000000',
                 linewidths=0.7)
manual_positions = [
    (1.,  2.0),   # upper
    # (1.0,  1.0),   # upper
    # ( 1.2, -2.0),   # lower
]
ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)
# ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f')

# axs[3].set_title(f"Temporal marginal ($\Gamma_e$={Gamma_e}, $\delta_a$={delta_a})")
ax3.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
# ax3 .set_ylabel(r"$t_2\Gamma_f$", fontsize=14)
cbar1 = plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
cbar1.set_label(r'$p(t_1, t_2)$',  labelpad=5, loc='center', fontsize=14)

ax3.locator_params(axis='both', nbins=5)
ax3.set_yticks(np.array([-2, -1, 0, 1, 2]))



plt.tight_layout()
plt.savefig('Plots/Ent_PDFinTimeFreq.png', dpi=900)

# plt.show()
