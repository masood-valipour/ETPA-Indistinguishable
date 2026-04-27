import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

N_T, N_W = 200, 200


def OptimumState_temporal(Gamma_e, marginal=False):
    T1, T2 = np.meshgrid(T_opt, T_opt)
    tStar = max(T_opt)
    Gamma_f = 1
    def P_sym(t1, t2):
        constant = Gamma_e * Gamma_f * np.exp(-Gamma_f * tStar) / 2
        chi1 = (t1 <= t2) & (t2 <= tStar)
        chi2 = (t2 <= t1) & (t1 <= tStar)
        term1 = np.exp((Gamma_f - Gamma_e) * t2 + Gamma_e * t1) * chi1
        term2 = np.exp((Gamma_f - Gamma_e) * t1 + Gamma_e * t2) * chi2
        return constant * (term1 + term2)
    def P_marg (t):
        p = np.exp((Gamma_f - Gamma_e)*tStar + Gamma_e*t) - np.exp(Gamma_f*t)
        if Gamma_f != Gamma_e:  
            p /= Gamma_f - Gamma_e
        else:
            p /= 1e-10  # avoid division by zero
        p += np.exp(Gamma_f*t)/Gamma_e
        p *= Gamma_e*Gamma_f*np.exp(-Gamma_f*tStar)/2
        return p
    if marginal:
        return P_marg(T_opt)
    else:
        return P_sym(T1, T2)

# === Optimum State Frequency ===
def OptimumState_frequency(Gamma_e, delta_a,marginal=False):
    W1, W2 = np.meshgrid(W_opt, W_opt)
    Gf = 1
    tStar = 0
    w_eg = -0.5 * delta_a
    w_fe = +0.5 * delta_a
    def PHI(w1, w2):
        phi = 1 / (1j * (w1 - w_eg) + Gamma_e / 2)
        phi += 1 / (1j * (w2 - w_eg) + Gamma_e / 2)
        phi *= np.sqrt(Gf * Gamma_e) * np.exp(1j * (w1 + w2 - w_fe - w_eg) * tStar)
        phi /= 4 * np.pi * (1j * (w1 + w2 - w_eg - w_fe) + Gf / 2)
        return phi
    def Analy_marginal(w):
        p = Gamma_e*(Gamma_e + Gf)* (Gamma_e + Gf/4) + Gamma_e*(w_fe - w_eg)**2 + Gf*(w-w_eg)**2
        p /= 4*np.pi*( (w-w_eg)**2 + Gamma_e**2/4) * ((w-w_fe)**2 + (Gamma_e + Gf)**2/4 )
        return p
    if marginal:
        return Analy_marginal(W_opt)
    else:
        return 2 * np.abs(PHI(W2, W1) ** 2)


fig = plt.figure(figsize=(12, 10))
gs = GridSpec(62, 106*5, figure=fig, hspace=0.0, wspace=0.1)
# parameters for examples
# Gamma_e_temp = 0.1


############################# First variable set #############################
# 1) Frequency joint probability (2D)
Gamma_e = 0.1
delta_a = 2.5
W_opt = np.linspace(-2, 3, N_W)


ax0 = fig.add_subplot(gs[0:8, 0:27*5+2])
ax0.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_marg_freq = OptimumState_frequency(Gamma_e, delta_a, marginal=True)
ax0.plot(W_opt, P_marg_freq, 'darkred', linewidth=2)
ax0.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax0.set_ylabel(r'$p(\omega)$', fontsize=12)
ax0.tick_params(labelbottom=False)
ax0.text(0.03, 0.97, '(a)', transform=ax0.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax0.set_yticks(np.array([ 0, 1, 2, 3]))


ax1 = fig.add_subplot(gs[8:28, 0:30*5],sharex=ax0)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_f1 = OptimumState_frequency(Gamma_e, delta_a, marginal=False)
im2 = ax1.imshow(np.real(P_f1), origin='lower',
                    extent=[W_opt.min(), W_opt.max(), W_opt.min(), W_opt.max()],
                    aspect='auto', cmap='hot_r')
Z = np.real(P_f1)
levels = np.percentile(Z, [ 90])
cs = ax1.contour(W_opt, W_opt, Z, levels=levels, colors='#000000', linewidths=0.4)
manual_positions = [( 0.8, -1.0)]
ax1.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)

ax1.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax1.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
ax1.set_ylabel(r"$(\omega_2 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
cbar0 = plt.colorbar(im2, ax=ax1, fraction=0.046, pad=0.04,shrink = 1)
ax1.text(0.68, 0.8, r'$\omega = \omega_{fe}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax1.text(0.18, 0.30, r'$\omega = \omega_{eg}$', transform=ax1.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')

ax1.locator_params(axis='both', nbins=5)
ax1.set_yticks(np.array([-2, -1,  0, 1, 2]))




Gamma_e_temp = 0.1
delta_a = 1.0
T_opt = np.linspace(-2, 0, N_T)


ax2 = fig.add_subplot(gs[34:42, 0:27*5+2])
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_marg_freq = OptimumState_temporal(Gamma_e_temp, marginal=True)
ax2.plot(T_opt, P_marg_freq, 'darkred', linewidth=2)
ax2.set_ylabel(r'$p(t)$', fontsize=12)
ax2.tick_params(labelbottom=False)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')

ax2.set_yticks(np.array([ 0, 0.25, 0.5,]))


ax3 = fig.add_subplot(gs[42:62, 0:30*5], sharex=ax2)
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_tt = OptimumState_temporal(Gamma_e_temp, marginal=False)
im0 = ax3.imshow(np.real(P_tt), origin='lower',
                    extent=[T_opt.min(), T_opt.max(), T_opt.min(), T_opt.max()],
                    aspect='auto', cmap='hot_r')
# Z = np.real(P_tt)
# levels = np.percentile(Z, [ 99])
# cs = ax3.contour(T_opt, T_opt, Z, levels=levels, colors='#000000', linewidths=0.7)
# manual_positions = [( 1, -1.0)]
# ax3.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)

ax3.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
ax3 .set_ylabel(r"$t_2\Gamma_f$", fontsize=14)
plt.colorbar(im0, ax=ax3, fraction=0.046, pad=0.04)

ax3.locator_params(axis='both', nbins=5)
ax3.set_yticks(np.array([ -2,-1.5,-1,-0.5]))


############################# Second variable set #############################
# 1) Frequency joint probability (2D)
Gamma_e = 1
delta_a = 2.5
W_opt = np.linspace(-3.5, 4, N_W)


ax4 = fig.add_subplot(gs[0:8, 38*5:65*5+2])
ax4.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_marg_freq = OptimumState_frequency(Gamma_e, delta_a, marginal=True)
ax4.plot(W_opt, P_marg_freq, 'darkred', linewidth=2)
ax4.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax4.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax4.tick_params(labelbottom=False)
ax4.text(0.03, 0.97, '(c)', transform=ax4.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax4.set_yticks(np.array([ 0, 0.2, 0.4]))

ax5 = fig.add_subplot(gs[8:28, 38*5:68*5],sharex=ax4)
ax5.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_f1 = OptimumState_frequency(Gamma_e, delta_a, marginal=False)
im2 = ax5.imshow(np.real(P_f1), origin='lower',
                    extent=[W_opt.min(), W_opt.max(), W_opt.min(), W_opt.max()],
                    aspect='auto', cmap='hot_r')
Z = np.real(P_f1)
levels = np.percentile(Z, [ 70])
cs = ax5.contour(W_opt, W_opt, Z, levels=levels, colors='#000000', linewidths=0.4)
manual_positions = [( -2, 3.0)]
ax5.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)

ax5.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax5.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax5.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax5.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax5.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
# ax5.set_ylabel("$(\omega_2 - \omega_{eg})/\Gamma_f$", fontsize=14)
cbar0 = plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04,shrink = 1)
ax5.locator_params(axis='both', nbins=5)
ax5.text(0.56, 0.82, r'$\omega = \omega_{fe}$', transform=ax5.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax5.text(0.13, 0.38, r'$\omega = \omega_{eg}$', transform=ax5.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')

ax5.set_yticks(np.array([-2, 0, 2]))


T_opt = np.linspace(-2, 0, N_T)


ax6 = fig.add_subplot(gs[34:42, 38*5:65*5+2])
ax6.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_marg_freq = OptimumState_temporal(Gamma_e, marginal=True)
ax6.plot(T_opt, P_marg_freq, 'darkred', linewidth=2)
ax6.tick_params(labelbottom=False)
ax6.text(0.03, 0.97, '(d)', transform=ax6.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax6.set_yticks(np.array([ 0, 0.25, 0.5,]))



ax7 = fig.add_subplot(gs[42:62, 38*5:68*5],sharex=ax6)
ax7.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_tt = OptimumState_temporal(Gamma_e, marginal=False)
im0 = ax7.imshow(np.real(P_tt), origin='lower',
                    extent=[T_opt.min(), T_opt.max(), T_opt.min(), T_opt.max()],
                    aspect='auto', cmap='hot_r')

ax7.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
# ax7 .set_ylabel(f"$t_2\Gamma_f$", fontsize=14)
plt.colorbar(im0, ax=ax7, fraction=0.046, pad=0.04)

ax7.locator_params(axis='both', nbins=5)
ax7.set_yticks(np.array([ -2,-1.5,-1,-0.5]))


############################# Third variable set #############################
# 1) Frequency joint probability (2D)
Gamma_e = 10
delta_a = 2.5
W_opt = np.linspace(-7, 7, N_W)


ax8 = fig.add_subplot(gs[0:8, 76*5:103*5+2])
ax8.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_marg_freq = OptimumState_frequency(Gamma_e, delta_a, marginal=True)
ax8.plot(W_opt, P_marg_freq, 'darkred', linewidth=2)
ax8.axvline(x=-delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax8.axvline(x=delta_a/2, color='BLACK', linestyle='-.', linewidth=1)
ax8.tick_params(labelbottom=False)

ax8.text(0.03, 0.97, '(e)', transform=ax8.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax8.set_yticks(np.array([ 0, 0.05, 0.1]))

ax9 = fig.add_subplot(gs[8:28, 76*5:106*5],sharex=ax8)
ax9.grid(True, alpha=1, color='0.7',linewidth=0.10)
P_f1 = OptimumState_frequency(Gamma_e, delta_a, marginal=False)
im2 = ax9.imshow(np.real(P_f1), origin='lower',
                    extent=[W_opt.min(), W_opt.max(), W_opt.min(), W_opt.max()],
                    aspect='auto', cmap='hot_r')
Z = np.real(P_f1)
levels = np.percentile(Z, [ 80])
cs = ax9.contour(W_opt, W_opt, Z, levels=levels, colors='#000000', linewidths=0.4)
manual_positions = [( -4, 5.0)]
ax9.clabel(cs, inline=True, fontsize=10, fmt='%.4f', manual=manual_positions)

ax9.plot([W_opt.min(), W_opt.max()], [W_opt.max(), W_opt.min()], color="#AEEBFD", linestyle='--', linewidth=2)
ax9.axvline(x=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax9.axvline(x=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax9.axhline(y=-delta_a/2, color='black', linestyle='-.', linewidth=1)
ax9.axhline(y=delta_a/2, color='black', linestyle='-.', linewidth=1)
ax9.set_xlabel(r"$(\omega_1 - \frac{\omega_{fg}}{2})/\Gamma_f$", fontsize=14)
# ax9.set_ylabel("$(\omega_2 - \omega_{eg})/\Gamma_f$", fontsize=14)
cbar0 = plt.colorbar(im2, ax=ax9, fraction=0.046, pad=0.04,shrink = 1)
cbar0.set_label(r'$p(\omega_1, \omega_2)$',  labelpad=5, loc='center', fontsize=14)
ax9.locator_params(axis='both', nbins=5)
ax9.text(0.62, 0.83, r'$\omega = \omega_{fe}$', transform=ax9.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax9.text(0.32, 0.2, r'$\omega = \omega_{eg}$', transform=ax9.transAxes, 
            fontsize=14, verticalalignment='center', rotation=90, color='black', weight='bold')
ax9.text(0.64, 0.22, r'$\omega_1 + \omega_2 = \omega_{fg}$', transform=ax9.transAxes, 
            fontsize=14, verticalalignment='center', rotation=-45, color='black', weight='bold')


T_opt = np.linspace(-2, 0, N_T)


ax_10 = fig.add_subplot(gs[34:42, 76*5:103*5+2])
ax_10.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_marg_freq = OptimumState_temporal(Gamma_e, marginal=True)
ax_10.plot(T_opt, P_marg_freq, 'darkred', linewidth=2)
# ax_10.set_ylabel(r'$p(t)$', fontsize=12)
ax_10.text(0.03, 0.97, '(f)', transform=ax_10.transAxes, 
            fontsize=16, verticalalignment='top', color='black', weight='bold')
ax_10.tick_params(labelbottom=False)
ax_10.set_yticks(np.array([ 0, 0.25, 0.5, 0.75]))



ax_11 = fig.add_subplot(gs[42:62, 76*5:106*5], sharex=ax_10)
ax_11.grid(True, alpha=1, color='0.7',linewidth=0.10)

P_tt = OptimumState_temporal(Gamma_e, marginal=False)
im0 = ax_11.imshow(np.real(P_tt), origin='lower',
                    extent=[T_opt.min(), T_opt.max(), T_opt.min(), T_opt.max()],
                    aspect='auto', cmap='hot_r')
ax_11.set_xlabel(r"$t_1\Gamma_f$", fontsize=14)
cbar1 = plt.colorbar(im0, ax=ax_11, fraction=0.046, pad=0.04)
cbar1.set_label(r'$p(t_1, t_2)$',  labelpad=5, loc='center', fontsize=14)

ax_11.locator_params(axis='both', nbins=5)
ax_11.set_yticks(np.array([ -2,-1.5,-1,-0.5]))




plt.tight_layout()
plt.savefig('Plots/OptimalState_PDFinTimeFreq.png', dpi=300)
# plt.show()