import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import os

os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')



############################### Entangled, delta_f nonZero ######################################
df = pd.read_csv('./DataSets/HeatMap_deltas_Entangled_No-Delay.csv')

df = df[df['P_max'] != 'P_max'].astype('float')
df.sort_values(by=['delta_a','Gamma_e'], ascending=True, inplace=True)
df = df[df['delta_a'] <20.01]
df.reset_index(drop=True, inplace=True)
delta_a = df['delta_a'].values
Gamma_e = df['Gamma_e'].values
P = df['P_max'].values
delta_a_unique = np.unique(delta_a)
Gamma_e = np.unique(Gamma_e)


fig = plt.figure(figsize=(3.7, 7))
gs = GridSpec(100, 1, figure=fig, hspace=0.0, wspace=0.1)

ax1 = fig.add_subplot(gs[0:32,0])
param1 = df['OmegaP'].values
param1 = param1.reshape( len(delta_a_unique), len(Gamma_e))
contour1 = ax1.contourf(Gamma_e, delta_a_unique, param1, origin='lower',
                    levels=np.linspace(0, param1.max()*1.1, 100), cmap='hot_r')
ax1.set_xscale('log')
ax1.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax1.set_yticks(np.array([0, 5, 10, 15, 20]))
ax1.tick_params(axis='x', which='both', labelbottom=False)
cbar1 = fig.colorbar(contour1, ax=ax1, orientation="vertical")
cbar1.set_label(r'$\Omega_+/\Gamma_f$', labelpad=1, fontsize=10)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
# cbar1.set_ticks(np.arange(0,param1.max(),0.2))
# cbar1.set_ticklabels(np.round(np.arange(0,param1.max(),0.2),2))
cbar1.set_ticks(np.array([0, 0.3, 0.600,  0.9 ]))
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', color='black', weight='bold')


ax2 = fig.add_subplot(gs[33:65,0],sharex=ax1)
param2 = (df['OmegaM'].values / (2*df['Gamma_e'].values + 1))
param2 = param2.reshape( len(delta_a_unique), len(Gamma_e))
contour2 = ax2.contourf(Gamma_e, delta_a_unique, param2, origin='lower',
                    levels=np.linspace(0, param2.max()*1.1, 100), cmap='hot_r')
ax2.set_xscale('log')
ax2.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax2.set_yticks(np.array([0, 5, 10, 15, ]))
cbar2 = fig.colorbar(contour2, ax=ax2, orientation="vertical")
cbar2.set_label(r'$\Omega_-/(2\Gamma_e+\Gamma_f)$', labelpad=1, fontsize=10)
# cbar2.set_ticks(np.arange(0,param2.max(),0.2))
# cbar2.set_ticklabels(np.round(np.arange(0,param2.max()*1.1,0.2),2))
cbar2.set_ticks(np.array([0, 0.2, 0.400,  0.6 ]))
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes,
            fontsize=12, verticalalignment='top', color='black', weight='bold')
ax2.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)
ax2.tick_params(axis='x', which='both', labelbottom=False)




ax3 = fig.add_subplot(gs[66:98,0],sharex=ax1)
param3 = -(df['delta_f'].values - df['delta_a'].values)
param3 = param3.reshape( len(delta_a_unique), len(Gamma_e))
contour3 = ax3.contourf(Gamma_e, delta_a_unique, param3, origin='lower',
                    levels=np.linspace(param3.min(), param3.max()*1, 100), 
                    cmap='hot_r')
ax3.set_xscale('log')
ax3.set_ylabel(r'$ \delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax3.set_yticks(np.array([0, 5, 10, 15, ]))
cbar3 = fig.colorbar(contour3, ax=ax3, orientation="vertical")
cbar3.set_label(r'$-(\delta_f - \delta_a)/\Gamma_f$', labelpad=1, fontsize=10)
# cbar3.set_ticks(np.arange(param3.min(),param3.max(),5))
# cbar3.set_ticklabels(np.round(np.arange(param3.min(),param3.max()*1,5),2))
ax3.text(0.03, 0.97, '(c)', transform=ax3.transAxes,
            fontsize=12, verticalalignment='top', color='black', weight='bold')
ax3.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)
cbar3.set_ticks(np.array([0, 5, 10, 15,   ]))
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)


plt.savefig('./Plots/Ent_Parameters_delta_Gamma.png', dpi=300)


############################### Entangled, delta_f=0  ######################################
df = pd.read_csv('./DataSets/HeatMap_deltas_Entangled_No-Delay_deltaF0.csv')

df = df[df['P_max'] != 'P_max'].astype('float')
df.sort_values(by=['delta_a','Gamma_e'], ascending=True, inplace=True)
df = df[df['delta_a'] <20.01]
df.reset_index(drop=True, inplace=True)
delta_a = df['delta_a'].values
Gamma_e = df['Gamma_e'].values
P = df['P_max'].values
delta_a_unique = np.unique(delta_a)
Gamma_e = np.unique(Gamma_e)


fig = plt.figure(figsize=(3, 6))
gs = GridSpec(100, 1, figure=fig, hspace=0.0, wspace=0.1)

ax1 = fig.add_subplot(gs[0:32,0])
param1 = df['OmegaP'].values
param1 = param1.reshape( len(delta_a_unique), len(Gamma_e))
contour1 = ax1.contourf(Gamma_e, delta_a_unique, param1, origin='lower',
                    levels=np.linspace(0, param1.max()*1.1, 100), cmap='hot_r')
ax1.set_xscale('log')
ax1.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax1.set_yticks(np.array([0, 5, 10, 15, 20]))
ax1.tick_params(axis='x', which='both', labelbottom=False)
cbar1 = fig.colorbar(contour1, ax=ax1, orientation="vertical")
cbar1.set_label(r'$\Omega_+/\Gamma_f$', labelpad=1, fontsize=10)
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
# cbar1.set_ticks(np.arange(0,param1.max(),0.2))
# cbar1.set_ticklabels(np.round(np.arange(0,param1.max(),0.2),2))
cbar1.set_ticks(np.array([0, 0.3, 0.600,  0.9 ]))
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', color='white', weight='bold')


ax2 = fig.add_subplot(gs[33:65,0],sharex=ax1)
param2 = (df['OmegaM'].values / (2*df['Gamma_e'].values + 1))
param2 = param2.reshape( len(delta_a_unique), len(Gamma_e))
contour2 = ax2.contourf(Gamma_e, delta_a_unique, param2, origin='lower',
                    levels=np.linspace(0, param2.max()*1.1, 100), cmap='hot_r')
ax2.set_xscale('log')
ax2.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax2.set_yticks(np.array([0, 5, 10, 15, ]))
cbar2 = fig.colorbar(contour2, ax=ax2, orientation="vertical")
cbar2.set_label(r'$\Omega_-/(2\Gamma_e+\Gamma_f)$', labelpad=1, fontsize=10)
# cbar2.set_ticks(np.arange(0,param2.max(),0.2))
# cbar2.set_ticklabels(np.round(np.arange(0,param2.max()*1.1,0.2),2))
cbar2.set_ticks(np.array([0, 5, 10,  15]))
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes,
            fontsize=12, verticalalignment='top', color='w', weight='bold')
ax2.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)



plt.savefig('./Plots/Ent_Parameters_delta_Gamma_deltaF0.png', dpi=300)


############################### UnEntangled, Delay ######################################
df = pd.read_csv('./DataSets/HeatMap_deltas_UnEnt_Delay.csv')

df = df[df['P_max'] != 'P_max'].astype('float')
df.sort_values(by=['delta_a','Gamma_e'], ascending=True, inplace=True)
df = df[df['delta_a'] <20.01]
df.reset_index(drop=True, inplace=True)

delta_a = df['delta_a'].values
Gamma_e = df['Gamma_e'].values
P = df['P_max'].values
delta_a_unique = np.unique(delta_a)
Gamma_e = np.unique(Gamma_e)


fig = plt.figure(figsize=(6.5, 3.5))
gs = GridSpec(100, 100, figure=fig, hspace=0.0, wspace=0.1)

ax1 = fig.add_subplot(gs[0:49,0:47])
param1 = df['OmegaA'].values/df['Gamma_e'].values
param1 = param1.reshape( len(delta_a_unique), len(Gamma_e))
contour1 = ax1.contourf(Gamma_e, delta_a_unique, param1, origin='lower',
                    levels=np.linspace(0, param1.max()*1.1, 100), cmap='hot_r')
ax1.set_xscale('log')
ax1.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax1.set_yticks(np.array([0, 5, 10, 15, ]))
ax1.tick_params(axis='x', which='both', labelbottom=False)
cbar1 = fig.colorbar(contour1, ax=ax1, orientation="vertical")
cbar1.set_label(r'$\Omega_a/\Gamma_e$', labelpad=1, fontsize=10)
cbar1.set_ticks(np.arange(0,param1.max(),0.5))
cbar1.set_ticklabels(np.round(np.arange(0,param1.max(),0.5),2))
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', color='white', weight='bold')
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)


ax2 = fig.add_subplot(gs[51:100,0:47],sharex=ax1)
param2 = (df['OmegaB'].values / (df['Gamma_e'].values + 1))
param2 = param2.reshape( len(delta_a_unique), len(Gamma_e))
contour2 = ax2.contourf(Gamma_e, delta_a_unique, param2, origin='lower',
                    levels=np.linspace(0, param2.max()*1.1, 100), cmap='hot_r')
ax2.set_xscale('log')
ax2.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax2.set_yticks(np.array([0, 5, 10, 15, ]))
# ax2.tick_params(axis='x', which='both', labelbottom=False)
cbar2 = fig.colorbar(contour2, ax=ax2, orientation="vertical")
cbar2.set_label(r'$\Omega_b/(\Gamma_e+\Gamma_f)$', labelpad=1, fontsize=10)
cbar2.set_ticks(np.arange(0,param2.max(),0.5))
cbar2.set_ticklabels(np.round(np.arange(0,param2.max(),0.5),2))
ax2.text(0.03, 0.97, '(c)', transform=ax2.transAxes,
            fontsize=12, verticalalignment='top', color='white', weight='bold')
ax2.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)


ax3 = fig.add_subplot(gs[0:49,53:100],sharex=ax1)

param3 = (df['Mu'].values*df['Gamma_e'].values)
param3 = param3.reshape( len(delta_a_unique), len(Gamma_e))
contour3 = ax3.contourf(Gamma_e, delta_a_unique, param3, origin='lower',
                    levels=np.linspace(param3.min(), param3.max(), 500), cmap='hot_r')
ax3.set_xscale('log')
# ax3.set_ylabel(r'$|\delta_a|/\Gamma_f$', labelpad=5, fontsize=10)
ax3.set_yticks(np.array([0, 5, 10, 15, ]))
cbar3 = fig.colorbar(contour3, ax=ax3, orientation="vertical")
cbar3.set_label(r'$\mu\Gamma_e$', labelpad=1, fontsize=10)
cbar3.set_ticks(np.array([0., 0.4, 0.8, 1.2, ]))

ax3.text(0.03, 0.97, '(b)', transform=ax3.transAxes,
            fontsize=12, verticalalignment='top', color='white', weight='bold')
# ax3.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)
ax3.tick_params(axis='x', which='both', labelbottom=False)
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)



ax4 = fig.add_subplot(gs[51:100,53:100],sharex=ax1)

param4 = (df['delta_f'].values - df['delta_a'].values)
param4 = param4.reshape( len(delta_a_unique), len(Gamma_e))
contour4 = ax4.contourf(Gamma_e, delta_a_unique, param4, origin='lower',
                    levels=np.linspace(param4.min(), param4.max(), 500), cmap='hot_r')
ax4.set_xscale('log')
ax4.set_yticks(np.array([0, 5, 10, 15, ]))
# ax4.set_ylabel(r'$|\delta_a|/\Gamma_f$', labelpad=5, fontsize=10)
cbar4 = fig.colorbar(contour4, ax=ax4, orientation="vertical")
cbar4.set_label(r'$(\delta_f-\delta_a)/\Gamma_f$', labelpad=1, fontsize=10)
# cbar4.set_ticks(np.array([-7, 0,7,]))
cbar4.set_ticks(np.array([-6,-3,  0,3,6, ]))
# cbar3.set_ticklabels(np.round(np.arange(param3.min(),param3.max(),2),0))

ax4.text(0.03, 0.97, '(d)', transform=ax4.transAxes,
            fontsize=12, verticalalignment='top', color='white', weight='bold')
ax4.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)
ax4.grid(True, alpha=1, color='0.7',linewidth=0.10)



plt.savefig('./Plots/UnEnt_Parameters_delta_Gamma_Delay.png', dpi=300)



############################### UnEntangled, deltaF0 ######################################
df = pd.read_csv('./DataSets/HeatMap_deltas_UnEnt_Delay_deltaF0.csv')

df = df[df['P_max'] != 'P_max'].astype('float')
df.sort_values(by=['delta_a','Gamma_e'], ascending=True, inplace=True)
df = df[df['delta_a'] <5.1]
df.reset_index(drop=True, inplace=True)

# delta_a = df['delta_a'].values
delta_a = df['delta_a'].values
Gamma_e = df['Gamma_e'].values
P = df['P_max'].values
delta_a_unique = np.unique(delta_a)
Gamma_e = np.unique(Gamma_e)


fig = plt.figure(figsize=(3.5, 7))
gs = GridSpec(100, 1, figure=fig, hspace=0.0, wspace=0.1)

ax1 = fig.add_subplot(gs[0:32,0])
param1 = df['OmegaA'].values/df['Gamma_e'].values
param1 = param1.reshape( len(delta_a_unique), len(Gamma_e))
contour1 = ax1.contourf(Gamma_e, delta_a_unique, param1, origin='lower',
                    levels=np.linspace(0, param1.max()*1, 100), cmap='hot_r')
ax1.set_xscale('log')
ax1.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax1.tick_params(axis='x', which='both', labelbottom=False)
cbar1 = fig.colorbar(contour1, ax=ax1, orientation="vertical")
cbar1.set_label(r'$\Omega_a/\Gamma_e$', labelpad=1, fontsize=10)
cbar1.set_ticks(np.array([0,50,  150,  250,  ]))
ax1.set_yticks(np.array([0, 2, 4 ]))
ax1.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', color='white', weight='bold')


ax2 = fig.add_subplot(gs[33:65,0],sharex=ax1)
param2 = (df['OmegaB'].values / (df['Gamma_e'].values + 1))
param2 = param2.reshape( len(delta_a_unique), len(Gamma_e))
contour2 = ax2.contourf(Gamma_e, delta_a_unique, param2, origin='lower',
                    levels=np.linspace(0, param2.max()*1.1, 100), cmap='hot_r')
ax2.set_xscale('log')
ax2.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax2.set_yticks(np.array([0, 2, 4 ]))
ax2.tick_params(axis='x', which='both', labelbottom=False)
cbar2 = fig.colorbar(contour2, ax=ax2, orientation="vertical")
cbar2.set_label(r'$\Omega_b/(\Gamma_e+\Gamma_f)$', labelpad=1, fontsize=10)
cbar2.set_ticks(np.array([0, 1, 2,3 ]))
ax2.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes,
            fontsize=12, verticalalignment='top', color='white', weight='bold')
ax2.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)



ax3 = fig.add_subplot(gs[66:98,0],sharex=ax1)

param3 = (df['Mu'].values * df['Gamma_e'].values)
param3 = param3.reshape( len(delta_a_unique), len(Gamma_e))
contour3 = ax3.contourf(Gamma_e, delta_a_unique, param3, origin='lower',
                    levels=np.linspace(param3.min(), param3.max(), 500), cmap='hot_r')
ax3.set_xscale('log')
ax3.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax3.set_yticks(np.array([0, 2, 4, ]))
cbar3 = fig.colorbar(contour3, ax=ax3, orientation="vertical")
cbar3.set_label(r'$\mu\Gamma_e$', labelpad=1, fontsize=10)
cbar3.set_ticks(np.array([0, 0.3, 0.6, 0.9,   ]))
ax3.grid(True, alpha=1, color='0.7',linewidth=0.10)

ax3.text(0.03, 0.97, '(c)', transform=ax3.transAxes,
            fontsize=12, verticalalignment='top', color='black', weight='bold')
ax3.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)



plt.savefig('./Plots/UnEnt_Parameters_delta_Gamma_deltaF0.png', dpi=300)





################################# Coherent, No-Delay ######################################
df = pd.read_csv('./DataSets/HeatMap_Coherent_P_Delta.csv')
df = df[df['P_max'] != 'P_max'].astype('float')
df.sort_values(by=['Delta_1','Gamma_e'], ascending=True, inplace=True)
df = df[df['Gamma_e']<100.1]
# df = df[df['Delta_1']<2.51]
df.reset_index(drop=True, inplace=True)
delta_a = df['Delta_1'].values*2
Gamma_e = df['Gamma_e'].values
Omega = df['Omega'].values
P = df['P_max'].values
delta_a_unique = np.unique(delta_a)
Gamma_e = np.unique(Gamma_e)


fig = plt.figure(figsize=(3, 6))
gs = GridSpec(100, 1, figure=fig, hspace=0.0, wspace=0.1)
ax1 = fig.add_subplot(gs[0:32,0])
param1 = df['Omega'].values
param1 = param1.reshape( len(delta_a_unique), len(Gamma_e))
contour1 = ax1.contourf(Gamma_e, delta_a_unique, param1, origin='lower',
                    levels=np.linspace(0, param1.max()*1.1, 100), cmap='hot_r')
ax1.set_xscale('log')
ax1.set_ylabel(r'$\delta_a/\Gamma_f$', labelpad=5, fontsize=10)
ax1.set_yticks(np.array([0, 5,10,15,20]))
# ax1.tick_params(axis='x', which='both', labelbottom=False)
cbar1 = fig.colorbar(contour1, ax=ax1, orientation="vertical")
cbar1.set_label(r'$\Omega/\Gamma_f$', labelpad=1, fontsize=10)
cbar1.set_ticks(np.arange(int(param1.min()),int(param1.max()),4))
cbar1.set_ticks(np.array([0,4,8,12]))
ax1.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)


plt.savefig('./Plots/Coherent_Parameters_delta_Gamma.png', dpi=300)

