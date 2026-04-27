from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')
scale = 0.81

fig = plt.figure(figsize=(3, 6))
gs = GridSpec(100, 100, figure=fig, hspace=0.0, wspace=0.1)


ax1 = fig.add_subplot(gs[0:30,:])
inputFile = f'P_heatMap_Gamma_Delta_Entangled.csv'
inputFile = './DataSets/' + inputFile
df = pd.read_csv(inputFile)
df = df[df['P_max'] != 'P_max'].astype('float')
df.sort_values(by=['Delta1','Gamma_e'], ascending=True, inplace=True)
df = df[df['Gamma_e']<60]
df = df[df['Delta1']<12.5]
df.reset_index(drop=True, inplace=True)
delta_a = df['Delta1'].values*2
Gamma_e = df['Gamma_e'].values
P = df['P_max'].values
delta_a_unique = np.unique(delta_a)
Gamma_e = np.unique(Gamma_e)
P_grid = P.reshape( len(delta_a_unique), len(Gamma_e))
contour1 = ax1.contourf(Gamma_e, delta_a_unique, P_grid, origin='lower',
                    levels=np.linspace(0, scale, 100), cmap='hot_r')
ax1.set_xscale('log')
ax1.set_ylabel(r'$|\delta_a|/\Gamma_f$', labelpad=5, fontsize=10)


ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', color='black', weight='bold')
Z = P_grid
levels = np.percentile(Z, [61])  
cs = ax1.contour(
    Gamma_e,
    delta_a_unique,
    Z,
    levels=levels,
    colors="black",
    linewidths=0.8
)
ax1.clabel(cs, inline=True, fontsize=10, fmt='%.2f')




ax2 = fig.add_subplot(gs[35:65,:])
inputFile = f'HeatMap_deltas_Entangled_No-Delay.csv'
inputFile = './DataSets/' + inputFile
df = pd.read_csv(inputFile)
df = df[df['P_max'] != 'P_max'].astype('float')
df.sort_values(by=['delta_a','Gamma_e'], ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
delta_a = df['delta_a'].values
Gamma_e = df['Gamma_e'].values
P = df['P_max'].values
delta_a_unique = np.unique(delta_a)
Gamma_e = np.unique(Gamma_e)
P_grid = P.reshape( len(delta_a_unique), len(Gamma_e))
contour2 = ax2.contourf(Gamma_e, delta_a_unique, P_grid, origin='lower',
                    levels=np.linspace(0, scale, 500), cmap='hot_r')
ax2.set_xscale('log')
ax2.set_ylabel(r'$|\delta_a|/\Gamma_f$', labelpad=5, fontsize=10)
ax2.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)
cbar2 = fig.colorbar(contour2, ax=[ax1, ax2], orientation="vertical")
cbar2.set_label(r'$P_f^{max}$', labelpad=1, fontsize=10)
cbar2.set_ticks(np.arange(0,scale,0.2))
cbar2.set_ticklabels(np.round(np.arange(0,scale,0.2),2))
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', color='black', weight='bold')
# choose contour levels (same idea as your imshow code)
Z = P_grid
levels = np.percentile(Z, [42,60])  
cs = ax2.contour(
    Gamma_e,
    delta_a_unique,
    Z,
    levels=levels,
    colors="#021815",
    linewidths=0.8
)
ax2.clabel(cs, inline=True, fontsize=10, fmt='%.2f')


outputFile = './Plots/'+ 'P_heatMap_Gamma_deltaA_Entangled.png'
plt.savefig(outputFile)
# plt.show()
plt.close()

