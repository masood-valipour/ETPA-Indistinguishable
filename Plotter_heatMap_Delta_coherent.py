import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

scale = 0.39

fig = plt.figure(figsize=(4, 2.5))
gs = GridSpec(1, 1, figure=fig, hspace=0.0, wspace=0.1)



ax1 = fig.add_subplot(gs[0,0])
inputFile = 'HeatMap_Coherent_P_Delta.csv' 
inputFile = './DataSets/' + inputFile
df = pd.read_csv(inputFile)
df = df[df['P_max'] != 'P_max'].astype('float')
df.sort_values(by=['Delta_1','Gamma_e'], ascending=True, inplace=True)
df = df[df['Gamma_e']<100]
df.reset_index(drop=True, inplace=True)
delta_a = df['Delta_1'].values*2
Gamma_e = df['Gamma_e'].values
P = df['P_max'].values
delta_a_unique = np.unique(delta_a)
Gamma_e = np.unique(Gamma_e)
P_grid = P.reshape( len(delta_a_unique), len(Gamma_e))
contour1 = ax1.contourf(Gamma_e, delta_a_unique, P_grid, origin='lower',
                    levels=np.linspace(0, scale, 100), cmap='hot_r')
ax1.set_xscale('log')
ax1.set_ylabel(r'$|\delta_a|/\Gamma_f$', labelpad=5, fontsize=10)
ax1.set_yticks(np.linspace(0,20,5,endpoint=True))
ax1.set_xlabel(r'$\Gamma_e/\Gamma_f$', labelpad=5, fontsize=10)
cbar1 = fig.colorbar(contour1, ax=ax1, orientation="vertical")
cbar1.set_label(r'$P_f^{max}$', labelpad=1, fontsize=13)
cbar1.set_ticks(np.arange(0,scale,0.1))
cbar1.set_ticklabels(np.round(np.arange(0,scale,0.1),2))


outputFile = './Plots/P_heatMap_Gamma_deltaA_Coherent.png'
plt.savefig(outputFile)
plt.close()



