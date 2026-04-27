import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
# os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')

fig = plt.figure(figsize=(9, 3))
gs = GridSpec(50, 109, figure=fig, hspace=0.0, wspace=0.1)





#################################################### Gamma_e = 0.1 ####################################################
df1 = pd.read_csv('DataSets/Dynamics/Optimal_dynamics_G0.1.csv')
df2 = pd.read_csv('DataSets/Dynamics/Entangled_dynamics_G0.1.csv')
df3 = pd.read_csv('DataSets/Dynamics/Unentangled_dynamics_G0.1.csv')


ax00 = fig.add_subplot(gs[0:20, 0:30])
mask1 = df1['t'] < 0
mask2 = df1['t'] > 0
t_max= df2[df2['P'] == df2['P'].max()]['t'].values[0]
df2['t'] -= t_max
t_max= df3[df3['P'] == df3['P'].max()]['t'].values[0]
df3['t'] -= t_max
line1, = ax00.plot(df1[mask1]['t'], df1[mask1]['Marginal'],linestyle = '--',linewidth=1.2, label=r'Optimal')
ax00.plot(df1[mask2]['t'], df1[mask2]['Marginal'],linestyle = '--',linewidth=1.2,color=line1.get_color())
ax00.plot(df2['t'], df2['Marginal'],linestyle = ':', linewidth=1.2, label=r'Entangled')
ax00.plot(df3['t'], df3['Marginal'],linestyle = '-.', linewidth=1.2, label=r'Unentangled')

# ax00.set_ylim(bottom= 0)
# ax00.set_ylim(0,0.58)

ax00.set_ylabel(r'$p(t)$')
ax00.text(0.07, 0.9, '(a)', transform=ax00.transAxes, 
            fontsize=10, verticalalignment='top', color='black', weight='bold')
ax00.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax00.tick_params(axis='x', labelbottom=False)


ax01 = fig.add_subplot(gs[20:50, 0:30],sharex=ax00)
ax01.plot(df1['t'], df1['P'],linestyle = '--', linewidth=1.2, label=r'Optimal')
ax01.plot(df2['t'], df2['P'],linestyle = ':', linewidth=1.2, label=r'Entangled')
ax01.plot(df3['t'], df3['P'],linestyle = '-.', linewidth=1.2, label=r'Unentangled')
ax01.set_xlabel(r'$t\Gamma_f$')
ax01.set_ylabel(r'$P_{f}(t)$')
ax01.grid(True, alpha=1, color='0.7',linewidth=0.10)
# ax01.set_ylim(0,1.1)
ax00.set_xlim(-17, 7)
ax01.set_xlim(-17, 7)

#################################################### Gamma_e = 1 ####################################################
df1 = pd.read_csv('DataSets/Dynamics/Optimal_dynamics_G1.csv')
df2 = pd.read_csv('DataSets/Dynamics/Entangled_dynamics_G1.csv')
df3 = pd.read_csv('DataSets/Dynamics/Unentangled_dynamics_G1.csv')


ax10 = fig.add_subplot(gs[0:20, 31:61],sharey=ax00)
mask1 = df1['t'] < 0
mask2 = df1['t'] > 0
t_max= df2[df2['P'] == df2['P'].max()]['t'].values[0]
df2['t'] -= t_max
t_max= df3[df3['P'] == df3['P'].max()]['t'].values[0]
df3['t'] -= t_max
line1, = ax10.plot(df1[mask1]['t'], df1[mask1]['Marginal'],linestyle = '--',linewidth=1.2, label=r'Optimal')
ax10.plot(df1[mask2]['t'], df1[mask2]['Marginal'],linestyle = '--',linewidth=1.2,color=line1.get_color())
ax10.plot(df2['t'], df2['Marginal'],linestyle = ':', linewidth=1.2, label=r'Entangled')
ax10.plot(df3['t'], df3['Marginal'],linestyle = '-.', linewidth=1.2, label=r'Unentangled')

# ax10.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
# ax10.set_ylim(bottom= 0)
# ax10.set_ylim(0,0.59)
# ax10.set_ylabel(r'$p$')

ax10.text(0.07, 0.9, '(b)', transform=ax10.transAxes, 
            fontsize=10, verticalalignment='top', color='black', weight='bold')
ax10.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax10.tick_params(axis='x', labelbottom=False)
ax10.tick_params(axis='y', labelleft=False)

ax11 = fig.add_subplot(gs[20:50, 31:61],sharex=ax10, sharey=ax01)
ax11.plot(df1['t'], df1['P'],linestyle = '--', linewidth=1.2, label=r'Optimal')
ax11.plot(df2['t'], df2['P'],linestyle = ':', linewidth=1.2, label=r'Entangled')
ax11.plot(df3['t'], df3['P'],linestyle = '-.', linewidth=1.2, label=r'Unentangled')
ax11.set_xlabel(r'$t\Gamma_f$')
# ax11.set_ylabel(r'$P_{f}$')
ax11.grid(True, alpha=1, color='0.7',linewidth=0.10)
# ax11.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
# ax11.set_ylim(0,1.1)
ax11.tick_params(axis='y', labelleft=False)

ax10.set_xlim(-6, 5)
ax11.set_xlim(-6, 5)

#################################################### Gamma_e = 10 ####################################################
df1 = pd.read_csv('DataSets/Dynamics/Optimal_dynamics_G10.csv')
df2 = pd.read_csv('DataSets/Dynamics/Entangled_dynamics_G10.csv')
df3 = pd.read_csv('DataSets/Dynamics/Unentangled_dynamics_G10.csv')


ax20 = fig.add_subplot(gs[0:20, 62:92],sharey=ax00)
mask1 = df1['t'] < 0
mask2 = df1['t'] > 0
t_max= df2[df2['P'] == df2['P'].max()]['t'].values[0]
df2['t'] -= t_max
t_max= df3[df3['P'] == df3['P'].max()]['t'].values[0]
df3['t'] -= t_max
line1, = ax20.plot(df1[mask1]['t'], df1[mask1]['Marginal'],linestyle = '--',linewidth=1.2, label=r'Optimal')
ax20.plot(df1[mask2]['t'], df1[mask2]['Marginal'],linestyle = '--',linewidth=1.2,color=line1.get_color())
ax20.plot(df2['t'], df2['Marginal'],linestyle = ':', linewidth=1.2, label=r'Entangled')
ax20.plot(df3['t'], df3['Marginal'],linestyle = '-.', linewidth=1.2, label=r'Unentangled')

# ax0.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
# ax20.set_ylim(bottom= 0)
# ax20.set_ylim(0,1.5)
# ax20.set_ylabel(r'$p$')
ax20.text(0.07, 0.9, '(c)', transform=ax20.transAxes, 
            fontsize=10, verticalalignment='top', color='black', weight='bold')
ax20.grid(True, alpha=1, color='0.7',linewidth=0.10)
ax20.tick_params(axis='x', labelbottom=False)
ax20.tick_params(axis='y', labelleft=False)

ax21 = fig.add_subplot(gs[20:50, 62:92],sharex=ax20, sharey=ax01)
ax21.plot(df1['t'], df1['P'],linestyle = '--', linewidth=1.2, label=r'Optimal')
ax21.plot(df2['t'], df2['P'],linestyle = ':', linewidth=1.2, label=r'Entangled')
ax21.plot(df3['t'], df3['P'],linestyle = '-.', linewidth=1.2, label=r'Unentangled')
ax21.set_xlabel(r'$t\Gamma_f$')
# ax21.set_ylabel(r'$P_{f}$')
ax21.grid(True, alpha=1, color='0.7',linewidth=0.10)
# ax21.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
# ax21.set_ylim(0,1.1)
ax21.tick_params(axis='y', labelleft=False)

ax20.set_xlim(-5, 5)
ax21.set_xlim(-5, 5)
# ax0.legend(loc='upper right',frameon=False  )


plt.tight_layout()
plt.savefig('Plots/dynamics_allCases', dpi=300)

