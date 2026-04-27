import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import List
import os
os.makedirs('Plots',exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.style.use('physrev.mplstyle')




def plotter (inputFile: List[str], outputFile: str, title: str = '', label: List[str] =[], fillBetween: bool = False):
    r"""
    Plots the maximum probability against the ratio of coupling constants 
    \(\frac{\Gamma_e}{\Gamma_f}\) on a logarithmic scale.

    This function reads data from two input CSV files, plots the maximum transition 
    probability \(P_f^{max}\) against the coupling ratio \(\Gamma_e/\Gamma_f\), and 
    allows for the option to fill the area between the two curves if desired.

    Parameters:
    -----------
    inputFile : List[str]
        A list of two strings representing the paths to the input CSV files containing 
        the data for the plots.
    
    outputFile : str
        The path where the output plot will be saved.
    
    title : str, optional (default='')
        The title of the plot.
    
    label : List[str], optional (default=[])
        A list of labels for the two datasets, used in the plot legend.
    
    fillBetween : bool, optional (default=False)
        If True, fills the area between the two curves where one is greater than the other. 
        (In the case of comparing Entangled and Unentangled cases)

    Returns:
    --------
    None
        The function saves the plot to the specified output file.
    """
    outputFile = './Plots/'+ outputFile
    inputFile = ['./DataSets/'+ inputFile for inputFile in inputFile]
    fig, ax = plt.subplots()
    df1 = pd.read_csv(inputFile[0]).iloc[:-2]
    df2 = pd.read_csv(inputFile[1]).iloc[:-2]
    
    ax.semilogx(df1['Gamma_e'],df1['P_max'], marker = '.', markersize=1, linestyle ='solid', label = label[0])
    ax.semilogx(df2['Gamma_e'],df2['P_max'], marker = '^', markersize=1,linestyle ='dashed', label = label[1])
    # ax.semilogx(df1['Gamma_e'],df1['P_max'],  linewidth=1, linestyle ='solid', label = label[0])
    # ax.semilogx(df2['Gamma_e'],df2['P_max'], linewidth=1,linestyle ='-.', label = label[1])
    if fillBetween:
        indx =  df1[df1['P_max'] > df2['P_max']].index
        indx = np.append(indx[0]-1, indx)
        plt.fill_between(df1['Gamma_e'][indx] , df1['P_max'][indx],df2['P_max'][indx],alpha = 0.2 )
    ax.set_title(title)
    ax.set_xlabel(r'$\Gamma_e/\Gamma_f$')
    ax.set_ylabel(r'$P_f^{max}$')
    # ax.set_ylim(-0.01,0.7)
    ax.legend(frameon=False)
    plt.savefig(outputFile)



# ########################## Gaussian Entangled pairs ############################################
inputFile = ['P_Gamma_Entangled_withDelay_highPrecision.csv',
             'P_Gamma_Entangled_withoutDelay_highPrecision.csv']
outputFile = 'P_Gamma_Entangled.png'
# title = 'Gaussian Entangled'
title = ''
label = [r'$\mu\neq0$', 
         r'$\mu=0$']
plotter(inputFile, outputFile, title = title, label = label)

# ########################## Gaussian UnEntangled pairs ############################################
inputFile = ['P_Gamma_Unentangled_withDelay_highPrecision.csv',
             'P_Gamma_Unentangled_withoutDelay_highPrecision.csv']
outputFile = 'P_Gamma_Unentangled.png'
# title = 'Gaussian Unentangled'
title = ''
label = [r'$\mu\neq0$', 
         r'$\mu=0$']
plotter(inputFile, outputFile, title = title, label = label)

