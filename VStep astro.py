# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:43:30 2023

@author: Angel.BAUDON
"""

import matplotlib.pyplot as plt, numpy as np, glob, pandas as pd, os
import scipy.stats as stat, seaborn as sns
from scipy.optimize import curve_fit
from ToolKit.IntraGrpStat import IntraGrpStat
from ToolKit.Tollbox_TG_shorten import Rawtrace, toto_filter


folder = r"C:\Angel.BAUDON\Exp\Patch\Data\2023_03_PatchAstro\VStep"
sub_folders = [x for x in glob.glob( rf'{folder}\*') if x.split('\\')[-1] != 'analysis']
if not os.path.exists(rf'{folder}\analysis'): os.makedirs(rf'{folder}\analysis')
show_fig, save_fig = False, True
 

for sub_folder in sub_folders:
    drug = sub_folder.split('\\') [-1]
    print('\n'*2, '='*30, '\n', drug, '\n', '='*30, '\n')
    cells = [x for x in glob.glob(rf'{sub_folder}\*') if x != 'analysis']

    Currents = np.empty((len(cells), 4, 11))
    for c, cell in enumerate(cells):
        times, cell_name = glob.glob(rf'{cell}\*.wcp'), cell.split('\\') [-1]
        print('\n', ':'*30, '\n', cell_name, '\n', ':'*30, '\n'*2)
    
        cell_analysis = rf'{cell}\analysis'
        if not os.path.exists(cell_analysis): os.makedirs(cell_analysis)
    
        current_cell = np.empty((4,11))
        time_label = ('_0', '_15')
        for f, tl in enumerate(time_label):
            current = []
            file = [x for x in times if tl in x]
            if not len(file):
                current_cell[f,:] = (np.nan,)*11
                continue
            
            file = file[0]
            file_name = (file.split('\\')[-1]).split('.')[0]
            print(file_name, '\n'*2)
    
            raw = Rawtrace(file)
            y_ax = raw.matrix[:,:int(.2*raw.sampling_rate)]
        
            
            plt.figure(), plt.title(file_name)
            for y in y_ax:
                plt.plot(np.linspace(0, .2, y_ax.shape[1]), y)
                current.append(np.nanmedian(y[30000:40000]) - np.nanmedian(y[:4000]))
            plt.close()
            current_cell[f,:] = current
        Currents[c,:] = current_cell
        
    plt.figure(), plt.title(drug), plt.ylabel('Current injected (pA)')
    x_ax = np.arange(11)
    mean, sem = np.nanmean(Currents, axis=0), stat.sem(Currents, axis=0, nan_policy='omit')

    for tl, m, s in zip(('0', '15'), mean, sem):
        plt.errorbar(x_ax, m, yerr=s, label=tl+' min')
    plt.legend(loc='upper left'), plt.xticks(x_ax, np.linspace(-100, 100, 11, dtype=int))
    plt.xlabel('Holding voltage (mV)')
    if save_fig: plt.savefig(rf'{folder}\analysis\IV curve {drug}.pdf')
    if not show_fig: plt.close()
























