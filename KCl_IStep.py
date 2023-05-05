# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:54:32 2022

@author: Angel.BAUDON
"""

import pyabf, matplotlib.pyplot as plt, glob, os, numpy as np, pandas as pd, seaborn as sns, shutil
from scipy.signal import find_peaks, savgol_filter
from collections import Counter
from ToolKit.IntraGrpStat import IntraGrpStat
import scipy.stats as stat


folder = r"C:\Angel.BAUDON\Exp\Patch\Data\2022_Puff KCl\IStep"
sub_folders = [x for x in glob.glob(rf'{folder}\*') if x.split('\\')[-1]!='analysis']
if not os.path.exists(rf'{folder}\analysis'): os.makedirs(rf'{folder}\analysis')
show_fig, save_fig = False, True

Output, all_profiles = {'Latencies':[], 'Number of spike':[], 'Thresholds':[]}, {}
for sub_folder in sub_folders:
    droj = sub_folder.split("\\")[-1]
    print('\n'*2, '='*30, '\n', droj, '\n', '='*30, '\n'*2)
    cell_folders = [x for x in glob.glob(rf'{sub_folder}\*') if x.split('\\')[-1]!='analysis']
    
    sub_folder_analysis = rf'{sub_folder}\analysis'
    if not os.path.exists(sub_folder_analysis): os.makedirs(sub_folder_analysis)

    trt_latencies, trt_n_spikes, trt_thresholds, trt_profiles = [], [], [], []
    for cell_folder in cell_folders:
        print('-'*30, '\n', cell_folder.split('\\')[-1], '\n', '-'*30, '\n'*2)
        
        cell_folder_analysis = rf'{cell_folder}\analysis'
        if not os.path.exists(cell_folder_analysis): os.makedirs(cell_folder_analysis)

        latencies, n_spikes, thresholds, profiles = [], [], [], []
        for f, file in enumerate(glob.glob(rf'{cell_folder}\*.abf')):
            file_name = file.split('\\')[-1].split('.')[0]
            abf = pyabf.ABF(file)
            
            sweep_indexes, sweep_thresholds = [], []
            plt.figure(), plt.title(f'{file_name} rec n°{f}')
            for sweepNumber in abf.sweepList:
                abf.setSweep(sweepNumber)
                plt.plot(abf.sweepX, abf.sweepY)
                sweep = abf.sweepY - savgol_filter(abf.sweepY, 1001, 1)
                indx, _ = find_peaks(sweep, height=10, prominence=10)
                if len(indx): sweep_thresholds.append(abf.sweepY[indx[0]-50])
                sweep_indexes.append(indx)
            if save_fig: plt.savefig(rf'{cell_folder_analysis}\{file_name}.pdf')
            if not show_fig: plt.close()
            
            active_sweep = [x for x in sweep_indexes if len(x)]
            latency = active_sweep[0][0]
            n_spike = len(active_sweep[1]) if len(active_sweep)>1 else len(active_sweep[0])
            
            border = 0.1*abf.dataRate
            profile = 'ES' if border > active_sweep[0][0] else 'LS'
            
            adaptborder = border > max([a for b in active_sweep for a in b])
            singlespike = 1 == max([len(x) for x in active_sweep])
            print(adaptborder, singlespike)
            if adaptborder or singlespike:
                profile += ' A'
            else: profile += ' NA'
            # profile += ' A' if border > max([a for b in active_sweep for a in b]) else ' NA'
            print(profile)
            latencies.append(latency), n_spikes.append(n_spike)
            thresholds.append(sweep_thresholds[0]), profiles.append(profile)
        trt_latencies.append(np.mean(latencies)), trt_n_spikes.append(np.mean(n_spikes))
        trt_thresholds.append(np.mean(thresholds))
        
        result = all(x == profiles[0] for x in profiles)
        
        if result: trt_profiles.append(profiles[0])
        else:
            w = [x[:2] for x in profiles]
            profile = max(set(w), key = w.count)
            profile += ' NA' if True in ['NA' in p for p in profiles] else ' A'
            trt_profiles.append(profile)
        
        new_name = cell_folder_analysis + profile
        if os.path.exists(new_name): shutil.rmtree(new_name)
        os.rename(cell_folder_analysis, new_name)

    for k, data in zip(Output.keys(), (trt_latencies, trt_n_spikes, trt_thresholds)): Output[k].append(data)
    all_profiles[droj] = trt_profiles
    c = Counter(trt_profiles)

    plt.figure(), plt.title(rf'{droj}    {len(trt_profiles)} cells')
    plt.pie(c.values(), labels = [rf'{k} ({c[k]} cells)' for k in c.keys()],
            autopct= '%1.1f%%')
    if save_fig: plt.savefig(rf'{sub_folder_analysis}\Profiles {droj}.pdf')
    if not show_fig: plt.close()

for k in Output.keys():
    #Create a DataFrame to plot it
    TRT = [x.split('\\')[-1] for x in sub_folders]
    Trt = [a for b in [(T,)*len(O) for T, O in zip(TRT, Output[k])] for a in b]
    df = pd.DataFrame({'Treatment': Trt, k: [a for b in Output[k] for a in b]})
    
    #Plot all that shit
    plt.figure(figsize=(5, 7))
    sns.barplot(data=df,  x='Treatment', y=k, color='lightgreen', capsize=.2, errcolor='k')
    sns.swarmplot(x='Treatment', y=k, data=df, size=5, color='w',
                  edgecolor='k', linewidth=1)
    for trt in set(df['Treatment']):
        t = df[df['Treatment'] == trt][k]
        mean, sem = np.nanmean(t), stat.sem(t)
        print(k, trt, '\n', mean, sem, '\n\n')
    df_stat = IntraGrpStat(df)
    
    if save_fig: plt.savefig(rf'{folder}\analysis\{k}.pdf')
    if not show_fig: plt.close()
    
    
    writer = pd.ExcelWriter(f'{folder}/analysis/{k}.xlsx')
    df.to_excel(writer, sheet_name = k)
    df_stat.to_excel(writer, sheet_name = 'Stats')
    writer.save()

plt.figure(figsize=(6, 9))
for i, cond in enumerate(all_profiles.keys()):
    c = Counter(all_profiles[cond])
    fig1 = plt.subplot(len(all_profiles.keys()),2,i*2+1)
    plt.pie(c.values(), labels = [rf'{k} ({c[k]} cells)' for k in c.keys()],
            autopct= '%1.1f%%')   
    
    fig2 = plt.subplot(len(all_profiles.keys()),2,i*2+2)
    fig2.text(0,0,rf'{cond}    {len(all_profiles[cond])} cells')
    fig2.axis('off')
    

if save_fig: plt.savefig(rf'{folder}\analysis\Profiles.pdf')
if not show_fig: plt.close()
