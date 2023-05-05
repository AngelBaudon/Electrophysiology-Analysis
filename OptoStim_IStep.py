# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:53:02 2023

@author: Angel.BAUDON
"""


import pyabf, matplotlib.pyplot as plt, glob, os, numpy as np, pandas as pd, seaborn as sns, shutil
from scipy.signal import find_peaks, savgol_filter
from collections import Counter
from ToolKit.IntraGrpStat import IntraGrpStat
from ToolKit.Tollbox_TG_shorten import Rawtrace


folder = r"C:\Angel.BAUDON\Exp\Patch\Data\2023_Rec CeL OptoStim BLA\IStep"
sub_folders = [x for x in glob.glob(rf'{folder}\*') if x.split('\\')[-1]!='analysis']
if not os.path.exists(rf'{folder}\analysis'): os.makedirs(rf'{folder}\analysis')
show_fig, save_fig = True, True

Output, all_profiles = {'Latencies':[], 'Number of spike': []}, {}
for sub_folder in sub_folders:
    droj = sub_folder.split("\\")[-1]
    print('\n'*2, '='*30, '\n', droj, '\n', '='*30, '\n'*2)
    files = [x for x in glob.glob(rf'{sub_folder}\*') if x.split('\\')[-1]!='analysis']
    
    sub_folder_analysis = rf'{sub_folder}\analysis'
    if not os.path.exists(sub_folder_analysis): os.makedirs(sub_folder_analysis)

    trt_latencies, trt_n_spikes, trt_profiles = [], [], []
    for file in files:
        file_name = file.split('\\')[-1].split('.')[0]
        print('-'*30, '\n', file_name, '\n', '-'*30, '\n'*2)
        raw = Rawtrace(file)
        plt.figure(), plt.title(file_name)

        latencies, n_spikes, profiles = [], [], []
        for f, istep in enumerate(raw.matrix):
            fig = plt.subplot(2,2,f+1)
            
            istep = istep[5000:]
            sweep_indexes = []
            for sweepNumber in range(10):
                step = istep[16000*sweepNumber:16000*(sweepNumber+1)]
                fig.plot(step)
                sweep = step - savgol_filter(step, 1001, 1)
                indexes, _ = find_peaks(sweep, height=10, prominence=10)
                sweep_indexes.append(indexes)
            
            active_sweep = [x for x in sweep_indexes if len(x)]
            if active_sweep:
                latency = active_sweep[0][0]
                n_spike = len(active_sweep[1]) if len(active_sweep)>1 else len(active_sweep[0])
                
                border = 5000 + 0.1*raw.sampling_rate
                profile = 'ES' if border > active_sweep[0][0] else 'LS'
                profile += ' A' if border > max([a for b in active_sweep for a in b]) else ' NA'
            else:
                latency, n_spike, profile = np.nan, np.nan, 'Not spiking'
            fig = plt.subplot(2,2,4)
            plt.text(0.3, 0.3, profile, dict(size=30))

            latencies.append(latency), n_spikes.append(n_spike), profiles.append(profile)
        trt_latencies.append(np.mean(latencies)), trt_n_spikes.append(np.mean(n_spikes))
        
        result = all(x == profiles[0] for x in profiles)
        if result: trt_profiles.append(profiles[0])
        else:
            w = [x[:2] for x in profiles]
            profile = max(set(w), key = w.count)
            profile += ' NA' if True in ['NA' in p for p in profiles] else ' A'
            trt_profiles.append(profile)
        
        
        if save_fig: plt.savefig(rf'{sub_folder_analysis}\{file_name} {profile}.pdf')
        if not show_fig: plt.close()

    for k, data in zip(Output.keys(), (trt_latencies, trt_n_spikes)): Output[k].append(data)
    trt_profiles = [a for a in trt_profiles if a != 'Not spiking']
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
