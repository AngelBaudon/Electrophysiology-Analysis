# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:44:11 2022

@author: Angel.BAUDON
"""
import pyabf, matplotlib.pyplot as plt, numpy as np, glob, pandas as pd, os
import scipy.stats as stat, seaborn as sns
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pyabf.filter import gaussian
from ToolKit.IntraGrpStat import IntraGrpStat

folder = r"C:\Angel.BAUDON\Exp\Patch\Data\2022_Puff KCl\Puff"
sub_folders = [x for x in glob.glob(rf'{folder}\*') if x.split('\\')[-1]!='analysis']
if not os.path.exists(rf'{folder}\analysis'): os.makedirs(rf'{folder}\analysis')
show_fig, save_fig = False, True

Params = ('Membrane voltage (mV)', 'AP frequency (Hz)', 'Delta membrane voltage (mV)',
          'Delta AP frequency (Hz)', 'Rise constant (s)', 'Decay constant (s)')
Output = {x:[] for x in Params}

for sub_folder in sub_folders:
    print('\n'*2, '='*30, '\n', sub_folder.split('\\')[-1], '\n', '='*30, '\n'*2)
    files = glob.glob(rf'{sub_folder}\*.abf')
    
    sub_folder_analysis = rf'{sub_folder}\analysis'
    if not os.path.exists(sub_folder_analysis): os.makedirs(sub_folder_analysis)
    
    all_vm, all_bin = np.empty((len(files), 3)), np.empty((len(files), 3))
    all_dvm, all_dbin = np.empty((len(files), 3)), np.empty((len(files), 3))
    all_vm[:], all_bin[:], all_dvm[:], all_dbin[:] = (np.nan,)*4
    all_lambdas = {'Rise':[], 'Decay':[]}
    
    for f, file in enumerate(files):
        file_name = (file.split('\\')[-1]).split('.')[0]
        print(file_name, '\n'*2)
        
        abf = pyabf.ABF(file)
        y_ax, x_ax = list(abf.sweepY), abf.sweepX
        
        indx = (20, 40, 80, 100, 140, 160)
        if file.split('\\')[-1][-5] == 'a': indx = [x*1.15 for x in indx]
        # if 'trunc' in file: indx = indx[:-2]
        indx = indx[:-2] if 'trunc' in file else indx[:-4] if 'trunk' in file else indx
        indx = [int(x*abf.dataRate) for x in indx]
        
        gaussian(abf, 100)
        abf.setSweep(0)
        envlp = abf.sweepY

        indexes, _ = find_peaks(y_ax-envlp, height=20, prominence=20)
        
        binary = np.zeros(len(y_ax))
        for i in indexes: binary[i]+=1
        
        splt_vm, splt_bin = np.split(np.asarray(envlp), indx), np.split(binary, indx)
        
        splt_data, all_deltas = [], []
        for name, splt in zip(('vm', 'bin'), (splt_vm, splt_bin)):
            data = [[splt[x+y] for y in (0,1,2)] for x in np.arange(len(indx))[::2]]
            if name == 'bin':
                data = [[sum(r)/(len(r)/abf.dataRate) for r in run] for run in data]
                
            elif name == 'vm':
                data = [[np.nanmedian(r) for r in run] for run in data]
            splt_data.append(np.nanmean(np.array(data), axis=0))
            all_deltas.append([y[1]-y[0] for y in data])
            
            
        all_vm[f,:], all_bin[f,:] = splt_data
        for deltas, stock in zip(all_deltas, (all_dvm, all_dbin)):
            for i, delta in enumerate(deltas): stock[f,i] = delta
        

        def expo(x, a, b, c): return a*np.exp(-b*x)+c #a=size, b=angle, c=intercept
        
        lambdas = {'Rise':[], 'Decay':[]}
        for i in range(1, len(splt_vm)):
            try:
                data = [x for x in splt_vm[i] if str(x) != 'nan']
                x = np.linspace(0,len(data),len(data))
                (size, loc, mark) = (-10, -1, .632) if i%2 else (10, 0, .368)
                popt, _ = curve_fit(expo, x, data, p0=[size, 0.00002, -60])
                fit = expo(x, *popt)-min(expo(x, *popt))
                tau = (np.where(fit<mark*max(fit))[0][loc])/abf.dataRate
                lambdas['Rise'].append(tau) if i%2 else lambdas['Decay'].append(tau)
                # plt.figure(), plt.plot(x, data), plt.plot(x, expo(x, *popt)), plt.title(tau)
            except RuntimeError: continue
        for k in all_lambdas.keys(): all_lambdas[k].append(np.mean(lambdas[k]))
        
        plt.figure(), plt.title(file_name)
        plt.plot(x_ax, y_ax, c='b', lw=.5, label='Raw')
        plt.plot(x_ax, envlp, c='k', label='Envlp')
        [plt.axvline(i/abf.dataRate, c='r') for i in indx]
        plt.plot(indexes/abf.dataRate, [y_ax[i] for i in indexes], 'xr')
        plt.legend(loc='upper right')
        if save_fig: plt.savefig(rf'{sub_folder_analysis}\{file_name}.pdf')
        if not show_fig: plt.close()

    for k, data in zip(Output.keys(),
                       (all_vm, all_bin,
                        *[np.nanmean(d, axis=1) for d in (all_dvm, all_dbin)],
                        *[all_lambdas[k] for k in all_lambdas])):
        Output[k].append(data)



for k in Output.keys():
    data = Output[k]
    plt.figure(), plt.title(k), plt.ylabel(k)
    TRT = [x.split('\\')[-1] for x in sub_folders]
    Trt = [a for b in [(T,)*len(O) for T, O in zip(TRT, Output[k])] for a in b]

    if k in Params[:2]:
        mean = [x for y in [np.mean(x, axis=0) for x in data] for x in y]
        sem = [x for y in [stat.sem(x, axis=0) for x in data] for x in y]
        x_ax = np.arange(len(sub_folders)*3)
        plt.bar(x_ax, mean, yerr=sem, width=.9, capsize=3)
        [[plt.plot(i, d, lw=.5, c='k', marker='o', mfc='w', mec='k') for d in dat]
          for i, dat in zip(np.split(x_ax, len(sub_folders)), data)]

    else:
        df = pd.DataFrame({'Treatment': Trt, k: [a for b in data for a in b]})
        sns.barplot(data=df, x='Treatment', y=k, color='lightgreen', capsize=.2)
        sns.swarmplot(x='Treatment', y=k, data=df, size=5, color='w',
                      edgecolor='k', linewidth=1)
        for trt in set(df['Treatment']):
            t = df[df['Treatment'] == trt][k]
            mean, sem = np.nanmean(t), stat.sem(t)
            print(k, trt, '\n', mean, sem, '\n\n')


        df_stat = IntraGrpStat(df)
        
        writer = pd.ExcelWriter(f'{folder}/analysis/{k}.xlsx')
        df.to_excel(writer, sheet_name = k)
        df_stat.to_excel(writer, sheet_name = 'Stats')
        writer.save()
    
    if save_fig: plt.savefig(rf'{folder}\analysis\{k}.pdf')
    if not show_fig: plt.close()
    
    


