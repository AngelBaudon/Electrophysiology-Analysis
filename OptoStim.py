# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:02:29 2023

@author: Angel.BAUDON

"""

import matplotlib.pyplot as plt, numpy as np, glob, pandas as pd, os
import scipy.stats as stat, seaborn as sns
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from ToolKit.IntraGrpStat import IntraGrpStat
from ToolKit.Tollbox_TG_shorten import Rawtrace, toto_filter


folder = r"C:\Angel.BAUDON\Exp\Patch\Data\2023_Rec CeL OptoStim BLA\IClamp"
sub_folders = [x for x in glob.glob(rf'{folder}\*') if x.split('\\')[-1]!='analysis']
if not os.path.exists(rf'{folder}\analysis'): os.makedirs(rf'{folder}\analysis')
show_fig, save_fig = False, True

Params = ('Membrane voltage (mV)', 'AP frequency (Hz)', 'Amplitude(pA)',
          'Failure rate', 'PPSE rises', 'PPSE decays', 'Delta membrane voltage (mV)',
          'Delta AP frequency (Hz)', 'Rise constant (s)', 'Decay constant (s)')
Output = {x:[] for x in Params}

for sub_folder in sub_folders:
    print('\n'*2, '='*30, '\n', sub_folder.split('\\')[-1], '\n', '='*30, '\n'*2)
    files = glob.glob(rf'{sub_folder}\*.wcp')
    
    sub_folder_analysis = rf'{sub_folder}\analysis'
    if not os.path.exists(sub_folder_analysis): os.makedirs(sub_folder_analysis)
    
    all_vm, all_bin = np.empty((len(files), 3)), np.empty((len(files), 3))
    all_dvm, all_dbin = np.empty((len(files), 3)), np.empty((len(files), 3))
    all_amps, all_fails = np.empty((len(files), 199)), np.empty((len(files), 199))
    all_rises, all_decays = np.empty((len(files), 199)), np.empty((len(files), 199))
    all_vm[:], all_bin[:], all_dvm[:], all_dbin[:] = (np.nan,)*4
    all_amps[:], all_fails[:], all_rises[:], all_decays[:] = (np.nan,)*4
    all_lambdas = {'Rise':[], 'Decay':[]}
    
    for f, file in enumerate(files):
        file_name = (file.split('\\')[-1]).split('.')[0]
        print(file_name, '\n'*2)
        
        raw = Rawtrace(file)
        y_ax = np.concatenate(raw.matrix)
        if file.split('\\')[-1] in ('P2_230201_001.2.wcp', 'P2_230201_006.wcp'): y_ax = y_ax/200
        sampling = raw.sampling_rate
        rec_len = len(y_ax)/sampling 
        
        print('j adore le caca')

        trace = toto_filter(y_ax, sample_rate=sampling,
                            freq_low=.01, freq_high=500)
        sampling = sampling/10

        indx = (20, 30, 99.8, 109.8, 176.6, 189.6)
        indx = np.array([int(x*sampling) for x in indx])
        sub_indx = [np.linspace(start, stop, 200, dtype=int)
                    for (start, stop) in np.split(indx, 3)]
        
        trace = trace[::10] - np.nanmedian(trace[:indx[0]])
        
        x_ax = np.linspace(0, rec_len, len(trace))

        envlp = toto_filter(trace, sample_rate=sampling,
                            freq_low=.01, freq_high=1)
        fltr = trace - envlp
        
        spikes, _ = find_peaks(fltr, height=20, prominence=20)
        ppse, _ = find_peaks(fltr, height=.5, prominence=1.5)
        
        # plt.figure(),
        # plt.subplot(2,1,1)
        # plt.plot(x_ax, trace)
        # plt.plot(x_ax, envlp, c='purple')
        # plt.plot(spikes/sampling, [trace[i] for i in spikes], 'og')
        # plt.plot(ppse/sampling, [trace[i] for i in ppse], 'xr')

        
        # plt.subplot(2,1,2)
        # plt.plot(x_ax, fltr)
        # plt.plot(x_ax, ppse, [fltr[i] for i in ppse], 'xr')
        
        
        binary = np.zeros(len(trace))
        for i in spikes: binary[i]+=1
        
        splt_vm, splt_bin = np.split(np.asarray(envlp), indx), np.split(binary, indx)
        
        splt_data, all_deltas = [], []
        for name, splt in zip(('vm', 'bin'), (splt_vm, splt_bin)):
            data = [[splt[x+y] for y in (0,1,2)] for x in np.arange(len(indx))[::2]]
            if name == 'bin':
                data = [[sum(r)/(len(r)/sampling) for r in run] for run in data]
                
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
                tau = (np.where(fit<mark*max(fit))[0][loc])/sampling
                lambdas['Rise'].append(tau) if i%2 else lambdas['Decay'].append(tau)
                # plt.figure(), plt.plot(x, data), plt.plot(x, expo(x, *popt)), plt.title(tau)
            except RuntimeError: continue
        for k in all_lambdas.keys(): all_lambdas[k].append(np.mean(lambdas[k]))
        
        
        cell_amps, cell_fails = [], []
        cell_rise, cell_decay = [], []
        for sub_i in sub_indx:
            stims = np.split(trace, sub_i)[1:-1]
            amps, fails, sweep_rise, sweep_decay = [], [], [np.nan,]*199, [np.nan,]*199
            
            for s, stim in enumerate(stims):
                start, stop = sub_i[s], sub_i[s+1]
                stim_ppse = ppse[np.where(np.logical_and(ppse>start, ppse<stop))]
                
                if len(stim_ppse):

                    indx_stim_ppse = stim_ppse[0] - start -10
                    stim = stim[10:]-stim[0]
                    rise, decay = stim[:indx_stim_ppse], stim[indx_stim_ppse:]
                    amp, fail = stim[indx_stim_ppse], False
                    
                    # plt.figure(), plt.title(indx_stim_ppse)
                    # plt.subplot(2,1,1), plt. plot(stim)
                    
                    for k, kin in enumerate((rise, decay)):
                        x, mark = np.arange(len(kin)), [.632 if k%2 else .368]

                        try:
                            popt, _ = curve_fit(expo, x, kin, p0=[-10, 0.02, 0])
                            fit = expo(x, *popt)-min(expo(x, *popt))
                            tau = (np.where(fit<mark[0]*max(fit))[0][k-1])/sampling
                            # plt.subplot(2, 2, 3+k), plt.plot(x, kin), plt.plot(x, expo(x, *popt))
                            # plt.text(1,1, str(tau))
                            if not k: sweep_rise[s] = tau
                            else: sweep_decay[s] = tau
                        except: continue

                else: amp, fail = np.nan, True
                
                amps.append(amp), fails.append(fail)
            
            cell_amps.append(amps), cell_fails.append(fails)
            cell_rise.append(sweep_rise), cell_decay.append(sweep_decay)

        all_amps[f,:] = np.nanmean(cell_amps, axis=0)
        all_fails[f,:] = np.nanmean(cell_fails, axis=0)
        all_rises[f,:] = np.nanmean(cell_rise, axis=0)
        all_decays[f,:] = np.nanmean(cell_decay, axis=0)        
        
        
        

        plt.figure(), plt.title(file_name)
        plt.plot(x_ax, trace, c='b', lw=.5, label='Raw')
        plt.plot(x_ax, envlp, c='k', label='Envlp')
        [plt.axvline(i/sampling, c='r') for i in indx]
        [plt.axvline(i/sampling, c='lightblue', zorder=0)
          for i in np.concatenate(sub_indx)]
        plt.plot(spikes/sampling, [trace[i] for i in spikes], 'og')
        plt.plot(ppse/sampling, [trace[i] for i in ppse], 'xr')
        plt.legend(loc='upper right')
        if save_fig: plt.savefig(rf'{sub_folder_analysis}\{file_name}.pdf')
        if not show_fig: plt.close()

    for k, data in zip(Output.keys(),
                        (all_vm, all_bin, all_amps, all_fails, all_rises, all_decays,
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

    elif k in Params[2:6]:
        means = [np.nanmean(x, axis=0) for x in data]
        sems = [stat.sem(x, axis=0) for x in data]
        x_ax = np.arange(199)
        for mean, sem in zip(means, sems):
            plt.plot(x_ax, mean)
            plt.fill_between(x_ax, mean-sem, mean+sem, alpha=0.5, zorder=1)

    else:
        df = pd.DataFrame({'Treatment': Trt, k: [a for b in data for a in b]})
        sns.barplot(data=df, x='Treatment', y=k, color='lightgreen', capsize=.2)
        sns.swarmplot(x='Treatment', y=k, data=df, size=5, color='w',
                      edgecolor='k', linewidth=1)

        # df_stat = IntraGrpStat(df)
        
        writer = pd.ExcelWriter(f'{folder}/analysis/{k}.xlsx')
        df.to_excel(writer, sheet_name = k)
        # df_stat.to_excel(writer, sheet_name = 'Stats')
        writer.save()
    
    if save_fig: plt.savefig(rf'{folder}\analysis\{k}.pdf')
    if not show_fig: plt.close()
    
    

