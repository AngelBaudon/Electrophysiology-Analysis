# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:37:52 2023

@author: Angel.BAUDON
"""
import pyabf, matplotlib.pyplot as plt, numpy as np, glob, pandas as pd, os
import scipy.stats as stat, seaborn as sns, scipy
from scipy.signal import find_peaks
from ToolKit.Tollbox_TG_shorten import  toto_filter
from statsmodels.stats.anova import AnovaRM
from scikit_posthocs import posthoc_conover
from statsmodels.stats.multicomp import pairwise_tukeyhsd

folder = r"C:\Angel.BAUDON\Exp\Patch\Data\2023_Puff OTR Ago on OTR GFP"
sub_folders = [x for x in glob.glob(rf'{folder}\*')
               if x.split('\\')[-1] not in ('analysis', 'ISteps', 'Raw')]
if not os.path.exists(rf'{folder}\analysis'): os.makedirs(rf'{folder}\analysis')
n_bin, Output, Raw_bl = 60, {}, []

for sub_folder in sub_folders:
    drug = sub_folder.split('\\')[-1]
    print('\n'*2, '='*30, '\n', drug, '\n', '='*30, '\n'*2)
    files = glob.glob(rf'{sub_folder}\*.abf')
    
    sub_folder_analysis = rf'{sub_folder}\analysis'
    if not os.path.exists(sub_folder_analysis): os.makedirs(sub_folder_analysis)
    
    matrix, raw_bl = np.zeros((len(files), n_bin)), []
    for f, file in enumerate(files):
        file_name = (file.split('\\')[-1]).split('.')[0]
        print(file_name, '\n'*2)
        
        abf = pyabf.ABF(file)
        y_ax = toto_filter(abf.sweepY, sample_rate=abf.sampleRate,
                           freq_low=1, freq_high=500)[:600 * abf.sampleRate]

        indexes, _ = find_peaks(y_ax, height = 10)
        binary = np.zeros(len(y_ax))
        for ind in indexes: binary[ind] = 1
        splt = np.sum(np.split(binary, n_bin), axis=1)
        matrix[f,:] = splt/max(splt)
        raw_bl.append(np.mean(splt[:6])/10)
        
        plt.figure(), plt.title(file_name)
        plt.subplot(311), plt.plot(y_ax), plt.title('Raw trace')
        plt.subplot(312), plt.plot(binary), plt.title('Binary')
        plt.subplot(313), plt.plot(matrix[f,:]), plt.title('Normalized')
        plt.savefig(rf'{sub_folder_analysis}\{file_name}.pdf'), plt.close()
        
    Output[drug] = matrix
    Raw_bl.append(raw_bl)

    plt.figure(), plt.suptitle(drug)
    plt.subplot(211), plt.ylabel('Time course of APs')
    mean, sem = np.nanmean(matrix, axis=0), stat.sem(matrix, axis=0)
    plt.fill_between(np.arange(len(mean)), mean-sem, mean+sem,
                     color='lightblue', alpha=0.25, zorder=1)
    plt.plot(mean, c='b', zorder=2), plt.xlabel('Time (10s bins)')
    plt.axvline(6, c='gold', lw = 2)
    plt.subplot(212), sns.heatmap(matrix, cmap="coolwarm")
    plt.savefig(rf'{sub_folder_analysis}\Heat course and Time map.pdf'), plt.close()

    

plt.figure()
writer = pd.ExcelWriter(f'{folder}/analysis/all data.xlsx')

for d, drug in enumerate(Output.keys()):
    data = [(cell[:6], cell[8:14], cell[-6:]) for cell in Output[drug]]
    data = np.asarray([[np.nanmean(x) for x in cell] for cell in data])
    
    Cell_id = [(i,)*3 for i in range(data.shape[0])]
    df = pd.DataFrame({'Cell_ID': [x for y in Cell_id for x in y],
                      'Time': ('Bl', drug, 'Wsh')*data.shape[0],
                      'Score':[x for y in data for x in y]})
    
    mean, sem = np.mean(data, axis=0), stat.sem(data, axis=0)
    print(mean, sem)
    x_ax = np.arange(d*3, d*3+3)
    
    plt.bar(x_ax, mean, yerr=sem, width=.9, capsize=3, label=drug)
    [plt.plot(x_ax, d, lw=.5, c='k', marker='o', mfc='w', mec='k')
     for d in data]
    plt.legend()
    
    pd.DataFrame(data).to_excel(writer, sheet_name=drug)
    
    
    
    
    # norm, grp = [], []
    # for time in set(df['Time']):        
    #     #Normality
    #     values = df[df['Time'] == time]['Score'].values
    #     S, p_val_s = scipy.stats.shapiro(values)
    #     norm.append(False) if p_val_s < 0.05 else norm.append(True)
    #     grp.append(values)
        
    # #Equality of variances
    # L, p_val_l = scipy.stats.levene(*grp)
    # norm.append(False) if p_val_l < 0.05 else norm.append(True)

    # if not False in norm:
    #     aovrm = AnovaRM(df, depvar='Score', subject='Cell_ID', within=['Time'])
    #     res = aovrm.fit().summary().tables[0]
    #     Stat, pval = float(res['F Value']), float(res['Pr > F'])
    #     ph = pairwise_tukeyhsd(df['Score'], df['Time'])

    # else:
    #     Stat, pval = scipy.stats.friedmanchisquare(*grp)
    #     ph = posthoc_conover(grp, p_adjust = 'bonferroni')
    # print('\n\n', drug, '\n', Stat, pval, '\n', ph, '\n\n')

plt.savefig(rf'{folder}\analysis\Histo Hz.pdf'), plt.close()
writer.save()



# mean_bl, sem_bl = [np.mean(x) for x in Raw_bl], [stat.sem(x) for x in Raw_bl]
# plt.figure(), plt.ylabel('Raw baseline Frequencies (Hz)')
# plt.bar((0,1,2), mean_bl, yerr=sem_bl, width=.9, capsize=3, label=drug)



