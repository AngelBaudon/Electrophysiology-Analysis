# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 08:51:36 2021

@author: Angel.BAUDON
"""
import matplotlib.pyplot as plt, pickle, pandas as pd
import numpy as np, seaborn as sns, scipy.stats as Stat
from ToolKit.IntraGrpStat import IntraGrpStat


class PlotFactory():
    def __init__(self, cell_value, rest, global_envlp, precise_envlp, pol,
                 threshold, indexes, file_name, exp_duration, raw = False):
        if raw: self.raw = (raw[0], [x*pol for x in raw[1]])
        self.threshold = -threshold
        self.data = pol*np.asarray(cell_value)
        self.rest = pol*rest
        self.indexes = indexes
        self.file_name = file_name.split('.')[0]
        self.exp_duration = exp_duration
        self.time = np.linspace(0, exp_duration, len(rest))
        self.global_envlp = pol*global_envlp
        self.precise_envlp = pol*precise_envlp

        
    def EventOverlay(self, individual_traces, show_fig = False):
        if show_fig:
            plt.figure(figsize=(5, 9))
            events = np.zeros((len(self.indexes), 70))
            events = [self.rest[i-20 : i+50] for i in self.indexes if i>20]
            for eve in events: plt.plot(eve, c='deepskyblue')
            plt.plot(np.nanmean(events, axis=0), c='r')

        
    def TracePloter(self, individual_traces, show_fig = False):
        plt.figure().suptitle(self.file_name)

        # fig = plt.subplot(3,1,1)
        # X = np.linspace(0, self.exp_duration, len(self.raw[0]))
        # fig.plot(X, self.raw[0], c='b', lw=.5, label='Raw')
        # fig.plot(X, self.raw[1], c='k', label='Filtered')
        # plt.legend(loc='upper right')

        fig = plt.subplot(3,1,1)
        fig.plot(self.time, self.data, c='b', lw=.5, label='Data')
        fig.plot(self.time, self.global_envlp, c='k', label='Global envlp')
        fig.plot(self.time, self.precise_envlp, c='purple', label='Precise envlp')
        plt.legend(loc='upper right')

        fig = plt.subplot(3,1,2)
        fig.plot(self.time, self.rest, c='b', lw=.5, label='Rest')
        plt.axhline(self.threshold)
        plt.legend(loc='upper right')
        for i in self.indexes: fig.scatter(self.time[i], self.rest[i], c='r', marker='x')
        
        pickle.dump(fig, open(rf'{individual_traces}/PLT/{self.file_name}', 'wb'))
        plt.savefig(rf'{individual_traces}/PDF/{self.file_name}.pdf')
        if not show_fig: plt.close()


    def Max_Ploter(self, X_label, Y_label, drug, drug_time, sampling_Hz,
                       event_type, max_min_traces, Max_Indx, show_fig = False):
        
        
        #Plot the trace with the Maximum & Minimum frames
        plt.figure(figsize = (15,10))
        plot, loc = plt.subplot(211), 4
        plot.plot(self.time, self.rest, c='b', lw=.5)
        plt.title('{}, {}'.format(drug, self.file_name))
        plt.ylabel(Y_label), plt.xlabel(X_label)
        plt.axvline(drug_time/sampling_Hz, c='gold', lw=2)
        for index in self.indexes:
            plt.scatter(self.time[index], self.rest[index], c='r',marker='o')
        for Start, Stop in Max_Indx: plt.axvspan(Start, Stop, alpha=0.25, color='palegreen')
        
        for (fr1, fr2) in Max_Indx:
            Start, Stop = int(fr1*sampling_Hz), int(fr2*sampling_Hz)
            interv, loc = plt.subplot(2,3,loc), loc+1
            interv.plot(self.time[Start:Stop], self.rest[Start:Stop], c='b', lw=.25)

            for idx in self.indexes: 
                if Start <= idx <= Stop:
                    plt.scatter(self.time[idx] , self.rest[idx], c='r',marker='o')
        
        pickle.dump(plot, open(rf'{max_min_traces}/PLT/{self.file_name}', 'wb'))
        plt.savefig(rf'{max_min_traces}/PDF/{self.file_name}.pdf')
        if not show_fig: plt.close()
        
        
class FinalPlots():
    def __init__(self, exp_duration, rec_duration, drug, colorz, folder, drug_time):
        self.exp_duration, self.rec_duration = exp_duration, rec_duration
        self.sampling_Hz = rec_duration/exp_duration
        self.x_ax = np.linspace(0, self.exp_duration, self.rec_duration)
        self.drug, self.colorz, self.folder = drug, colorz, folder
        self.drug_time = drug_time
    
    #Histos
    def Histo(self, Output, show_fig = False):
        M, S, P, n_cell = [], [], [], len(Output.index)
        titles = [x for x in Output.columns if x != 'Behavior']
        for title in titles:
            data, x_ax = Output[title], np.arange(3)
            data = np.asarray([[x[i] for x in data] for i in x_ax]).T
            bl, dr, wsh = [data[:,i] for i in range(3)]
            meanz = np.nanmean(data, axis=0)
            semz = [Stat.sem(data[:,i]) for i in x_ax]
            plt.figure(), plt.xticks(x_ax, ['Baseline', self.drug, 'Wash']), plt.title(title)
            plt.ylabel('Hz') if title in ('Max Hz', 'Min Hz') else plt.ylabel('Amplitude')
            plt.bar(x_ax, meanz, yerr=semz, color=self.colorz[1], capsize=10, zorder=0)
            
            for x in data:
                [plt.scatter(i, x, s=20, c='w', marker='o', edgecolor='k', zorder=2)
                 for i, x in enumerate(x)]
                plt.plot(x_ax, x, c='k', lw=0.5, zorder=1)
                
                
            try: stat, pval, test = IntraGrpStat([bl, dr, wsh], Paired=True)
            except ValueError: stat, pval, test = np.nan, np.nan, 'Error'
            
            print(stat, pval, test, '\n')
            
            
            def Q(pval):
                if pval<=0.001: star = '***'
                elif 0.001<pval<=0.01: star = '**'
                elif 0.01<pval<=0.05: star = '*'
                else: star = 'ns'
                return star
        
            def caca(x, y, pval):
                plt.plot((x[0]+.1, x[1]-.1), (y,)*2, c='k', lw=1)
                plt.text(np.mean(x), y, Q(pval), size=10, weight='bold') 
            
            bot, top = plt.ylim()
            plt.ylim(bot, top+top/5)
            (bot, top), k = plt.ylim(), list(test.keys())
            
            for x, y, p in zip(((0,2), (0,1), (1,2), (.5,1.5)),
                               (top-top/5, *(top-top/7,)*2, top-top/10),
                               (pval, test[k[1]], test[k[3]], test[k[2]])): caca(x, y, p)
            
            plt.text(-.4, top-(top/6),
                     f"Test: {test['Test']} \n Stat: {stat} \n p-val: {pval} \n n: {n_cell} cells")
            M.append(meanz), S.append(semz), P.append((pval, stat, test))
            plt.savefig(rf'{self.folder}\{title}.pdf')
            if not show_fig: plt.close()
        for x, y in zip(('Means', 'Sems', 'Stat'), (M, S, P)): Output.loc[x] = (*y, np.nan)
        return Output


    def HeatMap(self, bin_exp, win_10s, show_fig = False):
        plt.figure(), plt.title(f'{self.drug} heat map')
        fig = sns.heatmap(bin_exp,
                          cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
        fig.set_ylabel('Cells'), fig.set_xlabel('Bin (1min)')
        fig.axvline(self.drug_time/win_10s, c='gold', lw = 2)
        plt.savefig(rf'{self.folder}\Heat map.pdf')
        if not show_fig: plt.close()

    
    def ECDF(self, ecdf, show_fig = False):
        mean_ecdf, sem_ecdf = np.nanmean(ecdf, axis=0), Stat.sem(ecdf, axis=0)
        plt.figure(), plt.title(self.drug), plt.ylabel('Ecdf')
        
        plt.fill_between(self.x_ax, mean_ecdf-sem_ecdf, mean_ecdf+sem_ecdf,
                         color = self.colorz[1], alpha=0.25, zorder=1)
        
        plt.plot(self.x_ax, mean_ecdf, c='r', zorder=2)
        plt.axvline(self.drug_time/self.sampling_Hz, c='gold', lw = 2)
        
        for e in ecdf: plt.plot(self.x_ax, e, lw=.5, zorder=0, c='deepskyblue')
        plt.savefig(rf'{self.folder}\Ecdf.pdf')
        if not show_fig: plt.close()
    
    def TimeCourse(self, data, label, norm, show_fig = False):
        if norm:
            D = np.zeros((len(data), len(data[0])))
            for c, cell in enumerate(data): D[c,:] = [x/sum(cell) for x in cell]
            data = D
                
        plt.figure()
        for cell in data: plt.plot(cell, lw=1, zorder=0, alpha = 0.5)
        plt.plot(np.nanmean(data, axis = 0), c='r', lw=2, zorder=1, alpha=1)
        plt.axvline(self.drug_time/self.sampling_Hz, c='gold', lw = 5)
        plt.ylabel(label), plt.xlabel('time(seconds)')
        plt.savefig(rf'{self.folder}\{label}.pdf')
        if not show_fig: plt.close()

    
    def PieChart(self, data, show_fig = False):
        plt.figure()
        behavior = pd.Series([0]*4, index=('Incr & Decr', 'Incr', 'Decr', 'Not resp'))
        behavior.add(data.value_counts(), fill_value=0)
        plt.pie(behavior, explode = (0.1, 0, 0, 0), colors = self.colorz[:4],
                autopct = '%1.1f%%', shadow = True, startangle = 90)
        plt.legend(behavior.index, loc ='upper right'), plt.axis('equal')
        plt.savefig(rf'{self.folder}\Pie_Chart.pdf')
        if not show_fig: plt.close()
