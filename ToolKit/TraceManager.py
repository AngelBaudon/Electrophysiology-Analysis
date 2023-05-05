# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:55:24 2021

@author: Angel.BAUDON
"""
import numpy as np, pyabf, pyabf.filter, scipy
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os, pandas as pd, matplotlib.pyplot as plt


class FileManager():
    def __init__(self, folder_exp):
        self.folder_exp = folder_exp


    def FileMaker(self, bin_len, drug):
        """ Create 3 files to stock the analsis """
        analysis = f'{self.folder_exp}\Analysis_{bin_len}s'
        traces = f'{analysis}\Indiv traces'
        traces_min_max = f'{analysis}\Max_Min traces'
        Subfo1, Subfo2 = [[f'{a}\{b}' for b in ('PDF', 'PLT')] for a in (traces, traces_min_max)]
        [os.makedirs(path) for path in [analysis, traces, traces_min_max, *Subfo1, *Subfo2]
         if not os.path.exists(path)]
        writer = pd.ExcelWriter('{}/{}.xlsx'.format(analysis, drug))
        return analysis, traces, traces_min_max, writer
    
    
    def FileSeeker(self, forma):
        folder_exp_list = os.listdir(rf'{self.folder_exp}')
        if forma == 'xlsx': folder_exp_list = os.listdir(rf'{self.folder_exp}\dat')
        folder_exp_list = [x for x in folder_exp_list if x.split('.')[-1] == forma]
        return folder_exp_list, len(folder_exp_list)
    
    
class Trace():

    def __init__(self, file):
        self.file = file
    
    def Filter(self, *args):
        from timeit import default_timer as timer
        file, pol, drug_arrival, exp_duration, bin_len, event_type = args
        #Apply a gaussian filter
        abf = pyabf.ABF(self.file)
        if event_type != 'AP':  pyabf.filter.gaussian(abf, 10)
        abf.setSweep(0)
        
        #Extract info
        reduction_factor = 50 if abf.dataRate == 20000 else 100
        sampling_Hz, X_label, Y_label = int(abf.dataRate/reduction_factor), abf.sweepLabelX, abf.sweepLabelY
        drug_time, self.rec_duration = drug_arrival*sampling_Hz, exp_duration*sampling_Hz
        win = bin_len*sampling_Hz

        #Downsample the trace to reduce handling cost
        time, data = abf.sweepX, pol*abf.sweepY
        self.dat = data[::reduction_factor]
        self.data = [x for x in self.dat if str(x)!='nan'][:self.rec_duration]
        self.time = time[::reduction_factor]
        self.time = self.time[:len(self.data)]
 
        # raw = pyabf.ABF(self.file)
        # raw.setSweep(0)
        # raw = raw.sweepY
        # raw = [x for i, x in enumerate(raw) if not i%reduction_factor]
        # self.raw = (raw, self.dat)


        return sampling_Hz, X_label, Y_label, self.time, drug_time, self.rec_duration, win
    
    
    def ModeSubstractor(self):

        
        def Envlp(data, chunk_range):
            y_new = []
            for chunk in range(*chunk_range):
                lmin = (np.diff(np.sign(np.diff(data)))>0).nonzero()[0]+1
                low = lmin[[i+np.argmin(data[lmin[i:i+chunk]])
                                  for i in range(0,len(lmin),chunk)]]
                interp = interp1d(low, data[low], fill_value="extrapolate")
                y_new.append(interp(np.arange(len(data))))
            return np.nanmean(np.asarray(y_new), axis=0)


        #Find the mode
        global_mode = np.asarray(savgol_filter(self.data, 1001, 1))
        precise_mode = np.asarray(savgol_filter(self.data, 101, 1))
        
        #Find the envelope
        global_envlp = Envlp(global_mode, chunk_range=(20, 30))
        precise_envlp = Envlp(precise_mode, chunk_range=(5, 10))

        self.rest = self.data - precise_envlp
        
        # return self.raw, self.data, global_envlp, precise_envlp, self.rest
        return self.data, global_envlp, precise_envlp, self.rest
    
    
    def NoiseEvaluator(self, show_fig = False):
        from scipy.optimize import curve_fit
        from scipy.interpolate import UnivariateSpline
        
        distri = list(self.rest[:])
        distri.sort()
        
        plt.figure(), plt.title('Gaussian fit')
        y,bins,patches, = plt.hist(distri, bins=1000)
    
        x = bins[:-1]
        px = np.linspace(np.min(x),np.max(x),len(x))
        
        def gaussian(x, mu, sigma,A):
            return A*np.exp(-np.power(x-mu,2.)/(2*np.power(sigma,2.)))
        p_opt, _ = curve_fit(gaussian,x,y)
        py = gaussian(px, *p_opt)
        plt.plot(px, py, c='orange', lw=2, label='gaussian fit')
        
        spline = UnivariateSpline(px, py-np.max(py)/2,s=0)
        root_x = float(spline.roots()[0])
        root_y = gaussian(root_x, *p_opt)
    
        plt.axvline(root_x, c='r'), plt.axhline(root_y, c='r')
        plt.scatter(root_x, root_y, s=50, c='k', zorder=4)
        if not show_fig: plt.close()

        return -root_x
    

def WCPSweepReader(file):
    try:
        import neo
        file = neo.WinWcpIO(file)
        block = file.read_block(signal_group_mode='group-by-same-units')
        matrix = np.zeros((len(block.segments),
                           len(block.segments[0].analogsignals[0].magnitude)))
        sampling_Hz = block.segments[0].analogsignals[0].sampling_rate
        
        for sweep in range(len(block.segments)):
            matrix[sweep,:] = np.ravel(block.segments[sweep].analogsignals[0].magnitude)

    except UnicodeDecodeError:
        import pyabf
        abf = pyabf.ABF(file)
        sampling_Hz = abf.dataRate
        for i in abf.sweepList:
            abf.setSweep(i)
            if not i: matrix = abf.sweepY
            else: matrix = np.vstack([matrix, abf.sweepY])
                
    return matrix, int(sampling_Hz/1000)


def z_plotter(zs, time, idx, sweep):
    if zs<1: plt.scatter(time[idx], sweep[idx], c='y', marker='o', s=20, zorder=2)
    elif 1<=zs<2: plt.scatter(time[idx], sweep[idx], c='g', marker='o', s=20, zorder=2)
    elif 2<=zs<3: plt.scatter(time[idx], sweep[idx], c='b', marker='o', s=20, zorder=2)
    else: plt.scatter(time[idx], sweep[idx], c='r', marker='o', s=20, zorder=2)