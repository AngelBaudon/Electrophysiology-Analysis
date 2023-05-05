# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:54:34 2021

@author: Angel.BAUDON
"""
from scipy.signal import find_peaks
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d




class Analysor():
    def __init__(self, rec_duration, sampling_Hz, bin_len, drug_arrival, wash):
        self.rec_duration = rec_duration
        self.sampling_Hz = sampling_Hz
        self.exp_duration = int(rec_duration/sampling_Hz)
        self.bin_len = bin_len
        self.drug_arrival = drug_arrival
        self.wash = wash
    
    
    def PeakFinder(self, rest, root, event_type, pol, show_fig=False):
        threshold = root*4 if event_type != 'AP' else 10
        self.event_type = event_type
        
        # I1, _ = find_peaks(np.asarray(rest), height=threshold)
        # I2, _ = find_peaks(np.asarray(rest), prominence=threshold)
        I3, _ = find_peaks(np.asarray(rest), height=threshold, prominence=threshold)
        
        if show_fig:
            plt.figure(), plt.title(root), plt.axhline(-threshold*pol, c='y'), plt.plot(rest, c='b')
            [plt.scatter(np.arange(len(rest))[i], rest[i], c='r', marker=o, s=100) for i in I3]
            
            
            # [[plt.scatter(np.arange(len(rest))[i], rest[i], c=c, marker=m, s=100) for i in ind]
            #  for c, m, ind in zip(('c','k','r'), ('o','1','2'), (I3, I2, I1))]
        
        self.indexes = I3
        return threshold, self.indexes

    def Binary(self):
        #Binary list of events
        self.peak_binary = np.zeros(self.rec_duration)
        for index in self.indexes: self.peak_binary[index] += 1
        ecdf = np.cumsum(self.peak_binary)
        Hz_sec = [np.nansum(x) for x in np.split(self.peak_binary, self.exp_duration)]
        return self.peak_binary, ecdf/ecdf[-1], Hz_sec

    
    def CellExcludor(self):
        #Exclude cells with less than 1 event/min
        n_bin = int(self.exp_duration/60)
        splitted_cell_binary = np.split(self.peak_binary, n_bin)
        bin_cell_exclu = [np.nansum(scb) for scb in splitted_cell_binary]
        nb_0, nb_1 = bin_cell_exclu.count(0), bin_cell_exclu.count(1)
        if nb_0 + nb_1 > 20: return True
    
        
    def BinCalculator(self, bin_len=40):
        #Split the data into bins & attribute a binary value to each bin
        n_bin = int(self.rec_duration/(bin_len*self.sampling_Hz))
        splitted_cell_binary = np.split(self.peak_binary, n_bin)
        self.bin_cell = [np.nansum(scb) for scb in splitted_cell_binary]
        return self.bin_cell


    def AmpSeeker(self, rest):
        amp = np.full(self.rec_duration, np.nan)
    
        for ind in self.indexes:
            # self.peak = rest[ind-100:ind+100] if ind > 50 else rest[:200]
            
            # plt.figure(), plt.plot(self.peak)
            
            # peak_fltr = savgol_filter(self.peak, 101, 1)
            
            # plt.plot(peak_fltr)
            
            
            amp[ind] = rest[ind]
            
            
        amp_sec = [np.nanmean(x) for x in np.split(amp, self.exp_duration)]
        return amp_sec, amp




    def FrameSelector(self, frame, initial_burst):
        self.b = frame
        
        
        
        # self.SF = [np.sum(mini_bin[i:i+self.bin_len]) for i, j in enumerate(mini_bin)
        #             if i<(self.exp_duration-self.bin_len)]
        # self.sf = self.SF[self.drug_arrival : self.exp_duration-self.wash]
        
        self.Three_windows = (frame[120:self.drug_arrival],
                              frame[360:720],
                              frame[self.exp_duration-self.wash:self.exp_duration-120])
        
        self.SF_3_windows = [[sum(Tw[i:i+self.bin_len])
                             for i, j in enumerate(Tw)]
                             for Tw in self.Three_windows]


        def Seeker(s_f, MinOrMax):
            Indexes = []
            for sf in s_f:
                sf = sf[:len(sf)-self.bin_len]
                Indx = np.argmax(sf) if MinOrMax == 'Max' else np.argmin(sf)
                Indexes.append(Indx + self.bin_len//2)
            
            Indexes = [x+y for x, y in zip(Indexes, (120, 360, self.exp_duration-self.wash))]
            
            Indexes = [(I-self.bin_len//2, I+self.bin_len//2) for I in Indexes]
            
            Indexes = [(120, self.drug_arrival), Indexes[1],
                       (self.exp_duration-self.wash, self.exp_duration-120)]
            
            Hz = [np.nanmean(frame[I[0]:I[1]]) for I in Indexes]
            return Indexes, Hz
        
        return Seeker(self.SF_3_windows, 'Max'), Seeker(self.SF_3_windows, 'Min')



    def CellClassifier(self, Parameter):
        
        '''   To adapt with amplitude   '''
        P_Max, P_Min = Parameter
        
        #Caracterise the cell behavior
        if (P_Max[1]>1.2*P_Max[0]) and (P_Min[1]<1.2*P_Min[0]) : return 'Incr & Decr'
        elif P_Max[1]>1.2*P_Max[0]: return 'Incr'
        elif P_Min[1]<1.2*P_Min[0]: return 'Decr'
        else: return 'Not resp'
        
        

        
        
        # #Find the standart deviation 
        # sd_bl = np.std(self.bin_cell[:self.drug_arrival])
        # sd_wsh = np.std(self.bin_cell[self.exp_duration-self.wash:self.exp_duration])
        
        # #Find the delta between the drug vs baseline and the drug vs wash
        # Delta = [(P[1]-P[0], P[1]-P[2]) for P in Parameter]

        # #Verify if the delta is > or < to the threshold
        # Deltas_bool = [x for y in [(D[0]>3*sd_bl, D[1]>3*sd_wsh) for D in Delta] for x in y]
        
        # #Caracterise the cell behavior
        # if not False in Deltas_bool: return 'Incr & Decr'
        # elif Deltas_bool[0] and Deltas_bool[1]: return 'Incr'
        # elif Deltas_bool[2] and Deltas_bool[3]: return 'Decr'
        # else: return 'Not resp'


def envlp(data, chunk_range=(5,15)):
    y_new = []
    for chunk in range(*chunk_range):
        lmin = (np.diff(np.sign(np.diff(data)))>0).nonzero()[0]+1
        low = lmin[[i+np.argmin(data[lmin[i:i+chunk]])
                          for i in range(0,len(lmin),chunk)]]
        interp = interp1d(low, data[low], fill_value='extrapolate')
        y_new.append(interp(np.arange(len(data))))
    return np.nanmean(np.asarray(y_new), axis=0)