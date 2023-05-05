# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:02:59 2022

@author: Angel.BAUDON
"""


from scipy.signal import find_peaks
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import savgol_filter



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
            [plt.scatter(np.arange(len(rest))[i], rest[i], c='r', marker='o', s=100) for i in I3]
            
            
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




    def FrameSelector(self, frame, method, initial_burst):        
        '''
        Method can be:
            - FVF: Fix baseline, Variable drug time, Fix wash
                   last frame of the baseline
                   Max activity during drug
                   Last frame of the wash
                   
            - Vcell: Max activity cell by cell
            - FFF: 3 fixed frames
            - Vtot: Max frequency calculated on time course
        '''
        
        
        if method == 'FVF':
            idx_bl, idx_wsh = [x-self.bin_len for x in (self.drug_arrival, self.exp_duration)]

            dr = frame[self.drug_arrival : self.exp_duration-self.wash]
            SF = [sum(dr[i:i+self.bin_len]) for i in range(len(dr)-self.bin_len)]
            idx_dr = np.argmax(SF) + self.drug_arrival
            
            return ([(i, i+self.bin_len) for i in (idx_bl, idx_dr, idx_wsh)],
                    [np.nanmean(frame[i:i+self.bin_len]) for i in (idx_bl, idx_dr, idx_wsh)])
        
        elif method == 'Vcell':
            self.Three_windows = (frame[initial_burst : self.drug_arrival],
                                  frame[self.drug_arrival : self.exp_duration],
                                  frame[self.exp_duration-self.wash : self.exp_duration])
            
            self.SF_3_windows = [[sum(Tw[i : i+self.bin_len])
                                 for i in range(len(Tw)-self.bin_len)]
                                 for Tw in self.Three_windows]
            
            idxs = [np.argmax(SF) for SF in self.SF_3_windows]
            indexes = [x+y for x, y in zip(idxs, (initial_burst, self.drug_arrival,
                                             self.exp_duration-self.wash))]
            
            return ([(i, i+self.bin_len) for i in indexes],
                    [np.nanmean(frame[i:i+self.bin_len]) for i in indexes])


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
        