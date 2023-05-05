# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:50:04 2022

@author: Angel.BAUDON
"""

import neo, numpy as np, matplotlib.pyplot as plt

        
class Rawtrace: 
    def __init__ (self, file):
       self.file = neo.WinWcpIO(f'{file}')
       self.block = self.file.read_block(signal_group_mode='group-by-same-units')
       self.nb_points = len(self.block.segments[0].analogsignals[0].magnitude)
       self.nb_sweeps = len(self.block.segments)
       self.matrix = np.zeros((self.nb_sweeps,self.nb_points))
       self.sampling_rate = float(self.block.segments[0].analogsignals[0].sampling_rate)
       self.time = np.linspace(0,(self.nb_points/self.sampling_rate),self.nb_points)
       for sweep in range(len(self.block.segments)):
           self.matrix[sweep,:] = np.ravel(self.block.segments[sweep].analogsignals[0].magnitude)
    
    def __repr__ (self):
        return f'{self.matrix}'
    


def toto_filter(signal, order=8, sample_rate=20000, freq_low=400, freq_high=2000, axis=0):
    import scipy.signal
    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
    sos_coeff = scipy.signal.iirfilter(order, Wn, btype="band", ftype="butter", output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)
    return filtered_signal   


if __name__ == '__main__':
    raw = r"C:\Angel.BAUDON\Exp\Patch\Data\2023_Rec CeL OptoStim BLA\IRamp\0_Ctrl\P1_230127_001.2.wcp"
    
    file = Rawtrace(raw)
    plt.figure()
    plt.plot(file.matrix[2])