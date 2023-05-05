# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:17:22 2019

@author: Master5.INCI-NSN
"""

import neo 
import os
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd 
from scipy import signal

file = r"C:\Users\Angel.BAUDON\Desktop\OptoStim BLA Rec CeL\2022_11_Patch1\8_11\S1_C1\221108_001.2.wcp"
        
class Rawtrace: 
    def __init__ (self, file):
       self.file = neo.WinWcpIO(f'{file}')
       self.block = self.file.read_block(signal_group_mode='group-by-same-units')
       self.nb_points = len(self.block.segments[0].analogsignals[1].magnitude)
       self.nb_sweeps = len(self.block.segments)
       self.matrix = np.zeros((self.nb_sweeps,self.nb_points))
       self.sampling_rate = float(self.block.segments[0].analogsignals[1].sampling_rate)
       self.time = np.linspace(0,(self.nb_points/self.sampling_rate),self.nb_points)
       for sweep in range(len(self.block.segments)):
           self.matrix[sweep,:] = np.ravel(self.block.segments[sweep].analogsignals[1].magnitude)
           
    def filters(self, freq_low, freq_high, window, order):
        if freq_low + freq_high != 0:
            self.matrix = filter_signal(self.matrix[:], freq_low = freq_low, freq_high = freq_high,axis=1)
        if window + order !=0:
            self.matrix = signal.savgol_filter(self.matrix[:],window,order)
      
    
    def __repr__ (self):
        return f'{self.matrix}'
    
class Rawtrace3D: 
   
    def __init__ (self, folder):
        self.files = os.listdir(folder)
       
        for file,idx in zip (self.files,range(len(self.files))): 
           
            self.file = neo.WinWcpIO(f'{folder}\{file}')
            self.block = self.file.read_block(signal_group_mode='group-by-same-units')
            self.nb_points = len(self.block.segments[0].analogsignals[0].magnitude)
            self.nb_sweeps = len(self.block.segments)
            self.sampling_rate = float(self.block.segments[0].analogsignals[0].sampling_rate)
            self.time = np.linspace(0,(self.nb_points/self.sampling_rate),self.nb_points)
            if idx == 0:
                self.matrix = np.zeros((self.nb_sweeps,self.nb_points,len(self.files)))
             
            for sweep in range(len(self.block.segments)):
                self.matrix[sweep,:,idx] = np.ravel(self.block.segments[sweep].analogsignals[0].magnitude)
           
    def filters (self, freq_low=0, freq_high=0, window=0, order=0):
        if freq_low + freq_high != 0:
            self.matrix = filter_signal(self.matrix[:], freq_low = freq_low, freq_high = freq_high, axis = 1)
        if window + order !=0:
            self.matrix = signal.savgol_filter(self.matrix[:],window,order,axis=1)
    
          
    def __repr__ (self)    :
        return f'{self.matrix}'
           
def response_window (start, stop, time):
    
    # delta = stop - start
    start_idx = np.ravel(np.where(time >= start))[0]
    stop_idx = np.ravel(np.where(time >= stop))[0]     
    return start_idx, stop_idx

class TraceAnalysis:
    
    def __init__(self, trace, start , stop):
        self.trace = trace
        self.traced = trace
        self.start = start
        self.stop = stop
       
    def average (self, axis = 2, method_avg = 'median'):
        self.method_avg=method_avg
        if self.method_avg =='mean':
            self.trace = np.mean(self.trace, axis = axis)
            print('mean')
        if self.method_avg =='median':
            self.trace = np.median(self.trace, axis = axis)
            print('median')
    def leak_substraction(self,window,noise=False):
        if self.trace.ndim ==1:
            delta = np.mean(self.trace[self.start-(self.start-window):self.start]) 
            self.trace = self.trace - delta
            return self.trace
        if self.trace.ndim ==2:
            delta = np.mean(self.trace[:,self.start-(self.start-window):self.start],axis=1) 
            self.trace = self.trace - delta[:,None].astype(float)
            return self.trace
        if self.trace.ndim ==3:
            if noise==True:
                self.trace_baseline = np.mean(self.trace[:,self.start:,:],axis=1)
                self.trace_jump = np.mean(self.trace[:,int(self.start/2):self.start,:],axis=1)
                print(self.trace_baseline.shape)
                print(self.trace_jump.shape)
                self.delta = self.trace_baseline - self.trace_jump
                print(self.delta.shape)
                self.delta =  self.delta[:,np.newaxis,:]
                print(self.delta.shape)
                self.trace[:,int(self.start/2):self.start,:] = self.trace[:,int(self.start/2):self.start,:] + self.delta
            self.trace_leak = np.mean(self.trace[:,self.start-(self.start-window):self.start,:],axis=1)
            self.trace_leak = self.trace_leak[:,np.newaxis,:]
            self.trace = self.trace - self.trace_leak
            return self.trace
        # if self.trace.ndim==2:
        #     delta = np.mean(self.trace[:,self.start-(self.start-window):self.start],axis=1) 
        #     self.trace = self.trace - delta[:,None]
        #     return self.trace
        
    def mod_substraction(self):
        import gaussian_fit
        trace_mod=np.array([gaussian_fit.gaussianfit(sweep,plot=False,bins=25) for sweep in self.trace])
        self.trace=self.trace-trace_mod[:,None].astype(float)
        return trace_mod 
        
    def spikes_detection(self,time,url,threshold_factor=3,showfig=False):
        from scipy import signal
        validation_threshold=  np.std(self.trace) * threshold_factor
        spikes_df=pd.DataFrame(columns=['Sweep','Nb spikes','Spikes time','Iteration'])
        for sweep in range(self.trace.shape[0]):
            print(f"sweep {sweep}")
            if self.trace.ndim ==3:
                for i in range(self.trace.shape[2]):
                    print(f"i {i}")
                    peaks, _ = signal.find_peaks(self.trace[sweep,:,i],height=validation_threshold,distance=50)
                    peaks_time = time[peaks]
                    spikes_df.loc[len(spikes_df)] = [sweep,len(peaks),peaks_time,i]
                    if showfig == True :
                        fig,ax=plt.subplots()
                        ax.plot(time,self.trace[sweep,:,i],color='black',alpha=0.7)
                        ax.scatter(peaks_time,self.trace[sweep,:,i][peaks],color='orange')
                        fig.suptitle(f"Sweep {sweep+1} Iteration {i+1}")
                        if not os.path.exists(url):
                             os.makedirs(url)
                        fig.savefig(f"{url}/Sweep {sweep+1} Iteration {i+1}.pdf")
                        plt.close()
            else:
                peaks, _ = signal.find_peaks(self.trace[sweep,:],height=validation_threshold,distance=50) 
                spikes_df.loc[len(spikes_df)] = [sweep,len(peaks),peaks]
        print(spikes_df['Spikes time'])
        return spikes_df
    def charge (self,sampling_rate, w1=0,w2=0):
        if w1-w2 == 0:
            w1=self.start
            w2=self.stop
        if self.trace.ndim == 1:
            self.q = np.trapz(self.trace[w1:w2],dx=1.0/sampling_rate)
            return self.q
        
        if self.trace.ndim == 2:
            print(w1,w2)
            print('ici mecton')
            self.q = np.trapz(self.trace[:,w1:w2],dx=1.0/sampling_rate,axis=1)
            return self.q
        
    def amplitude (self, w1=0,w2=0 ,data = 0, polarity = 'inward'):
       
        if w1-w2 == 0:
            w1=self.start
            w2=self.stop
        if type(data) == int:
            data=self.trace
            
        if self.trace.ndim == 1:
             if polarity == 'inward':
                '''MIN'''
                self.idx = np.argmin(data[w1:w2]) 
                # self.min_val = np.array([np.mean(self.trace[i[0],int(self.start + idx - 5) : int(self.start + idx + 5)]) for i,idx in np.ndenumerate(self.min_idx)])
                min_peak = np.min(data[w1:w2])
                baseline =  np.mean(data[w1-200:w1-100])
                self.amp = min_peak - baseline
                print('inward 1d')
                return self.amp
            
             if polarity == 'outward':
                '''MAX'''
                self.idx = np.argmax(data[w1:w2]) 
                max_peak = np.max(data[w1:w2])
                baseline =  np.mean(data[w1-200:w1-100])
                self.amp = max_peak - baseline
                print('outward 1d')
                return self.amp
            
        if self.trace.ndim == 2:
            if polarity == 'inward':
                '''MIN'''
                print(w1,w2)
                self.idx = np.argmin(data[:,w1:w2], axis=1) 
                # min_peak = np.min(self.trace[:,w1:w2],axis=1)
                min_peak = np.array([np.mean(data[sweep[0],int(w1 + peak_idx - 10) : int(w1 + peak_idx + 10)]) for sweep,peak_idx in np.ndenumerate(self.idx)])
                baseline =  np.mean(data[:,w1-200:w1-100],axis=1)
                self.amp = min_peak - baseline
                # self.amp = min_peak
                print('inward 2d')
                return self.amp
              
            if polarity == 'outward':
                '''MAX'''
                self.idx = np.argmax(data[:,w1:w2], axis=1) 
                max_peak = np.max(data[:,w1:w2],axis=1)
                baseline =  np.mean(data[:,w1-200:w1-100],axis=1)
                self.amp = max_peak - baseline
                print('outward 2d')
                return self.amp
           
            
        if self.trace.ndim == 3:
            if polarity == 'inward':
                '''MIN'''
                print(w1,w2)
                self.idx = np.argmin(data[:,w1:w2,:], axis=1) 
                print('index =',self.idx.shape)
                # min_peak = np.min(self.trace[:,w1:w2],axis=1)
                min_peak = np.array([np.mean(data[site,int(w1 + self.idx[site,sweep] - 10) : int(w1 + self.idx[site,sweep] + 10),sweep]) for site in range(data.shape[0]) for sweep in range(data.shape[2])] )
                min_peak = min_peak.reshape((data.shape[0],data.shape[2]))
                baseline =  np.mean(data[:,w1-200:w1-100,:],axis=1)
               
                print(self.trace.shape)
                # self.amp = min_peak - baseline
                self.amp = min_peak
                print('inward 3d')
                return self.amp
              
            # if polarity == 'outward':
            #     '''MAX'''
            #     self.idx = np.argmax(self.trace[:,w1:w2], axis=1) 
            #     max_peak = np.max(self.trace[:,w1:w2],axis=1)
            #     baseline =  np.mean(self.trace[:,w1-200:w1-100],axis=1)
            #     self.amp = max_peak - baseline
            #     print('outward 2d')
            #     return self.amp
            # print('outward 2d')
    def std(self,sampling_rate):
       if self.trace.ndim==2:
           self.amp_raw = self.amplitude()
           self.sigma_amp = np.std(self.amp_raw)
           self.q_raw = self.charge(sampling_rate)
           self.sigma_q = np.std(self.q_raw)
           return self.sigma_amp,self.sigma_q
       
    def rs(self,title,url,w1=999,w2=1102,vj=-3): 
        min_peak = np.min(self.traced[:,w1:w2,:],axis=1)
        baseline =  np.mean(self.traced[:,w1-50:w1,:],axis=1)
        ip = min_peak - baseline
        self.r = vj*1000/ip.T
        ravel_r = np.ravel(self.r)
        fig,ax=plt.subplots(1,1)
        idx_r=np.linspace(0,2*len(ravel_r-1),len(ravel_r))
        ax.plot(idx_r,ravel_r)
        fig.savefig(f"{url}/{title}.pdf")
        #pour check que le seal test est bien dans la fenetre
        # fig2,ax2=plt.subplots(1,1)
        # ax2.plot(np.linspace(0,len(self.traced[0,w1:w2,0]-1),len(self.traced[0,w1:w2,0])),self.traced[0,w1:w2,0])
        return ravel_r
    
    def figure(self,time,title,saveurl=0,color='black'):
        
        self.traced_leak = np.mean(self.traced[:,self.start-2000:self.start,:],axis=1)
        self.traced_leak = self.traced_leak[:,np.newaxis,:]
        self.traced_ls = self.traced - self.traced_leak
        # fig,subplot=plt.subplots(1,figsize=[19,13])
        
        # for j in range(self.traced.shape[2]):
        #     subplot.plot(time,self.traced_ls[0,:,j],color='0.7',zorder=1)
    
        # subplot.plot(time,self.trace[0,:],color='red',zorder=2)
      
        # subplot.set_xlabel('Time (s)')
        # subplot.set_ylabel('Amplitude (pA)')
        # fig.suptitle(f"{title} Evoked raw currents and averaged ")
      
        
        fig1,ax1=plt.subplots(6,7,figsize=[19,13])
        
        for i in range(self.traced.shape[0]):
            for j in range(self.traced.shape[2]):
                ax1[i//7,i%7].plot(time[self.start:self.stop],self.traced_ls[i,self.start:self.stop,j],color='0.7',zorder=1)
        
            ax1[i//7,i%7].plot(time[self.start:self.stop],self.trace[i,self.start:self.stop],color='red',zorder=2)
            # subplot[i//7,i%7].scatter(time[self.start+self.idx],self.amp,color='black',zorder=3)
            ax1[i//7,i%7].set_xlabel('Time (s)')
            ax1[i//7,i%7].set_ylabel('Amplitude (pA)')
        
           
        min_peak = np.min(self.traced[:,self.start:self.stop,:],axis=1)
        baseline =  np.mean(self.traced[:,self.start-200:self.start-100,:],axis=1)
        self.raw_amp = min_peak - baseline   
        fig2,ax2 = plt.subplots(6,7,sharex=True,sharey=True,figsize=(19,13))
        fig3,ax3 = plt.subplots(6,7,sharex=True,sharey=True,figsize=(19,13))
        for (row,col),sweep in np.ndenumerate(np.arange(self.traced.shape[0]).reshape((6,7))):#a opti pour le reshape
            ax3[row,col].boxplot(self.raw_amp[sweep,:])
            for i in range(self.traced.shape[2]):
                ax1[row,col].set_title(str(sweep+1))
                ax2[row,col].scatter(i,self.raw_amp[sweep,i],color=color)
                ax2[row,col].set_title(str(sweep+1))
                ax3[row,col].set_title(str(sweep+1))
                if col==0:
                    ax2[row,col].set_ylabel('Amplitude (pA)')
                    
        fig1.suptitle(f"{title} Evoked raw currents and averaged ")            
        fig2.suptitle(f"{title}")
        fig3.suptitle(f"{title}")
        if saveurl:
            fig1.savefig(f"{saveurl} courants bruts.pdf")
            fig2.savefig(f"{saveurl} amp raw.pdf")
            fig3.savefig(f"{saveurl} amp raw bam.pdf")
        # plt.close()   
        return self.raw_amp
        
    def zscore (self,noise_start,noise_stop,amplitude = 0,aorq='amp',sampling_rate=0, sigma_method='std_bis',bins=25,plot=False,polarity='inward'): 
        '''Get the value of the noise'''
        print(polarity)
        if aorq == 'amp':
            self.noise = self.amplitude(w1=noise_start,w2=noise_stop,polarity = polarity) 
        else:
            self.noise = self.charge(sampling_rate,noise_start,noise_stop)
        
        self.xbarre = np.mean(self.noise)
       
        
        '''Get the sigma of the noise'''
        if sigma_method == 'std':
            import noiseFit 
            self.sigma = noiseFit.noiseFit(self.noise,bins=bins,plot=plot) 
            print(self.sigma)
        elif sigma_method == 'std_bis':
            self.sigma = np.std(self.noise) 
        elif sigma_method == 'mad':
            self.sigma = MAD(self.noise)
            print(self.sigma)
        
        '''Get the evoked value eventually'''
        if aorq == 'amp':
            self.evok = amplitude 
            if not amplitude.any():
                self.evok = self.amplitude(polarity = polarity) 
        else:
            self.evok = amplitude
            if not amplitude.any():
                self.evok = self.charge(sampling_rate=sampling_rate)
        '''Zscore and Threshold'''
        self.zscore_grid = np.abs((np.abs(self.evok)-np.abs(self.xbarre))/np.abs(self.sigma))
        self.threshold = np.abs(self.sigma)*3+np.abs(self.xbarre)
      
        
        '''Output'''
        self.output_dic = {}       
        self.output_dic.update({'amp evok': [self.evok]}) 
        self.output_dic.update({'amp noise': [self.noise]}) 
        self.output_dic.update({'mean noise': [np.full_like(self.evok,self.xbarre)]})      
        self.output_dic.update({'std noise': [np.full_like(self.evok,self.sigma)]})   
        self.output_dic.update({'zscore threshold': [np.full_like(self.evok,self.threshold)]})   
        self.output_dic.update({'zscore amp': [self.zscore_grid]})
               
        return self.zscore_grid,self.output_dic   
   
    
# def zscore (data,sigma,mean): 
#     zscore_grid = (np.abs(data)-np.abs(mean))/np.abs(sigma)
#     return zscore_grid

def map_builder(data,map_grid):
    array_map = np.zeros((map_grid.shape))
    for (row,col),sweep in np.ndenumerate(map_grid-1):
        array_map[row,col]=data[sweep]
    return array_map

def big_matrix(arrays, grid_order):
    for i,grid in np.ndenumerate(grid_order):
        if i[0] == 0:
            the_matrix = arrays[grid-1]
        if i[0] != 0:
            the_matrix = np.concatenate((the_matrix,arrays[grid-1]),axis=1)   
    return the_matrix

def flipped_matrix(data,flip):
    if flip == 1:
        f_matrix = np.fliplr(data)       
    elif flip == 4 or flip == '3 ou 4':
        f_matrix = np.flip(data, axis=0)       
    elif flip == 3:
        f_data = np.flip(data, axis=1)
        f_matrix = np.flip(f_data, axis=0)       
    else:
        f_matrix = data
    return f_matrix

def h1_mask (data,threshold):
    mask = data >= threshold
    binary =np.zeros((mask.shape))
    for (i,j),x in np.ndenumerate(mask):
       if x == True:
           binary[i,j] = 1.0
       else:
           binary[i,j] = 0.0
    return binary

'''MAPS PART2'''

#info immuno

class Immuno:
    def __init__(self, xl_file, sheet_name):
        self.df=pd.read_excel(rf"{xl_file}", index_col = 0, sheet_name= sheet_name)
        
    def grid_order(self, cell_name):
        self.raw_order = int(self.df.loc[cell_name,'Ordre des grilles'])
        self.order = [int(i) for i in str(int(self.raw_order))]
        return self.order
    
    def orientation(self, cell_name):
        self.flip = self.df.loc[cell_name,'Orientation patch']
        return self.flip    
    def cell_pos(self, cell_name):
        self.cell = self.df.loc[cell_name,'Position de la cellule (numéro de case)']
        return self.cell
    
    def map_position (self,cell_name,square=20.0,len_row=7,nb_grids=4):
        self.index_cell = np.where(self.df.index == cell_name) 
        self.cell_norm = self.df.iloc[self.index_cell[0][0]+1,:]
        self.P1mI_microns = self.df.loc[cell_name,"P1- ipsi"]
        self.P2mC = self.cell_norm.loc["P2- contra"]
        self.P2pC = self.cell_norm.loc["P2+ contra"]
        self.P1mC = self.cell_norm.loc["P1- contra"]
        self.P1p = self.cell_norm.loc["P1+"]
        self.P1mI = 100
        self.P2pI = self.cell_norm.loc["P2+ ipsi"]
        self.P2mI = self.cell_norm.loc["P2- ipsi"]
        self.PC = self.cell_norm.loc["Pos_cell (à partir de P1+)"] 
        
        pos_cell = np.arange(0,len_row*nb_grids)[int(self.cell_pos(cell_name))]
        
        #Bord de la cart en nombre de site de stim 
        map_left_side = pos_cell+0.5
        map_right_side = len_row*nb_grids-map_left_side
        
        flip = self.orientation(cell_name)
        
        if flip == 1 or flip == 3:
            map_right_side = pos_cell+0.5
            map_left_side = len_row*nb_grids-map_right_side
           
                
        #Bord de la carte en %
        self.left_side = map_left_side*square*100/self.P1mI_microns
        self.right_side = map_right_side*square*100/self.P1mI_microns
        
        #POSITION ARRAY 
        positions_pc_norm = np.linspace(self.PC-self.left_side,self.PC+self.right_side,len_row*nb_grids)
       
        return positions_pc_norm
        
'''MAPS PART3'''

def cumsum_1d (path,save=True,showfig=True,saveurl=0):
   
    url_xl = path
     
    maps_1d_e = pd.read_excel(url_xl,sheet_name='Amp E 1D',index_col = 0).values.ravel()/1000
    maps_1d_i = pd.read_excel(url_xl,sheet_name='Amp I 1D',index_col = 0).values.ravel()/1000
    x = pd.read_excel(url_xl,sheet_name='Positions pc norm',index_col = 0).values.ravel()
    cumsum_e = np.cumsum(maps_1d_e)
    cumsum_i = np.cumsum(maps_1d_i)
        
    fig,subplot=plt.subplots(2,1, figsize=[19,13])
    subplot[0].plot(x,cumsum_e,color='seagreen')
    subplot[1].plot(x,cumsum_i,color='cornflowerblue')
    subplot[1].set_xlabel('Distance to midline (% of P1-)')
    subplot[0].set_ylabel('Cumulative synaptic weight (nA)')
    subplot[1].set_ylabel('Cumulative synaptic weight (nA)')
    
    if save == True: 
         plt.savefig(f"{saveurl}\Cumulative synaptic weight.pdf")
         
    if showfig == False:
        plt.close()


# zebrine = pd.read_excel(r'//equipe2-nas1/Theo.GAGNEUX/Theo.GAGNEUX/Data 1-P/Maps vermis/exploitable/Cluster 1/Distance-zébrines v2.xlsx',index_col =0,sheet_name='Cl sym 100µM RuBi')  
  #   print(zebrine) 
    
  #   flip = zebrine[2*index,11]
  #   pos_cell = zebrine[2*index,13]   
