# -*- coding: utf-8 -*-
"""
Code Developed by Pedro Gomes for Myelin-H company

Windows 11
MNE 1.2.1
Python 3.9

"""

#################### Import #######################
import mne
import scipy.signal as signal
import epochs
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "browser"
from mne.io.constants import FIFF
from scipy.signal import find_peaks

################## Auxiliary Function #############
def numpy_to_evoked(data,channels,sfreq,tmin):
    """
    Convert Evoked.array into MNE Epochs Structure
    
    Parameters
    ----------
    data : Evoked data in array
    channels : List of Strings
    sfreq : Float (Hz)
    tmin : Float (s)

    Returns
    -------
    evoked : Evoked in MNE Structure.

    """
    info = mne.create_info(channels,sfreq,len(channels)*["eeg"])
    info.set_montage('standard_1020')
    evoked = mne.EvokedArray(data, info, tmin=tmin)
    return evoked

################### Evoked #######################################
class Evoked():
    #Class for Evoked handling and processing
    def __init__(self, evoked, tmin_e, csd):
        self.evoked_mne=evoked #evoked in MNE structure
        self.data=evoked.get_data() #evoked in numpy structure
        self.sfreq = evoked.info["sfreq"] #sampling frequency
        self.channels = evoked.info["ch_names"] #names of the channels
        self.tmin=tmin_e #tmin evoked
        self.csd=csd #heritage csd application info
    
    def psd_plot(self):
        """
        PSD plot for the evoked data

        """
        self.evoked_mne.compute_psd().plot()
    def timeseries_topo_plot(self):
        """
        MNE plot_topo

        """
        self.evoked_mne.plot_topo(color=None)
    
    def power_topographic_plot(self,time=None):
        """
        Computes the power topographic plot at the specified times
        
        Parameters
        ----------
        time : List of Floats, optional
            Times for computing the plots. The default is None.
        """
        evoked_copy=self.evoked_mne.copy()
        
        if self.csd==True: #doesnt recognize csd coil so we have to change to EEG
            with  evoked_copy.info._unlock():
                evoked_copy.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
            for i in range(len(evoked_copy.info['chs'])):
                evoked_copy.info['chs'][i].update(coil_type=FIFF.FIFFV_COIL_EEG,unit=FIFF.FIFF_UNIT_V_M2)
        if time==None:
            evoked_copy.plot_topomap(times="peaks", ch_type='eeg', cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4,colorbar=False)
        else:
              evoked_copy.plot_topomap(times=time, ch_type='eeg', cmap='Spectral_r', res=32,
                           outlines='skirt', contours=4,colorbar=False) 
            
    def image_plot(self,picks=None):
        """
        Plot MNE Image Plot over the evoked
        
        Parameters
        ----------
        picks : List of Strings, optional
           List of the Channels to compute the image plot. The default is None which is all tge channels.
        """
        self.evoked_mne.plot_image(picks=picks)
        
    def evoked_plot(self,picks=None):
        """
        Plot MNE evoked.plot()

        """
        self.evoked_mne.plot(picks=picks,spatial_colors=True, gfp=True)
    
    def joint_plot(self,times="peaks"):
        """
        Plot MNE joint_plot()

        """
        self.evoked_mne.plot_joint(times=times)
        
    def Band_Topomap(self, band=None, time=None):
        """
        Plot topomaps for each brainwave band or a specific band on over the specified times

        Parameters
        ----------
        band : Str, optional
            Name of the band to be computed (Delta,Theta,Alpha,Beta,Gamma). 
            The default is None, case in which all bands topomap is computed.
        time : List of Floats, optional
            Times where the topomap shall be computed. The default is None, where the time=="peaks".
        """
        if time==None:
            time="peaks"
        alpha=[8,12]
        theta=[4,8]
        beta=[12,30]
        delta=[1,4]
        gamma=[30,70]
        bands=[delta,theta,alpha,beta,gamma]
        if band==None: #Case where all the bands topomaps are computed
            bands_filtered=[]
            for e in bands:
                evoked_copy=self.data.copy()
                sos= signal.butter(4, e, btype='bandpass', fs=self.sfreq, output="sos")
                filtered = signal.sosfilt(sos, evoked_copy)
                filtered_mne=numpy_to_evoked(filtered,self.channels,self.sfreq,self.tmin)
                bands_filtered.append(filtered_mne)
            bands_filtered[0].plot_topomap(times=time, ch_type='eeg', title="Delta",colorbar=False)
            bands_filtered[1].plot_topomap(times=time, ch_type='eeg', title="Theta",colorbar=False)
            bands_filtered[2].plot_topomap(times=time, ch_type='eeg', title="Alpha",colorbar=False)
            bands_filtered[3].plot_topomap(times=time, ch_type='eeg', title="Beta",colorbar=False)
            bands_filtered[4].plot_topomap(times=time, ch_type='eeg', title="Gamma",colorbar=False)             
        else: #case where only a specific band topomap is computed
            if band=="Delta":
                evoked_copy=self.data.copy()
                sos= signal.butter(4, delta, btype='bandpass', fs=self.sfreq, output="sos")
                filtered = signal.sosfilt(sos, evoked_copy)
                filtered_mne=numpy_to_evoked(filtered,self.channels,self.sfreq,self.tmin)
                filtered_mne.plot_topomap(times=time, ch_type='eeg', title="Delta",colorbar=False)
            elif band=="Theta":
                evoked_copy=self.data.copy()
                sos= signal.butter(4, theta, btype='bandpass', fs=self.sfreq, output="sos")
                filtered = signal.sosfilt(sos, evoked_copy)
                filtered_mne=numpy_to_evoked(filtered,self.channels,self.sfreq,self.tmin)
                filtered_mne.plot_topomap(times=time, ch_type='eeg', title="Theta",colorbar=False)
            elif band=="Alpha":
                evoked_copy=self.data.copy()
                sos= signal.butter(4, alpha, btype='bandpass', fs=self.sfreq, output="sos")
                filtered = signal.sosfilt(sos, evoked_copy)
                filtered_mne=numpy_to_evoked(filtered,self.channels,self.sfreq,self.tmin)
                filtered_mne.plot_topomap(times=time, ch_type='eeg', title="Alpha",colorbar=False)
            elif band=="Beta":
                evoked_copy=self.data.copy()
                sos= signal.butter(4, beta, btype='bandpass', fs=self.sfreq, output="sos")
                filtered = signal.sosfilt(sos, evoked_copy)
                filtered_mne=numpy_to_evoked(filtered,self.channels,self.sfreq,self.tmin)
                filtered_mne.plot_topomap(times=time, ch_type='eeg', title="Beta",colorbar=False)
            else:
                evoked_copy=self.data.copy()
                sos= signal.butter(4, gamma, btype='bandpass', fs=self.sfreq, output="sos")
                filtered = signal.sosfilt(sos, evoked_copy)
                filtered_mne=numpy_to_evoked(filtered,self.channels,self.sfreq,self.tmin)
                filtered_mne.plot_topomap(times=time, ch_type='eeg', title="Gamma",colorbar=False)
    
    def gfp(self):
        """
        GFP of the evoked object
        
        Returns:
            GFP array of the Evoked data across channels
        """
        mean=np.mean(self.data) #common average
        new_evoked=self.data-mean
        gfp=(new_evoked**2)
        gfp_squared = np.sum(gfp, axis=0)
        gfp_root=np.sqrt(gfp_squared)/len(self.channels)
        return gfp_root
    
    def mean(self):
        """
        Mean of the evoked object
        
        Returns:
           Mean array of the Evoked data across channels
        """
        mean_evoked=np.mean(self.data, axis=0)
        return mean_evoked
    
    def median(self):
        """
        Median of the evoked

        Returns:
           Median array of the Evoked data across channels
        """
        median_evoked=np.median(self.data, axis=0)
        return median_evoked      
    
def gfp_peaks(gfp,number_of_peaks, tmin, sfreq):
    """
    Finds the three maximum peaks of a gfp array
    
    Parameters
    ----------
    gfp : np.array
        global field power.
    number_of_peaks : int 
        Number of the maximum peaks to display.
    tmin : float
        time in seconds.
    sfreq : int
        Sampling freauency.

    Returns
    -------
    peaks_list : list of tuples
        list of len = number_of_peaks and with [:,0] = amplitude and [:,1]= onset.

    """
    srate=1/sfreq
    peaks = find_peaks(gfp,distance=20)[0]
    amp=gfp[peaks]
    peaks_t=[]
    for i in range(len(peaks)):
        peaks_t.append(np.round(tmin+srate*peaks[i],decimals=3))
    peaks_list=sorted(zip(amp, peaks_t), reverse=True)[:number_of_peaks]
    # plt.plot(gfp)
    # plt.plot(peaks, gfp[peaks], "x")
    # plt.show()
    return peaks_list
    
def evoked_comparison(evoked_1,evoked_2, method="gfp"):
    """
    Plots both evoked groups combined responses
    
    Parameters
    ----------
    evoked_1 : Class Evoked
    evoked_2 : Class Evoked
    method : Str, optional
       Method to combine and compare evoked data. The default is "gfp", yet "mean" and "median" comparison are available.

    Returns
    -------
    dif : Np.Array
        Array resulting from the difference of both Evoked responses

    """
    tmin=evoked_1.tmin
    sfreq=evoked_1.sfreq
    t_step=1/sfreq
    if method=="gfp": #when gfp method
        evoked_1_gfp=evoked_1.gfp()
        evoked_2_gfp=evoked_2.gfp()
        duration=len(evoked_1_gfp)*t_step
        timeseries=np.arange(tmin,tmin+duration,t_step) #rebuild timeseries for plot
        dif=evoked_1_gfp-evoked_2_gfp
        fig=plt.figure()
        plt.title('Evoked Comparison through Global Field Power')
        plt.plot(timeseries, evoked_1_gfp, label = "Evoked_1") #plot both evoked data in same fig
        plt.plot(timeseries,evoked_2_gfp, label = "Evoked_2")
        plt.xlabel("Time(s)")
        plt.ylabel("gfp (V)")
        plt.legend()
        plt.show()

        return dif
    elif method=="mean": #when mean method
        evoked_1_mean=evoked_1.mean()
        evoked_2_mean=evoked_2.mean()
        duration=len(evoked_1_mean)*t_step
        timeseries=np.arange(tmin,tmin+duration,t_step)#rebuild timeseries for plot
        dif=evoked_1_mean-evoked_2_mean
        fig=plt.figure()
        plt.title('Evoked Comparison through Mean behaviour across channels')
        plt.plot(timeseries, evoked_1_mean, label = "Evoked_1") #plot both evoked data in same fig
        plt.plot(timeseries,evoked_2_mean, label = "Evoked_2")
        plt.xlabel("Time(s)")
        plt.ylabel("Mean(V)")
        plt.legend()
        plt.show()

        return dif
    else: #when median method
        evoked_1_median=evoked_1.median()
        evoked_2_median=evoked_2.median()
        duration=len(evoked_1_median)*t_step
        timeseries=np.arange(tmin,tmin+duration,t_step)#rebuild timeseries for plot
        dif=evoked_1_median-evoked_2_median
        fig=plt.figure()
        plt.title('Evoked Comparison through median behaviour across channels')
        plt.plot(timeseries, evoked_1_median, label = "Evoked_1") #plot both evoked data in same fig
        plt.plot(timeseries,evoked_2_median, label = "Evoked_2")
        plt.xlabel("Time(s)")
        plt.ylabel("Median(V)")
        plt.legend()
        plt.show()

        return dif
    
    
def topological_evoked_dif(evoked_1,evoked_2):
    """
    MNE plot_topo from the difference of the evoked one and evoked two
    
    Return:
        Evoked Class diference data

    """
    dif=evoked_1.data-evoked_2.data
    dif_evoked=numpy_to_evoked(dif,evoked_1.channels,evoked_1.sfreq,evoked_1.tmin)        
    dif_evoked=Evoked(dif_evoked,evoked_1.tmin,evoked_1.csd)
    dif_evoked.timeseries_topo_plot()
    return dif_evoked


    