# -*- coding: utf-8 -*-
"""
Code Developed by Pedro Gomes for Myelin-H company

Windows 11
MNE 1.2.1
Python 3.9

"""

########### Imports ###################
import mne
import mne.viz
import glob
import os
import numpy as np
import scipy.signal as signal
from mne.io.constants import FIFF
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import itertools
import collections
from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator
from .BodyMirror import BodyMirror

######### Functions ###################
def string_checker(channels_list, modality):
    """
    Function to obtain indexes of channels belonging to specific regions
    
    Parameters
    ----------
    channels_list : List of str
        List of all the Recording Channels.
    modality : str
        Region of interest, can be 'rh' for right hemisphere, 'lh' for left hemisphere, 'pos' for posterior region and 'ant' for anterior region.

    Returns
    -------
    res : list of ints
        List of the indexes of the channels.

    """
    if modality=="rh": #right hemisphere
        list_status=[]
        for string in channels_list:
            status=False
            for e in string:
                try:
                    n=int(e)
                    if n % 2 == 0: #looking if the number of the channel is even
                       status=True
                       break
                    else:
                       status=status
                except: 
                    None
            list_status.append(status)
        res = [i for i, val in enumerate(list_status) if val]
        return res
    elif modality=="lh": #left hemisphere
        list_status=[]
        for string in channels_list:
            status=False
            for e in string:
                try:
                    n=int(e)
                    if n % 2 != 0: #looking if the number of the channel is odd
                       status=True
                       break
                    else:
                       status=status
                except: 
                    None
            list_status.append(status)
        res = [i for i, val in enumerate(list_status) if val]
        return res
    elif modality=="ant": #anterior
        list_status = [True if i[0] == "F" or i[0] == "A" else False for i in channels_list] #frontal or AF channels
        res = [i for i, val in enumerate(list_status) if val]              
        return res    
    else:
        list_status = [True if i[1] == "P" or i[0] == "P" or i[0]=="O" or i[0]=="I" else False for i in channels_list] #check if the channel is in parietal, occipital or inion area 
        res = [i for i, val in enumerate(list_status) if val]              
        return res
  

def read_epochs_set(filepath):
    """
    Read epochs saved in .set file
    
    Parameters
    ----------
    filepath : Path for the .set file of interest containing the epoch data.

    Returns
    -------
    epochs : Returns the epoch data in MNE structure.

    """
    epochs=mne.read_epochs_eeglab(filepath)
    return epochs


#Notch filtering  - Q factor=8
def notch_filter(epochs_array,cut_freq,sfreq):
    """
    Apply notch filter on an Epoch array
    
    Parameters
    ----------
    epochs_array : Epochs of Numpy.Array type.
    cut_freq : Float (Hz).
    sfreq : Float (Hz)

    Returns
    -------
    outputsignal_notch : Epochs of Numpy.Array type.
    """
    b_notch, a_notch=signal.iirnotch(cut_freq, 8, sfreq)
    outputsignal_notch = signal.filtfilt(b_notch, a_notch, epochs_array)
    return outputsignal_notch

#Bandpass Filtering
def bp_filter(epochs_array, sfreq, l_freq, h_freq):
    """
    Apply Bandpass filter on an Epoch array
    
    Parameters
    ----------
    epochs_array : Epochs of Numpy.Array type.
    sfreq : Float (Hz)
    l_freq : Float (Hz)
    h_freq : Float (Hz)

    Returns
    -------
    outputsignal_bp : Epochs of Numpy.Array type.
    
    """
    outputsignal_bp=mne.filter.filter_data(data=epochs_array, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,h_trans_bandwidth=1)
    return outputsignal_bp

#Highpass Filtering
def hp_filter(epochs_array, sfreq, l_freq):
    """
    Apply highpass filter on an Epoch array

    Parameters
    ----------
    epochs_array : Epochs of Numpy.Array type
    sfreq : Float (Hz)
    l_freq : Float (Hz)
    
    Returns
    -------
    outputsignal_hp : Epochs of Numpy.Array type
    """
    outputsignal_hp=mne.filter.filter_data(data=epochs_array, sfreq=sfreq, l_freq=l_freq, h_freq=None)
    return outputsignal_hp

#Lowpass Filtering
def lp_filter(epochs_array, sfreq, h_freq):
    """
    Apply lowpass filter on an Epoch array
    
    Parameters
    ----------
    epochs_array : Epochs of Numpy.Array type
    sfreq : Float (Hz)
    h_freq : Float (Hz)
    h_trans_bandwidth : Float (Hz)

    Returns
    -------
    outputsignal_lp : Epochs of Numpy.Array type
    """
    outputsignal_lp=mne.filter.filter_data(data=epochs_array, sfreq=sfreq, l_freq=None, h_freq=h_freq,h_trans_bandwidth=1)
    return outputsignal_lp


def numpy_to_epochs(data,ch_names,sfreq,tmin,events=None, event_id=None):
    """
    Convert epochs.array into MNE Epochs Structure
    
    Parameters
    ----------
    data : Epochs of Numpy.Array type
    channels :List of Strings
    sfreq : Float (Hz)
    tmin : Float (s)
    events : List of events as ndarray, optional. The default is None. 
    event_id : event_id dict, optional. The default is None. 

    Returns
    -------
    epochs : Epochs in MNE Structure
    """
    info=mne.create_info(ch_names, sfreq, ch_types=len(ch_names)*['eeg'])
    info.set_montage('standard_1020')
    epochs = mne.EpochsArray(data, info,tmin=tmin,events=events, event_id=event_id)
    return epochs

def numpy_to_evoked(data,channels,sfreq,tmin):
    """
    Convert epochs.array into MNE Epochs Structure
    
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

################# Epochs ###################################
class Epochs():
    #Class for Epoch handling, preprocessing and processing
    
    def __init__(self, data,csd_status=None):
        self.data = data.get_data() #epochs in numpy.array
        self.sfreq = data.info["sfreq"] #sampling frequency
        self.channels = data.info["chs"] #list of channels
        self.events=data.events #events
        self.event_id=data.event_id#events id in epochs
        self.info=data.info #epochs info
        self.tmin =data.tmin #epochs starting time
        self.epochs_mne=data #epochs in mne
        self.ch_names=data.info["ch_names"] #names of the channel
        self.filtered_status=None #status of filtering
        self.ica_status=None #info about wether an ICA was performed
        self.ica=None #ica
        self.ica_n_comps=None #ica number of components
        self.csd=csd_status #info about wether a csd was performed
        self.drop_ch=None #drop_channels_info
        self.drop_ind=None #drop_ind_info
        self.auto_comps=None #components for automatic removal
        if self.csd==True:
            with  self.epochs_mne.info._unlock():
                self.epochs_mne.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
            for i in range(len(self.epochs_mne.info['chs'])):
                self.epochs_mne.info['chs'][i].update(coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                      unit=FIFF.FIFF_UNIT_V_M2)
    def average_reference(self):
        """
        Re-references the Epochs to average, modifies data in place.
        
        Returns
        -------
        self : Modified Epochs
            Common average re-referenced epochs
        """
        new_epochs=[]
        for i in range(len(self.data)):
            mean=np.mean(self.data[i]) #common average
            new_epochs.append(self.data[i]-mean)
        self.data=np.array(new_epochs)
        self.epochs_mne=numpy_to_epochs(self.data,self.ch_names,self.sfreq,self.tmin,self.events,self.event_id)
        return self
    
    def filtering_preprocessing(self,filters,n_cut_freq=None,l_freq=None,h_freq=None ):
        """
        Filtering Process to the data. From Epochs object, modifies the object according to the selected filters. Updatews filtered status of the object
        
        Parameters
        ----------
        filters : List of str. Filters are encoded by n, for notch, bp, for bandpass, hp, for highpass, lp, for lowpass.
            When a list is inputed it should be already sorted by order of application.

        Output
        -------
        Modifies self.data in-place and returns the modified object

        """
        sfreq=self.sfreq
        epochs_array=self.data
        epochs_filt=epochs_array
        for e in filters:
            if e=="n":
                cut_freq=n_cut_freq
                epochs_filt=notch_filter(epochs_filt,cut_freq,sfreq)
            elif e=="bp":
                l_freq=l_freq
                h_freq=h_freq
                epochs_filt=bp_filter(epochs_filt, sfreq, l_freq, h_freq)
            elif e=="lp":
                h_freq=h_freq
                epochs_filt=lp_filter(epochs_filt, sfreq, h_freq)
            else:
                l_freq=l_freq
                epochs_filt=hp_filter(epochs_filt, sfreq, l_freq)
        
        self.data=epochs_filt     
        self.epochs_mne=numpy_to_epochs(self.data,self.ch_names,self.sfreq,self.tmin,self.events,self.event_id)
        if self.csd==True:
            with  self.epochs_mne.info._unlock():
                self.epochs_mne.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
            for i in range(len(self.epochs_mne.info['chs'])):
                self.epochs_mne.info['chs'][i].update(coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                      unit=FIFF.FIFF_UNIT_V_M2)
        self.filtered_status=True
        
        return self
        
    def average_epochs(self):
        """
        Calculates the Average Epoch
        
        Returns
        -------
        evoked : Evoked MNE
        """
        mean=np.mean(self.data, axis=0)
        evoked_resp=mne.EvokedArray(mean, self.info, self.tmin)
        if self.csd==True: #correct info on Evoked after csd implementation because default considers that no csd was applied
            with  evoked_resp.info._unlock():
                evoked_resp.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
            for i in range(len(evoked_resp.info['chs'])):
                evoked_resp.info['chs'][i].update(coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                      unit=FIFF.FIFF_UNIT_V_M2)
        return evoked_resp

    def image_plot(self,combine=None, vmin=None,vmax=None,picks=None):
        """
        Plot Image across epochs
        
        Parameters
        ----------
        combine: string, 
            Method to combine different channels ‘mean’, ‘median’, ‘std’ (standard deviation) or ‘gfp’ (global field power). If None gfp is used
        vmin : Float, optional
            The default is None. So it will be automatically defined according to the minimum
        vmax : Float, optional
            The default is None. So it will be automatically defined according to the maximum
        picks : List of Strings, optional
           List of Channels. The default is None, and computes for all the channels

        """
        mne.viz.plot_epochs_image(self.epochs_mne, picks=picks, vmin=vmin, vmax=vmax,combine=combine)
        
    def psd_plot(self):
        """
        Average Psd plot of the epochs
        
        """
        self.epochs_mne.compute_psd().plot()
    
    def epochs_plot(self,picks=None, scalings=None):
        """
        Simple plot for the Epochs object
        
        Parameters
        ----------
        picks : List of Str, optional
            Channels to plot. The default is None, which plots all channels.
        scalings : float(microvolts), optional.
            The default is 20 microvolts.
        """
        self.epochs_mne.plot(picks=picks, scalings=None)
                
    def psd_topo_plot(self):
        """
        Plots topomap of the psd-Topomap over all the brainwave bands
        
        Order of the plots= delta, theta, alpha,beta,gamma
        """
        # defining each band
        alpha=[8,12]
        theta=[4,8]
        beta=[12,30]
        delta=[1,4]
        gamma=[30,70]
        bands=[delta,theta,alpha,beta,gamma]
        bands_filtered=[]
        for e in bands: #filter the data for each band
            print(e)
            epoched_copy=self.data.copy()
            sos= signal.butter(4, e, btype='bandpass', fs=self.sfreq, output="sos")
            filtered = signal.sosfilt(sos, epoched_copy)
            filtered_mean=np.mean(filtered,axis=0) #mean across epochs
            filtered_mean_2=np.mean(filtered_mean,axis=1) #mean across each channel
            bands_filtered.append(filtered_mean_2)
        mne.viz.plot_topomap(bands_filtered[0],self.info,ch_type='eeg',names=self.ch_names)
        mne.viz.plot_topomap(bands_filtered[1],self.info,ch_type='eeg',names=self.ch_names)       
        mne.viz.plot_topomap(bands_filtered[2],self.info,ch_type='eeg',names=self.ch_names)
        mne.viz.plot_topomap(bands_filtered[3],self.info,ch_type='eeg',names=self.ch_names)
        mne.viz.plot_topomap(bands_filtered[4],self.info,ch_type='eeg',names=self.ch_names)
    
    def topo_plot(self):
        """

        Plot the mean topomap of the epochs

        """
        mean_epochs=np.mean(self.data,axis=0)
        mean_epochs_2=np.mean(mean_epochs,axis=1)
        mne.viz.plot_topomap(mean_epochs_2,self.info,ch_type='eeg',names=self.ch_names)

    def morlet_tfr(self,fmin,fmax,tmin=None,tmax=None,power_plot=None,joint_plot=None, itc_plot=None, timefreqs=None, baseline=None):
        """
        Method which computes tfr analysis plots. Atleast 1s of data should be introduced, for that its adviced to cut 0.5s before and after stimuli"

        Parameters
        ----------
        fmin : float (Hz)
        fmax : float (Hz)
        tmin : float (s)
        tmax : float (s)
        power_plot : str, optional
            Input y to plot tfr power plot. The default is None.
        joint_plot : str, optional
            Input y to plot joint plot. The default is None.
        itc_plot : str, optional
            Input y to plot itc topo plot. The default is None.
        timefreqs : tuple of tuples e.g ((1, 10), (.3, 3)), for joint plot computation
            The default is None.
        baseline: tuple with (tmin, tmax) for baseline.
        """
        
        freqs = np.logspace(*np.log10([fmin, fmax]), num=8) #compute frequencies in the log domain
        n_cycles = freqs / 2.  # different number of cycle per frequency
        power_p, itc_p = mne.time_frequency.tfr_morlet(self.epochs_mne, freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=True, decim=3, n_jobs=1)
        power_p.plot_topo(baseline=baseline, mode='logratio', title='Average power')
        if power_plot=="y":
            if self.csd==True: #plot_topomap of averageTFR doesnt consider csd so i changed the way it interprets the coils
                with  power_p.info._unlock():
                    power_p.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
                for i in range(len(power_p.info['chs'])):
                    power_p.info['chs'][i].update(coil_type=FIFF.FIFFV_COIL_EEG,
                                              unit=FIFF.FIFF_UNIT_V_M2)
                power_p.plot_topomap(baseline=baseline, ch_type='eeg', fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, mode='logratio')
            else:
                power_p.plot_topomap(baseline=baseline, ch_type='eeg', fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, mode='logratio')
        if joint_plot=="y":
            try:
                power_p.plot_joint(baseline=baseline, tmin=tmin, tmax=tmax, mode='mean',timefreqs=timefreqs)
            except:
                print("Maximum peak is not in the selected interval please consider to use all the acquisition")
        if itc_plot=="y":
            itc_p.plot_topo(title='Inter-Trial coherence',vmin=0, vmax=1,cmap='Reds')
    
    def crop_epochs(self, tmin_crop, tmax_crop):
        """
        Method to crop the epochs. Updates the event of the pulse the tmin of the data, modifies in place. Adviced to do copy before applying crop.

        Parameters
        ----------
        tmin_crop : float (s)
        tmax_crop : float (s)

        Output
        -------
        Epoch object cropped and with events updated

        """

        epochs_array=self.data
        s_rate=1/self.sfreq
        if self.tmin<0 and tmin_crop<=0: 
            start_delay=abs(self.tmin)-abs(tmin_crop)
        elif self.tmin<0 and tmin_crop>0:
            start_delay=abs(self.tmin)+abs(tmin_crop)
        else:
            start_delay=abs(tmin_crop)-abs(self.tmin) 
        start_delay=np.round(start_delay, decimals=3) #solve some numerical comparison problems from python
        crop_begin=int(start_delay/s_rate) #onset
        duration=int(np.round((tmax_crop-tmin_crop)/s_rate,decimals=1))
        epochs_crop=[]
        for i in range(len(epochs_array)):
            channels_crop=[]
            for j in range(len(epochs_array[i])):
                channels_crop.append(epochs_array[i][j][crop_begin:crop_begin+duration]) #+1 because of the python indexing
            epochs_crop.append(channels_crop)
        self.data=np.array(epochs_crop)
        self.tmin=tmin_crop
        events_to_change=self.events
        for i in range(len(events_to_change)):
            if tmin_crop<=0 and tmax_crop>=0:
                events_to_change[i][0]=abs(tmin_crop)/s_rate+1+duration*i
            else:
                events_to_change[i][0]=0+duration*i+1
        self.events=events_to_change
        self.epochs_mne=numpy_to_epochs(self.data,self.ch_names,self.sfreq,self.tmin,events_to_change,self.event_id)
        if self.csd==True:
            with  self.epochs_mne.info._unlock():
                self.epochs_mne.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
            for i in range(len(self.epochs_mne.info['chs'])):
                self.epochs_mne.info['chs'][i].update(coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                      unit=FIFF.FIFF_UNIT_V_M2)
        return self
    
    def baseline_correction(self, split_epoch_for_baseline_correction=1):
        """
        Baseline Correction for the epochs object. Modifies in-place.       

        Parameters
        ----------
        split_epoch_for_baseline_correction : Int, optional
            Parameter for the epoch spliting for baseline correction. The default is None, 
            no split is done and the entire epoch channel's mean is considered for the baseline correction.

        Returns
        -------
        Modifies in place the data for epoch

        """
        epochs_array=self.data
        epochs_bc=[]
        for i in range(len(epochs_array)):
            channels_bc=[]
            for j in range(len(epochs_array[i])):
                array_to_bc=epochs_array[i][j]
                split_array=np.array_split(array_to_bc,split_epoch_for_baseline_correction)
                array_rec=[]
                for e in split_array: 
                    mean=np.mean(e)
                    array_rec.extend(e-mean)
                channels_bc.append(array_rec)
            epochs_bc.append(channels_bc)
        self.data=epochs_bc
        self.epochs_mne=numpy_to_epochs(self.data,self.ch_names,self.sfreq,self.tmin,self.events,self.event_id)
        if self.csd==True:
            with  self.epochs_mne.info._unlock():
                self.epochs_mne.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
            for i in range(len(self.epochs_mne.info['chs'])):
                self.epochs_mne.info['chs'][i].update(coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                      unit=FIFF.FIFF_UNIT_V_M2)
        return self
    
    
    def ICA_part_1(self, n_comps=None):
        """
        Creates ICA fiting. Modifies ICA-related Object attributes 

        Parameters
        ----------
        n_comps : int, optional
            number of ica components. The default is half of the len of channels.

        Returns
        -------
        Epochs.class
        epochs with ica fitted attribute ready to the second part

        """
        self.epochs_mne.set_eeg_reference() #for better performance
        if n_comps==None:
            n_comps=len(self.ch_names)//2 #default half of the len of channels
        ica=mne.preprocessing.ICA(n_components=n_comps,method='infomax', fit_params=dict(extended=True)) #infomax better for autolabelling
        ica.fit(self.epochs_mne)
        self.ica=ica
        self.ica_status=True
        self.ica_n_comps=n_comps
        return self
        
    def ICA_plots(self,picks=None):
        """
        ICA plot sources and properties for the specified picks 

        Parameters
        ----------
        picks : List of int, optional
            list of components. The default is None.
        """
        if self.ica_status==True:
             self.ica.plot_sources(self.epochs_mne,picks=picks)
             if picks==None:
                 picks=np.arange(0,self.ica.n_components_)
                 self.ica.plot_properties(self.epochs_mne,picks=picks)
        else:
            print("Run ICA_part_1 before running ICA_plots")
    
    def Auto_ICA_Eval(self):
        """
        Automatic ICA evaluation and selection of artifactfull components
        
        Returns
        -------
        
        automatic_detection : list of tuples with the ICAlabeling
        r_comps_to_exclude : list of components to remove according

        """
        labels_dict=label_components(self.epochs_mne,self.ica, method='iclabel')
        labels_prob=labels_dict['y_pred_proba']
        labels=labels_dict['labels']
        index=[*range(0,self.ica_n_comps)]
        automatic_detection=list(zip(index,labels_prob, labels))
        # return automatic_detection
        
        r_comps_to_exclude=[]
                
        if self.filtered_status!=None:
            for i in range(len(automatic_detection)):
                if automatic_detection[i][1]>0.70 and automatic_detection[i][2]=='eye blink':
                    r_comps_to_exclude.append(i)
                elif automatic_detection[i][1]>0.85 and automatic_detection[i][2]=='muscle artifact':
                    r_comps_to_exclude.append(i)    
                elif automatic_detection[i][1]>0.70 and automatic_detection[i][2]=='line noise':
                    r_comps_to_exclude.append(i)
            sources=self.ica.get_sources(self.epochs_mne)
            picks = [*range(0, np.shape(sources)[1], 1)]
            sources.plot(picks=picks)
            
            #looking for pulses components
            srs_copy_pulse=sources.copy()
            srs_copy_pulse.crop(-0.050,0.050)
            pulse_data=srs_copy_pulse.get_data()
            sources_data=sources.get_data()
            pulse_data_sum=np.sum(abs(pulse_data),axis=-1)
            sources_sum=np.sum(abs(sources_data),axis=-1)
            div=np.divide(pulse_data_sum,sources_sum)
            avg=np.mean(div, axis=0)
            index=np.argwhere(np.array(abs(avg))>0.30)
            index=index[:,0]
            for e in index:
                    r_comps_to_exclude.append(e)          
            
            #looks for slow drift components
            srs_copy_drift=sources.copy()
            srs_copy_drift_2=sources.copy()
            srs_copy_drift.crop(srs_copy_drift.tmin,srs_copy_drift.tmin+0.015)
            srs_copy_drift_2.crop(srs_copy_drift_2.tmax-0.015,srs_copy_drift_2.tmax)
            srs_copy_drift=srs_copy_drift.get_data()
            srs_copy_drift_2=srs_copy_drift_2.get_data()
            sources_data=sources.get_data()
            srs_copy_drift_sum=np.sum(abs(srs_copy_drift),axis=-1)
            srs_copy_drift_2_sum=np.sum(abs(srs_copy_drift_2),axis=-1)
            sources_sum=np.sum(abs(sources_data),axis=-1)
            srs_copy_drift_total=srs_copy_drift_sum+srs_copy_drift_2_sum
            div=np.divide(srs_copy_drift_total,sources_sum)
            avg=np.mean(div, axis=0)
            index=np.argwhere(np.array(abs(avg))>0.25)
            index=index[:,0]
            for e in index:
                    r_comps_to_exclude.append(e)
            
            #low power components usually non brain components
            sources_sum=np.sum(abs(sources_data),axis=-1)
            mean_source_sum=np.mean(abs(sources_sum),axis=0)
            xmax=max(mean_source_sum)
            for i, x in enumerate(mean_source_sum):
                mean_source_sum[i] = x / xmax
            
            index=np.argwhere(mean_source_sum<0.75)
            for e in index:
                    r_comps_to_exclude.append(e[0])
        else:
            for i in range(len(automatic_detection)):
                if automatic_detection[i][1]>0.70 and automatic_detection[i][2]=='eye blink':
                    r_comps_to_exclude.append(i)
                elif automatic_detection[i][1]>0.80 and automatic_detection[i][2]=='muscle artifact':
                    r_comps_to_exclude.append(i)    
        
        r_comps_to_exclude = list(set(r_comps_to_exclude))
        
        if r_comps_to_exclude != []:
            evoked =self.epochs_mne.average()
            self.ica.plot_overlay(evoked, exclude=[0], picks='eeg')
        
        self.auto_comps=r_comps_to_exclude
        
        return automatic_detection, r_comps_to_exclude
    
    def ICA_part_2(self, comps_to_exclude=None, auto_removal='y'):
        """
        ICA components removal, can be automaticall or manual according to the str in auto_removal
            
        Parameters
        ----------
        comps_to_exclude : list of int
            list of comps_to_exclude. should be inputed when auto_removal its different than 'y'
        
        auto_removal : str
            Specifies wheter mannual or automatic ICA removal will be done. if 'y' automatic removal is performed 
        
        Returns
        -------
        epochs after ica removal

        """
        try:
            if auto_removal == 'y':
                comps_to_exclude=self.auto_comps
            else:
                comps_to_exclude=comps_to_exclude
                if comps_to_exclude==None:
                    print('ERROR:  Introduce List of ints of components')
            self.ica.exclude=comps_to_exclude
            epoched_ica=self.ica.apply(self.epochs_mne)
            self.epochs_mne=epoched_ica
            self.data=epoched_ica.get_data()
            return self
        except:
            if self.csd==True:
                print("Perform ICA before Surface Laplcian")
            else:
                print("You need to run first the method ICA_part_1")
    
    def bad_channel_interpolation(self, bad_channels_list):
        """
        Interpolation of Bad Channels. Modifies in Place

        Parameters
        ----------
        bad_channels_list : List of Str
            List of channels to interpolate.

        Returns
        -------
        Epochs Object
            Epochs Object with interpolated bad channels.

        """
        self.epochs_mne.info['bads'].extend(bad_channels_list)
        simulated_epochs_interp=self.epochs_mne.interpolate_bads(reset_bads=True)
        self.epochs_mne=simulated_epochs_interp
        self.data = simulated_epochs_interp.get_data()
        return self
        
    def surface_laplacian_transform(self):
        """
        Re-reference data through Surface Laplacian. If you want to do source analysis copy Epoch object before applying as the source analysis already does csd.
        Modifies object in place
        
        Returns
        -------
        Epochs Object
            Epochs Object after csd re-referencing.

        """
        
        epoched_ref= mne.preprocessing.compute_current_source_density(self.epochs_mne)
        self.epochs_mne=epoched_ref
        self.data = epoched_ref.get_data()
        self.csd=True
        self.info=self.epochs_mne.info
        return self
    
    
    def bad_epochs_rejection(self,threshold=220e-6):
        """
        Rejection of epochs based on Maximum-Minimum Peak to peak thresholding
        Modifies in Place
        
        Parameters
        ----------
        threshold : float, optional
            threshold for the bad epochs rejection. The default is 220e-6.

        Returns
        -------
        Epochs Object
           Epochs object after removal of epochs thresholded.

        """
        
        
        epochs_array=self.epochs_mne.get_data()
        dropped_epochs=[]
        channels_thresholded=[]
        for i in range(len(epochs_array)):
            channels=[]
            for j in range(len(epochs_array[i])):
                maxValue = np.max(epochs_array[i][j])
                minValue = np.min(epochs_array[i][j])
                if maxValue-minValue>threshold:
                    channels.append(self.ch_names[j])
            if channels!=[]:
                    dropped_epochs.append(i)
                    channels_thresholded.append(channels)
        epochs_array_copy=epochs_array.copy()
        if dropped_epochs!=[]:
            new_epochs_array=np.delete(epochs_array_copy,dropped_epochs,axis=0)
            self.data=np.array(new_epochs_array)
            events_1=self.events
            
            events_ind_to_drop=[*range(-1*len(dropped_epochs),0, 1)]
            events_removed=np.delete(events_1,events_ind_to_drop,axis=0)
            
            #we can do this because the onset is evenly spaced, if not we had to change to len
            self.epochs_mne=numpy_to_epochs(self.data,self.ch_names,self.sfreq,self.tmin,events_removed,self.event_id)
            self.events=events_removed
            self.drop_ch=channels_thresholded
            self.drop_ind=dropped_epochs
            print(dropped_epochs)
            if self.csd==True:
                with  self.epochs_mne.info._unlock():
                    self.epochs_mne.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
                for i in range(len(self.epochs_mne.info['chs'])):
                    self.epochs_mne.info['chs'][i].update(coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                          unit=FIFF.FIFF_UNIT_V_M2)     
        return self
    
    
    def bad_epochs_plot(self):
        """
        Plots Removed epochs info, channels whose surpassed threshold

        """
        
        if self.drop_ind!=None:
            print("here")
            nr_epochs=len(self.data)
            nr_drop_epochs=len(self.drop_ind)
            total_epochs=nr_epochs+nr_drop_epochs
            flat_channels = list(itertools.chain(*self.drop_ch))
            w = collections.Counter(flat_channels)
            counts=list(w.values())
            array = (np.array(counts)/total_epochs)*100
            multiplied = list(array)
            channels=list(w.keys())
            plt.figure()
            plt.bar(channels, multiplied)
            plt.grid(True)
            plt.xlabel('Channels')
            plt.ylabel('% of epochs removed')
            plt.title('{} of {} epochs were removed ({}%)'.format(nr_drop_epochs,total_epochs,round((nr_drop_epochs/total_epochs)*100,1)))
            plt.show()
        else:
            print("No Channels were dropped")
    
    def psd_bands(self):
        """
        Computes mean relative PSD for the different band across epochs
       
        Returns
        -------
        list of floats
            Each band relative PSD mean value

        """
        psd_total=np.sum(mne.time_frequency.psd_array_welch(self.data,self.sfreq, fmin=0, fmax=70, n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_delta=np.sum(mne.time_frequency.psd_array_welch(self.data, self.sfreq, fmin=0, fmax=4,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_theta=np.sum(mne.time_frequency.psd_array_welch(self.data, self.sfreq, fmin=4, fmax=8,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_alpha=np.sum(mne.time_frequency.psd_array_welch(self.data, self.sfreq, fmin=8, fmax=12,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_beta=np.sum(mne.time_frequency.psd_array_welch(self.data, self.sfreq, fmin=12, fmax=30,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_gamma=np.sum(mne.time_frequency.psd_array_welch(self.data, self.sfreq, fmin=30, fmax=80,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        
        r_psd_delta=np.divide(psd_delta, psd_total)
        r_psd_theta=np.divide(psd_theta, psd_total)
        r_psd_alpha=np.divide(psd_alpha, psd_total)
        r_psd_beta=np.divide(psd_beta, psd_total)
        r_psd_gamma=np.divide(psd_gamma, psd_total)
        
        mean_r_psd_delta=np.mean(r_psd_delta,axis=0)
        mean_r_psd_theta=np.mean(r_psd_theta,axis=0)
        mean_r_psd_alpha=np.mean(r_psd_alpha,axis=0)
        mean_r_psd_beta=np.mean(r_psd_beta,axis=0)
        mean_r_psd_gamma=np.mean(r_psd_gamma,axis=0)
        
        mean_r_psd_delta_2=np.mean(mean_r_psd_delta,axis=0)
        mean_r_psd_theta_2=np.mean(mean_r_psd_theta,axis=0)
        mean_r_psd_alpha_2=np.mean(mean_r_psd_alpha,axis=0)
        mean_r_psd_beta_2=np.mean(mean_r_psd_beta,axis=0)
        mean_r_psd_gamma_2=np.mean(mean_r_psd_gamma,axis=0)
        return [mean_r_psd_delta_2,mean_r_psd_theta_2,mean_r_psd_alpha_2,mean_r_psd_beta_2,mean_r_psd_gamma_2]
    
    def psd_bands_region(self, modality):
        """
        Computes mean relative PSD for the different band across epochs for the specified region.; Uses String_Checker to know the channels from the region
        
        Parameters
        ---------
        modality : str
            Region of interest, can be 'rh' for right hemisphere, 'lh' for left hemisphere, 'pos' for posterior region and 'ant' for anterior region.
        
        Returns
        -------
        list of floats
            Each band relative PSD mean value

        """
        index_list=string_checker(self.ch_names, modality)
        epochs_channels=self.data[:,index_list]    

        psd_total=np.sum(mne.time_frequency.psd_array_welch(epochs_channels,self.sfreq, fmin=0, fmax=70, n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_delta=np.sum(mne.time_frequency.psd_array_welch(epochs_channels, self.sfreq, fmin=0, fmax=4,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_theta=np.sum(mne.time_frequency.psd_array_welch(epochs_channels, self.sfreq, fmin=4, fmax=8,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_alpha=np.sum(mne.time_frequency.psd_array_welch(epochs_channels, self.sfreq, fmin=8, fmax=12,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_beta=np.sum(mne.time_frequency.psd_array_welch(epochs_channels, self.sfreq, fmin=12, fmax=30,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)
        psd_gamma=np.sum(mne.time_frequency.psd_array_welch(epochs_channels, self.sfreq, fmin=30, fmax=80,  n_fft=256, n_overlap=64,n_per_seg=128)[0],axis=-1)

        r_psd_delta=np.divide(psd_delta, psd_total)
        r_psd_theta=np.divide(psd_theta, psd_total)
        r_psd_alpha=np.divide(psd_alpha, psd_total)
        r_psd_beta=np.divide(psd_beta, psd_total)
        r_psd_gamma=np.divide(psd_gamma, psd_total)

        mean_r_psd_delta=np.mean(r_psd_delta,axis=0)
        mean_r_psd_theta=np.mean(r_psd_theta,axis=0)
        mean_r_psd_alpha=np.mean(r_psd_alpha,axis=0)
        mean_r_psd_beta=np.mean(r_psd_beta,axis=0)
        mean_r_psd_gamma=np.mean(r_psd_gamma,axis=0)

        mean_r_psd_delta_2=np.mean(mean_r_psd_delta,axis=0)
        mean_r_psd_theta_2=np.mean(mean_r_psd_theta,axis=0)
        mean_r_psd_alpha_2=np.mean(mean_r_psd_alpha,axis=0)
        mean_r_psd_beta_2=np.mean(mean_r_psd_beta,axis=0)
        mean_r_psd_gamma_2=np.mean(mean_r_psd_gamma,axis=0)
        return [mean_r_psd_delta_2,mean_r_psd_theta_2,mean_r_psd_alpha_2,mean_r_psd_beta_2,mean_r_psd_gamma_2]
    
    def source_space(self,src_filepath='freesurfer/subjects/src',bem_sol_filepath='freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',trans_path='freesurfer/subjects/daniel/daniel-trans-z-trans.fif', subjects_dir='freesurfer/subjects/',method="eLORETA"):
        """
        Source Analysis computation. 
        
        Important note: Please do not perform this analysis on Surface laplacian re-referenced data, as the eLORETA method already computes csd.
        Instead use a copy of the data prior to the csd.
        
        Also freesurfer folder should be in this package
        
        
        Parameters
        ----------
        src_filepath : str
            Path for the src file. 
        bem_sol_filepath : str
            path for the bem_sol.
        trans_path : str
            path for the bem_sol.
        subjects_dir : str
            Directory of Inverse solution subject .
        method : r , optional
            Method to compute inverse solution. The default is "eLORETA". other strings like 'dSPM', 'sLoreta' and 'MNE' are available

        Returns
        -------
        stc_p2 : Source MNE data.

        """
        mne.utils.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)
        src_dat=mne.read_source_spaces(src_filepath)
        bem_sol= mne.read_bem_solution(bem_sol_filepath)
        trans=trans_path
        epochs=self.epochs_mne.copy()
        epochs.set_eeg_reference('average', projection=True)
        epochs.apply_proj()
        epochs.apply_baseline(baseline=(None, None))
        fwd = make_forward_solution(self.info, trans, src_dat, bem_sol)
        cov_p = mne.compute_covariance(epochs, method='auto') 
        inv_p = make_inverse_operator(epochs.info, fwd, cov_p, loose=0.2, depth=0.8)
        stc_p2= BodyMirror.EEG_deep_analysis.source_localization(epochs.average(), inverse=inv_p, subjects_dir=subjects_dir, method=method)
        return stc_p2
    
    # def source_space_test(self,method="eLORETA"):
    #     epochs=self.epochs_mne.copy()
    #     fs_dir = fetch_fsaverage(verbose=True)
    #     subjects_dir = op.dirname(fs_dir)
    #     subject = 'fsaverage'
    #     trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    #     src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    #     bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    #     src_dat=mne.read_source_spaces(src)
    #     bem_sol= mne.read_bem_solution(bem)
    #     labels = mne.read_labels_from_annot('fsaverage', parc='aparc',subjects_dir=subjects_dir)
    #     del labels[-1]
    #     epochs.set_eeg_reference('average', projection=True)
    #     epochs.apply_proj()
    #     epochs.apply_baseline(baseline=(None, None))
    #     fwd = make_forward_solution(self.info, trans, src_dat, bem_sol)
    #     cov_p = mne.compute_covariance(epochs, method='auto') 
    #     inv_p = make_inverse_operator(epochs.info, fwd, cov_p, loose=0.2, depth=0.8)
    #     snr = 3.
    #     lambda2 = 1. / snr ** 2
    #     stcs = apply_inverse_epochs(epochs, inv_p, lambda2,method=method, pick_ori="normal", return_generator=True)
    #     label_ts = mne.extract_label_time_course(stcs, labels, src_dat, mode='mean_flip', return_generator=True)
    #     self.source=stcs
    #     self.source_labels=labels
    #     self.source_labels_ts=label_ts
    #     stc_p2= BodyMirror.EEG_deep_analysis.source_localization(epochs.average(), inverse=inv_p, subjects_dir=subjects_dir, method=method)
    #     return self
    
def read_concatenate_epochs_set(folder_path, subjects="Hv", csd_status=None):
    """
    Read a dataset containing epochs saved in .set file.
    
    Parameters
    ----------
    folder_path : path for the .set file.
    subjects : Str, Subjects define whose the group search - Name of group should be in filename 
        DESCRIPTION. The default is "Hv".

    Returns
    -------
    data : Concatenated epochs from the whole dataset as an Epochs Object.
    """
    new_filepath=os.path.join(folder_path+"\*.set") #create a path for the directory where the set files will be read
    file_list = glob.glob(new_filepath) #finds every .set file in the directory
    if subjects=="Hv": #Comparison to read healthy volunteers or MS patients
        file_list = [item for item in file_list if "HV" in item.upper()]
    else:
        file_list= [item for item in file_list if "TYS" in item.upper()]
    data=[]
    for i in range(len(file_list)):
          data.append(read_epochs_set(file_list[i]))
    data=mne.concatenate_epochs(data) #concatenated epochs from the whole sample
    epochs=Epochs(data, csd_status=csd_status) 
    return epochs

def ERSC_barplot_comparison(l_1,l_2):
    """
    Comparison barplot for Event Related Spectral Changes, whether in specific regions or in all the brain.
    
    Parameters
    ----------
    l_1 : list of floats
        Mean Band Relative Amplitude, before the stimulus.
    l_2 : list of floats
        Mean Band Relative Amplitude, after the stimulus.

    """

    X=["Delta","Theta","Alpha","Beta","Gamma"]
    X_axis = np.arange(len(X))
    plt.figure()
    plt.bar(X_axis - 0.2, l_1, 0.4, label = "Before Stimulus")
    plt.bar(X_axis + 0.2, l_2, 0.4, label = 'After Stimulus')
    plt.xticks(X_axis, X)
    plt.xticks(rotation = 90)
    plt.xlabel("Groups")
    plt.ylabel("Mean Relative Amplitude")
    plt.title("ERSC")
    plt.legend()
    plt.show()
    
def read_list_epochs_set(folder_path, subjects="Hv", csd_status=None):
    """
    Read a dataset containing epochs saved in .set file.
    
    Parameters
    ----------
    folder_path : path for the .set file.
    
    Returns
    -------
    data : List of per subject epochs.Epochs.
    """
    new_filepath=os.path.join(folder_path+"\*.set") #create a path for the directory where the set files will be read
    file_list = glob.glob(new_filepath) #finds every .set file in the directory
    if subjects=="Hv": #Comparison to read healthy volunteers or MS patients
        file_list = [item for item in file_list if "HV" in item.upper()]
    else:
        file_list= [item for item in file_list if "TYS" in item.upper()]
    data=[]
    for i in range(len(file_list)):
        data.append(Epochs(read_epochs_set(file_list[i]), csd_status=csd_status))
    return data

def epochs_in_list_splitter(list_epochs, n=2):
    """
    Split epochs in list into n partitions. Such that it returns a list of all partitions of all the epochs.

    Parameters
    ----------
    list_epochs : list of epochs
        List of the epochs to split.
    n : Int. number of partitions of each epoch, optional
        The default is 2.

    Returns
    -------
    new_list_epochs : list of Epochs
        list of all the partitioned epochs. len of new_list_epochs will be equal to n times the len of the list_epochs.

    """
    new_list_epochs=[]
    for i in range(len(list_epochs)):
        arr_epochs=list_epochs[i].data
        split_array=np.array_split(arr_epochs, n)
        split_epochs=[]
        for j in range(len(split_array)):
            split_epochs.append(Epochs(numpy_to_epochs(split_array[j],list_epochs[i].ch_names,list_epochs[i].sfreq,list_epochs[i].tmin)))
        new_list_epochs.extend(split_epochs)
        print(len(new_list_epochs))
    return new_list_epochs

def epochs_splitter(epochs_concatenated, n=2):
    """
    Split epochs into n partitions. Such that it returns a list of the "partition epochs".

    Parameters
    ----------
    epochs_concatenated : epochs.Epochs
    n : Int, optional
        number of partitions of each epoch. The default is 2.

    Returns
    -------
    split_epochs : list of Epochs
        list of all the partitioned epochs.
    """
    arr_epochs=epochs_concatenated.data
    split_array=np.array_split(arr_epochs, n)
    split_epochs=[]
    for j in range(len(split_array)):
            split_epochs.append(Epochs(numpy_to_epochs(split_array[j],epochs_concatenated.ch_names,epochs_concatenated.sfreq,epochs_concatenated.tmin)))
    
    return split_epochs