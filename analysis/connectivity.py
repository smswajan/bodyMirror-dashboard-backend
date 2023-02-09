# -*- coding: utf-8 -*-
"""
Code Developed by Pedro Gomes for Myelin-H company

Windows 11
MNE 1.2.1
Python 3.9

"""


##################### Imports ###########################
from .epochs import Epochs
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity.viz import plot_sensors_connectivity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import bct
#################### Functions and Classes ###########################

class Connectivity():
    #### Class for connectivity computation and connectivity plot
    def __init__(self, epochs): #epochs its an MNE Python object, so when starting this use epochs.epochs_mne atribute
        self.epochs_mne=epochs.epochs_mne #epochs in mne format
        self.sfreq=epochs.sfreq #sampling frequency
        self.info=epochs.info #epochs info
        self.channels=epochs.channels #channels info
        self.ch_names=self.epochs_mne.info["ch_names"] #channel names
        self.con = None #attribute to access connectivity
        self.con_reg=None #regional_con  
        self.reg_channels=None #regional ch
        self.threshold=None #threshold for Network Analysis

    def compute_connectivity(self,fmin=None,fmax=np.inf,tmin=None, tmax=None, method='imcoh',mode='fourier'):
        """
        Computes functional connectivity for the specified frequency band and tmin and tmax. Modifies Threshold and con in place
        Important: Always run this analysis before regional connectivity analyis and Network analysis as this code gives the threshold for network analysis
        
        Parameters
        ----------
        fmin : float, optional
            Minimum frequency. The default is None, Which is 0
        fmax : float, optional
            Maximum freauency. The default is np.inf.
        tmin : float, optional
            Minimum time. The default is None which starts at the beggining of the epoch
        tmax : float, optional
            Maximum time. The default is None which ends at the end of the epoch.
        method : str, optional
            ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased']. The default is 'imcoh'.
        mode: method to compute evaluate spectrum and phase. Default is fourier, due to time cost, yet multitaper is adviced.
        
        Returns
        -------
        con_array : np.array
            matrix of pairwise correlation.

        """
        con = spectral_connectivity_epochs(self.epochs_mne, method=method, mode=mode,faverage=True, sfreq=self.sfreq,
                                           fmin=fmin, fmax=fmax,tmin=tmin, n_jobs=1) #mode fourier cause its faster
        con_array=abs(con.get_data(output='dense')[:, :, 0]) #abs because imcoh returns signed values
        self.con=con_array
        arr_to_thresh=con_array[np.tril_indices((np.shape(con_array)[0]),k=-1)]
        self.threshold=np.quantile(arr_to_thresh,0.20)
        return con_array
    
    def connectivity_plots(self):
        """Plots both connectivity circle and sensors connectivity - Uses data from all to all connectivity and should be runned after the compute connectivity method"""
        plot_connectivity_circle(self.con, self.ch_names, title='All-to-All Connectivity')
        plot_sensors_connectivity(self.epochs_mne.info,self.con)
        
    def regional_connectivity(self,region,fmin=None,fmax=np.inf,tmin=None, tmax=None, method='imcoh', mode='fourier'):
        """
        Computes regional connectivity, again uses epochs.string_checker to view for the channels belonging to the region

        Parameters
        ----------
        region : str
            'lh' for left hemisphere, 'rh' for right hemisphere, 'pos' for posterior, 'ant' for anterior.
        fmin : float, optional
            Minimum frequency. The default is None, Which is 0
        fmax : float, optional
            Maximum freauency. The default is np.inf.
        tmin : float, optional
            Minimum time. The default is None which starts at the beggining of the epoch
        tmax : float, optional
            Maximum time. The default is None which ends at the end of the epoch.
        method : str, optional
            ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased']. The default is 'imcoh'.
        mode: method to compute evaluate spectrum and phase. Default is fourier, due to time cost, yet multitaper is adviced.

        Returns
        -------
        con_array : np.array
            connectivity matrix.

        """        
        ch_list=[]
        index_list=epochs.string_checker(self.ch_names, region)
        if len(index_list) <=1:
            print("You dont have enough elements to analyze")
        else:
            epochs_channels=self.epochs_mne.get_data()[:,index_list]
            con = spectral_connectivity_epochs(epochs_channels, method=method, mode=mode,faverage=True, sfreq=self.sfreq, fmin=fmin, fmax=fmax,tmin=tmin, n_jobs=1) #mode fourier cause its faster
            con_array=abs(con.get_data(output='dense')[:, :, 0]) #abs because imcoh returns signed values
            ch_names_array = np.array(self.ch_names)
            ch_names_list=list(ch_names_array[index_list])
            self.con_reg=con_array #regional_con  
            self.reg_channels=ch_names_list #regional ch
            return con_array
    
    def con_heatmap(self):
        """
        Plot connectivity heatmaps

        """
        plt.figure()
        ax = sns.heatmap(self.con, linewidth=0.5,cmap="YlGnBu",xticklabels=self.epochs_mne.info["ch_names"], yticklabels=self.epochs_mne.info["ch_names"])
        sns.set(font_scale = 1)
        ax.set_title('Functional Connectivity', fontsize = 20)
        ax.set(xlabel='Channel', ylabel='Channel')
        plt.show()
    
    def reg_con_heatmap(self):
        """
        Plot reg connectivity heatmaps

        """
        plt.figure()
        ax = sns.heatmap(self.con_reg, linewidth=0.5,cmap="YlGnBu",xticklabels=self.reg_channels, yticklabels=self.reg_channels)
        sns.set(font_scale = 1)
        ax.set_title('Functional Connectivity', fontsize = 20)
        ax.set(xlabel='Channel', ylabel='Channel')
        plt.show()

class Network_Analysis():
    #Class for the Network analysis
    
    def __init__(self, con_array,ch_names,threshold):
        self.con_array=(con_array > threshold) * con_array #connectivity array given by compute_connectity method
        self.ch_names=ch_names #list of channels names given by Connectivity.ch_names
    
    def Centrality(self): #Betweenness Centrality - Importance of a node
        bc=bct.betweenness_wei(self.con_array)
        plt.figure()
        plt.bar(np.arange(0, len(self.ch_names)),bc, tick_label=self.ch_names)
        plt.xticks(rotation = 90)
        plt.ylabel('Betweenness Centrality')
        plt.xlabel('Channel')
        plt.show()
        return bc
    
    def Strengths(self): #Strength - Importance of a node
        st=bct.strengths_und(self.con_array)
        plt.figure() 
        plt.bar(np.arange(0, len(self.ch_names)),st, tick_label=self.ch_names)
        plt.xticks(rotation = 90)
        plt.ylabel('Strength')
        plt.xlabel('Channel')
        plt.show()
        return st
    
    # def Modularity(self): #Quality of separation among clusters
    #     Com,Q=bct.community_louvain(self.con_array)
    #     return Com,Q
    
    def Transitivity(self): #Transitivity-Clustered activity
        T=bct.transitivity_wd(self.con_array)
        return T
    def Charpath_GE(self): #Characteristic Path/ Global Efficiency  - Integrated Activity
        D=bct.distance_wei_floyd(self.con_array, transform='inv')[0]
        Charpath, GE=bct.charpath(D, include_diagonal=False, include_infinite=False)[0:2]
        return Charpath, GE 
        
    
def Comparison_barplot_nr(nr_1,nr_2, Metric="Transitivity"):
    """
    Comparison plot for single valued metrics, like Transitivity, Charpath and GE

    Parameters
    ----------
    nr_1 : float
        Group 1 metric value.
    nr_2 : float
        Group 1 metric value.
    Metric : str, optional
        Metric name. The default is "Transitivity".

    """
    l_1=[nr_1]
    l_2=[nr_2]
    fig_3=plt.figure()
    plt.bar(1 - 0.2, l_1, 0.2, label = Metric+' Group 1')
    plt.bar(1 + 0.2, l_2, 0.2, label = Metric+' Group 2') 
    plt.xticks([0.8,1.2], ["Group 1", "Group 2"])
    plt.xlabel("Groups")
    plt.ylabel(Metric)
    plt.title(Metric+" Comparison")
    plt.legend()
    plt.show()
    
def Comparison_barplot_list(l_1,l_2,chs_names,Metric="Betweenness Centrality"):
    """
    Comparison for list valued metrics like Strength and Betweenness Centrality
    
    Parameters
    ----------
    l_1 : list
        group 1 list of values for such metric.
    l_2 : list
        group 2 list of values for such metric.
    chs_names : list of str
        list of channel names.
    Metric : str , optional
        Metric name. The default is "Betweenness Centrality".

    """
    X = chs_names
    X_axis = np.arange(len(X))
    fig_4=plt.figure()
    plt.bar(X_axis - 0.1, l_1, 0.2, label = 'Group_1')
    plt.bar(X_axis + 0.1, l_2, 0.2, label = 'Group 2')
    plt.xticks(X_axis, X)
    plt.xticks(rotation = 90)
    plt.xlabel("Groups")
    plt.ylabel(Metric)
    plt.title(Metric+" Comparison")
    plt.legend()
    plt.show()
    
# class Source_Connectivity():
    
#     def __init__(self,epochs):
#         self.source=epochs.source #epochs in mne format
#         self.source_labels=epochs.source_labels
#         self.source_labels_ts=epochs.source_labels_ts
#         self.sfreq=epochs.sfreq #sampling frequency
#         self.info=epochs.info #epochs info
#         self.con = None #attribute to access connectivity
#         self.threshold=None
#         self.label_names=[label.name for label in self.source_labels]
        
        
#     def compute_connectivity(self, fmin=None, fmax=np.inf, mode='fourier'):
#         con = spectral_connectivity_epochs(self.source_labels_ts, method='imcoh', mode=mode, sfreq=self.sfreq, fmin=fmin, fmax=fmax, faverage=True, n_jobs=1)
#         con_array=abs(con.get_data(output='dense')[:, :, 0])
#         self.con=con_array
#         arr_to_thresh=con_array[np.tril_indices((np.shape(con_array)[0]),k=-1)]
#         self.threshold=np.quantile(arr_to_thresh,0.20)
#         return con_array
    
#     def circle_plot(self):
#         label_names = [label.name for label in self.source_labels]
#         label_colors = [label.color for label in self.source_labels]
#         lh_labels = [name for name in label_names if name.endswith('lh')]
        
#         # Get the y-location of the label
#         label_ypos = list()
#         for name in lh_labels:
#             idx = label_names.index(name)
#             ypos = np.mean(self.source_labels[idx].pos[:, 1])
#             label_ypos.append(ypos)
        
#         # Reorder the labels based on their location
#         lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
        
#         # For the right hemi
#         rh_labels = [label[:-2] + 'rh' for label in lh_labels]
        
#         # Save the plot order and create a circular layout
#         node_order = list()
#         node_order.extend(lh_labels[::-1])  # reverse the order
#         node_order.extend(rh_labels)
        
#         node_angles = circular_layout(label_names, node_order, start_pos=90,
#                                       group_boundaries=[0, len(label_names) / 2])
        
#         # Plot the graph using node colors from the FreeSurfer parcellation. We only
#         # show the 300 strongest connections.
#         fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
#                                subplot_kw=dict(polar=True))
#         plot_connectivity_circle(self.con, label_names, n_lines=300,
#                                  node_angles=node_angles, node_colors=label_colors,
#                                  title='All-to-All Connectivity', ax=ax)
#         fig.tight_layout()

def connectivity_circle_subplots(con_array_1,con_array_2,ch_names):
    """
    Computes connectivity subplots comparing two connectivity arrays which share the same channels

    Parameters
    ----------
    con_array_1 : np.array
        connectivity matrix one.
    con_array_2 : np.array
        connectivity matrix one.
    ch_names : list of str
        list of channels names
    """
    con_array=[con_array_1,con_array_2]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='black',
                             subplot_kw=dict(polar=True))
    i=0
    for ax, array in zip(axes, con_array):
        if i==0:
            plot_connectivity_circle(array, ch_names, title='Healthy',ax=ax)
            i=1
        else:
            plot_connectivity_circle(array, ch_names, title='MS',ax=ax)