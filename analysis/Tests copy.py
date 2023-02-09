# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:42:00 2022

@author: testm
"""

import pandas as pd
import numpy as np
import scipy
import connectivity
import epochs
import evoked
import ML
import statistics
import matplotlib.pyplot as plt
import copy
import seaborn as sns
# %matplotlib qt


########################## Loading cleaned folder preprocessed data ##################################
Epochs_HV = epochs.read_concatenate_epochs_set(
    r'C:\Users\testm\Documents\New_TMS_Data', subjects="Hv", csd_status=False)
Epochs_TYS = epochs.read_concatenate_epochs_set(
    r'C:\Users\testm\Documents\New_TMS_Data', subjects="TYS", csd_status=False)

Epochs_HV_src = copy.copy(Epochs_HV)
Epochs_HV_src.crop_epochs(-0.015, 0.300)
Epochs_TYS_src = copy.copy(Epochs_TYS)
Epochs_TYS_src.crop_epochs(-0.015, 0.300)

# # # ##################### For Pedro's preprocessed data ########################
# Epochs_HV=epochs.read_concatenate_epochs_set(r'C:\Users\testm\Documents\Cleaned_Epochs_csd', subjects="Hv", csd_status=None)
# Epochs_TYS=epochs.read_concatenate_epochs_set(r'C:\Users\testm\Documents\Cleaned_Epochs_csd', subjects="TYS", csd_status=None)

# Epochs_HV_src=epochs.read_concatenate_epochs_set(r'C:\Users\testm\Documents\Cleaned_Epochs', subjects="Hv", csd_status=None)
# Epochs_TYS_src=epochs.read_concatenate_epochs_set(r'C:\Users\testm\Documents\Cleaned_Epochs', subjects="TYS", csd_status=None)

# Epochs_HV=epochs.Epochs(epochs.numpy_to_epochs(Epochs_HV.data,Epochs_HV.ch_names,Epochs_HV.sfreq,-1),True)
# Epochs_TYS=epochs.Epochs(epochs.numpy_to_epochs(Epochs_TYS.data,Epochs_TYS.ch_names,Epochs_TYS.sfreq,-1),True)

# Epochs_HV_src=epochs.Epochs(epochs.numpy_to_epochs(Epochs_HV_src.data,Epochs_HV_src.ch_names,Epochs_HV_src.sfreq,-1))
# Epochs_TYS_src=epochs.Epochs(epochs.numpy_to_epochs(Epochs_TYS_src.data,Epochs_TYS_src.ch_names,Epochs_TYS_src.sfreq,-1))

# # ####################################### Individual loading #################################################
#Epochs_HV=epochs.read_concatenate_epochs_set(r'C:\Users\testm\Documents\cleaned', subjects="Hv", csd_status=None)
# Epochs_TYS=epochs.Epochs(epochs.read_epochs_set(r'C:\Users\testm\Documents\cleaned\FAT_Hv_13_FAT_T0_Fat_TEP_pre_final_avref.set'))

# Epochs_HV_src=copy.copy(Epochs_HV) #copy for source analysis
# Epochs_TYS_src=copy.copy(Epochs_TYS) #copy for source analysis


############################################ copy ##########################################
Epochs_HV_copy = copy.copy(Epochs_HV)
Epochs_TYS_copy = copy.copy(Epochs_TYS)
Epochs_HV_copy_2 = copy.copy(Epochs_HV)  # copy for Morlet analysis
Epochs_TYS_copy_2 = copy.copy(Epochs_TYS)  # copy for Morlet analysis

###################################
Epochs_HV.crop_epochs(0.013, 0.300)  # Crop epochs for post stimuli analysis
Epochs_TYS.crop_epochs(0.013, 0.300)  # Crop epochs for post stimuli analysis

# Crop epochs for pre stimuli analysis
Epochs_HV_copy.crop_epochs(-0.650, -0.350)
# Crop epochs for pre stimuli analysis
Epochs_TYS_copy.crop_epochs(-0.650, -0.350)

######################## Evoked ######################
Evoked_TYS = evoked.Evoked(Epochs_TYS.average_epochs(),
                           Epochs_TYS.tmin, Epochs_HV.csd)
Evoked_HV = evoked.Evoked(Epochs_HV.average_epochs(),
                          Epochs_HV.tmin, Epochs_HV.csd)

####################### Plot Topomap #################
Evoked_HV.power_topographic_plot()
Evoked_HV.power_topographic_plot(time=[0.080, 0.100])

Evoked_TYS.power_topographic_plot()
Evoked_TYS.power_topographic_plot(time=[0.080])

######################### Plot Evoked Image####################
Evoked_HV.image_plot()
Evoked_TYS.image_plot()
Evoked_HV.image_plot(picks=['Pz', "Iz", "Cz"])
Evoked_TYS.image_plot(picks=['Pz', "Iz", "Cz"])  # Not equal due to averaging
# ####################### Plot Epoched Image ###################
Epochs_HV.image_plot(vmin=-20.0, vmax=20.0, picks=['Cz'])
Epochs_TYS.image_plot(vmin=-20.0, vmax=20.0, picks=['Cz'])
# ####################### Plot Evoked ######################
Evoked_HV.evoked_plot()
Evoked_TYS.evoked_plot()
####################### Topo Plot #########################
Epochs_HV.topo_plot()
Epochs_TYS.topo_plot()
# ####################### Joint Plots ######################
Evoked_HV.joint_plot(times="peaks")
Evoked_TYS.joint_plot(times="peaks")
# ####################### Band Plots #######################
Evoked_HV.Band_Topomap(time=[0.015, 0.020, 0.025, 0.030, 0.035,
                       0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.090, 0.1, 0.12, 0.15])
Evoked_TYS.Band_Topomap(time=[0.015, 0.020, 0.025, 0.030, 0.035,
                        0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.090, 0.1, 0.12, 0.15])
####################### Bands Topo #########################
Epochs_HV.psd_topo_plot()
Epochs_TYS.psd_topo_plot()
# ####################### TFR ##################################
Epochs_HV_copy_2.crop_epochs(-0.500, 0.500)  # crop for Morlet
Epochs_TYS_copy_2.crop_epochs(-0.500, 0.500)  # crop for Morlet
Epochs_HV_copy_2.morlet_tfr(6, 35, -0.500, 0.500, power_plot="y",
                            joint_plot="y", itc_plot="y", timefreqs=None, baseline=(-0.500, 0.015))
Epochs_TYS_copy_2.morlet_tfr(6, 35, -0.500, 0.500, power_plot="y",
                             joint_plot="y", itc_plot="y", timefreqs=None, baseline=(-0.500, 0.015))
########################### Compare Evoked ###################
evoked.evoked_comparison(Evoked_HV, Evoked_TYS, method="gfp")
evoked.evoked_comparison(Evoked_HV, Evoked_TYS, method="mean")
evoked.evoked_comparison(Evoked_HV, Evoked_TYS, method="median")
evoked.topological_evoked_dif(Evoked_HV, Evoked_TYS)
########################## gfp peaks ###############################
gfp = Evoked_TYS.gfp()
print(evoked.gfp_peaks(gfp, 3, Evoked_TYS.tmin, Evoked_TYS.sfreq))

gfp = Evoked_HV.gfp()
print(evoked.gfp_peaks(gfp, 3, Evoked_TYS.tmin, Evoked_TYS.sfreq))


########################## Source ##############################
HV_Source = Epochs_HV_src.source_space('freesurfer/subjects/src', 'freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',
                                       'freesurfer/subjects/daniel/daniel-trans-z-trans.fif', 'freesurfer/subjects/', method="eLORETA")
TYS_Source = Epochs_TYS_src.source_space('freesurfer/subjects/src', 'freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',
                                         'freesurfer/subjects/daniel/daniel-trans-z-trans.fif', 'freesurfer/subjects/', method="eLORETA")

########################## Band Source activity ###############

######### Low Frequency #############
Epochs_HV_low = copy.copy(Epochs_HV_src)
Epochs_TYS_low = copy.copy(Epochs_TYS_src)
Epochs_HV_low.filtering_preprocessing(["bp"], l_freq=0, h_freq=12)
Epochs_TYS_low.filtering_preprocessing(["bp"], l_freq=0, h_freq=12)
HV_Source = Epochs_HV_low.source_space('freesurfer/subjects/src', 'freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',
                                       'freesurfer/subjects/daniel/daniel-trans-z-trans.fif', 'freesurfer/subjects/', method="eLORETA")
TYS_Source = Epochs_TYS_low.source_space('freesurfer/subjects/src', 'freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',
                                         'freesurfer/subjects/daniel/daniel-trans-z-trans.fif', 'freesurfer/subjects/', method="eLORETA")


######### Beta Frequency #############
Epochs_HV_low = copy.copy(Epochs_HV_src)
Epochs_TYS_low = copy.copy(Epochs_TYS_src)
Epochs_HV_low.filtering_preprocessing(["bp"], l_freq=12, h_freq=30)
Epochs_TYS_low.filtering_preprocessing(["bp"], l_freq=12, h_freq=30)
HV_Source = Epochs_HV_low.source_space('freesurfer/subjects/src', 'freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',
                                       'freesurfer/subjects/daniel/daniel-trans-z-trans.fif', 'freesurfer/subjects/', method="eLORETA")
TYS_Source = Epochs_TYS_low.source_space('freesurfer/subjects/src', 'freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',
                                         'freesurfer/subjects/daniel/daniel-trans-z-trans.fif', 'freesurfer/subjects/', method="eLORETA")

######### gamma #############
Epochs_HV_low = copy.copy(Epochs_HV_src)
Epochs_TYS_low = copy.copy(Epochs_TYS_src)
Epochs_HV_low.filtering_preprocessing(["bp"], l_freq=30, h_freq=80)
Epochs_TYS_low.filtering_preprocessing(["bp"], l_freq=30, h_freq=80)
HV_Source = Epochs_HV_low.source_space('freesurfer/subjects/src', 'freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',
                                       'freesurfer/subjects/daniel/daniel-trans-z-trans.fif', 'freesurfer/subjects/', method="eLORETA")
TYS_Source = Epochs_TYS_low.source_space('freesurfer/subjects/src', 'freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif',
                                         'freesurfer/subjects/daniel/daniel-trans-z-trans.fif', 'freesurfer/subjects/', method="eLORETA")


######################## ERSC ######################################

epochs.ERSC_barplot_comparison(
    Epochs_HV_copy.psd_bands(), Epochs_HV.psd_bands())
epochs.ERSC_barplot_comparison(
    Epochs_TYS_copy.psd_bands(), Epochs_TYS.psd_bands())

######################## Regional ERSC ####################################
# Right Hemisphere
epochs.ERSC_barplot_comparison(Epochs_HV_copy.psd_bands_region(
    'rh'), Epochs_HV.psd_bands_region('rh'))
epochs.ERSC_barplot_comparison(Epochs_TYS_copy.psd_bands_region(
    'rh'), Epochs_TYS.psd_bands_region('rh'))

# Left Hemisphere
epochs.ERSC_barplot_comparison(Epochs_HV_copy.psd_bands_region(
    'lh'), Epochs_HV.psd_bands_region('lh'))
epochs.ERSC_barplot_comparison(Epochs_TYS_copy.psd_bands_region(
    'lh'), Epochs_TYS.psd_bands_region('lh'))

# Posterior
epochs.ERSC_barplot_comparison(Epochs_HV_copy.psd_bands_region(
    'pos'), Epochs_HV.psd_bands_region('pos'))
epochs.ERSC_barplot_comparison(Epochs_TYS_copy.psd_bands_region(
    'pos'), Epochs_TYS.psd_bands_region('pos'))

# Posterior
epochs.ERSC_barplot_comparison(Epochs_HV_copy.psd_bands_region(
    'ant'), Epochs_HV.psd_bands_region('ant'))
epochs.ERSC_barplot_comparison(Epochs_TYS_copy.psd_bands_region(
    'ant'), Epochs_TYS.psd_bands_region('ant'))


########################### Functional Connectivity #####################
#### Allfreq pre ###############
c_theta = connectivity.Connectivity(Epochs_HV_copy)
tys_theta = connectivity.Connectivity(Epochs_TYS_copy)
arr_con_theta_hv = c_theta.compute_connectivity(
    fmin=0, fmax=80, mode='multitaper')
arr_con_theta_tys = tys_theta.compute_connectivity(
    fmin=0, fmax=80, mode='multitaper')
c_theta.connectivity_plots()
tys_theta.connectivity_plots()
connectivity.connectivity_circle_subplots(
    arr_con_theta_hv, arr_con_theta_tys, c_theta.ch_names)
c_theta.con_heatmap()
tys_theta.con_heatmap()
hv_theta_n = connectivity.Network_Analysis(
    arr_con_theta_hv, c_theta.ch_names, c_theta.threshold)
tys_theta_n = connectivity.Network_Analysis(
    arr_con_theta_tys, tys_theta.ch_names, tys_theta.threshold)
connectivity.Comparison_barplot_list(hv_theta_n.Centrality(
), tys_theta_n.Centrality(), c_theta.ch_names, Metric="Betweenness Centrality")
connectivity.Comparison_barplot_list(hv_theta_n.Strengths(
), tys_theta_n.Strengths(), c_theta.ch_names, Metric="Strengths")
connectivity.Comparison_barplot_nr(
    hv_theta_n.Transitivity(), tys_theta_n.Transitivity())
connectivity.Comparison_barplot_nr(
    hv_theta_n.Charpath_GE()[0], tys_theta_n.Charpath_GE()[0], "Charpath")
connectivity.Comparison_barplot_nr(
    hv_theta_n.Charpath_GE()[1], tys_theta_n.Charpath_GE()[1], "GE")

#### All freq post ###############
c_theta = connectivity.Connectivity(Epochs_HV)
tys_theta = connectivity.Connectivity(Epochs_TYS)
arr_con_theta_hv = c_theta.compute_connectivity(
    fmin=0, fmax=80, mode='multitaper')
arr_con_theta_tys = tys_theta.compute_connectivity(
    fmin=0, fmax=80, mode='multitaper')
c_theta.connectivity_plots()
tys_theta.connectivity_plots()
connectivity.connectivity_circle_subplots(
    arr_con_theta_hv, arr_con_theta_tys, c_theta.ch_names)
c_theta.con_heatmap()
tys_theta.con_heatmap()
hv_theta_n = connectivity.Network_Analysis(
    arr_con_theta_hv, c_theta.ch_names, c_theta.threshold)
tys_theta_n = connectivity.Network_Analysis(
    arr_con_theta_tys, tys_theta.ch_names, tys_theta.threshold)
connectivity.Comparison_barplot_list(hv_theta_n.Centrality(
), tys_theta_n.Centrality(), c_theta.ch_names, Metric="Betweenness Centrality")
connectivity.Comparison_barplot_list(hv_theta_n.Strengths(
), tys_theta_n.Strengths(), c_theta.ch_names, Metric="Strengths")
connectivity.Comparison_barplot_nr(
    hv_theta_n.Transitivity(), tys_theta_n.Transitivity())
connectivity.Comparison_barplot_nr(
    hv_theta_n.Charpath_GE()[0], tys_theta_n.Charpath_GE()[0], "Charpath")
connectivity.Comparison_barplot_nr(
    hv_theta_n.Charpath_GE()[1], tys_theta_n.Charpath_GE()[1], "GE")


#### Low freq ###############
c_theta = connectivity.Connectivity(Epochs_HV_copy)
tys_theta = connectivity.Connectivity(Epochs_TYS_copy)
arr_con_theta_hv = c_theta.compute_connectivity(fmin=0, fmax=12)
arr_con_theta_tys = tys_theta.compute_connectivity(fmin=0, fmax=12)
c_theta.connectivity_plots()
tys_theta.connectivity_plots()
connectivity.connectivity_circle_subplots(
    arr_con_theta_hv, arr_con_theta_tys, c_theta.ch_names)
c_theta.con_heatmap()
tys_theta.con_heatmap()
hv_theta_n = connectivity.Network_Analysis(
    arr_con_theta_hv, c_theta.ch_names, c_theta.threshold)
tys_theta_n = connectivity.Network_Analysis(
    arr_con_theta_tys, tys_theta.ch_names, tys_theta.threshold)
connectivity.Comparison_barplot_list(hv_theta_n.Centrality(
), tys_theta_n.Centrality(), c_theta.ch_names, Metric="Betweenness Centrality")
connectivity.Comparison_barplot_list(hv_theta_n.Strengths(
), tys_theta_n.Strengths(), c_theta.ch_names, Metric="Strengths")
connectivity.Comparison_barplot_nr(
    hv_theta_n.Transitivity(), tys_theta_n.Transitivity())
connectivity.Comparison_barplot_nr(
    hv_theta_n.Charpath_GE()[0], tys_theta_n.Charpath_GE()[0], "Charpath")
connectivity.Comparison_barplot_nr(
    hv_theta_n.Charpath_GE()[1], tys_theta_n.Charpath_GE()[1], "GE")


#### Low Freq post ##############
c_theta = connectivity.Connectivity(Epochs_HV)
tys_theta = connectivity.Connectivity(Epochs_TYS)
arr_con_theta_hv = c_theta.compute_connectivity(
    fmin=0, fmax=12, mode='multitaper')
arr_con_theta_tys = tys_theta.compute_connectivity(
    fmin=0, fmax=12, mode='multitaper')
c_theta.connectivity_plots()
tys_theta.connectivity_plots()
connectivity.connectivity_circle_subplots(
    arr_con_theta_hv, arr_con_theta_tys, c_theta.ch_names)
c_theta.con_heatmap()
tys_theta.con_heatmap()
hv_theta_n = connectivity.Network_Analysis(
    arr_con_theta_hv, c_theta.ch_names, c_theta.threshold)
tys_theta_n = connectivity.Network_Analysis(
    arr_con_theta_tys, tys_theta.ch_names, tys_theta.threshold)
connectivity.Comparison_barplot_list(hv_theta_n.Centrality(
), tys_theta_n.Centrality(), c_theta.ch_names, Metric="Betweenness Centrality")
connectivity.Comparison_barplot_list(hv_theta_n.Strengths(
), tys_theta_n.Strengths(), c_theta.ch_names, Metric="Strengths")
connectivity.Comparison_barplot_nr(
    hv_theta_n.Transitivity(), tys_theta_n.Transitivity())
connectivity.Comparison_barplot_nr(
    hv_theta_n.Charpath_GE()[0], tys_theta_n.Charpath_GE()[0], "Charpath")
connectivity.Comparison_barplot_nr(
    hv_theta_n.Charpath_GE()[1], tys_theta_n.Charpath_GE()[1], "GE")


#### beta pre ##############
c_beta = connectivity.Connectivity(Epochs_HV_copy)
tys_beta = connectivity.Connectivity(Epochs_TYS_copy)
arr_con_beta_hv = c_beta.compute_connectivity(fmin=12, fmax=30)
arr_con_beta_tys = tys_beta.compute_connectivity(fmin=12, fmax=30)
c_beta.connectivity_plots()
tys_beta.connectivity_plots()
connectivity.connectivity_circle_subplots(
    arr_con_beta_hv, arr_con_beta_tys, c_beta.ch_names)
c_beta.con_heatmap()
tys_beta.con_heatmap()
hv_beta_n = connectivity.Network_Analysis(
    arr_con_beta_hv, c_beta.ch_names, c_beta.threshold)
tys_beta_n = connectivity.Network_Analysis(
    arr_con_beta_tys, tys_beta.ch_names, tys_beta.threshold)
connectivity.Comparison_barplot_list(hv_beta_n.Centrality(
), tys_beta_n.Centrality(), c_beta.ch_names, Metric="Betweenness Centrality")
connectivity.Comparison_barplot_list(hv_beta_n.Strengths(
), tys_beta_n.Strengths(), c_beta.ch_names, Metric="Strengths")
connectivity.Comparison_barplot_nr(
    hv_beta_n.Transitivity(), tys_beta_n.Transitivity())
connectivity.Comparison_barplot_nr(
    hv_beta_n.Charpath_GE()[0], tys_beta_n.Charpath_GE()[0], "Charpath")
connectivity.Comparison_barplot_nr(
    hv_beta_n.Charpath_GE()[1], tys_beta_n.Charpath_GE()[1], "GE")

#### beta ##############
c_beta = connectivity.Connectivity(Epochs_HV)
tys_beta = connectivity.Connectivity(Epochs_TYS)
arr_con_beta_hv = c_beta.compute_connectivity(fmin=12, fmax=30)
arr_con_beta_tys = tys_beta.compute_connectivity(fmin=12, fmax=30)
c_beta.connectivity_plots()
tys_beta.connectivity_plots()
connectivity.connectivity_circle_subplots(
    arr_con_beta_hv, arr_con_beta_tys, c_beta.ch_names)
c_beta.con_heatmap()
tys_beta.con_heatmap()
hv_beta_n = connectivity.Network_Analysis(
    arr_con_beta_hv, c_beta.ch_names, c_beta.threshold)
tys_beta_n = connectivity.Network_Analysis(
    arr_con_beta_tys, tys_beta.ch_names, tys_beta.threshold)
connectivity.Comparison_barplot_list(hv_beta_n.Centrality(
), tys_beta_n.Centrality(), c_beta.ch_names, Metric="Betweenness Centrality")
connectivity.Comparison_barplot_list(hv_beta_n.Strengths(
), tys_beta_n.Strengths(), c_beta.ch_names, Metric="Strengths")
connectivity.Comparison_barplot_nr(
    hv_beta_n.Transitivity(), tys_beta_n.Transitivity())
connectivity.Comparison_barplot_nr(
    hv_beta_n.Charpath_GE()[0], tys_beta_n.Charpath_GE()[0], "Charpath")
connectivity.Comparison_barplot_nr(
    hv_beta_n.Charpath_GE()[1], tys_beta_n.Charpath_GE()[1], "GE")


#### gamma pre##############
c_gamma = connectivity.Connectivity(Epochs_HV_copy)
tys_gamma = connectivity.Connectivity(Epochs_TYS_copy)
arr_con_gamma_hv = c_gamma.compute_connectivity(fmin=30, fmax=80)
arr_con_gamma_tys = tys_gamma.compute_connectivity(fmin=30, fmax=80)
c_gamma.con_heatmap()
tys_gamma.con_heatmap()
c_gamma.connectivity_plots()
tys_gamma.connectivity_plots()
connectivity.connectivity_circle_subplots(
    arr_con_gamma_hv, arr_con_gamma_tys, c_gamma.ch_names)
hv_gamma_n = connectivity.Network_Analysis(
    arr_con_gamma_hv, c_gamma.ch_names, c_gamma.threshold)
tys_gamma_n = connectivity.Network_Analysis(
    arr_con_gamma_tys, tys_gamma.ch_names, tys_gamma.threshold)
connectivity.Comparison_barplot_list(hv_gamma_n.Centrality(
), tys_gamma_n.Centrality(), c_gamma.ch_names, Metric="Betweenness Centrality")
connectivity.Comparison_barplot_list(hv_gamma_n.Strengths(
), tys_gamma_n.Strengths(), c_gamma.ch_names, Metric="Strengths")
connectivity.Comparison_barplot_nr(
    hv_gamma_n.Transitivity(), tys_gamma_n.Transitivity())
connectivity.Comparison_barplot_nr(
    hv_gamma_n.Charpath_GE()[0], tys_gamma_n.Charpath_GE()[0], "Charpath")
connectivity.Comparison_barplot_nr(
    hv_gamma_n.Charpath_GE()[1], tys_gamma_n.Charpath_GE()[1], "GE")

#### gamma ##############
c_gamma = connectivity.Connectivity(Epochs_HV)
tys_gamma = connectivity.Connectivity(Epochs_TYS)
arr_con_gamma_hv = c_gamma.compute_connectivity(fmin=30, fmax=80)
arr_con_gamma_tys = tys_gamma.compute_connectivity(fmin=30, fmax=80)
c_gamma.con_heatmap()
tys_gamma.con_heatmap()
c_gamma.connectivity_plots()
tys_gamma.connectivity_plots()
connectivity.connectivity_circle_subplots(
    arr_con_gamma_hv, arr_con_gamma_tys, c_gamma.ch_names)
hv_gamma_n = connectivity.Network_Analysis(
    arr_con_gamma_hv, c_gamma.ch_names, c_gamma.threshold)
tys_gamma_n = connectivity.Network_Analysis(
    arr_con_gamma_tys, tys_gamma.ch_names, tys_gamma.threshold)
connectivity.Comparison_barplot_list(hv_gamma_n.Centrality(
), tys_gamma_n.Centrality(), c_gamma.ch_names, Metric="Betweenness Centrality")
connectivity.Comparison_barplot_list(hv_gamma_n.Strengths(
), tys_gamma_n.Strengths(), c_gamma.ch_names, Metric="Strengths")
connectivity.Comparison_barplot_nr(
    hv_gamma_n.Transitivity(), tys_gamma_n.Transitivity())
connectivity.Comparison_barplot_nr(
    hv_gamma_n.Charpath_GE()[0], tys_gamma_n.Charpath_GE()[0], "Charpath")
connectivity.Comparison_barplot_nr(
    hv_gamma_n.Charpath_GE()[1], tys_gamma_n.Charpath_GE()[1], "GE")

######################## ML Prep ###################################
X, y, info = ML.prep_epochs_for_ML(Epochs_HV, Epochs_TYS)

##################### ML Test #####################################
C = ML.Classification(X, y, info)
clf = C.Classifier()

Epochs = epochs.Epochs(epochs.read_epochs_set(
    r'C:\Users\testm\Documents\cleaned\FAT_Hv_04_FAT_T0_Fat_TEP_pre_final_avref.set'))
X = ML.prep_epochs_for_Predict(Epochs, crop="y")
print(ML.Prediction_Task(clf, X))






















######################## Statistical Analysis ##################
HV_list = epochs.read_list_epochs_set(
    r'C:\Users\testm\Documents\New_TMS_Data', subjects="Hv", csd_status=None)
TYS_list = epochs.read_list_epochs_set(
    r'C:\Users\testm\Documents\New_TMS_Data', subjects="TYS", csd_status=None)

############################################ Preparation ##########################################
HV_list_copy = copy.deepcopy(HV_list)
TYS_list_copy = copy.deepcopy(TYS_list)
HV_list_copy_2 = copy.deepcopy(HV_list)
TYS_list_copy_2 = copy.deepcopy(TYS_list)
###################################
for i in range(len(HV_list)):
    # Crop epochs for post stimuli analysis
    HV_list[i].crop_epochs(0.013, 0.300)
    HV_list_copy[i].crop_epochs(-0.650, -0.200)
    HV_list_copy_2[i].crop_epochs(0.013, 0.700)
for i in range(len(TYS_list)):
    # Crop epochs for post stimuli analysis
    TYS_list[i].crop_epochs(0.013, 0.300)
    TYS_list_copy[i].crop_epochs(-0.650, -0.350)
    TYS_list_copy_2[i].crop_epochs(0.013, 0.700)



###### Peaks ######
HV_ind=[]
HV_amps=[]

for i in range(len(HV_list)):
    Evoked_HV = evoked.Evoked(HV_list[i].average_epochs(),
                          HV_list[i].tmin, HV_list[i].csd)
    gfp = Evoked_HV.gfp()
    peaks=evoked.gfp_peaks(gfp, 1, HV_list[i].tmin, HV_list[i].sfreq)
    for e in peaks:
        
        HV_ind.append(e[1])
        HV_amps.append(e[0])
        
TYS_ind=[]
TYS_amps=[]


for i in range(len(TYS_list)):
    Evoked_TYS = evoked.Evoked(TYS_list[i].average_epochs(),
                          TYS_list[i].tmin, TYS_list[i].csd)
    gfp = Evoked_TYS.gfp()
    peaks=evoked.gfp_peaks(gfp, 1, TYS_list[i].tmin, TYS_list[i].sfreq)
    for e in peaks:
        TYS_ind.append(e[1])
        TYS_amps.append(e[0])

box_plot_data_max_ind=[HV_ind,TYS_ind]
plt.boxplot(box_plot_data_max_ind)      
U, p= scipy.stats.mannwhitneyu(HV_ind, TYS_ind,alternative='less')    
print(p)

##### Peaks ######
HV_15_25_ind=[]
HV_25_37_ind=[]
HV_37_52_ind=[]
HV_52_80_ind=[]
HV_71_100_ind=[]
HV_80_140_ind=[]
HV_150_220_ind=[]

HV_15_25_amps=[]
HV_25_37_amps=[]
HV_37_52_amps=[]
HV_52_80_amps=[]
HV_71_100_amps=[]
HV_80_140_amps=[]
HV_150_220_amps=[]
for i in range(len(HV_list)):
    Evoked_HV = evoked.Evoked(HV_list[i].average_epochs(),
                          HV_list[i].tmin, HV_list[i].csd)
    gfp = Evoked_HV.gfp()
    peaks=evoked.gfp_evoked(gfp, HV_list[i].tmin, HV_list[i].sfreq)
    for e in peaks:
        if e[1]>=0.015 and e[1]<0.025:
            HV_15_25_ind.append(e[1])
            HV_15_25_amps.append(e[0])
        if e[1]>=0.025 and e[1]<0.037:
            HV_25_37_ind.append(e[1])
            HV_25_37_amps.append(e[0])
        if e[1]>=0.037 and e[1]<0.055:
            HV_37_52_ind.append(e[1])
            HV_37_52_amps.append(e[0])
        if e[1]>=0.055 and e[1]<0.070:
            HV_52_80_ind.append(e[1])
            HV_52_80_amps.append(e[0])
        if e[1]>=0.071 and e[1]<0.100:
            HV_71_100_ind.append(e[1])
            HV_71_100_amps.append(e[0])
        if e[1]>=0.090 and e[1]<0.140:
            HV_80_140_ind.append(e[1])
            HV_80_140_amps.append(e[0])
        if e[1]>=0.150 and e[1]<0.220:
            HV_150_220_ind.append(e[1])
            HV_150_220_amps.append(e[0]) 

TYS_15_25_ind=[]
TYS_25_37_ind=[]
TYS_37_52_ind=[]
TYS_52_80_ind=[]
TYS_71_100_ind=[]
TYS_80_140_ind=[]
TYS_150_220_ind=[]

TYS_15_25_amps=[]
TYS_25_37_amps=[]
TYS_37_52_amps=[]
TYS_52_80_amps=[]
TYS_71_100_amps=[]
TYS_80_140_amps=[]
TYS_150_220_amps=[]
for i in range(len(TYS_list)):
    Evoked_TYS = evoked.Evoked(TYS_list[i].average_epochs(),
                          TYS_list[i].tmin, TYS_list[i].csd)
    gfp = Evoked_TYS.gfp()
    peaks=evoked.gfp_evoked(gfp, TYS_list[i].tmin, TYS_list[i].sfreq)
    for e in peaks:
        if e[1]>=0.015 and e[1]<0.025:
            TYS_15_25_ind.append(e[1])
            TYS_15_25_amps.append(e[0])        
        if e[1]>=0.025 and e[1]<0.037:
            TYS_25_37_ind.append(e[1])
            TYS_25_37_amps.append(e[0])
        if e[1]>=0.037 and e[1]<0.055:
            TYS_37_52_ind.append(e[1])
            TYS_37_52_amps.append(e[0])
        if e[1]>=0.055 and e[1]<0.070:
            TYS_52_80_ind.append(e[1])
            TYS_52_80_amps.append(e[0])
        if e[1]>=0.071 and e[1]<0.100:
            TYS_71_100_ind.append(e[1])
            TYS_71_100_amps.append(e[0])
        if e[1]>=0.090 and e[1]<0.140:
            TYS_80_140_ind.append(e[1])
            TYS_80_140_amps.append(e[0])
        if e[1]>=0.150 and e[1]<0.220:
            TYS_150_220_ind.append(e[1])
            TYS_150_220_amps.append(e[0])

box_plot_data_P20_ind=[HV_15_25_ind,TYS_15_25_ind]
plt.boxplot(box_plot_data_P20_ind)
box_plot_data_P20_amps=[HV_15_25_amps,TYS_15_25_amps]
plt.boxplot(box_plot_data_P20_ind)

U, p= scipy.stats.mannwhitneyu(HV_15_25_ind, TYS_15_25_ind,alternative='less')    
print(p)
U, p= scipy.stats.mannwhitneyu(HV_15_25_amps, TYS_15_25_amps,alternative='less')    
print(p)

plt.close('all')
plt.Figure()
box_plot_data_P30_ind=[HV_25_37_ind,TYS_25_37_ind]
plt.boxplot(box_plot_data_P30_ind)
box_plot_data_P30_amps=[HV_25_37_amps,TYS_25_37_amps]
plt.boxplot(box_plot_data_P30_amps)

U, p= scipy.stats.mannwhitneyu(HV_25_37_ind, TYS_25_37_ind,alternative='less')    
print(p)
U, p= scipy.stats.mannwhitneyu(HV_25_37_amps, TYS_25_37_amps,alternative='less')    
print(p)

box_plot_data_N45_ind=[HV_37_52_ind,TYS_37_52_ind]
plt.boxplot(box_plot_data_N45_ind)
box_plot_data_N45_amps=[HV_37_52_amps,TYS_37_52_amps]
plt.boxplot(box_plot_data_N45_amps)

U, p= scipy.stats.mannwhitneyu(HV_37_52_ind, TYS_37_52_ind,alternative='less')    
print(p)
U, p= scipy.stats.mannwhitneyu(HV_37_52_amps, TYS_37_52_amps,alternative='less')    
print(p)


box_plot_data_P60_ind=[HV_52_80_ind,TYS_52_80_ind]
plt.boxplot(box_plot_data_P60_ind)
box_plot_data_P60_amps=[HV_52_80_amps,TYS_52_80_amps]
plt.boxplot(box_plot_data_P60_amps)

U, p= scipy.stats.mannwhitneyu(HV_52_80_ind, TYS_52_80_ind,alternative='less')    
print(p)
U, p= scipy.stats.mannwhitneyu(HV_52_80_amps, TYS_52_80_amps,alternative='less')    
print(p)

box_plot_data_P90_ind=[HV_71_100_ind,TYS_71_100_ind]
plt.boxplot(box_plot_data_P90_ind)
box_plot_data_P90_amps=[HV_71_100_amps,TYS_71_100_amps]
plt.boxplot(box_plot_data_P90_amps)

U, p= scipy.stats.mannwhitneyu(HV_71_100_ind, TYS_71_100_ind,alternative='less')    
print(p)
U, p= scipy.stats.mannwhitneyu(HV_71_100_amps, TYS_71_100_amps,alternative='less')    
print(p)

box_plot_data_N100_ind=[HV_80_140_ind,TYS_80_140_ind]
plt.boxplot(box_plot_data_N100_ind)
box_plot_data_N100_amps=[HV_80_140_amps,TYS_80_140_amps]
plt.boxplot(box_plot_data_N100_amps)

U, p= scipy.stats.mannwhitneyu(HV_80_140_ind, TYS_80_140_ind,alternative='less')    
print(p)
U, p= scipy.stats.mannwhitneyu(HV_80_140_amps, TYS_80_140_amps,alternative='less')    
print(p)


box_plot_data_P180_ind=[HV_150_220_ind,TYS_150_220_ind]
plt.boxplot(box_plot_data_P180_ind)
box_plot_data_P180_amps=[HV_150_220_amps,TYS_150_220_amps]
plt.boxplot(box_plot_data_P180_amps)

U, p= scipy.stats.mannwhitneyu(HV_150_220_ind, TYS_150_220_ind,alternative='less')    
print(p)
U, p= scipy.stats.mannwhitneyu(HV_150_220_amps, TYS_150_220_amps,alternative='less')    
print(p)


###### PSD Bands ######
HV_ERSC = []
for i in range(len(HV_list)):
    HV_ERSC.append(np.array(HV_list[i].psd_bands()) -
                   np.array(HV_list_copy[i].psd_bands()))

HV_ERSC_df = pd.DataFrame(HV_ERSC)

TYS_ERSC = []
for i in range(len(TYS_list)):
    TYS_ERSC.append(
        np.array(TYS_list[i].psd_bands())-np.array(TYS_list_copy[i].psd_bands()))

HV_ERSC_df = pd.DataFrame(HV_ERSC, columns =['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
TYS_ERSC_df = pd.DataFrame(TYS_ERSC, columns =['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])

U1, p=scipy.stats.mannwhitneyu(HV_ERSC_df["Delta"], TYS_ERSC_df["Delta"],alternative='less')
print(p)
combined_dfs = pd.DataFrame({'df1': HV_ERSC_df["Delta"],
                             'df2': TYS_ERSC_df["Delta"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

U1, p=scipy.stats.mannwhitneyu(HV_ERSC_df["Theta"], TYS_ERSC_df["Theta"],alternative='less')
print(p)
combined_dfs = pd.DataFrame({'df1': HV_ERSC_df["Theta"],
                             'df2': TYS_ERSC_df["Theta"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

U1, p=scipy.stats.mannwhitneyu(HV_ERSC_df["Alpha"], TYS_ERSC_df["Alpha"],alternative='less')
print(p)
combined_dfs = pd.DataFrame({'df1': HV_ERSC_df["Alpha"],
                             'df2': TYS_ERSC_df["Alpha"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

U1, p=scipy.stats.mannwhitneyu(HV_ERSC_df["Beta"], TYS_ERSC_df["Beta"],alternative='greater')
print(p)
combined_dfs = pd.DataFrame({'df1': HV_ERSC_df["Beta"],
                             'df2': TYS_ERSC_df["Beta"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

U1, p=scipy.stats.mannwhitneyu(HV_ERSC_df["Gamma"], TYS_ERSC_df["Gamma"],alternative='greater')
print(p)
combined_dfs = pd.DataFrame({'df1': HV_ERSC_df["Gamma"],
                             'df2': TYS_ERSC_df["Gamma"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

###### Connectivity ######
HV_Connectivity_post = []

for i in range(len(HV_list)):
    c = connectivity.Connectivity(HV_list_copy_2[i])
    c_array= c.compute_connectivity(fmin=0, fmax=80, mode='multitaper')
    hv = connectivity.Network_Analysis(c_array, c.ch_names, c.threshold)
    t=hv.Transitivity()
    char=hv.Charpath_GE()[0]
    ge=hv.Charpath_GE()[1]
    HV_Connectivity_post.append([t,char,ge])

TYS_Connectivity_post = []

for i in range(len(TYS_list)):
    c = connectivity.Connectivity(TYS_list_copy_2[i])
    c_array= c.compute_connectivity(fmin=0, fmax=80, mode='multitaper')
    hv = connectivity.Network_Analysis(c_array, c.ch_names, c.threshold)
    t=hv.Transitivity()
    char=hv.Charpath_GE()[0]
    ge=hv.Charpath_GE()[1]
    TYS_Connectivity_post.append([t,char,ge])

HV_Connectivity_pos_df = pd.DataFrame(HV_Connectivity_post, columns =['Transitivity', 'Charpath', 'GE'])

TYS_Connectivity_pos_df = pd.DataFrame(TYS_Connectivity_post, columns =['Transitivity', 'Charpath', 'GE'])

U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["Transitivity"], TYS_Connectivity_pos_df["Transitivity"], alternative="less")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["Transitivity"],
                             'df2': TYS_Connectivity_pos_df["Transitivity"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()


U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["Charpath"], TYS_Connectivity_pos_df["Charpath"], alternative="greater")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["Charpath"],
                             'df2': TYS_Connectivity_pos_df["Charpath"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["GE"], TYS_Connectivity_pos_df["GE"], alternative="less")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["GE"],
                             'df2': TYS_Connectivity_pos_df["GE"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

##################### Low Frequency #################################################
HV_Connectivity_post = []

for i in range(len(HV_list)):
    c = connectivity.Connectivity(HV_list_copy_2[i])
    c_array= c.compute_connectivity(fmin=8, fmax=12, mode='multitaper')
    hv = connectivity.Network_Analysis(c_array, c.ch_names, c.threshold)
    t=hv.Transitivity()
    char=hv.Charpath_GE()[0]
    ge=hv.Charpath_GE()[1]
    HV_Connectivity_post.append([t,char,ge])

TYS_Connectivity_post = []

for i in range(len(TYS_list)):
    c = connectivity.Connectivity(TYS_list_copy_2[i])
    c_array= c.compute_connectivity(fmin=8, fmax=12, mode='multitaper')
    hv = connectivity.Network_Analysis(c_array, c.ch_names, c.threshold)
    t=hv.Transitivity()
    char=hv.Charpath_GE()[0]
    ge=hv.Charpath_GE()[1]
    TYS_Connectivity_post.append([t,char,ge])

HV_Connectivity_pos_df = pd.DataFrame(HV_Connectivity_post, columns =['Transitivity', 'Charpath', 'GE'])
TYS_Connectivity_pos_df = pd.DataFrame(TYS_Connectivity_post, columns =['Transitivity', 'Charpath', 'GE'])

U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["Transitivity"], TYS_Connectivity_pos_df["Transitivity"], alternative="less")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["Transitivity"],
                             'df2': TYS_Connectivity_pos_df["Transitivity"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()


U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["Charpath"], TYS_Connectivity_pos_df["Charpath"], alternative="greater")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["Charpath"],
                             'df2': TYS_Connectivity_pos_df["Charpath"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["GE"], TYS_Connectivity_pos_df["GE"], alternative="less")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["GE"],
                             'df2': TYS_Connectivity_pos_df["GE"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

##################### Beta #################################################
HV_Connectivity_post = []

for i in range(len(HV_list)):
    c = connectivity.Connectivity(HV_list_copy_2[i])
    c_array= c.compute_connectivity(fmin=12, fmax=30, mode='multitaper')
    hv = connectivity.Network_Analysis(c_array, c.ch_names, c.threshold)
    t=hv.Transitivity()
    char=hv.Charpath_GE()[0]
    ge=hv.Charpath_GE()[1]
    HV_Connectivity_post.append([t,char,ge])

TYS_Connectivity_post = []

for i in range(len(TYS_list)):
    c = connectivity.Connectivity(TYS_list_copy_2[i])
    c_array= c.compute_connectivity(fmin=12, fmax=30, mode='multitaper')
    hv = connectivity.Network_Analysis(c_array, c.ch_names, c.threshold)
    t=hv.Transitivity()
    char=hv.Charpath_GE()[0]
    ge=hv.Charpath_GE()[1]
    TYS_Connectivity_post.append([t,char,ge])

HV_Connectivity_pos_df = pd.DataFrame(HV_Connectivity_post, columns =['Transitivity', 'Charpath', 'GE'])
TYS_Connectivity_pos_df = pd.DataFrame(TYS_Connectivity_post, columns =['Transitivity', 'Charpath', 'GE'])

U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["Transitivity"], TYS_Connectivity_pos_df["Transitivity"], alternative="less")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["Transitivity"],
                             'df2': TYS_Connectivity_pos_df["Transitivity"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()


U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["Charpath"], TYS_Connectivity_pos_df["Charpath"], alternative="greater")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["Charpath"],
                             'df2': TYS_Connectivity_pos_df["Charpath"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["GE"], TYS_Connectivity_pos_df["GE"], alternative="less")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["GE"],
                             'df2': TYS_Connectivity_pos_df["GE"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()
##################### Gamma #################################################
HV_Connectivity_post = []

for i in range(len(HV_list)):
    c = connectivity.Connectivity(HV_list_copy_2[i])
    c_array= c.compute_connectivity(fmin=30, fmax=80, mode='multitaper')
    hv = connectivity.Network_Analysis(c_array, c.ch_names, c.threshold)
    t=hv.Transitivity()
    char=hv.Charpath_GE()[0]
    ge=hv.Charpath_GE()[1]
    HV_Connectivity_post.append([t,char,ge])

TYS_Connectivity_post = []

for i in range(len(TYS_list)):
    c = connectivity.Connectivity(TYS_list_copy_2[i])
    c_array= c.compute_connectivity(fmin=30, fmax=80, mode='multitaper')
    hv = connectivity.Network_Analysis(c_array, c.ch_names, c.threshold)
    t=hv.Transitivity()
    char=hv.Charpath_GE()[0]
    ge=hv.Charpath_GE()[1]
    TYS_Connectivity_post.append([t,char,ge])
HV_Connectivity_pos_df = pd.DataFrame(HV_Connectivity_post, columns =['Transitivity', 'Charpath', 'GE'])
TYS_Connectivity_pos_df = pd.DataFrame(TYS_Connectivity_post, columns =['Transitivity', 'Charpath', 'GE'])

U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["Transitivity"], TYS_Connectivity_pos_df["Transitivity"], alternative="less")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["Transitivity"],
                             'df2': TYS_Connectivity_pos_df["Transitivity"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()


U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["Charpath"], TYS_Connectivity_pos_df["Charpath"], alternative="greater")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["Charpath"],
                             'df2': TYS_Connectivity_pos_df["Charpath"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

U1, p=scipy.stats.mannwhitneyu(HV_Connectivity_pos_df["GE"], TYS_Connectivity_pos_df["GE"], alternative="less")
print(p)
combined_dfs = pd.DataFrame({'df1': HV_Connectivity_pos_df["GE"],
                             'df2': TYS_Connectivity_pos_df["GE"]})
sns.set_style('white')
sns.boxplot(data=combined_dfs, palette='flare')
sns.despine()
plt.show()

################################ New ML ##########################################
# from random import sample
# import random


# accu_per_sample=[]
# stdev_per_sample=[]
# conf_matrix_per_sample=[]
# for number_of_samples in range(60,80):
#     acc=[]
#     conf_mat_l=[]
#     HV_Trains=[]
#     TYS_Trains=[]
#     HV_Leave_Outs_Epochs=[]
#     TYS_Leave_Outs_Epochs=[]
#     for i in range(10):
#         shuffled_HV = random.sample(HV_list, len(HV_list))
#         HV_Training = shuffled_HV[:len(shuffled_HV)-4]
#         HV_Leave_Out = shuffled_HV[len(shuffled_HV)-4:]
#         HV_sample=epochs.epochs_in_list_splitter(HV_Training)
#         HV_sample=sample(HV_sample,len(HV_sample))
#         Evoked_HV_list=[]
#         for i in range(len(HV_sample)):
#             Evoked_HV = evoked.Evoked(HV_sample[i].average_epochs(),
#                                   HV_sample[i].tmin, HV_sample[i].csd)
#             Evoked_HV_list.append(Evoked_HV.data)    
#         Epochs_HV_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_HV_list),Evoked_HV.channels,Evoked_HV.sfreq,Evoked_HV.tmin))
#         HV_Trains.append(Epochs_HV_list)
        
#         # HV_Leave_Out = shuffled_HV[len(shuffled_HV)-4:]
#         # HV_test=epochs.epochs_in_list_splitter(HV_Leave_Out)
#         # Evoked_HV_list_test=[]
#         # for i in range(len(HV_test)):
#         #     Evoked_HV_test = evoked.Evoked(HV_test[i].average_epochs(),
#         #                           HV_test[i].tmin, HV_test[i].csd)
#         #     Evoked_HV_list_test.append(Evoked_HV_test.data)    
#         # Epochs_HV_test_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_HV_list_test),Evoked_HV_test.channels,Evoked_HV_test.sfreq,Evoked_HV_test.tmin))
        
        
#         shuffled_TYS = random.sample(TYS_list, len(TYS_list))
#         TYS_Training = shuffled_TYS[:len(shuffled_TYS)-4]
#         TYS_Leave_Out = shuffled_TYS[len(shuffled_TYS)-4:]
#         TYS_sample=epochs.epochs_in_list_splitter(TYS_Training)
#         TYS_sample=sample(TYS_sample,number_of_samples)
#         Evoked_TYS_list=[]
#         for i in range(len(TYS_sample)):
#             Evoked_TYS = evoked.Evoked(TYS_sample[i].average_epochs(),
#                                   TYS_sample[i].tmin, TYS_sample[i].csd)
#             Evoked_TYS_list.append(Evoked_TYS.data)    
#         Epochs_TYS_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_TYS_list),Evoked_TYS.channels,Evoked_TYS.sfreq,Evoked_TYS.tmin))
#         TYS_Trains.append(Epochs_TYS_list)
        
        
#         X, y, info = ML.prep_epochs_for_ML(Epochs_HV_list, Epochs_TYS_list)
#         C = ML.Classification(X, y, info)
#         clf,accuracy, conf_mat = C.Classifier()
#         acc.append(accuracy)
#         conf_mat_l.append(conf_mat)

    
#     accu_per_sample.append(statistics.mean(acc))
#     stdev_per_sample.append(statistics.stdev(acc))
#     conf_matrix_per_sample.append(conf_mat_l)


from random import sample
import random

acc=[]
conf_mat_l=[]
HV_Trains=[]
TYS_Trains=[]
HV_Leave_Outs_Epochs=[]
TYS_Leave_Outs_Epochs=[]
labels_per_model=[]
for i in range(100):
    shuffled_HV = random.sample(HV_list, len(HV_list))
    HV_Training = shuffled_HV[:len(shuffled_HV)-4]
    HV_Leave_Out = shuffled_HV[len(shuffled_HV)-4:]
    HV_sample=epochs.epochs_in_list_splitter(HV_Training)
    HV_sample=sample(HV_sample,len(HV_sample))
    Evoked_HV_list=[]
    for i in range(len(HV_sample)):
        Evoked_HV = evoked.Evoked(HV_sample[i].average_epochs(),
                              HV_sample[i].tmin, HV_sample[i].csd)
        Evoked_HV_list.append(Evoked_HV.data)    
    Epochs_HV_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_HV_list),Evoked_HV.channels,Evoked_HV.sfreq,Evoked_HV.tmin))
    HV_Trains.append(Epochs_HV_list)
    
    HV_test=epochs.epochs_in_list_splitter(HV_Leave_Out)
    Evoked_HV_list_test=[]
    for i in range(len(HV_Leave_Out)):
        Evoked_HV_test = evoked.Evoked(HV_Leave_Out[i].average_epochs(),
                              HV_Leave_Out[i].tmin, HV_Leave_Out[i].csd)
        Evoked_HV_list_test.append(Evoked_HV_test.data)    
    Epochs_HV_test_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_HV_list_test),Evoked_HV_test.channels,Evoked_HV_test.sfreq,Evoked_HV_test.tmin))
    
    shuffled_TYS = random.sample(TYS_list, len(TYS_list))
    TYS_Training = shuffled_TYS[:len(shuffled_TYS)-4]
    TYS_Leave_Out = shuffled_TYS[len(shuffled_TYS)-4:]
    TYS_sample=epochs.epochs_in_list_splitter(TYS_Training)
    TYS_sample=sample(TYS_sample,70)
    Evoked_TYS_list=[]
    for i in range(len(TYS_sample)):
        Evoked_TYS = evoked.Evoked(TYS_sample[i].average_epochs(),
                              TYS_sample[i].tmin, TYS_sample[i].csd)
        Evoked_TYS_list.append(Evoked_TYS.data)    
    Epochs_TYS_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_TYS_list),Evoked_TYS.channels,Evoked_TYS.sfreq,Evoked_TYS.tmin))
    TYS_Trains.append(Epochs_TYS_list)
    
    TYS_test=epochs.epochs_in_list_splitter(TYS_Leave_Out)
    Evoked_TYS_list_test=[]
    for i in range(len(TYS_Leave_Out)):
        Evoked_TYS_test = evoked.Evoked(TYS_Leave_Out[i].average_epochs(),
                              TYS_Leave_Out[i].tmin, TYS_Leave_Out[i].csd)
        Evoked_TYS_list_test.append(Evoked_TYS_test.data)    
    Epochs_TYS_test_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_TYS_list_test),Evoked_TYS_test.channels,Evoked_TYS_test.sfreq,Evoked_TYS_test.tmin))
    
    X, y, info = ML.prep_epochs_for_ML(Epochs_HV_list, Epochs_TYS_list)
    C = ML.Classification(X, y, info)
    clf,accuracy, conf_mat = C.Classifier()
    acc.append(accuracy)
    conf_mat_l.append(conf_mat)
    
    X, y, info = ML.prep_epochs_for_ML(Epochs_HV_test_list, Epochs_TYS_test_list)
    labels=ML.Prediction_Task_Evoked(clf, X)
    labels_per_model.append(labels)
print(statistics.mean(acc))
print(statistics.stdev(acc))
print(conf_mat_l)

Expected=["Healthy","Healthy","Healthy","Healthy","MS","MS","MS","MS"]
NR=len(Expected)
test_accuracy=[]
for i in range(len(labels_per_model)):
    print(labels_per_model[i])
    s=0
    for j in range(len(labels_per_model[i])):
        if labels_per_model[i][j]==Expected[j]:
            print(labels_per_model[i][j])
            s=s+1
            print(s)
    test_accuracy.append(s/NR)

print(statistics.mean(test_accuracy))            

MS_class_MS=0
HV_class_MS=0
HV_class_HV=0
MS_class_HV=0
for i in range(len(labels_per_model)):
    s=0
    for j in range(len(labels_per_model[i])):
        if labels_per_model[i][j]=="Healthy" and Expected[j]=="Healthy":
            HV_class_HV=HV_class_HV+1
        elif labels_per_model[i][j]=="MS" and Expected[j]=="MS":
            MS_class_MS=MS_class_MS+1
        elif labels_per_model[i][j]=="MS" and Expected[j]=="Healthy":
            HV_class_MS=HV_class_MS+1
        elif labels_per_model[i][j]=="Healthy" and Expected[j]=="MS":
            MS_class_HV=MS_class_HV+1

print("MS_class_MS:", MS_class_MS)
print("HV_class_HV:", HV_class_HV)
print("MS_class_HV:", MS_class_HV)
print("HV_class_MS:", HV_class_MS)







##################### Preprocessing ##########################################




E = epochs.Epochs(epochs.read_epochs_set(
    r'C:\Users\testm\Documents\TMS_EEG_SM_DL-before-cleaning\FAT_HV_04_minus5to10CubicInterpolated_dwnsampled1000.set'))

epochs_cropped = E.crop_epochs(-1, 1)
epochs_cropped.epochs_plot()
epochs_1 = epochs_cropped.ICA_part_1(20)
# epochs_1.ICA_plots()
print(epochs_1.Auto_ICA_Eval())
epochs_2 = epochs_1.ICA_part_2()
epochs_2.epochs_plot()
filt_epochs = epochs_2.filtering_preprocessing(
    ["n", "bp"], n_cut_freq=50, l_freq=2, h_freq=90)
filt_epochs.epochs_plot()
# filt_epochs.psd_plot()
epochs_1 = filt_epochs.ICA_part_1(20)
# epochs_1.ICA_plots()
print(epochs_1.Auto_ICA_Eval())
epochs_2 = epochs_1.ICA_part_2()
epochs_2.epochs_plot()
epochs_2.bad_epochs_rejection(threshold=220e-6)
interp_epochs = epochs_2.bad_channel_interpolation([])

# avg_epochs=interp_epochs.average_reference()
epochs_ref = epochs_2.surface_laplacian_transform()
epochs_ref.epochs_plot()
epochs_bc = epochs_ref.baseline_correction()
epochs_bc.epochs_plot()
# epochs_bc.epochs_plot()
# epochs_ref=epochs_bc.surface_laplacian_transform()

epochs_ref.epochs_plot()
epochs_ref.image_plot()
epochs_bc.psd_plot()
epochs_bc.psd_topo_plot()
epochs_bc.topo_plot()
epochs_bc.morlet_tfr(4, 30, power_plot="y", joint_plot="y",
                     itc_plot="y", timefreqs=None, baseline=[-1, 0])
