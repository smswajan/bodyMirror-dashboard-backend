# -*- coding: utf-8 -*-
"""
Code Developed by Pedro Gomes for Myelin-H company

Windows 11
MNE 1.2.1
Python 3.9

"""

############## Imports ###################
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import statistics as st
import evoked
import epochs

def prep_epochs_for_ML(HV_epochs, MS_epochs, crop="y"):
    """
    Epochs preparation for the fitting

    Parameters
    ----------
    HV_epochs : epochs.Epochs
        HV subjects
    MS_epochs : epochs.Epochs
        MS Subjects
    crop: string
        If 'y' data will be cropped from the 0.015 seconds to the 0.250 which is the period where classification is maximized

    Returns
    -------
    X : numpy.array
        Training Set.
    y : numpy.array
        Training labels Set.
    info : Epochs info

    """
    # crop epochs if specified on the interval of usual TMS EEG stimulation
    if crop=="y":
        HV_epochs.crop_epochs(0.015,0.250)
        MS_epochs.crop_epochs(0.015,0.250)
    #get data for the fittings    
    info=HV_epochs.info
    X1 = HV_epochs.data
    X2 = MS_epochs.data
    y1 = np.ones(X1.shape[0])
    y2 = np.zeros(X2.shape[0])
    X=np.concatenate((X1, X2), axis=0)
    y=np.concatenate((y1, y2), axis=0)
    return X, y,info
    
def prep_epochs_for_Predict(Epochs, crop="y"):
    """
    Epochs preparation for the fitting

    Parameters
    ----------
    Epochs : epochs.Epochs
        Epochs to predict


    crop: string
        If 'y' data will be cropped from the 0.015 seconds to the 0.250 which is the period where classification is maximized

    Returns
    -------
    X : numpy.array
        Set to predict.

    """
    # crop epochs if specified on the interval of usual TMS EEG stimulation
    if crop=="y":
        Epochs.crop_epochs(0.015,0.250)
    #get data for the fittings    
    info=Epochs.info
    X = Epochs.data
    return X

def prep_list_of_epochs(list_epochs_HV,list_epochs_TYS, crop="y"):
    """
    List of Epochs preparation for the evoked ML fitting

    Parameters
    ----------
    list_epochs_HV : list of epochs.Epochs objects
        HV subjects epochs.
    list_epochs_TYS : list of epochs.Epochs objects
        MS subjects epochs.
    crop: string
        If 'y' data will be cropped from the 0.015 seconds to the 0.250 which is the period where classification is maximized

    Returns
    -------
    X : numpy.array
        Training Set.
    y : numpy.array
        Training labels Set.
    info : Epochs info

    """
    Evoked_HV_list=[]
    for i in range(len(list_epochs_HV)):
        Evoked_HV = evoked.Evoked(list_epochs_HV[i].average_epochs(),
                              list_epochs_HV[i].tmin, list_epochs_HV[i].csd)
        Evoked_HV_list.append(Evoked_HV.data)    
    Epochs_HV_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_HV_list),Evoked_HV.channels,Evoked_HV.sfreq,Evoked_HV.tmin))
    Evoked_TYS_list=[]
    for i in range(len(list_epochs_TYS)):
        Evoked_TYS = evoked.Evoked(list_epochs_TYS[i].average_epochs(),
                              list_epochs_TYS[i].tmin, list_epochs_TYS[i].csd)
        Evoked_TYS_list.append(Evoked_TYS.data)    
    Epochs_TYS_list=epochs.Epochs(epochs.numpy_to_epochs(np.array(Evoked_TYS_list),Evoked_TYS.channels,Evoked_TYS.sfreq,Evoked_TYS.tmin))
    X, y, info = prep_epochs_for_ML(Epochs_HV_list, Epochs_TYS_list, crop=crop)
    return X, y, info
    
class Classification():
    # Class where all the classifiers will be saved
    def __init__(self, X, y, info): 
        self.X=X #data
        self.y=y #label
        self.info=info #Info of the epochs
        
    def Classifier(self, solver="liblinear"):
        """
        Classifier using Scaler, Vectorizer, LogisticRegression. Classifies if each epoch is from a MS Patient or a HV

        Parameters
        ----------
        solver : str, optional
            Solver for the  Logistic Regression. The default is "liblinear". Check sklearn.linear_model.LogisticRegression solver documentation.

        Returns
        -------
        clf : Machine Learning model.

        """
        #assembly classifier
        clf = make_pipeline(Scaler(self.info),Vectorizer(),LogisticRegression(solver=solver))
        
        #fitting training data
        clf.fit(self.X, self.y)
        
        #cross validation on training data
        scores = cross_val_multiscore(clf, self.X, self.y, cv=10, n_jobs=1)
        y_pred = cross_val_predict(clf, self.X, self.y, cv=10)
        conf_mat = confusion_matrix(self.y, y_pred)
        print(conf_mat)
        print(np.mean(scores))
    
        
        return clf, np.mean(scores), conf_mat
    
    def Classifier_fit(self, solver="liblinear"):
        """
        Classifier using Scaler, Vectorizer, LogisticRegression. Classifies if each epoch is from a MS Patient or a HV

        Parameters
        ----------
        solver : str, optional
            Solver for the  Logistic Regression. The default is "liblinear". Check sklearn.linear_model.LogisticRegression solver documentation.

        Returns
        -------
        clf : Machine Learning model.

        """
        #assembly classifier
        clf = make_pipeline(Scaler(self.info),Vectorizer(),LogisticRegression(solver=solver))
        
        #fitting training data
        clf.fit(self.X, self.y)
        
        return clf

def Prediction_Task(clf,X):
    """
    Predicting if the epochs belong to a Healthy or MS patient.

    Parameters
    ----------
    clf : sklearn.model
        model where the data was fitted.
    X : epochs in np.array
        Set for prediction.

    Returns
    -------
    label : str
        "Healthy or MS".

    """
    y=clf.predict(X)
    y_mode=st.mode(y)
    if y_mode==1:
        label="Healthy"
    else:
        label="MS"
    return label

def Prediction_Task_Evoked(clf,X):
    """
    Predicting if the evoked belong to a Healthy or MS patient.

    Parameters
    ----------
    clf : sklearn.model
        model where the data was fitted.
    X : evoked in np.array
        Set for prediction.

    Returns
    -------
    label : str
        "Healthy or MS".

    """
    y=clf.predict(X)
    labels=[]
    for e in y:
        if e==1:
            labels.append("Healthy")
        else:
            labels.append("MS")
    return labels