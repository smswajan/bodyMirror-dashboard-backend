from django.shortcuts import render
import requests
# Create your views here.
import mne
from mne import Epochs, pick_types, events_from_annotations
import matplotlib.pyplot as plt 
import os
import urllib.request
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import pyrebase





firebaseConfig = {
    "apiKey": "AIzaSyBWvRHlqzYpPIySvieuTSQdaRN1gi19GCA",
    "authDomain": "myelin-ltd.firebaseapp.com",
    "databaseURL": "https://myelin-ltd-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "myelin-ltd",
    "storageBucket": "myelin-ltd.appspot.com",
    "messagingSenderId": "614148529430",
    "appId": "1:614148529430:web:c6ead0a80ce6f76d9a0437"
}
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()


def analysis(request):
    return render(request, 'index.html')

@api_view(['POST'])
def api_test(request):
    responses = {}
    if request.method == 'POST':
        data=request.data
        # timeDomainAnalysis(data)


        print(data)
        analysisID = data.get('analysisID',None)
        rawData = data.get('rawData',None)
        if rawData is None:
            responses ['message'] = "Raw data is required"
            return Response(responses, status=status.HTTP_400_BAD_REQUEST)
        timeDomainAndFrequencyAnalysis(data)
        responses ['data'] = data
    responses['message'] = 'response successful'
    return Response(responses, status=status.HTTP_200_OK)



def sourceLocalization(data):
    subjects_dir = 'freesurfer/subjects/'
    subject = 'daniel'
    bem_dir = op.join(subjects_dir, subject, 'bem')
    mne.utils.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)

    maps = mne.make_field_map(evks['Healthy'], trans='freesurfer/subjects/daniel/daniel-trans-z-trans.fif',
                            subject= 'daniel', subjects_dir=subjects_dir)
    evks['Healthy'].plot_field(maps, time=0.01)
    evks['Healthy'].plot_field(maps, time=-0.5)
    evks['Healthy'].plot_field(maps, time=0)
    evks['Healthy'].plot_field(maps, time=0.05)



def timeDomainAndFrequencyAnalysis(data):
    # retriving data from firebase
    analysisID = data.get('analysisID',None)
    rawData = data.get('rawData',None)
    urllib.request.urlretrieve(rawData, '/static/dataOne.set')
    h1= r"static/dataOne.set"
    raw_h1=mne.read_epochs_eeglab(h1)
    raw_h1=raw_h1.set_eeg_reference('average', projection=True)

    epochs_h1=mne.read_epochs_eeglab(h1)
    epochs_h1.set_eeg_reference('average', projection=True)
    epochs_h1.apply_proj()
    montage=raw_h1.get_montage()
    epochs_h1.set_montage(montage)
    epochs_h1=epochs_h1.filter(l_freq=1, h_freq=None)
    epochs_h1=epochs_h1.filter(l_freq=1, h_freq=100)

    ########################################################################################
    #using a second set from firebase
    urllib.request.urlretrieve('https://firebasestorage.googleapis.com/v0/b/myelin-ltd.appspot.com/o/data%2FFAT_Hv_01_FAT_T0_Fat_TEP_pre_final_avref.set?alt=media&token=31ca2f75-36a8-4292-bc91-97aad33aee1d', '/static/dataTwo.set')

    h2= r"static/dataTwo.set"

    raw_h2=mne.read_epochs_eeglab(h2)
    raw_h2=raw_h2.set_eeg_reference('average', projection=True)

    # topographic maps
    epochs_h2=mne.read_epochs_eeglab(h2)
    epochs_h2.set_eeg_reference('average', projection=True)
    epochs_h2.apply_proj()
    montage=raw_h2.get_montage()
    epochs_h2.set_montage(montage)
    epochs_h2=epochs_h2.filter(l_freq=1, h_freq=None)
    epochs_h2=epochs_h2.filter(l_freq=1, h_freq=100)

    epochs_h=mne.concatenate_epochs([epochs_h1, epochs_h2]) 

    evoked_h = epochs_h.average()
    evoked_h.apply_proj()

    evoked_h.power_topographic_plot(time=[0.080, 0.100]).savefig("pp.jpg")

    times=[0.080, 0.100]
    #uploading to firebase
    storage.child("analysis/" + analysisID + "/topographic-maps.jpg").put("pp.jpg")


    # plot particular electrodes
    plot_epochs_image(epochs_h, picks=[ 'FC1'], vmin=-20.0, vmax=20.0).savefig("plot-particular.jpg")
    storage.child("analysis/" + analysisID + "/plot-particular-electrodes.jpg").put("plot-particular.jpg")

    # topographic joint plot
    evoked_h.plot_topomap(times=np.linspace(0.05, 0.12, 5), ch_type='eeg').savefig("topographic-joint.jpg")
    storage.child("analysis/" + analysisID + "/topographic-joint.jpg").put("topographic-joint.jpg")

    # power plot joint
    power_p.plot_joint(baseline=(-1.3, 0), mode='mean', tmin=-.9, tmax=1, timefreqs=[(.1, 10)]).savefig("power-plot.jpg")
    storage.child("analysis/" + analysisID + "/power-plot.jpg").put("power-plot.jpg")



    


def detection(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if request.method == 'POST':
        selectedModel = request.POST["model"]



        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name,uploaded_file)
        url = fs.url(name)
        print(url)
        path = os.path.join(base_dir,url[1:])
        print(path)
        # img = image.load_img(path,target_size=(500,500))
        print(selectedModel)
        message = "Data Uploaded"
            
        return render(request,'clinical-data-analysis/index.html',{'message':message,'url':url[1:],'uploaded_data':'media/'})
            
    else:
        message = "Please choose an building image and upload to detect building"
        


def clinicalDataAnalysis(request):
    return render(request,'clinical-data-analysis/index.html')
def generateReport(request):
    return render(request, 'analysis/generate-medical-report.html')

def groupSubjectsAnalysis(request):
    return render(request, 'clinical-data-analysis/group-subjects.html')

def particularSubjectAnalysis(request):
    return render(request, 'clinical-data-analysis/particular-subject.html')

def loadData(request):
    return render(request, 'index.html')

def concatenateEpochs(epochs_h1,epochs_h2):
    epochs_h=mne.concatenate_epochs([epochs_h1, epochs_h2])
    print(epochs_h.get_data().shape)