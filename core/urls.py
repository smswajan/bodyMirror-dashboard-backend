"""core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from analysis import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.analysis, name='index'),
    path('upload/', views.loadData, name='upload'),
    path('clinical-data-analysis/', views.clinicalDataAnalysis, name='analysis'),
    # path('time-domain-analysis/', views.testAnalysis),
    path('time-domain-analysis/', views.timeDomainAnalysis),
    path('generate-medical-report/', views.generateReport),
    path('clinical-data-analysis/group-subjects/', views.groupSubjectsAnalysis), 
    path('clinical-data-analysis/particular-subject/', views.particularSubjectAnalysis),
    path('test-api/', views.api_test, name="APITest")
]
