from django.urls import include, path
from . import views

urlpatterns=[
    path('getNcData/',views.getNcData),
    path('getNeonData/',views.getNeonData),
    path('getReanalysisData/',views.getReanalysisData),
    path('getMeanData/',views.getMeanData),
    path('getMeanNcarData/',views.getMeanNcarData),
    path('getColorMapJsonData/',views.getColorMapJsonData),
]
