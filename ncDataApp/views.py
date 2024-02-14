from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from .models import NcData,neonData,ReanalysisData,MeanData_class,MeanNcarData_class,Ncar_colorMap_Class

# Create your views here.
@csrf_exempt
def getNcData(request):
    nc = NcData()
    nc_json = nc.nc_read_fun(request)
    return HttpResponse(nc_json)

# Create your views here.
@csrf_exempt
def getNeonData(request):
    neon = neonData()
    neon_json = neon.neon_read_fun(request)
    return HttpResponse(neon_json)

@csrf_exempt
def getReanalysisData(request):
    ReanalyseData = ReanalysisData()
    ReanalyseData_json = ReanalyseData.ReanalyseData_read_fun(request)
    return HttpResponse(ReanalyseData_json)

@csrf_exempt
def getMeanData(request):
    MeanData = MeanData_class()
    MeanData_json = MeanData.MeanData_read_fun(request)
    return HttpResponse(MeanData_json)

@csrf_exempt
def getMeanNcarData(request):
    MeanNcarData = MeanNcarData_class()
    MeanNcarData_json = MeanNcarData.MeanNcarData_read_fun(request)
    return HttpResponse(MeanNcarData_json)

@csrf_exempt
def getColorMapJsonData(request):
    getColorMapJsonData = Ncar_colorMap_Class()
    getColorMapJson = getColorMapJsonData.Ncar_colorMap_read_fun(request)
    return HttpResponse(getColorMapJson)