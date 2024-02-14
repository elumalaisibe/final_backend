from django.db import models
from .predictions import PredictionModel  # Import the model from prediction.py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from geojson import Polygon, Feature, FeatureCollection, dumps
import xarray as xr
import pandas as pd
import time
import json
import glob
from datetime import datetime, timedelta
from dateutil import rrule
from dateutil.relativedelta import relativedelta
import os
import calendar
import s3fs
import boto3

# Create your models here.

# -----------------code for read file from AWS S3---------------------------------------------------#

#bucket name and file key on S3
bucket_name = 'djangoprojects3'
file_key = 'Nc_data/CAM6_SM_0_5m_0_45_leadday_1999_2021_organize_Ens_mean_normalized.nc'
file_key1='ERA5/ERA5_SM_0_5m_0_45_leadday_1999_2021_organize_normalized.nc'

# Create an s3fs.S3FileSystem object with your AWS credentials if needed
# If your AWS credentials are already configured externally (e.g., using environment variables or IAM roles),
# you can skip this step.
s3 = s3fs.S3FileSystem(anon=False, key='AKIAU43XJTI3JDYLIYXJ', secret='1FzV+utG054s592F121AiH4DBp3Hbp7p2xVPGdgJ')

# Use xarray's open_dataset function with the s3fs.S3FileSystem object to read the file from S3
with s3.open(f"{bucket_name}/{file_key}", mode='rb') as s3_file:
    data_cam6 = xr.open_dataset(s3_file, engine='scipy')
    
#read ERA5 file
with s3.open(f"{bucket_name}/{file_key1}", mode='rb') as s3_file1:
    data_Era5 = xr.open_dataset(s3_file1, engine='scipy')

class NcData(models.Model):
    def nc_read_fun(self,request):
         
        # Conversion of request from django.core.handlers.wsgi.WSGIRequest to dict data type
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        #print(body['lat'])
        lat = np.array(body['lat'])
        lon = np.array(body['lon'])
        soilLevel = np.array(body['soilLevel'])
        # dayCount = np.array(body['DayCount'])
        displayDate = str(np.array(body['Display_Date']))
        # displayEndDate = str(np.array(body['Display_End_Date']))
        dates = pd.to_datetime(displayDate)
        displayEndDate = str(pd.to_datetime(displayDate)+ timedelta(days=46))
        
        # To set the soil level value
        if(soilLevel != ""):
            j = int(soilLevel)
        else:
            j = 7
        
        # To set the Day Count
        # if(dayCount != ""):
        #     dCount = int(dayCount)
        # else:
        #     dCount = 10

        # Longitude Conversion
        lon = lon+360

        # H2OSOI Data Formation

        #Reading the netcdf file (.nc file) using xarray
        ds_CAM6 = data_cam6
        ds_CAM6 = ds_CAM6.fillna(0)
        ds_Modified_CAM6 = ds_CAM6.transpose("lat", "lon", "year", "month", "week", "lead")

        H2OSOI_data = ds_Modified_CAM6.H2OSOI.sel(lat=lat, lon=lon, method='nearest').values
        # Create a DataFrame
        H2OSOI_List = []  # Empty List

        # Loop For 23 years i.e From 1999 to 2021
        for i in range(23):
            # Loop For 12 months i.e From Jan to Dec
            for j in range(12):
                # Calculate the number of Mondays in the current month
                year = 1999 + i
                month = j + 1
                first_day, last_day = calendar.monthrange(year, month)
                num_mondays = sum(1 for day in range(1, last_day + 1) if calendar.weekday(year, month, day) == calendar.MONDAY)
                
                # Limit num_mondays to a maximum of 5
                num_mondays = min(num_mondays, 5)
                
                # Loop for 'num_mondays' weeks
                for k in range(num_mondays):
                    H2OSOI_List.append(H2OSOI_data[i][j][k])

        H2OSOI_df = pd.DataFrame(H2OSOI_List)
        
        # ERA-5 Data formation
        ds_ERA5 = data_Era5
        ds_ERA5 = ds_ERA5.fillna(0)
        ds_Modified_ERA5 = ds_ERA5.transpose("lat", "lon", "year", "month", "week", "lead")
        ERA5_data = ds_Modified_ERA5.OBS.sel(lat=lat, lon=lon, method='nearest').values
        # Create a DataFrame
        ERA5_List = []  # Empty List

        # Loop For 23 years i.e From 1999 to 2021
        for i in range(23):
            # Loop For 12 months i.e From Jan to Dec
            for j in range(12):
                # Calculate the number of Mondays in the current month
                year = 1999 + i
                month = j + 1
                first_day, last_day = calendar.monthrange(year, month)
                num_mondays = sum(1 for day in range(1, last_day + 1) if calendar.weekday(year, month, day) == calendar.MONDAY)
                
                # Limit num_mondays to a maximum of 5
                num_mondays = min(num_mondays, 5)
                
                # Loop for 'num_mondays' weeks
                for k in range(num_mondays):
                    ERA5_List.append(ERA5_data[i][j][k])

        ERA5_df = pd.DataFrame(ERA5_List)

        # Date DataFrame Creation
        start_date = datetime(1999, 1, 1)
        end_date = datetime(2021, 12, 31)

        date_range = list(rrule.rrule(freq=rrule.DAILY, dtstart=start_date, until=end_date))
        date_str = []
        for date_obj in date_range:
            date_str.append(date_obj.strftime('%Y/%m/%d'))
            
        # Converting it into datetimeIndex
        date_Index = pd.to_datetime(date_str)
        # Converting timestamp into DataFrame
        date_dataFrame = pd.DataFrame(date_Index)
        date_dataFrame.columns = ['Date']

        # Concatinating H2OSOI data and Date
        data_frame = pd.concat([date_dataFrame,H2OSOI_df,ERA5_df], axis =1 )
        data_frame = data_frame.set_index('Date')

        
        # Class and Object Creation for prediction
        pM = PredictionModel()
        predicted_data,evaluation_metric_df = pM.nc_prediction(request,ERA5_df,H2OSOI_df,dates)
        predicted_data_Final_df = pd.concat([predicted_data,evaluation_metric_df],axis=1)
        Json_dump = predicted_data_Final_df.to_json()
        return Json_dump

class neonData(models.Model):
    def neon_read_fun(self,request):

        # Conversion of request from django.core.handlers.wsgi.WSGIRequest to dict data type
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        print(body['lat'])
        lat = np.array(body['lat'])
        lon = np.array(body['lon'])
        site = np.array(body['SiteName'])
        displayDate = str(np.array(body['Display_Date']))
        # displayEndDate = str(np.array(body['Display_End_Date']))
        dates = pd.to_datetime(displayDate)
        end_dates = dates.date()
        # displayEndDate = str(pd.to_datetime(displayDate)+ timedelta(days=30))
        # Convert the input date to a datetime object
        

        # Find the end date of the month
        displayEndDate = str(end_dates + relativedelta(day=31))
        
        # Longitude Conversion
        lon = lon+360

        #Reading the netcdf file (.nc file) using xarray
        ds_CAM6 = data_cam6
        
        ds_Modified_CAM6 = ds_CAM6.transpose("lat", "lon", "year", "month", "week", "lead")

        H2OSOI_data = ds_Modified_CAM6.H2OSOI.sel(lat=lat, lon=lon, method='nearest').values
        H2OSOI_List = [] # Empty List
        Leap_Year_List = [1,5,9,13,17,19] # Leap Year List(1999 to 2021)=> [index 1-> 2000, index 5-> 2004,.......,index 19-> 2020]

        for i in range(23):
            for j in range(12):
                for k in range(5):
                    if (j % 2) == 0 and j not in [8,10]:
                        if (k != 4):
                            for l in range(7):
                                H2OSOI_List.append(H2OSOI_data[i][j][k][l])
                        else:
                            for l in range(3):
                                H2OSOI_List.append(H2OSOI_data[i][j][k][l])  
                    elif j == 1:
                        if i in Leap_Year_List:
                            if (k != 4):
                                for l in range(7):
                                    H2OSOI_List.append(H2OSOI_data[i][j][k][l])
                            else:
                                for l in range(1):
                                    H2OSOI_List.append(H2OSOI_data[i][j][k][l])
                        else:
                            if (k != 4):
                                for l in range(7):
                                    H2OSOI_List.append(H2OSOI_data[i][j][k][l])
                            else:
                                pass
                    elif j in [7,9,11]:
                        if (k != 4):
                            for l in range(7):
                                H2OSOI_List.append(H2OSOI_data[i][j][k][l])
                        else:
                            for l in range(3):
                                H2OSOI_List.append(H2OSOI_data[i][j][k][l])
                    else:
                        if (k != 4):
                            for l in range(7):
                                H2OSOI_List.append(H2OSOI_data[i][j][k][l])
                        else:
                            for l in range(2):
                                H2OSOI_List.append(H2OSOI_data[i][j][k][l])
        data_H2OSOI_5_mean_table = pd.DataFrame(H2OSOI_List)
        

        # Creating a collection of data frame to store various sites in diffrent frames
        dataframe_collection = {}

        # Creating a temporary data frame to store temp values for processing
        all_data = pd.DataFrame() 

        #Setting the date range using pandas
        klist= pd.date_range('2018-01','2018-02',freq='M').strftime("%Y-%m").tolist()
        if(site == "ABBY"):
            site_name = "ABBY_site"
            site_num = "D16.ABBY"
        elif (site == "BLAN"):
            site_name = "BLAN_site"
            site_num = "D02.BLAN"
        elif (site == "CLBJ"):
            site_name = "CLBJ_site"
            site_num = "D11.CLBJ"
        elif (site == "WOOD"):
            site_name = "WOOD_site"
            site_num = "D09.WOOD"
        elif (site == "TALL"):
            site_name = "TALL_site"
            site_num = "D08.TALL"
        elif(site == "CLBJ"):
            site_name = "CLBJ_site"
        elif (site == "CPER"):
            site_name = "CPER_site"
        elif (site == "KONZ"):
            site_name = "KONZ_site"
        elif (site == "NIWO"):
            site_name = "NIWO_site"
        elif (site == "ONAQ"):
            site_name = "ONAQ_site"
        elif(site == "ORNL"):
            site_name = "ORNL_site"
        elif (site == "OSBS"):
            site_name = "OSBS_site"
        elif (site == "SCBI"):
            site_name = "SCBI_site"
        elif (site == "SRER"):
            site_name = "SRER_site"
        elif (site == "UNDE"):
            site_name = "UNDE_site"
        else:
            site_name = "Default"
            site_num = "Default"

        if site_name == "Default":
            dictionary = {'Reason':'Clicked Out of Five Sites'}
            json_val = json.dumps(dictionary)
            return json_val
            
        year_str = str(dates.year)
        
        month_str = dates.month_name()
        

        # Create a date range for the entire month
        neon_start_date = pd.Timestamp(dates.year, dates.month, 1)
        neon_end_date = neon_start_date + pd.offsets.MonthEnd()
        neon_date_range = pd.date_range(start=neon_start_date, end=neon_end_date, freq='D')

        # Create a DataFrame with the date range
        neon_date_df = pd.DataFrame({'Date': neon_date_range})

        # Reading Neon csv based on given sites
        neon_site = pd.read_excel('Neon_data/'+site_name+'/'+site_name+'.xlsx', sheet_name=year_str)
        neon_site = neon_site[month_str[0:3]]
        neon_data = pd.DataFrame(neon_site)
        neon_data.columns = ['NEON_VSWC_Mean']

        neon_data_new = pd.concat([neon_date_df,neon_data], axis = 1)
        neon_data_new = neon_data_new.set_index('Date')
        # neon_data_new = neon_data_new.drop(['index'],axis=1)
        
        # # displayEndDate
        # endDay_count = len(neon_site)
        # displayEndDate = str(pd.to_datetime(displayDate)+ timedelta(days=endDay_count))
        # print(endDay_count)

        # ERA-5 Data formation
        ds_ERA5 = data_Era5
        ds_Modified_ERA5 = ds_ERA5.transpose("lat", "lon", "year", "month", "week", "lead")
        ERA5_data = ds_Modified_ERA5.OBS.sel(lat=lat, lon=lon, method='nearest').values
        ERA5_List = [] # Empty List
        Leap_Year_List = [1,5,9,13,17,19] # Leap Year List(1999 to 2021)=> [index 1-> 2000, index 5-> 2004,.......,index 19-> 2020]

        for i in range(23):
            for j in range(12):
                for k in range(5):
                    if (j % 2) == 0 and j not in [8,10]:
                        if (k != 4):
                            for l in range(7):
                                ERA5_List.append(ERA5_data[i][j][k][l])
                        else:
                            for l in range(3):
                                ERA5_List.append(ERA5_data[i][j][k][l])  
                    elif j == 1:
                        if i in Leap_Year_List:
                            if (k != 4):
                                for l in range(7):
                                    ERA5_List.append(ERA5_data[i][j][k][l])
                            else:
                                for l in range(1):
                                    ERA5_List.append(ERA5_data[i][j][k][l])
                        else:
                            if (k != 4):
                                for l in range(7):
                                    ERA5_List.append(ERA5_data[i][j][k][l])
                            else:
                                pass
                    elif j in [7,9,11]:
                        if (k != 4):
                            for l in range(7):
                                ERA5_List.append(ERA5_data[i][j][k][l])
                        else:
                            for l in range(3):
                                ERA5_List.append(ERA5_data[i][j][k][l])
                    else:
                        if (k != 4):
                            for l in range(7):
                                ERA5_List.append(ERA5_data[i][j][k][l])
                        else:
                            for l in range(2):
                                ERA5_List.append(ERA5_data[i][j][k][l])
        ERA5_df = pd.DataFrame(ERA5_List)
        ERA5_df.columns = ['ERA5']

        

        # Combining ERA5 
        combine_df = pd.concat([ERA5_df], axis = 1)
        mean = combine_df.mean(axis = 1)
        # Date Formation
        Date = pd.date_range(start="1999-01-01",end="12-31-2021")
        Date_df = pd.DataFrame(Date)
        Date_df.columns = ['Date']

        Mean_data = pd.concat([Date_df,mean,data_H2OSOI_5_mean_table],axis= 1)
        Mean_data.columns = ['Date','ERA_5','NCAR_H2OSOI_Mean']

        mask = (Mean_data['Date'] >= displayDate) & (Mean_data['Date'] <= displayEndDate)
        Mean_data_df = Mean_data.loc[mask]
        Mean_data_df = Mean_data_df.reset_index()
        Mean_data_df_new = Mean_data_df.set_index('Date')
        Mean_data_df_new = Mean_data_df_new.drop(['index'],axis=1)
        
        data_combine_new = pd.concat([Mean_data_df_new,neon_data_new],axis=1)
        data_combine_new = data_combine_new.reset_index()
        mask_new = (data_combine_new['Date'] >= displayDate) & (data_combine_new['Date'] <= displayEndDate)
        data_combine_new_1 = data_combine_new.loc[mask_new]
        
        # Final_df = Final_df.set_index('Date')
        # Mean_data_df = Mean_data_df.drop(['Date'],axis=1)
        Mean_data_df = Mean_data_df.drop(['index'],axis=1)
        data_combine = pd.concat([Mean_data_df, neon_data], axis =1)
        # print(data_combine)
        # mask = (data_combine['Date'] >= displayDate) & (data_combine['Date'] <= displayEndDate)
        # Ncar_Neon_df = data_combine.loc[mask]
        # Ncar_Neon_df = Ncar_Neon_df.reset_index()
        # Final_dataFrame = pd.concat([Ncar_Neon_df,Mean_data_df], axis =1)
        json_val = data_combine_new_1.to_json()
        return json_val
    
# Class not in use
class ReanalysisData(models.Model):
    def ReanalyseData_read_fun(self,request):
        # Conversion of request from django.core.handlers.wsgi.WSGIRequest to dict data type
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        print(body['lat'])
        lat = np.array(body['lat'])
        lon = np.array(body['lon'])
        site = np.array(body['SiteName'])
        YearCount = np.array(body['YearCount'])
        displayDate = str(np.array(body['Display_Date']))
        # displayEndDate = str(np.array(body['Display_End_Date']))
        displayEndDate = str(pd.to_datetime(displayDate)+ timedelta(days=30))

        # Longitude Conversion
        lon = lon+360

        # To set the Year Count
        if(YearCount != ""):
            YCount = int(YearCount)
        else:
            YCount = 5
        # ERA5 Data formation
        ds_ERA5 = data_Era5

        #swvRZ variable (location of 5 Site)
        if site == 'TALL':
            ERA5 =ds_ERA5.swvRZ.sel(lat=32.950, lon=272.607, method='nearest')
        elif site == 'ABBY':
            ERA5 =ds_ERA5.swvRZ.sel(lat=45.762, lon=237.67, method='nearest')
        elif site == 'BLAN':
            ERA5 =ds_ERA5.swvRZ.sel(lat=39.060, lon=281.928, method='nearest')
        elif site == 'CLBJ':
            ERA5 =ds_ERA5.swvRZ.sel(lat=33.401, lon=262.430, method='nearest')
        elif site == 'WOOD':
            ERA5 =ds_ERA5.swvRZ.sel(lat=47.128, lon=260.759, method='nearest')
        else:
            ERA5 =ds_ERA5.swvRZ.sel(lat=lat, lon=lon, method='nearest')

        ERA5_data_merge = xr.merge([ERA5], compat = 'override')
        ERA5_data_swvRZ = pd.DataFrame(ERA5_data_merge.swvRZ)
        ERA5_data_swvRZ.columns = ['ERA5']
        # Converting dataset to DataFrame
        #swvRZ_df = ERA5.to_dataframe()

        # MEERA_2 Data formation
        ds_MEERA_2 = xr.open_dataset("MEERA_2/MERRA2_SM_0_5m_1999_2018_organize.nc", engine="netcdf4")
        #RZMC variable (location of 5 Site)
        if site == 'TALL':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=32.950, lon=272.607, method='nearest')
        elif site == 'ABBY':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=45.762, lon=237.67, method='nearest')
        elif site == 'BLAN':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=39.060, lon=281.928, method='nearest')
        elif site == 'CLBJ':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=33.401, lon=262.430, method='nearest')
        elif site == 'WOOD':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=47.128, lon=260.759, method='nearest')
        else:
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=lat, lon=lon, method='nearest')

        MEERA_data_merge = xr.merge([MEERA_2], compat = 'override')
        MEERA_data_RZMC = pd.DataFrame(MEERA_data_merge.RZMC)
        MEERA_data_RZMC.columns = ['MEERA_2']

        # Date Formation
        Date = pd.date_range(start="1999-01-01",end="12-31-2018")
        Date_df = pd.DataFrame(Date)
        Date_df.columns = ['Date']
        # Combining ERA5 and MEERA_2
        combine_df = pd.concat([Date_df,ERA5_data_swvRZ.iloc[:7305], MEERA_data_RZMC], axis = 1)
        mask = (combine_df['Date'] >= displayDate) & (combine_df['Date'] <= displayEndDate)
        Final_df = combine_df.loc[mask]    
        
        json_data = Final_df.to_json()

        return json_data

# Class not in use
class MeanData_class(models.Model):
    def MeanData_read_fun(self,request):
        # Conversion of request from django.core.handlers.wsgi.WSGIRequest to dict data type
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        print(body['lat'])
        lat = np.array(body['lat'])
        lon = np.array(body['lon'])
        site = np.array(body['SiteName'])
        YearCount = np.array(body['YearCount'])
        displayDate = str(np.array(body['Display_Date']))
        # displayEndDate = str(np.array(body['Display_End_Date']))
        displayEndDate = str(pd.to_datetime(displayDate)+ timedelta(days=30))

        # Longitude Conversion
        lon = lon+360

        # To set the Year Count
        if(YearCount != ""):
            YCount = int(YearCount)
        else:
            YCount = 5
        # ERA5 Data formation
        ds_ERA5 = data_Era5

        #swvRZ variable (location of 5 Site)
        if site == 'TALL':
            ERA5 =ds_ERA5.swvRZ.sel(lat=32.950, lon=272.607, method='nearest')
        elif site == 'ABBY':
            ERA5 =ds_ERA5.swvRZ.sel(lat=45.762, lon=237.67, method='nearest')
        elif site == 'BLAN':
            ERA5 =ds_ERA5.swvRZ.sel(lat=39.060, lon=281.928, method='nearest')
        elif site == 'CLBJ':
            ERA5 =ds_ERA5.swvRZ.sel(lat=33.401, lon=262.430, method='nearest')
        elif site == 'WOOD':
            ERA5 =ds_ERA5.swvRZ.sel(lat=47.128, lon=260.759, method='nearest')
        else:
            ERA5 =ds_ERA5.swvRZ.sel(lat=lat, lon=lon, method='nearest')

        ERA5_data_merge = xr.merge([ERA5], compat = 'override')
        ERA5_data_swvRZ = pd.DataFrame(ERA5_data_merge.swvRZ)
        ERA5_data_swvRZ.columns = ['ERA5']
        # Converting dataset to DataFrame
        #swvRZ_df = ERA5.to_dataframe()

        # MEERA_2 Data formation
        ds_MEERA_2 = xr.open_dataset("MEERA_2/MERRA2_SM_0_5m_1999_2018_organize.nc", engine="netcdf4")
        #RZMC variable (location of 5 Site)
        if site == 'TALL':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=32.950, lon=272.607, method='nearest')
        elif site == 'ABBY':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=45.762, lon=237.67, method='nearest')
        elif site == 'BLAN':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=39.060, lon=281.928, method='nearest')
        elif site == 'CLBJ':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=33.401, lon=262.430, method='nearest')
        elif site == 'WOOD':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=47.128, lon=260.759, method='nearest')
        else:
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=lat, lon=lon, method='nearest')

        MEERA_data_merge = xr.merge([MEERA_2], compat = 'override')
        MEERA_data_RZMC = pd.DataFrame(MEERA_data_merge.RZMC)
        MEERA_data_RZMC.columns = ['MEERA_2']

        # Combining ERA5 and MEERA_2
        combine_df = pd.concat([ERA5_data_swvRZ.iloc[:7305], MEERA_data_RZMC], axis = 1)
        
        mean = combine_df.mean(axis = 1)
        # Date Formation
        Date = pd.date_range(start="1999-01-01",end="12-31-2018")
        Date_df = pd.DataFrame(Date)
        Date_df.columns = ['Date']

        final_data = pd.concat([Date_df,mean],axis= 1)
        final_data.columns = ['Date','Mean']

        mask = (final_data['Date'] >= displayDate) & (final_data['Date'] <= displayEndDate)
        Final_df = final_data.loc[mask]

        json_data = Final_df.to_json()

        return json_data

# Class not in use
class MeanNcarData_class(models.Model):
    def MeanNcarData_read_fun(self,request):
        # Conversion of request from django.core.handlers.wsgi.WSGIRequest to dict data type
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        print(body['lat'])
        lat = np.array(body['lat'])
        lon = np.array(body['lon'])
        site = np.array(body['SiteName'])
        YearCount = np.array(body['YearCount'])
        displayDate = str(np.array(body['Display_Date']))
        # displayEndDate = str(np.array(body['Display_End_Date']))
        displayEndDate = str(pd.to_datetime(displayDate)+ timedelta(days=30))
        dates = pd.to_datetime(displayDate)
        
        # Longitude Conversion
        lon = lon+360

        " ERA5 Soil Moisture Data "
        #Reading the netcdf file (.nc file) using xarray
        ds_ERA5 =data_Era5

        #swvRZ variable (location of 5 Site)
        if site == 'TALL':
            ERA5 =ds_ERA5.swvRZ.sel(lat=32.950, lon=272.607, method='nearest')
        elif site == 'ABBY':
            ERA5 =ds_ERA5.swvRZ.sel(lat=45.762, lon=237.67, method='nearest')
        elif site == 'BLAN':
            ERA5 =ds_ERA5.swvRZ.sel(lat=39.060, lon=281.928, method='nearest')
        elif site == 'CLBJ':
            ERA5 =ds_ERA5.swvRZ.sel(lat=33.401, lon=262.430, method='nearest')
        elif site == 'WOOD':
            ERA5 =ds_ERA5.swvRZ.sel(lat=47.128, lon=260.759, method='nearest')
        else:
            ERA5 =ds_ERA5.swvRZ.sel(lat=lat, lon=lon, method='nearest')

        ERA5_data_merge = xr.merge([ERA5], compat = 'override')
        ERA5_data_swvRZ = pd.DataFrame(ERA5_data_merge.swvRZ)
        ERA5_data_swvRZ.columns = ['RootZone Soil Moisture']
        
        " MERRA_2 Soil Moisture Data "
        ds_MEERA_2 = xr.open_dataset("MEERA_2/MERRA2_SM_0_5m_1999_2018_organize.nc", engine="netcdf4")
        #RZMC variable (location of 5 Site)
        if site == 'TALL':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=32.950, lon=272.607, method='nearest')
        elif site == 'ABBY':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=45.762, lon=237.67, method='nearest')
        elif site == 'BLAN':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=39.060, lon=281.928, method='nearest')
        elif site == 'CLBJ':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=33.401, lon=262.430, method='nearest')
        elif site == 'WOOD':
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=47.128, lon=260.759, method='nearest')
        else:
            MEERA_2 =ds_MEERA_2.RZMC.sel(lat=lat, lon=lon, method='nearest')

        MEERA_data_merge = xr.merge([MEERA_2], compat = 'override')
        MEERA_data_RZMC = pd.DataFrame(MEERA_data_merge.RZMC)
        MEERA_data_RZMC.columns = ['Water Root Zone']

        " Combining ERA-5 and MEERA-2 data "
        combine = pd.concat([ERA5_data_swvRZ.iloc[:7305], MEERA_data_RZMC], axis = 1)

        ####### Date for Mean Data
        Date = pd.date_range(start="1999-01-01",end="12-31-2018")
        Date_df = pd.DataFrame(Date)
        Date_df.columns = ['Date']

        " Mean Value for ERA-5 and MEERA-2 data "
        mean = combine.mean(axis = 1)
        final_data = pd.concat([Date_df,mean],axis= 1)
        final_data.columns = ['Date','Mean']
        final_data = final_data.set_index('Date')

        # Get the list of all files and directories
        path = "Nc_data/"+str(dates.year)+"/"+str(dates.month_name())
        dir_list = os.listdir(path)

        # week_number
        week_number = (dates.day-1)//7 + 1
        
        Folder_date = dir_list[week_number-1]
        File_list_ds = []
        for file in glob.glob('Nc_data/'+str(dates.year)+'/'+str(dates.month_name())+'/'+str(Folder_date)+'/*.nc'):
            File_list_ds.append(file)
        ds_List = []
        data_H2OSOI_5_mean = []    
        for i in range(len(File_list_ds)):
            file_name = File_list_ds[i]
            ds_List.append(xr.open_dataset(file_name, engine="netcdf4"))
            # H2OSOI variable (location of 5 Site)
            if site == 'TALL':
                H2OSOI_ds=ds_List[i].H2OSOI.sel(lat=32.950, lon=272.607, method='nearest')
            elif site == 'ABBY':
                H2OSOI_ds=ds_List[i].H2OSOI.sel(lat=45.762, lon=237.67, method='nearest')
            elif site == 'BLAN':
                H2OSOI_ds=ds_List[i].H2OSOI.sel(lat=39.060, lon=281.928, method='nearest')
            elif site == 'CLBJ':
                H2OSOI_ds=ds_List[i].H2OSOI.sel(lat=33.401, lon=262.430, method='nearest')
            elif site == 'WOOD':
                H2OSOI_ds=ds_List[i].H2OSOI.sel(lat=47.128, lon=260.759, method='nearest')
            else:
                H2OSOI_ds=ds_List[i].H2OSOI.sel(lat=lat, lon=lon, method='nearest')
            H2OSOI_df = pd.DataFrame(H2OSOI_ds)
            data_H2OSOI_5 = H2OSOI_df.iloc[ :46 , : 5]
            data_H2OSOI_5_mean.append(data_H2OSOI_5.mean(axis = 1, skipna = True))

        # Converting List to Dataframe
        All_file_H2OSOI_df = pd.DataFrame(data_H2OSOI_5_mean).transpose()

        # H2OSOI Mean 
        All_file_H2OSOI_df_mean = All_file_H2OSOI_df.mean(axis = 1, skipna = True)
        # Converting into timestamp
        datetimeindex = ds_List[0].indexes['time'].to_datetimeindex()
        ds_List[0]['time'] = datetimeindex
        date_df = pd.to_datetime(ds_List[0]['time'])

        # Converting timestamp into DataFrame
        date_dataFrame = pd.DataFrame(date_df)
        date_dataFrame.columns = ['Date']
        H2OSOI_dataFrame = pd.DataFrame(All_file_H2OSOI_df_mean)
        H2OSOI_dataFrame.columns = ['H2OSOI']
        temp_df = pd.concat([date_dataFrame,H2OSOI_dataFrame], axis =1 )
        temp_df =temp_df.set_index('Date')
        dataset = pd.concat([temp_df,final_data], axis =1 )
        dataset = dataset.reset_index()
        mask = (dataset['Date'] >= displayDate) & (dataset['Date'] <= displayEndDate)
        Final_df = dataset.loc[mask]
        Final_df = Final_df.reset_index()
        Final_df = Final_df.drop(['index'],axis=1)
        json_data = Final_df.to_json()

        return json_data
    
class Ncar_colorMap_Class(models.Model):
    def Ncar_colorMap_read_fun(self,request):
        # Conversion of request from django.core.handlers.wsgi.WSGIRequest to dict data type
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        Week_count = int(np.array(body['Week_count']))
        displayDate = str(np.array(body['Display_Date']))
        Color_Map_Date = str(np.array(body['Nc_Color_Map_Date']))
        YearCount = Color_Map_Date[0:4]
        
        # Color Map Date input
        input_date = datetime.strptime(Color_Map_Date, "%Y-%m-%d")
        days_since_monday = (input_date.weekday() - 7) % 7
        if (Week_count == 1):
            monday_date = input_date - timedelta(days=days_since_monday)
            Next_15_date = monday_date+timedelta(days=15)
        elif(Week_count == 2):
            monday_date = input_date - timedelta(days=days_since_monday)
            monday_date = monday_date + timedelta(days=15)
            Next_15_date = monday_date+timedelta(days=30)
        elif(Week_count == 3):
            monday_date = input_date - timedelta(days=days_since_monday)
            monday_date = monday_date + timedelta(days=30)
            Next_15_date = monday_date+timedelta(days=45)
        Next_15_date_str = Next_15_date.strftime("%d-%m-%Y")
        monday_date_str = monday_date.strftime("%d-%m-%Y")
        # Selecting Date
        if(monday_date_str[0:1] == '0'):
            w_s = monday_date_str[1:2]
        else:
            w_s = monday_date_str[0:2]
        
        if(Next_15_date_str[0:1] == '0'):
            w_e = str(int(monday_date_str[1:2])+15)
        else:
            w_e = str(int(monday_date_str[0:2])+15)

        if(monday_date_str[3:4] == '0'):
            MonthCount = monday_date_str[4:5]
        else:
            MonthCount = monday_date_str[3:5]
        

        # Reading Ncar File using Xarray
        ds = data_cam6
        ds = ds.fillna(0)
        ds = ds.transpose("lat", "lon", "year", "month", "week", "lead")

        # Define the latitude and longitude values for the grid
        x = np.arange(24.0, 50.0, 1)
        lat_list = [25.916231, 26.858639, 27.801046, 28.743456, 29.685863, 30.628273,
                    31.57068 , 32.51309 , 33.455498, 34.397907, 35.340313, 36.282722,
                    37.225132, 38.167538, 39.109947, 40.052357, 40.994766, 41.937172,
                    42.87958 , 43.82199 , 44.764397, 45.706806, 46.649216, 47.59162 ,
                    48.53403]

        # Creating Longitude List
        lon_list = [236.25, 237.5 , 238.75, 240, 241.25, 242.5 , 243.75, 245.  , 246.25,
                    247.5, 248.75, 250, 251.25, 252.5, 253.75, 255, 256.25, 257.5 ,
                    258.75, 260, 261.25, 262.5, 263.75, 265, 266.25, 267.5 , 268.75,
                    270, 271.25, 272.5 , 273.75, 275, 276.25, 277.5 , 278.75, 280.  ,
                    281.25, 282.5 , 283.75, 285, 286.25, 287.5, 288.75, 290, 291.25,
                    292.5]
        lat = np.array(lat_list)
        lon = np.array(lon_list)

        # Longitude Conversion
        lon = ((lon+180)%360)-180
        Nc_lon = lon+360



        # Create a GeoJSON polygon for each grid cell
        features = []
        for i in range(len(lat)-1):
            for j in range(len(lon)-1):
                coords = [
                    [lon[j], lat[i]],
                    [lon[j+1], lat[i]],
                    [lon[j+1], lat[i+1]],
                    [lon[j], lat[i+1]],
                    [lon[j], lat[i]]
                ]
                H2OSOI_ds = ds.H2OSOI.sel(lat=lat[i], lon=Nc_lon[j], method='nearest').values
                #print(H2OSOI_ds)
                
                # Year count
                year = int(YearCount)
                #print(year)
                year_arr = np.arange(1999, 2022, 1)
                Year_df = pd.DataFrame(year_arr)
                year_index = Year_df.loc[Year_df[0] == year].index[0]

                # Month_Count
                month = int(MonthCount)
                #print(month)
                month_arr = np.arange(1, 13, 1)
                month_df = pd.DataFrame(month_arr)
                month_index = month_df.loc[month_df[0] == month].index[0]

                data_H2OSOI_15 = H2OSOI_ds[year_index][month_index][0][int(w_s):int(w_e)]
                data_H2OSOI_15_mean = data_H2OSOI_15.mean()
                data_H2OSOI_15_mean = round(data_H2OSOI_15_mean, 5)
                feature = Feature(geometry=Polygon([coords]), properties={"H2OSOI": float(data_H2OSOI_15_mean)})
                if(feature['properties']['H2OSOI'] != 0.0):
                    features.append(feature)

        # Create a FeatureCollection and save to a file
        feature_collection = FeatureCollection(features)
        Json_data = dumps(feature_collection)
        return Json_data