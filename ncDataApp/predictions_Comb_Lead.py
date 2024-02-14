from django.db import models
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import pandas as pd
import numpy as np
from dateutil import rrule
from datetime import datetime, timedelta
#import the necessary libraries and modules for prediction
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import s3fs
import boto3



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


class PredictionModel():
    def nc_prediction(self,request):
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        lat = round(float(np.array(body['lat'])))
        lon = round(float(np.array(body['lon'])))
        Lead_Val = str(np.array(body['soilLevel']))
        # Fetching Tall site eco region location from CSV
        Tall_Eco_df= pd.read_excel('Neon_data/TALL_site/TALL_site_Eco_region.xlsx')
        Tall_Eco_dict = Tall_Eco_df.to_dict(orient='index')
        # Fetching Blan site eco region location from CSV
        WOOD_Eco_df= pd.read_excel('Neon_data/WOOD_site/WOOD_site_Eco_region.xlsx')
        Wood_Eco_dict = WOOD_Eco_df.to_dict(orient='index')
        # To set the Lead value
        # if(Lead_Val == '7'):
        #     Lead = int(Lead_Val)
        #     Look_back = 10
        # elif(Lead_Val == '21'):
        #     Lead = int(Lead_Val)
        #     Look_back = 30
        # elif(Lead_Val == '35'):
        #     Lead = int(Lead_Val)
        #     Look_back = 40
        # else:
        #     Lead = 7
        #     Look_back = 10
        # Formation of ERA-5 Nc Data 
        ds_ERA5 = data_Era5
        ds_ERA5 = ds_ERA5.fillna(0)
        ds_Modified_ERA5 = ds_ERA5.transpose("lat", "lon", "year", "month", "week", "lead")

        # Formation of H2OSOI dataset
        #Reading the netcdf file (.nc file) using xarray
        ds_CAM6 = data_cam6
        ds_CAM6 = ds_CAM6.fillna(0)
        ds_Modified_CAM6 = ds_CAM6.transpose("lat", "lon", "year", "month", "week", "lead")

        # Date formation
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

        displayDate = str(np.array(body['Display_Date']))
        date = pd.to_datetime(displayDate)
        Latitude = lat
        Longitude = lon+360
        Lead_Dict = {}
        Lead_Dict_1 = {}

        start_date = str(date)[0:10]
        end_date = str(date + timedelta(days=46))[0:10]
        dataset_List = []
        
        for a in range(3):
            # Load in a model using the .h5 format
            if a == 0:
                Lead = 7
                Look_back = 10
                if any(round(entry['Latitude']) == round(Latitude) and round(entry['Longitude Converted']) == round(Longitude) for entry in Tall_Eco_dict.values()):
                    print("Tall Site")
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/TALL_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")
                elif any(round(entry['Latitude']) == round(Latitude) and round(entry['Longitude Converted']) == round(Longitude) for entry in Wood_Eco_dict.values()):
                    print("WOOD Site")
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/WOOD_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")
                else:
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/TALL_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")
                    
                
                start_date = str(date)[0:10]
                end_date = str(date + timedelta(days=14))[0:10]
                Lead_name = 'Lead_7/Past_10'
            elif a == 1:
                Lead = 21
                Look_back = 30
                if any(round(entry['Latitude']) == round(Latitude) and round(entry['Longitude Converted']) == round(Longitude) for entry in Tall_Eco_dict.values()):
                    print("Tall Site")
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/TALL_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")
                elif any(round(entry['Latitude']) == round(Latitude) and round(entry['Longitude Converted']) == round(Longitude) for entry in Wood_Eco_dict.values()):
                    print("WOOD Site")
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/WOOD_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")
                else:
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/TALL_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")    
                start_date = str(date+ timedelta(days=15))[0:10]
                end_date = str(date + timedelta(days=28))[0:10]
                Lead_name = 'Lead_21/Past_30'
            elif a == 2:
                Lead = 35
                Look_back = 40
                if any(round(entry['Latitude']) == round(Latitude) and round(entry['Longitude Converted']) == round(Longitude) for entry in Tall_Eco_dict.values()):
                    print("Tall Site")
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/TALL_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")
                elif any(round(entry['Latitude']) == round(Latitude) and round(entry['Longitude Converted']) == round(Longitude) for entry in Wood_Eco_dict.values()):
                    print("WOOD Site")
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/WOOD_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")
                else:
                    loaded_model_1 = tf.keras.models.load_model("Neon_data/TALL_site/Lead "+str(Lead)+"/best_model_saved_for_Lead_"+str(Lead)+"_Retrained.h5")
                start_date = str(date+ timedelta(days=29))[0:10]
                end_date = str(date + timedelta(days=45))[0:10]
                Lead_name = 'Lead_35/Past_40'

            ERA5_data = ds_Modified_ERA5.OBS.sel(lat=Latitude, lon=Longitude, method='nearest').values

            # ERA-5 Data formation
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
            ERA5_df.columns = ['OBS / ERA5']

            H2OSOI_data = ds_Modified_CAM6.H2OSOI.sel(lat=Latitude, lon=Longitude, method='nearest').values

            # H2OSOI List Formation
            H2OSOI_List = [] 
            Leap_Year_List = [1,5,9,13,17,19] 

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
            H2OSOI_df = pd.DataFrame(H2OSOI_List)
            H2OSOI_df.columns = ['Climate Model Only (H2OSOI)']

            # Concatinating ERA-5 data and Date 
            data_frame = pd.concat([date_dataFrame,ERA5_df,H2OSOI_df], axis =1 )
            data_frame = data_frame.set_index('Date')

            cols = list(data_frame)
            #New dataframe with only training data columns converted into float (We do this to convert any int data to float inorder to get accurate value)
            df_for_training = data_frame[cols].astype(float)

            # Normalization Takes place (0 to 1)
            scaler = StandardScaler()
            scaler = scaler.fit(df_for_training)
            df_for_training_scaled = scaler.transform(df_for_training)

            #Empty lists to be populated using formatted training data
            trainX = []
            trainY = []
            # Number of days we want to look into the future based on the past days.
            n_lead = Lead
            # Number of past days we want to use to predict the future.
            n_LookBack = Look_back

            trainX = []
            trainY = []

            for i in range(n_LookBack, len(df_for_training_scaled) - n_lead + 1):
                x = df_for_training_scaled[i - n_LookBack:i, 0]  # Select the 1st column for trainX
                y = df_for_training_scaled[i:i + n_lead, 1]  # Select the 2nd column for trainY
                trainX.append(np.concatenate((x, y)))
                trainY.append(df_for_training_scaled[i + n_lead - 1, 0])  # 0 represents the 1st column of the dataset

            trainX, trainY = np.array(trainX), np.array(trainY)
            trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
            trainY = trainY.reshape(trainY.shape[0], -1)

            # Based on past 10 days values next 36 days values are predicted
            n_past = 6575
            n_days_for_prediction= 1826

            # Using Loaded Model Make prediction
            prediction = loaded_model_1.predict(trainX[-n_days_for_prediction:])

            # Inverse for normalized value takes place here
            prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
            y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]

            # Prediction dataframe is created
            y_pred_future_array = y_pred_future.reshape(-1, 1)
            y_pred_future_df = pd.DataFrame(y_pred_future_array)
            y_pred_future_df.columns = ["predictions"]

            # Date Df to array conversion
            pred_date = date_dataFrame[-n_days_for_prediction:]
            date_df = pd.DataFrame(np.array(pred_date))

            # Concatination of date and y pred value
            df_pred=pd.concat([date_df,y_pred_future_df],axis=1)
            df_pred.columns = ['Date','predictions']
            df_predictions = df_pred.set_index('Date')

            Actual_value = df_for_training['OBS / ERA5']
            Actual_df = pd.DataFrame(Actual_value)
            H2OSOI_Value = df_for_training['Climate Model Only (H2OSOI)']
            H2OSOI_dataframe = pd.DataFrame(H2OSOI_Value)
            dataset_plt_final = pd.concat([Actual_df,df_predictions,H2OSOI_dataframe], axis = 1)
            # Extract the predicted and actual values
            predicted_values = df_predictions.values.flatten()
            actual_values = Actual_df.values.flatten()
            actual_values = actual_values[-1826:]

            OBS_ERA5 = df_for_training['OBS / ERA5']
            OBS_ERA5_df = pd.DataFrame(OBS_ERA5[-1826:])
            H2OSOI_values = dataset_plt_final['Climate Model Only (H2OSOI)'][-1826:].values.flatten()
            dataset_plt_final_1 = dataset_plt_final[start_date:end_date]
            dataset_List.append(dataset_plt_final_1)

        # Combined Data with Lead 7, 21, 35
        combined_data = pd.concat(dataset_List)
        combined_data.reset_index(inplace=True)
        def anomaly_correlation_coefficient(model_data, observed_data):
            # Calculate the mean along the time axis
            model_mean = np.mean(model_data, axis=0)
            observed_mean = np.mean(observed_data, axis=0)

            # Calculate the anomalies by subtracting the mean
            model_anomalies = model_data - model_mean
            observed_anomalies = observed_data - observed_mean

            # Calculate the ACC
            numerator = np.sum(model_anomalies * observed_anomalies)
            denominator = np.sqrt(np.sum(model_anomalies**2) * np.sum(observed_anomalies**2))
            acc = numerator / denominator

            return acc
        # Calculate Mean Absolute Error (MAE)
        actual_values = combined_data['OBS / ERA5'].values
        predicted_values = combined_data['predictions'].values
        mae = round(mean_absolute_error(actual_values, predicted_values),2)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = round(np.sqrt(mean_squared_error(actual_values, predicted_values)),2)

        acc = round(anomaly_correlation_coefficient(predicted_values, actual_values),2)
        Evaluation_metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'ACC': acc
        }
        evaluation_metrics_df = pd.DataFrame([Evaluation_metrics])

        # Access the 'Date', 'Prediction', and 'OBS / ERA5' columns
        df_predictions = combined_data[['Date', 'predictions']]
        df_predictions.set_index('Date', inplace=True)
        return df_predictions,evaluation_metrics_df
