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

class PredictionModel():
    def nc_prediction(self,request, ERA5_df, H2OSOI_df, input_date):
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        lat = round(float(np.array(body['lat'])))
        lon = round(float(np.array(body['lon'])))
        Latitude = lat
        Longitude = lon+360
        Lead_Val = str(np.array(body['soilLevel']))
        if(Lead_Val == ''):
            Lead_Val = 7
        else:
            Lead_Val = int(Lead_Val)
        
        def gen_data(X, y, num_steps=1): # def gen_data(1200, 1200, 7):
            Xs, ys = [], [] # two empty lists
            for i in range(len(X) - num_steps): # for i in range(1200 - 7):
                stacked_data = X.iloc[i:num_steps+i-1, :num_steps].stack() # stacked_data = X.iloc[0:7+0-1, :7].stack()
                Merged_data = stacked_data.reset_index(drop=True) # reset index values
                Xs.append(Merged_data) # Append data to Xs
                ys.append(y.iloc[num_steps+i-1, :]) # ys.append(np.array(y.iloc[7+0-1, :]))
            return Xs, ys # return Xs and ys to dataX and dataY

        # Num of row steps
        num_steps = int(Lead_Val) # because we took only first 7 columns (7 days)
        dataX, dataY = gen_data(ERA5_df, ERA5_df, num_steps) # dataX, dataY = gen_data(1200, 1200, 7)
        dataX = pd.DataFrame(dataX)
        dataY = np.array(dataY, dtype=np.float32)
        if(Lead_Val == 7):
            model = tf.keras.models.load_model("Best_Model/Lead_7/model__saved_for_1200_rows_Prediction_Lead_"+str(Lead_Val)+".h5")
            # model = tf.keras.models.load_model("Best_Model/model__saved_for_1200_rows_Prediction.h5")
        elif(Lead_Val == 21):
            model = tf.keras.models.load_model("Best_Model/Lead_21/model__saved_for_1200_rows_Prediction_Lead_"+str(Lead_Val)+".h5")
        predicted_data = model.predict(dataX)
        pred = pd.DataFrame(predicted_data)
        dataY_1 = pd.DataFrame(dataY)
        
        def get_monday_date(df, input_date):
            # Parse the input date
            # input_date = datetime.strptime(str(input_date), "%Y-%m-%d")
            print(input_date)
            print(type(input_date))

            # Get the start date of the DataFrame
            start_date = datetime(1999, 1, 4)  # Assuming the 0th row starts on 4/1/1999

            # Calculate the number of days between the input date and the start date
            days_difference = (input_date - start_date).days

            # Calculate the number of weeks
            week_number = days_difference // 7

            # Calculate the Monday date for the given week
            monday_date = start_date + timedelta(days=week_number * 7)

            return monday_date.strftime("%d-%m-%Y"),week_number
        monday_date,week_number = get_monday_date(H2OSOI_df, input_date)
        # Date Formation
        end_46th_date = str(pd.to_datetime(monday_date)+ timedelta(days=45))
        monday_date = str(monday_date)
        Date = pd.date_range(start= monday_date,end= end_46th_date)
        
        Date_df = pd.DataFrame(Date)
        pred = pred.iloc[week_number]    
        ERA5_df = ERA5_df.iloc[week_number]
        H2OSOI_df = H2OSOI_df.iloc[week_number]

        Combined_df = pd.concat([Date_df,ERA5_df,H2OSOI_df,pred],axis=1)
        Combined_df.columns = ['Date','ERA5','H2OSOI','predictions']
        print(Combined_df)

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
        mae = round(mean_absolute_error(Combined_df['ERA5'], Combined_df['predictions']),2)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = round(np.sqrt(mean_squared_error(Combined_df['ERA5'], Combined_df['predictions'])),2)

        acc = round(anomaly_correlation_coefficient(Combined_df['ERA5'], Combined_df['predictions']),2)
        Evaluation_metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'ACC': acc
        }
        evaluation_metrics_df = pd.DataFrame([Evaluation_metrics])
        return Combined_df,evaluation_metrics_df
