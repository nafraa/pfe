# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:59:09 2022

@author: ahmed
"""

import sqlalchemy
import matplotlib.pyplot as plt 
import pandas as pd
import autokeras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import RootMeanSquaredError
import numpy as np
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import mean_squared_error
add_data_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\convertcsv.csv'
add_d21_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\d2021.csv'



pd.option_context('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)




#Importation des données de la base MSSQL avec sqlalchemy
server="localhost"
database="GIPA_T"
driver ="ODBC Driver 17 for SQL Server"
con = f'mssql://@{server}/{database}?driver={driver}'
engine = sqlalchemy.create_engine(con,fast_executemany=True)
con = engine.connect()
sql = "select Article,DateFacture,CA from GIPA.dbo.faits_resultat"
df = pd.read_sql(sql,con)  #mise en place des données dans un DataFrame 



df['DateFacture'] = pd.to_datetime(df['DateFacture']) #formatage du champ Date
#groupement de la dataframe en mois
per = df.DateFacture.dt.to_period("M") 
g = df.groupby(per) 
df_clean = g.sum()


#importation des données supplementaires
data_test = pd.read_csv(add_data_path)
data_test.set_index('Date',inplace=True)
data21 = pd.read_csv(add_d21_path)

#jointure des deux df &
df_finale = pd.concat([data_test,df_clean])


df_finale.reset_index(drop=True,inplace=True)
scaler = MinMaxScaler(feature_range=(0,1)) #le scaler est un outil qui permet de normaliser les données en transformant 
dataset = df_finale.values #conversion des valeurs en numpy array   
dataset = scaler.fit_transform(dataset) #application du scaler sur la dataset
train_size = int(len(df_finale)*0.6) #ajustement de la taille d'entrainement du model
test_size = len(df_finale) - train_size #ajustement de la taille de test du mode
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

def prepare_data(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)
    
trainX , trainY = prepare_data(dataset,12) #les données d'entrainement
testX , testY = prepare_data(test,12) #les données de test
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

clf = autokeras.TimeseriesForecaster(
   
    predict_from= 59,
    predict_until= 71,
    max_trials=1,
    objective="val_loss",
)
# Train the TimeSeriesForecaster with train data
clf.fit(
    x=trainX,
    y=trainY,
    
    batch_size=32,
    epochs=100
)