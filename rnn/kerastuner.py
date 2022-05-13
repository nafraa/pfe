# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:30:00 2022

@author: ahmed
"""

import math
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy
from keras.layers import LSTM, Bidirectional, Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator

add_data_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\convertcsv.csv'
add_d21_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\d2021.csv'



pd.option_context('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)




#Importation des données de la base MSSQL avec sqlalchemy
server="localhost"
database="GIPA"
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

#preprocessing de la df pour l'implementer dans le model

dataset = df_finale.values #conversion des valeurs en numpy array   
train_size = int(len(df_finale)*0.6) #ajustement de la taille d'entrainement du model
test_size = len(df_finale) - train_size #ajustement de la taille de test du model
scaler = MinMaxScaler(feature_range=(0,1)) #le scaler est un outil qui permet de normaliser les données en transformant 
#les données entre 0 et 1 ( 0 étant la valeur minime et 1 la valeur maximale) car les LSTM sont plus performants
#avec des valeurs normalisées

dataset = scaler.fit_transform(dataset) #application du scaler sur la dataset



train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

def conversion_en_window(dataset,temps): #convertir les données en format comprehensible par la couche d'input LSTM ; 
# 2 3 4 .. len(dataset) -1 -> derniere valeure
# 3 4 5 .. len(dataset) -1 -> derniere valeure
 X, Y =[], []
 for i in range(len(dataset)-temps):
  d=i+temps  
  X.append(dataset[i:d,0])
  Y.append(dataset[d,0])
 return np.array(X), np.array(Y)
    
train_gen = TimeseriesGenerator(dataset,dataset,12,1)
test_gen = TimeseriesGenerator(test,test,12,1)







# ----- Initialisation du model ----- # 

def build_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(hp.Int('input_units',min_value=32,max_value=256,step=12),input_shape=(12,1), activation='relu',return_sequences=True)))
    model.add(Bidirectional(LSTM(hp.Int('first_layer',min_value=32,max_value=256,step=12),activation='relu')))
    model.add(Dense(1,'linear'))
    model.compile(optimizer=Adam(learning_rate=0.001),loss="mse")
    return model
    
    
    
tuner= kt.RandomSearch(
    build_model,
    objective='loss',
    max_trials=3,
    executions_per_trial=1)
   

tuner.search(train_gen,epochs=1000,validation_data=(test_gen),verbose=2)
best_model = tuner.get_best_hyperparameters()[0].values
print(best_model)



