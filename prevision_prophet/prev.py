from sklearn.metrics import mean_squared_error
import sqlalchemy
import matplotlib.pyplot as plt 
import pandas as pd
pd.option_context('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import numpy as np
from math import sqrt, ceil
import fbprophet
add_data_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\convertcsv.csv'
add_d21_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\d2021.csv'

#Importation des données de la base MSSQL avec sqlalchemy
def connexion_bd():
    server="localhost"
    database="GIPA"
    driver ="ODBC Driver 17 for SQL Server"
    con = f'mssql://@{server}/{database}?driver={driver}'
    engine = sqlalchemy.create_engine(con,fast_executemany=True)
    con = engine.connect()
    sql = "select Article,DateFacture,CA from GIPA.dbo.faits_resultat"
    df = pd.read_sql(sql,con)  #mise en place des données dans un DataFrame 
    return df



def transformation_données_2021():
    data21 = pd.read_csv(add_d21_path)
    data21.drop("Date",axis=1,inplace=True)
    array_d21 = np.array(data21)
    return array_d21

df = connexion_bd() #connexion  a la base
df['DateFacture'] = pd.DatetimeIndex(df['DateFacture']) #conversion de la colonne de type object a datetimeIndex
df.index = df['DateFacture']
group = df.groupby(pd.Grouper(freq='M'))#grouper les dates en forme de mois
df_c = group.sum() #faire la somme de chaque mois
#importation des données de validation
data_test = pd.read_csv(add_data_path)
data_test.set_index('Date',inplace=True)
#jointure des deux df 
df_finale = pd.concat([data_test,df_c])
df_finale.reset_index(inplace=True)
df_finale['index'] = pd.to_datetime(df_finale['index'])
df_finale.columns = ['ds','y'] #prophet spécifie que les colonnes soient appelées ds et y
model = fbprophet.Prophet() #instanciation de prophet
model.fit(df_finale) #ajustement de la base au modéle
future_dates=model.make_future_dataframe(periods=12,freq='M') #prophet permet de créer
#une dataframe qui contient les dates et valeurs
prevision = model.predict(future_dates[60:72])#cette dataframe contient les anciennes valeurs donc
#on choisit les 12 derniers valeurs , du 1 janvier 2021 au 12 decembre 2021 
df_resultats_fb = pd.DataFrame() 
array_d21 = transformation_données_2021()
#dataframe pour faire la comparaison des resultats
df_resultats_fb['donnees reelles'] = pd.DataFrame(array_d21)
df_resultats_fb['previsions'] = prevision['yhat']
model.plot(prevision) 
prophet_score = sqrt(mean_squared_error(array_d21,prevision['yhat']))
print('RMSE des prévisions par rapport au données réelles: %.2f RMSE' % (prophet_score))