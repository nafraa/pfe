from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import sqlalchemy
import matplotlib.pyplot as plt 
import pandas as pd
pd.option_context('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import numpy as np
from math import sqrt, ceil
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

df = connexion_bd()
df['DateFacture'] = pd.DatetimeIndex(df['DateFacture'])
df.index = df['DateFacture']
group = df.groupby(pd.Grouper(freq='M'))  
df_c = group.sum()
data_test = pd.read_csv(add_data_path)
data_test.set_index('Date',inplace=True)
#jointure des deux df 
df_finale = pd.concat([data_test,df_c])
df_finale.reset_index(inplace=True)
df_finale['index'] = pd.to_datetime(df_finale['index'])
df_finale


model=sm.tsa.statespace.SARIMAX(df_finale['CA'],order=(2, 1, 2),seasonal_order=(0,1,1,12))#création du modéle SARIMAX
#les valeurs d'ordre on été choisie selon une méthode brute-force qui boucle de 0 a 5 pour chaque attribut et compare les resultats
results=model.fit() #ajustement des données au modele
array_d21 = pd.DataFrame(transformation_données_2021())
df_resultat = pd.DataFrame()
df_resultat['donnees reelles'] = array_d21
res = results.predict(start=59,end=70,dynamic=False)
df_res = pd.DataFrame(res)
df_res.reset_index(inplace=True)
df_resultat['previsions'] = df_res['predicted_mean']
df_resultat.plot()
score_sarimax = sqrt(mean_squared_error(array_d21,df_resultat['previsions']))
print('RMSE des prévisions par rapport au données réelles: %.2f RMSE' % (score_sarimax))