import sqlalchemy
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import numpy as np
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import mean_squared_error

pd.option_context('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#Importation des données de la base MSSQL avec sqlalchemy
server="localhost"
database="GIPA"
driver ="ODBC Driver 17 for SQL Server"
con = f'mssql://@{server}/{database}?driver={driver}'
engine = sqlalchemy.create_engine(con,fast_executemany=True)
con = engine.connect()
sql = "select NumFacture,CodeArticle,QteFacturee,CA from GIPA.dbo.dimension_facture"
df = pd.read_sql(sql,con)  #mise en place des données dans un DataFrame 
df_finale = df.groupby(['NumFacture','CodeArticle'])['QteFacturee'].sum().unstack().reset_index().fillna(0).set_index('NumFacture') #transformer la dataset en format convenable pour l'algorithme apriori
df_finale.applymap(lambda x:abs(x))






def hot_encode(resultat):
    if resultat  == 0:
        return 0
    else:
        return 1
df_finale = df_finale.applymap(lambda x:hot_encode(x))
from mlxtend.frequent_patterns import apriori,association_rules
article_frequence = apriori(df_finale,min_support=0.07,use_colnames=True)
regles = association_rules(article_frequence,metric='lift',min_threshold=1)
regles['antecedents'] = regles['antecedents'].apply(list)
regles['consequents'] = regles['consequents'].apply(list)

regles.to_csv('article_similarité')
