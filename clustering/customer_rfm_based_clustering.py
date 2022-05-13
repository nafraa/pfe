import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style = "darkgrid")

import sqlalchemy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime
from sklearn.metrics import silhouette_score


import warnings
warnings.filterwarnings("ignore")
add_data_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\convertcsv.csv'
pd.option_context('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
server="localhost"
database="GIPA_T"
driver ="ODBC Driver 17 for SQL Server"
con = f'mssql://@{server}/{database}?driver={driver}'
engine = sqlalchemy.create_engine(con,fast_executemany=True)
con = engine.connect()
sql = "select   Client,DateFacture,CA from GIPA.dbo.faits_resultat"
df_clients = pd.read_sql(sql,con)
df_clients['Date'] = pd.DatetimeIndex(df_clients.DateFacture).date
derniere_date = datetime.date(2021,1,1) #date de comparaison pour la Recency

# ---- Recency ---- 
df_clients_recency = df_clients.groupby(['Client'],as_index=False)['Date'].max()
#grouper la dataset en fonction des client en affichant les dates
df_clients_recency.columns = ['Client','Dernier_jour_dachat']
#changement du nom des colonnes
df_clients_recency['Recency'] = df_clients_recency.Dernier_jour_dachat.apply(lambda x:(derniere_date - x).days)
#methode lambda pour soustraire la date de comparaison a chaque ligne <date> pour compter le nombre de jours 
#qui les sépare


# ---- Frequency et Monetary ---- 
df_FM = df_clients.groupby('Client').agg({'DateFacture' :lambda x:len(x),'CA':lambda x:x.sum()})
#utilisation de la méthode agg sur une series ; compter la longueur de la colonne
#datefacture pour savoir combien de factures ( nombre d'achat pour frequency) 
#méthode sum pour faire la somme des achats 
df_FM.rename(columns={'DateFacture':'Frequency','CA':'Monetary'},inplace=True)

df_RFM = df_clients_recency.merge(df_FM,left_on='Client',right_on='Client')
#dt qui contient tous les criteres
df_RFM.drop('Dernier_jour_dachat',inplace=True,axis=1)
scaler = StandardScaler()
rfm_standardisée = scaler.fit_transform(df_RFM[['Recency','Frequency','Monetary']])
#transformation des valeurs en valeurs normalisées pour avoir
#une meilleure performance
n_clusters_range = [2,3,4,5,7,8]#liste qui contient les nombre de clusters logiques
#on boucle sur chaque nombre de clusters et on ajuste le KMeans en fonction
#de ce nombre pour savoir quel est le meilleur n
for cluster in n_clusters_range:
    kmeans=KMeans(n_clusters=cluster,max_iter=50)#initiation du modele
    kmeans.fit(rfm_standardisée)#ajustement
    cluster_labels = kmeans.labels_#labels désigne le nombre de cluster
    silhouette_moy = silhouette_score(rfm_standardisée,cluster_labels)
#Methode Silhouette pour déterminer le nombre optimale de clusters
    print("n = {} , score = {}".format(cluster,silhouette_moy)  )
        

kmeans=KMeans(n_clusters=4,max_iter=100)#instanciation du KMeans
kmeans.fit(rfm_standardisée) #ajustement des données
rfm_standardisée = pd.DataFrame(rfm_standardisée)#conversion en df
rfm_standardisée.loc[:,'Client'] = df_RFM['Client']#affection de chaque client 
rfm_standardisée['cluster num'] = kmeans.labels_  #affectation des nombre de clusters
rfm_standardisée.columns = ['Recency','Frequency','Monetary','Client','cluster num']
rfm_standardisée.drop('Client',axis=1,inplace=True)
#création de la df finale pour la visualistion 
#avec des données non standardisées
#---- Table Finale nettoyée ----w
df_RFM['num of cluster '] = kmeans.labels_ 
rfm = scaler.inverse_transform(df_RFM[['Recency','Frequency','Monetary']])
rfm = pd.DataFrame(rfm)
rfm['num cluster'] = rfm_standardisée['cluster num']
rfm.columns=(['Recency','Frequency','Monetary','cluster num'])
plt.rcParams['figure.figsize'] = (10,10) 
sns.scatterplot(x=rfm['Recency'], y=rfm['Frequency'], palette="Set1",hue=rfm['cluster num'])
count = rfm['cluster num'].value_counts()
print(type(count))
print(count[0])

# ---- 3D 
plt.rcParams['figure.figsize'] = (15,10) 


fig = plt.figure(1)
fig.add_subplot(projection="3d")
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.scatter(rfm['Recency'],rfm['Frequency'],rfm['Monetary'],c=rfm_standardisée['cluster num'],s=200,cmap='spring',alpha=0.5,edgecolor='darkgrey')
ax.set_xlabel('Frequency',fontsize=16)
ax.set_ylabel('Recency',fontsize=16)
ax.set_zlabel('Monetary',fontsize=16)
plt.show()
