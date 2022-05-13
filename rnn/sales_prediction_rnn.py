import sqlalchemy
import matplotlib.pyplot as plt 
import pandas as pd
pd.option_context('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Bidirectional,GRU,Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt, ceil

add_data_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\convertcsv.csv'
add_d21_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\d2021.csv'
scaler = MinMaxScaler(feature_range=(0,1)) #le scaler est un outil qui permet de normaliser les données en transformant 
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

def pretraitement_données(df):
    df['DateFacture'] = pd.to_datetime(df['DateFacture']) #formatage du champ Date
    #groupement de la dataframe en mois
    per = df.DateFacture.dt.to_period("M") 
    g = df.groupby(per) 
    df_clean = g.sum()
    #importation des données de validation
    data_test = pd.read_csv(add_data_path)
    data_test.set_index('Date',inplace=True)
    
    #jointure des deux df 
    df_finale = pd.concat([data_test,df_clean])
    return df_finale

def transformation_données_2021():
    data21 = pd.read_csv(add_d21_path)
    data21.drop("Date",axis=1,inplace=True)
    array_d21 = np.array(data21)
    return array_d21

#preprocessing de la df pour l'implementer dans le model
def division_données_train_test(df_finale):
    
    dataset = df_finale.values #conversion des valeurs en numpy array   
    train_size = int(len(df_finale)*0.6) #ajustement de la taille d'entrainement du model
    test_size = len(df_finale) - train_size
     #ajustement de la taille de test du model
    #les données entre 0 et 1 ( 0 étant la valeur minime et 1 la valeur maximale) car les LSTM sont plus performants
    #avec des valeurs normalisées
    dataset = scaler.fit_transform(dataset) #application du scaler sur la dataset
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return dataset ,train, test #division de notre dataset en 2 parties entrainement et test


def conversion_en_window(dataset,temps): #convertir les données en format comprehensible par la couche d'input LSTM ; 
# 2 3 4 .. len(dataset) -1 -> derniere valeure
# 3 4 5 .. len(dataset) -1 -> derniere valeure
 X, Y =[], []
 for i in range(len(dataset)-temps):
  d=i+temps  
  X.append(dataset[i:d,0])
  Y.append(dataset[d,0])
 return np.array(X), np.array(Y)

#-------------------- WKEFT HOUNI 
def pretraitement_données_lstm(dataset,test): 
    trainX , trainY = conversion_en_window(dataset,12) #les données d'entrainement
    testX , testY = conversion_en_window(test,12) #les données de test
    
    #conversion pour l'input
    trainX = trainX.reshape((trainX.shape[0],1,trainX.shape[1]))
    testX = testX.reshape((testX.shape[0],1,testX.shape[1]))
    return trainX,trainY,testX,testY



# ----- Initialisation du model ----- # 
def création_model(train_gen,test_gen):
    model = Sequential() #utilisation du model de sequence puisque nos données sont en format de series temporelles  
    model.add(Bidirectional(LSTM(68,input_shape=(24,1,12), activation='relu',return_sequences=True))) 
    model.add(Dropout(0.2))#les hyperparameters optimisées avec kerastuner
    model.add(Bidirectional(LSTM(200,activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(1,'linear'))
    model.compile(loss="mse",optimizer=Adam(learning_rate=0.001),metrics=[RootMeanSquaredError()]) #compilation du model avec la loss function du mean squared error et le adam optimizer
    model.fit(train_gen,verbose=2,validation_data=test_gen,epochs=100,batch_size=256)  #ajustement des donnees dans le model
    return model


 
def test_model(model,trainX,trainY,testX,testY):
   
    trainPredict= model.predict(trainX) #prévision sur les données d'entrainement
    testPredict = model.predict(testX) ##prévision sur les données de test
    trainPredict = scaler.inverse_transform(trainPredict) #inversement du scaler pour retrouver les valeurs initiales
    trainY = scaler.inverse_transform([trainY])
    trainY = np.transpose(trainY,(1,0)) #ajustement de la forme du tableau
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    testY = np.transpose(testY,(1,0))
    return trainX,trainY,testX,testY,trainPredict,testPredict

# -----Prevision------- #

    
def affichage_des_plots(array_d21,prev,trainY,trainPredict):
    plt.figure(figsize=(20,20))
    ax1 = plt.subplot(3, 1, 1)
    plt.title('Données réelles vs prévision 2021')
    ax1.plot(array_d21,'r-o', label="données réelles")
    ax1.plot(prev,'b-o', label="prévisions") # Showing top Horizontal 
    plt.legend()
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.title("Données d'entrainement réelles vs prédites")
    ax2.plot(trainY,'-o', label="Données d'entrainement")
    ax2.plot(trainPredict,'.:', label="données prédites d'entrainement") 
    plt.legend()
    plt.show()
     
    result = pd.concat([pd.DataFrame(prev),pd.DataFrame(array_d21)],axis=1)
    l = ['Prévision','Données 2021']
    result.columns = l
    return result

     
df = connexion_bd() #création de notre dataframe
df_finale = pretraitement_données(df) #prétraitement de la base
array_d21 = transformation_données_2021() #importation des données de 2021

dataset , train ,test = division_données_train_test(df_finale) #division de notre dataset en 
#entrainement et test
train_gen = TimeseriesGenerator(dataset,dataset,12,1) #conversion des données de test et
#entrainement au format convenable
test_gen = TimeseriesGenerator(test,test,12,1)
#model_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\best_model'
model = création_model(train_gen,test_gen) #création de notre modele
trainPredict = model.predict(train_gen) #prévision des données d'entrainement
testPredict = model.predict(test_gen) #prévision des données de test
#prévision des valeurs de l'année prochaine (2021)
prévision = []
current_batch = train[-12:]  
current_batch = current_batch.reshape(1,12,1) 
future = 12

for i in range(len(test) + future):
    
    current_pred = model.predict(current_batch)[0]
    prévision.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
df_prev = pd.DataFrame(columns=["réelles","prev"]) #création de la dataframe
#pour les resultat
prev_rescaled = scaler.inverse_transform(prévision)
df_prev.loc[:,"predicted"] = prev_rescaled[:,0]

p = pd.DataFrame(prev_rescaled[24:])
p.columns = ['prev']
p['réelles'] = array_d21

prevScore = sqrt(mean_squared_error(array_d21,p['prev'])) #application de la RMSE 
print('RMSE des prévisions par rapport au données réelles: %.2f RMSE' % (prevScore))
p.plot() 



        