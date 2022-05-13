import sys
path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev'
sys.path.append(path)
#imporation des elements de 
import streamlit as st
#st.set
import pandas as pd
import matplotlib.pyplot as plt
from rnn.sales_prediction_rnn import df_finale,p,prevScore
from recommender_system.recommendation import top3_similarite_client_code,recommend,df_finale_rec 
from clustering.customer_rfm_based_clustering import df_RFM ,rfm_standardisée ,rfm
from prevision_prophet.prev import prophet_score,df_resultats_fb
import numpy as np
import seaborn as sns
add_d21_path = r'C:\Users\ahmed\OneDrive\Desktop\CODE_PFE_ETL\prev\rnn\data\d2021.csv'



from streamlit_option_menu import option_menu
def transformation_données_2021():
    data21 = pd.read_csv(add_d21_path)
    data21.drop("Date",axis=1,inplace=True)
    array_d21 = np.array(data21)
    return array_d21
from streamlit_ace import st_ace



col1, col2 = st.beta_columns([1,1]) #alligner la page selon les colonnes
#Visualisation
plt.rcParams['figure.figsize'] = (10,10) 
fig0,ax0  = plt.subplots()
ax0 = sns.scatterplot(x=rfm['Recency'], y=rfm['Frequency'], palette="Set1",hue=rfm['cluster num'])
# ---- 3D 
plt.rcParams['figure.figsize'] = (10,10) 
fig = plt.figure(1)
fig.add_subplot(projection="3d")
plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')

ax1.scatter(rfm['Recency'],rfm['Frequency'],rfm['Monetary'],c=rfm_standardisée['cluster num'],s=200,cmap='spring',alpha=0.5,edgecolor='darkgrey')
ax1.set_xlabel('Frequency',fontsize=16)
ax1.set_ylabel('Recency',fontsize=16)
ax1.set_zlabel('Monetary',fontsize=16)
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(p['réelles'],label='Données réelles')
ax2.plot(p['prev'],label='prévisions')
ax2.set_xlabel('mois')
ax2.set_ylabel("Chiffre d'affaires")
legend = ax2.legend(loc='upper right', shadow=True, fontsize='x-large')

fig3, ax3 = plt.subplots()
ax3.plot(df_resultats_fb['donnees reelles'],label='Données réelles')
ax3.plot(df_resultats_fb['previsions'],label='prévisions')
ax3.set_xlabel('mois')
ax3.set_ylabel("Chiffre d'affaires")
legend = ax3.legend(loc='upper right', shadow=True, fontsize='x-large')

fig4 = plt.figure()
ax4 = fig4.add_axes([0,0,1,1])
clust = [0,1,2,3]
number = rfm['cluster num'].value_counts()
ax4.bar(clust,number)   
plt.show()
with col1:
    st.title("Application analytique de GIPA")

menu = st.sidebar.selectbox("Navigation",['Prévision','Segmentation','Recommandation']) #sidebar de navigation
if menu == "Prévision":
    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.subheader('Prévision avec les reseaux de neuronnes récurentes')
        st.write("---")
        st.write("Prévisions pour l'année 2021")
        st.dataframe(p)
        st.write("Graphique des LSTM")
        st.pyplot(fig2)
        st.write('---')
        st.write('Score du RMSE de la prévision : ')
        st.write(prevScore)
        st.write("---")
    with col2:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        
        st.subheader('Prévision avec la méthode Prophet')
        st.write("---")
    
        st.write("Prévisions pour l'année 2021")
        st.dataframe(df_resultats_fb)
        st.write("Graphique de Prohpet")
        st.pyplot(fig3)
        st.write('---')
        st.write('Score du RMSE de la prévision : ')
        st.write(prophet_score)
        st.write("---")
    with col2:
        st.write('---')
        
    with col1:
        st.write("Code des LSTM")
        code = r'''
        
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

        preprocessing de la df pour l'implementer dans le model
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
        p.plot() '''
        st.code(code,language='python')
        code_prophet ='''
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
        print('RMSE des prévisions par rapport au données réelles: %.2f RMSE' % (prophet_score))'''
    with col2:
        st.write(" ")
        st.write("Code de la méthode Prophet")
        
        st.write(" ")
        

        st.code(code_prophet,language="python")
        
if menu == "Segmentation":
    
    
    with col1:
        st.subheader('Segmentation des clients')
        st.write('segmentation des clients en groupe selon leurs RFM respectifs')
        st.dataframe(df_RFM)
        st.write('')
        st.pyplot(fig)
    
        """
        les clusters sont classées de 0 a 3 , 3 étant la meilleure classe et 0 la pire.
        """
        st.write("---")
        st.write('repartition des clients selon les cluster')
        st.pyplot(fig4)
    
    with col2:
        st.write("Code de la procédure de segmentation")
        st.write("---")
        code_clust = r'''
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
        plt.show()'''
        st.code(code_clust,language="python")    
    
if menu == "Recommandation":
    with col1:
        st.subheader('Recommendation des articles selon le Code Client')
        st.write("---")
        st.write("Liste des clients et articles")
        st.dataframe(df_finale_rec)
        st.write('---')
        code = st.text_input(label="Saisir le Code du Client",value="00060")
        top3_client_code_list = top3_similarite_client_code(code)
        recommendation = recommend(top3_client_code_list)
        st.write('liste des articles similaires')
        st.write(recommendation)
    with col2:
        code_rec = '''pd.option_context('display.max_rows',None)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        #Importation des données de la base MSSQL avec sqlalchemy
        server="localhost"
        database="GIPA"
        driver ="ODBC Driver 17 for SQL Server"
        con = f'mssql://@{server}/{database}?driver={driver}'
        engine = sqlalchemy.create_engine(con,fast_executemany=True)
        con = engine.connect()
        sql = "select NumFacture,CodeClient,CodeArticle,QteFacturee,CA from GIPA.dbo.dimension_facture"
        sql_article = "select CodeArticle ,Article from GIPA.dbo.faits_resultat "
        df = pd.read_sql(sql,con)  #mise en place des données dans un DataFrame 
        df_article = pd.read_sql(sql_article,con)
        df_finale_rec = df.groupby(['CodeClient','CodeArticle'])['QteFacturee'].sum().unstack().fillna(0) #transformer la dataset en format convenable pour la matrice
        df_finale_rec.applymap(lambda x:abs(x))


        client_similarite_matrice = pd.DataFrame(cosine_similarity(df_finale_rec)) #application de la similarite
        #nettoyage
        client_similarite_matrice.columns = df_finale_rec.index
        client_similarite_matrice['CodeClient'] = df_finale_rec.index
        client_similarite_matrice = client_similarite_matrice.set_index('CodeClient')


        def top3_similarite_client_code(CodeArticle):
            #méthode qui retourne les codes des 3 premiers clients
            clients_code = client_similarite_matrice.loc[CodeArticle].sort_values(ascending=False).index
            #methode loc retourne les lignes de la valeur en parametre 
            clients_code = clients_code[1:4] #top 3
            return list(clients_code)

        def top_3_similarite_client_articles(liste_code):
            #methode qui retourne les articles achetees par les 3 premiers clients
            #client 1
            articles_1 = set(df_finale_rec.loc[liste_code[0]].iloc[
            df_finale_rec.loc[liste_code[0]].to_numpy().nonzero()].index)
            #client2
            articles_2 = set(df_finale_rec.loc[liste_code[1]].iloc[
            df_finale_rec.loc[liste_code[1]].to_numpy().nonzero()].index)
            #client3
            articles_3 = set(df_finale_rec.loc[liste_code[2]].iloc[
            df_finale_rec.loc[liste_code[2]].to_numpy().nonzero()].index)
            return articles_1,articles_2,articles_3

        def recommend(liste):
            #method qui prend en parametre la liste des achats des clients et renvoi une recommandation
            article_client1 , article_client2 , article_client3 = top_3_similarite_client_articles(liste)
            recommendation = set.difference(article_client1,article_client2,article_client3)#utilisation 
            #de la SET pour pouvoir utiliser ses methodes tel que la difference entre chaque set
            l = list(recommendation)
            rec=[]
            for i in l:
                rec.append(df_article.loc[df_article['CodeArticle']== i ,'Article'].iloc[0])
            return rec'''
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write('Code du systeme')
        st.code(code_rec,language="python")
   
    
    
