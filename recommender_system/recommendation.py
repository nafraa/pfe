import sqlalchemy
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  


pd.option_context('display.max_rows',None)
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
    return rec

