#chargement des packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import *

def main():
     
     st.title("Application de Machine Learning pour la prédiction de puissance électrique avec les modèles SARIMA(p,d,q)")
     st.subheader("Auteur: Jean Baptiste Love KPOSSOU")

     #Fonction d'importation des données
     @st.cache_data(persist=True)
     def load_data():
          chemin="/home/user/Téléchargements/Données mois.xlsx"
          data=pd.read_excel(chemin)
          return data
     
     #affichage des données
     df=load_data()
     df_sample=df.sample(20)
     #st.write(df)

     #afficher les données avec un checkbox dans la barre latérale gauche
     if st.sidebar.checkbox("Afficher les données brutes",False):
          st.title("Jeu de données 'puissance électrique': Echantillon de 20 observations")
          st.write(df_sample)

     
     classififier=st.sidebar.selectbox("Variables",
                                       ("Puissance 63kv", "Puissance 15kv")
                                       )


    # CHARGEMENT DE DONNÉES
     chemin = "/home/user/Téléchargements/Données mois.xlsx"
     data = pd.read_excel(chemin)

# Transformation de la colonne 'month' en datetime
     data['month'] = pd.to_datetime(data['month'], infer_datetime_format=True)

# Affichage des données
     st.subheader('Affichage de la base de données')
     st.write(data)

# PARTAGE DES DONNÉES
     df1 = data[["month", "big"]]
     df2 = data[["month", "small"]]

# Mettre la date en index de la table
     for j in [df1, df2]:
          j.set_index('month', inplace=True)

# ANALYSE DES DONNÉES

# Affichage des graphiques
     st.subheader('Évolution du nombre mensuel de 63kv')
     st.line_chart(df1)

     st.subheader('Évolution du nombre mensuel de 15kv')
     st.line_chart(df2)

# MODELE AUTOMATIQUE

# Séparer les données en ensemble d'entraînement et ensemble de test
     train_data1 = df1['big'][:-15]
     train_data2 = df2['small'][:-15]
     test_data1 = df1['big'][-15:]
     test_data2 = df2['small'][-15:]

# Utiliser auto_arima pour trouver le meilleur modèle ARIMA
     model1 = pm.auto_arima(train_data1)
     model2 = pm.auto_arima(train_data2)

     st.subheader('Résumé des modèles ARIMA')

     st.write("Modèle pour 63kv")
     st.write(model1.summary())

     st.write("Modèle pour 15kv")
     st.write(model2.summary())

# PRÉDICTION DE 2024 À 2060

# Entraînement du modèle ARIMA(1,1,0) pour 63kv
     model1 = ARIMA(df1['big'], order=(1, 1, 0))
     model1_fit = model1.fit()
     forecast1 = model1_fit.forecast(steps=468)
     dates_forecast1 = pd.date_range(start='2024-01-01', periods=468, freq='MS')
     forecast_df1 = pd.DataFrame({'month': dates_forecast1, 'big': forecast1})

# Entraînement du modèle ARIMA(0,1,0) pour 15kv
     model2 = ARIMA(df2['small'], order=(0, 1, 0))
     model2_fit = model2.fit()
     forecast2 = model2_fit.forecast(steps=468)
     dates_forecast2 = pd.date_range(start='2024-01-01', periods=468, freq='MS')
     forecast_df2 = pd.DataFrame({'month': dates_forecast2, 'small': forecast2})

# Mise en commun des données réelles et des données prédictes
     data_pred = pd.merge(forecast_df1, forecast_df2)

# Visualisation de l'ensemble des données (de 2021 à 2060) pour big(63kv) et small(15kv)
     Data_total_2021_2060 = data_pred.loc[data_pred["month"] < "2061-01-01"]
     Data_total_2021_2060.set_index('month', inplace=True)

     st.subheader('Visualisation de l\'ensemble des données (de 2021 à 2060)')

# Visualisation pour big(63kv)
     st.write("Évolution du nombre mensuel de 63kv de janvier 2021 à décembre 2060")
     st.line_chart(Data_total_2021_2060['big'])

# Visualisation pour small(15kv)
     st.write("Évolution du nombre mensuel de 15kv de janvier 2021 à décembre 2060")
     st.line_chart(Data_total_2021_2060['small'])

# Export des données totales de janvier 2021 à décembre 2060
     import base64
     if st.button('Exporter data.csv'):
          csv = Data_total_2021_2060.to_csv(index=True)
          b64 = base64.b64encode(csv.encode()).decode()
          href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Télécharger data.csv</a>'
          st.markdown(href, unsafe_allow_html=True)
if __name__=="__main__":
     main()