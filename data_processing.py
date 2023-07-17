#!/usr/bin/env python
# coding: utf-8

# In[76]:


import streamlit as st
import pandas as pd
import pickle
import gzip
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn

# Importation du fichier clients (fichier original nettoyé)
df = pd.read_csv('application_clean_sample.csv', sep=';')

# Importation du fichier de données filtrées
df_filtered = pd.read_csv('df_filtered_sample.csv', sep=';')
data = df_filtered.copy()

# On ouvre le fichier pickel contenant les informations relatives à notre modèle
with gzip.open('modele_optimal.pickle.gz', 'rb') as file:
    modele_save = pickle.load(file)

# Accédez aux informations du modèle
y_train = modele_save['y_train']
y_test = modele_save['y_test']
X_train = modele_save['X_train']
X_test = modele_save['X_test']
y_pred_prob_test = modele_save['y_pred_prob_test']
y_pred_prob_train = modele_save['y_pred_prob_train']
trained_model = modele_save['trained_model']

# Ajout des probabilités aux données de test et d'entraînement
X_train['Proba'] = y_pred_prob_train
X_test['Proba'] = y_pred_prob_test

# On concatène nos df
data_prob = pd.concat([X_train, X_test])
data_prob = data_prob[data_prob['SK_ID_CURR'].isin(df['SK_ID_CURR'])]

# Calcul de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_test)

# Calcul de l'indice J de Youden pour chaque seuil
j_scores = tpr + (1 - fpr) - 1

# Trouver l'indice du seuil optimal qui minimise le FNR et maximise le TPR
optimal_threshold_index = j_scores.argmax()

# Seuil optimal
optimal_threshold = thresholds[optimal_threshold_index]

# Prédiction des classes avec le seuil optimal
y_pred_optimal = (y_pred_prob_test >= optimal_threshold).astype(int)

# Calcul des métriques de qualité de la classification avec le seuil optimal
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
precision_optimal = precision_score(y_test, y_pred_optimal)
recall_optimal = recall_score(y_test, y_pred_optimal)
f1_optimal = f1_score(y_test, y_pred_optimal)
AUC_optimal = roc_auc_score(y_test, y_pred_prob_test)

# On crée une fonction affichant la jauge de prédiction du seuil pour chaque client
def jauge(value, optimal_threshold):
    if value > optimal_threshold:
        color='red'
    else:
        color='green'
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = value,
        mode = "gauge+number",
        title = {'text': "Probabilité prédite"},
        gauge = {'axis': {'range': [None, 1]}, 
                 'steps': [
                     {'range': [0, 0.5], 'color': "lightgray"},
                     {'range': [0.5, 1], 'color': "gray"}],
                 'threshold': {'line' : {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': optimal_threshold},
                 'bar' : {'color' : color}}))
    return fig

# On importe le fichier contenant les valeurs de shap
with open('shap_values_sample.pickle', 'rb') as file:
    shap_file_sample = pickle.load(file)

shap_values = shap_file_sample['shap_values']
data_shap_scaled = shap_file_sample['shap_scaled']
explainer = shap_file_sample['explainer']
data_shap = shap_file_sample['data_shap']

# On crée une fonction qui affiche 2 graphiques de la distribution d'une feature sélectionnée
# pour les 2 classes

def distri_features(df, optimal_threshold, feature, client_value):
    # Clients de la classe 1 (prêt non accordé) en utilisant le seuil
    class_1_data = df[df['Proba'] >= optimal_threshold]
    
    # Clients de la classe 0 (prêt accordé) en utilisant le seuil
    class_0_data = df[df['Proba'] < optimal_threshold]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribution de la feature pour la classe 0 (prêt accordé)
    sns.histplot(class_0_data[feature], kde=True, color='green', ax=axes[0])
    axes[0].axvline(client_value, color='blue', linestyle='dashed', linewidth=1)
    axes[0].set_title(f"Distribution de la feature {feature} pour la classe 0 (prêt accordé)")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Count')

    # Distribution de la feature pour la classe 1 (prêt non accordé)
    sns.histplot(class_1_data[feature], kde=True, color='red', ax=axes[1])
    axes[1].axvline(client_value, color='blue', linestyle='dashed', linewidth=1)
    axes[1].set_title(f"Distribution de la feature {feature} pour la classe 1 (prêt non accordé)")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    
    fig = plt.gcf()
    st.pyplot(fig)
    
# On crée une fonction affichant un nuage de points entre 2 features sélectionnées

def bivarié_plot(feature1, feature2, df, client_value):
    # Scores des clients
    score = df['Proba']
    
    # Données des deux fonctionnalités
    data_features = df[[feature1, feature2, 'SK_ID_CURR']]
    
    # Positions des clients
    client_positions = data_features[data_features['SK_ID_CURR'] == client_value]
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_features[feature1], data_features[feature2], c=score, cmap='coolwarm')
    plt.colorbar(scatter, label='Scores')
    plt.scatter(client_positions[feature1], client_positions[feature2], color='black', marker='*', s=200, label='Client')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Analyse bi-variée entre {} et {} pour le client {}'.format(feature1, feature2, client_value))
    plt.legend(loc='upper right')
    
    fig = plt.gcf()
    st.pyplot(fig)

