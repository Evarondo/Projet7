# Projet 7 : Implémentez un modèle de scoring
## Mission
En tant que Data Scientist au sein de l'entreprise "Prêt à dépenser", proposant des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt, l'idée est de mettre en oeuvre un outil de "**scoring crédit**" pour calculer la probabilité qu'un client rembourse son crédit, et ainsi, de classifier la demande en crédit *accordé* ou *refusé*. Pour cela, il est nécessaire de développer un algorithme de classification en s'appuyant sur différentes sources de données (comportement, provenant d'institutions financières, ...).
La création d'un dashboard interactif est développé pour plus de transparence lors de l'octroi de crédit et pour que les clients aient accès à leurs données personnelles et puissent les explorer plus facilement. 

Pour cela, le modèle de scoring de prédiction est mis en production à l'aide d'une API, puis le dashboard interactif appelle l'API pour les prédictions. 

## Données
Pour mener à bien le projet, différents fichiers .csv contenant les informations sont nécessaires téléchargés ici : https://www.kaggle.com/c/home-credit-default-risk/data. Ces fichiers sont les suivants:
- Fichier HomeCredit_columns_description.csv
- Fichier application_train.csv
- Fichier application_test.csv
- Fichier bureau.csv
- Fichier bureau_balance.csv
- Fichier credit_card_balance.csv 
- Fichier installments_payments.csv
- Fichier POS_CASH_balance.csv
- Fichier previous_application.csv
- Fichier sample_submission.csv

## Description du répertoire
Le répertoire contient d'abord le notebook jupyter de nettoyage et modélisation : Projet7_nettoyage_modelisation.ipynb.
Puis, un dossier "api" contenant les éléments nécessaires au déploiement de l'API et un dossier "dashboard" contenant les éléments nécessaires au déploiement du dashboard Streamlit à partir de l'API. 
