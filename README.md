# **Projet 7 : Implémentez un modèle de scoring**
## <u>Mission</u>
En tant que Data Scientist au sein de l'entreprise "Prêt à dépenser", proposant des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt, l'idée est de mettre en oeuvre un outil de "**scoring crédit**" pour calculer la probabilité qu'un client rembourse son crédit, et ainsi, de classifier la demande en crédit *accordé* ou *refusé*. Pour cela, il est nécessaire de développer un algorithme de classification en s'appuyant sur différentes sources de données (comportement, provenant d'institutions financières, ...).
La création d'un **dashboard interactif** est développé pour plus de transparence lors de l'octroi de crédit et pour que les clients aient accès à leurs données personnelles et puissent les explorer plus facilement. 

Pour cela, le modèle de scoring de prédiction est mis en production à l'aide d'une API, puis le dashboard interactif appelle l'API pour les prédictions. 

## <u>Données</U>
Pour mener à bien le projet, différents fichiers .csv contenant les informations nécessaires sont téléchargés [ici](https://www.kaggle.com/c/home-credit-default-risk/data). Les fichiers sont les suivants:
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

Pour les ouvrir et les lire dans le notebook ` Projet7_nettoyage_modelisation.ipynb`, il faut les enregistrer dans un fichier "datas".  

## <u>Description du répertoire</u>
Le répertoire contient d'abord le notebook jupyter de nettoyage et modélisation : `Projet7_nettoyage_modelisation.ipynb`, ainsi que la ``note_methodologique.pdf`` et d'un fichier `data_processing.py`.

Le fichier ``Projet7_nettoyage_modelisation.ipynb`` est constitué de 2 parties : 
- Une partie concernant l'analyse des données, le nettoyage (valeurs manquantes, aberrantes, ...) pour chaque fichier .csv, puis un feature engineering en se basant sur le kernel ayant remporté la compétition [kaggle](https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features). Les fichiers sont ensuite joins entre eux en se basant sur l'identifiant client. 
- Une seconde partie de modélisation en comparant différents modèles entre eux (Logistic Regression, Decision Tree Classification et LightGBM), ainsi que différentes méthodes de gestion de déséquilibre des classes (SMOTE, RandomUnderSampler et class_weight) en raison du fait que les classes binaires sont fortement déséquilibrées (target 1 = 8%, target 0 = 92%). Plusieurs métriques ont été choisies pour l'évaluation des différents modèles de classification. Une fois le modèle optimal choisi, un seuil de probabilité optimal est calculé, puis une analyse de la feature importance globale et locale est effectuée avec SHAP (Shapley Additive exPlanation). 

Le fichier `data_processing.py` récupère les données échantillonnées ainsi que le modèle optimal et les valeurs de shap enregistrés au format .pickle. Les fonctions créées permettent la visualisation de graphiques. L'idée, pour notre api et dashboard, est d'importer ce fichier et les données et variables traitées en vue du déploiement de nos applications. 

Un dossier `tests` contient le test unitaire Pytest pour l'api. 

## <u>Information concernant les fichiers lourds</u>
Dans le cas où nos données dépassent 100 Mo, ce qui est le cas si l'on garde les informations relatives aux 307 511 clients, Git LFS (Large File Storage), qui est une extension de Git, peut être utilisé pour la gestion des fichiers trop volumineux. Néanmoins, nous avons préféré échantillonner nos clients en ne gardant que 100 clients pour le déploiement sur Heroku afin d'éviter les erreurs comme il est explicité dans la documentation relative à [Heroku](https://devcenter.heroku.com/articles/git#deploy-from-a-branch-besides-main). Heroku ne supportant pas les fichiers chargés par l'extension Git LFS.

## <u>Prérequis</u>
* [Python](https://www.python.org/downloads/release/python-3109/) : version 3.10.9
* [Windows](https://www.microsoft.com/fr-fr/software-download/windows11) 11 
* [Git](https://git-scm.com/downloads) : version 2.41.0

Ce projet `Projet7_nettoyage_modelisation.ipynb` est à la base pour le déploiement de l'api et du dashboard.