#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import uvicorn
from fastapi import FastAPI
import pickle
import nest_asyncio
nest_asyncio.apply()

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

# Importation du fichier clients (fichier original nettoyé)
df = pd.read_csv('application_clean.csv', sep=';')

from pydantic import BaseModel

class ClientSearch(BaseModel):
    client_id: int

app = FastAPI()

@app.get("/clients/{client_id}")
async def get_client_info(client_id: int):
    # Récupérer les informations du client à partir du DataFrame (df)
    client_info = df[df['SK_ID_CURR'] == client_id]
    
    if not client_info.empty:
        identifiant = int(client_info.iloc[0]['SK_ID_CURR'])
        genre = str(client_info.iloc[0]['CODE_GENDER'])
        age = int(client_info.iloc[0]['AGE'])
        profession = str(client_info.iloc[0]['OCCUPATION_TYPE'])
        revenu = float(client_info.iloc[0]['AMT_INCOME_TOTAL'])
        nb_enfants = int(client_info.iloc[0]['CNT_CHILDREN'])
        statut_fam = str(client_info.iloc[0]['NAME_FAMILY_STATUS'])
        
        type_contrat = str(client_info.iloc[0]['NAME_CONTRACT_TYPE'])
        montant_credit = float(client_info.iloc[0]['AMT_CREDIT'])
        
        # Dictionnaire avec les informations client
        response = {
            "Identifiant client :": identifiant,
            "Genre :": genre,
            "Age :": age,
            "Type de profession :": profession,
            "Revenu total :": revenu,
            "Nombre d'enfants:": nb_enfants,
            "Statut familial:": statut_fam,
            "Type de contrat:": type_contrat,
            "Montant du crédit:": montant_credit,
        }

        return response
    else:
        return {"message": "Client non trouvé"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8007)


# In[ ]:




