#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from data_processing import df, data_prob, optimal_threshold

class ClientSearch(BaseModel):
    client_id: int

app = FastAPI()

@app.get("/clients/{client_id}")
async def get_client_info(client_id: int, optimal_threshold: float = optimal_threshold):
    # On récupère les informations du client à partir du DataFrame (df)
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
        
        # On récupère la probabilité client
        client_proba = float(data_prob.loc[data_prob['SK_ID_CURR'] == client_id, 'Proba'])
        
        # Comparaison de la probabilité prédite avec le seuil optimal
        if client_proba > optimal_threshold:
            position = "Refusé"
            position_color = "red"
        else:
            position = "Accepté"
            position_color = "green"

        # Dictionnaire avec les informations client
        response = {
            "Identifiant client:": identifiant,
            "Genre :": genre,
            "Age :": age,
            "Type de profession :": profession,
            "Revenu total :": revenu,
            "Nombre d'enfants:": nb_enfants,
            "Statut familial:": statut_fam,
            "Type de contrat:": type_contrat,
            "Montant du crédit:": montant_credit,
            "Score du client:": client_proba,
            "Position du client par rapport au seuil:": position
        }

        return response
    else:
        return {"message": "Client non trouvé"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)


# In[ ]:




