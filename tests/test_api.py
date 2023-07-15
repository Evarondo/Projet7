#!/usr/bin/env python
# coding: utf-8


# In[9]:


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api import app


# In[18]:


client = TestClient(app)

def test_get_client_info():
    response = client.get("/clients/289313")
    assert response.status_code == 200
    data = response.json()

    assert "Identifiant client:" in data
    assert "Genre :" in data
    assert "Age :" in data
    assert "Type de profession :" in data
    assert "Revenu total :" in data
    assert "Nombre d'enfants:" in data
    assert "Statut familial:" in data
    assert "Type de contrat:" in data
    assert "Montant du crédit:" in data
    assert "Score du client:" in data
    assert "Position du client par rapport au seuil:" in data


# In[ ]:




