#!/usr/bin/env python
# coding: utf-8

# In[45]:


import matplotlib.pyplot as plt
import shap
import streamlit as st
import requests
from PIL import Image
from data_processing import data_prob, shap_values, data_shap_scaled, explainer, data_shap, optimal_threshold, jauge, distri_features, bivarié_plot


# In[46]:


def get_client_info(client_id):
    response = requests.get(f"https://fastapi-projet7-24875f0688c4.herokuapp.com/clients/{client_id}")
    return response.json()

# Créez une interface utilisateur Streamlit
def main(data_shap):
    st.title("Application tableau de bord interactif")
    
    image = Image.open('image_P7.png')
    st.image(image, caption='Logo entreprise "prêt à dépenser"', use_column_width=True)
    
    # Saisie de l'identifiant client
    client_id = st.selectbox("Sélectionnez l'identifiant client :", data_prob['SK_ID_CURR'].unique())
    
    # Vérifiez si l'identifiant client est saisi
    if client_id:
        # Appel de la fonction pour récupérer les informations du client
        client_info = get_client_info(client_id)
        
        # Vérifiez si le client est trouvé
        if "message" in client_info:
            st.error("Client non trouvé")
        else:
            # Affichage des informations du client
            st.subheader("Informations du client")
            
            # Formatage des informations en texte
            info_text = ""
            for key, value in client_info.items():
                formatted_key = f"__{key}__"
                info_text += f"{formatted_key}: {value}\n"
            
            # Affichage des informations formatées
            st.markdown(info_text)
            
            # Récupération du score du client
            score = client_info.get("Score du client:")
                
            # Comparaison de la probabilité prédite avec le seuil optimal
            if score > optimal_threshold:
                position = "Refusé"
                position_color = "red"
            else:
                position = "Accepté"
                position_color = "green"
                    
            # Affichage de la jauge
            fig = jauge(score, optimal_threshold)
            fig.update_layout(annotations=[
                dict(
                    x=0.5,
                    y=0.45,
                    text=position,
                    showarrow=False,
                    font=dict(color=position_color, size=24)
                )
            ])
            st.plotly_chart(fig)
            
            st.subheader(f"Feature importance globale (à gauche) et locale du client {client_id} (à droite)")
            
            # Feature importance globale
            fig, ax = plt.subplots()
            summary_plot = shap.summary_plot(shap_values, data_shap_scaled, feature_names=data_shap.columns, show=False)
            columns = st.columns(2)

            with columns[0]:
                st.pyplot(fig)
                
            # Réindexer le DataFrame avec des indices continus
            data_shap = data_shap.reset_index(drop=True)

            # Feature importance locale
            client_data_shap = data_shap[data_shap['SK_ID_CURR'] == int(client_id)].index.item()
            explanation_client = shap.Explanation(values=shap_values[1][client_data_shap], base_values=explainer.expected_value[1], feature_names=data_shap.columns)

            # Création du waterfall plot pour le client spécifié
            fig2, ax2 = plt.subplots()
            waterfall_plot = shap.plots.waterfall(explanation_client, show=False)
            with columns[1]:
                st.pyplot(fig2)
            
            st.subheader("Distribution de la feature sélectionnée dans la liste")

            # Sélection d'une feature pour le graphique de distribution
            feature = st.selectbox("Sélectionnez une feature:", data_prob.columns)

            # Récupération de la valeur de la feature pour le client sélectionné
            client_data = data_prob[data_prob['SK_ID_CURR'] == int(client_id)]
            if not client_data.empty:
                client_value = client_data[feature].values[0]
                # Affichage des graphiques de distribution
                distri_features(data_prob, optimal_threshold, feature, client_value)
            else:
                st.warning("Aucune donnée disponible pour cet identifiant client")
            
            st.subheader("Analyse bi-variée entre 2 features sélectionnées (dégradé de couleur selon le score)")
            
            # Sélection de 2 features
            selected_features = st.multiselect("Sélectionnez deux features:", data_prob.columns, default=[])
            
            # Vérification
            if len(selected_features) != 2:
                st.warning("Veuillez sélectionner exactement deux features.")
            else:
                feature1, feature2 = selected_features[0], selected_features[1]
                # Affichage scatter plot
                if not client_data.empty:
                    bivarié_plot(feature1, feature2, data_prob, int(client_id))
                else:
                    st.warning("Aucune donnée disponible pour cet identifiant client")

if __name__ == "__main__":
    main(data_shap)


# In[ ]:




