import streamlit as st
import requests
import json

# Page Title
st.title("Prédiction de Joueur de Basketball")

# Image of the NBA
st.image("logo-NBA.jpg", caption="NBA", use_column_width=True)

# Sidebar Title
st.sidebar.title("Entrez les données du joueur")

# Define the input features with their descriptions
FEATURES = {
    'GP': 'Nombre de matchs joués',
    'MIN': 'Minutes jouées par match',
    'PTS': 'Points marqués par match',
    'FGM': 'Tirs réussis par match',
    'FGA': 'Tentatives de tir par match',
    'FG%': 'Pourcentage de réussite des tirs',
    '3P Made': 'Paniers à 3 points réussis par match',
    '3PA': 'Tentatives de paniers à 3 points par match',
    '3P%': 'Pourcentage de réussite des paniers à 3 points',
    'FTM': 'Lancers francs réussis par match',
    'FTA': 'Tentatives de lancers francs par match',
    'FT%': 'Pourcentage de réussite des lancers francs',
    'OREB': 'Rebonds offensifs par match',
    'DREB': 'Rebonds défensifs par match',
    'REB': 'Total des rebonds par match',
    'AST': 'Passes décisives par match',
    'STL': 'Interceptions par match',
    'BLK': 'Contres par match',
    'TOV': 'Balles perdues par match'
}

# Sidebar Inputs with tooltips
input_data = {}
for feature, description in FEATURES.items():
    input_data[feature] = st.sidebar.number_input(f"{feature} ({description})", value=0.0)

# Predict Button
if st.sidebar.button("Prédire"):

    response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
    
    if response.status_code == 200:
        result = response.json()
        prediction_message = "Joueur de basket à fort potentiel pour investissement à long terme. 🏀💰" if result['prediction'] == 1 else "Joueur de basket peu prometteur pour investissement à long terme. 🏀📉"
        st.success(f"Prédiction: {prediction_message}")
        st.info(f"Probabilité: {result['probability'][0][1]*100:.2f}%")
    else:

        st.error("Erreur lors de la prédiction")
