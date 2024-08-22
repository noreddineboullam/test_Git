[33mcommit 2abe26e8ee3bbe7b8c8246b5b5a64a2de7617b1f[m[33m ([m[1;36mHEAD -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: noreddineboullam <boullam08@gmail.com>
Date:   Thu Aug 22 15:02:05 2024 +0200

    Add files via upload

[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/NBA_Interface.png b/Boullam_API_FLASK _STREAMLIT/NBA_Interface.png[m
[1mnew file mode 100644[m
[1mindex 0000000..3019cbf[m
Binary files /dev/null and b/Boullam_API_FLASK _STREAMLIT/NBA_Interface.png differ
[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/README.md b/Boullam_API_FLASK _STREAMLIT/README.md[m
[1mnew file mode 100644[m
[1mindex 0000000..e4db688[m
[1m--- /dev/null[m
[1m+++ b/Boullam_API_FLASK _STREAMLIT/README.md[m	
[36m@@ -0,0 +1,53 @@[m
[32m+[m[32m# NBA Player Prediction[m[41m[m
[32m+[m[41m[m
[32m+[m[32m## Overview[m[41m[m
[32m+[m[32mThis project aims to predict the future performance of NBA players based on various statistical features. It utilizes a machine learning model trained on historical NBA player data to make predictions about whether a player will continue their career successfully or not.[m[41m[m
[32m+[m[41m[m
[32m+[m[32m## Features[m[41m[m
[32m+[m[32mThe following features are used as inputs to the prediction model:[m[41m[m
[32m+[m[32m- GP: Number of games played by the player[m[41m[m
[32m+[m[32m- MIN: Average number of minutes played per game[m[41m[m
[32m+[m[32m- PTS: Average number of points scored per game[m[41m[m
[32m+[m[32m- FGM: Average number of field goals made per game[m[41m[m
[32m+[m[32m- FGA: Average number of field goal attempts per game[m[41m[m
[32m+[m[32m- FG%: Field goal percentage (percentage of successful field goals)[m[41m[m
[32m+[m[32m- 3P Made: Average number of three-pointers made per game[m[41m[m
[32m+[m[32m- 3PA: Average number of three-point attempts per game[m[41m[m
[32m+[m[32m- 3P%: Three-point percentage (percentage of successful three-point attempts)[m[41m[m
[32m+[m[32m- FTM: Average number of free throws made per game[m[41m[m
[32m+[m[32m- FTA: Average number of free throw attempts per game[m[41m[m
[32m+[m[32m- FT%: Free throw percentage (percentage of successful free throws)[m[41m[m
[32m+[m[32m- OREB: Average number of offensive rebounds per game[m[41m[m
[32m+[m[32m- DREB: Average number of defensive rebounds per game[m[41m[m
[32m+[m[32m- REB: Average number of total rebounds per game[m[41m[m
[32m+[m[32m- AST: Average number of assists per game[m[41m[m
[32m+[m[32m- STL: Average number of steals per game[m[41m[m
[32m+[m[32m- BLK: Average number of blocks per game[m[41m[m
[32m+[m[32m- TOV: Average number of turnovers per game[m[41m[m
[32m+[m[41m[m
[32m+[m[32m## Prediction[m[41m[m
[32m+[m[32mThe model predicts whether a player will continue their career successfully or not based on the input features. It provides a probability score along with the prediction.[m[41m[m
[32m+[m[41m[m
[32m+[m[32m- Successful Prediction: If the model predicts that the player will continue their career successfully, it means the player is likely to have a successful long-term career in the NBA.[m[41m[m
[32m+[m[32m- Unsuccessful Prediction: If the model predicts that the player will not continue their career successfully, it means the player's future performance may not meet the criteria for a successful long-term career in the NBA.[m[41m[m
[32m+[m[41m[m
[32m+[m[32m## API FLASK + STREAMLIT[m[41m[m
[32m+[m[32min the File notebook_boullam you can find :[m[41m [m
[32m+[m[41m[m
[32m+[m[32m### 1. Frontend[m[41m [m
[32m+[m[32mThe file `frontend.py` contains the Streamlit code for the graphical user interface.[m[41m[m
[32m+[m[41m[m
[32m+[m[32m### 2. Backend API[m[41m[m
[32m+[m[32mThe file `backend_api.py` contains the code for the Flask API, which returns predictions from the `best_model.h5` model.[m[41m[m
[32m+[m[41m[m
[32m+[m[32m## Setup[m[41m[m
[32m+[m[32m1. Install requirements: `pip install -r requirements.txt`[m[41m[m
[32m+[m[32m2. Turn Flask API on: `python3 backend_api.py`[m[41m[m
[32m+[m[32m3. Execute Streamlit: `streamlit run frontend.py`[m[41m[m
[32m+[m[32m4. Fill in variables, and click submit.[m[41m[m
[32m+[m[41m[m
[32m+[m[32m## Model and Scaler[m[41m[m
[32m+[m[32mThe files `best_model.pkl` and `scaler.pkl` contain the trained model and scaler, respectively, in pkl format.[m[41m[m
[32m+[m[41m[m
[32m+[m[32m## Credits[m[41m[m
[32m+[m[32mThis project was developed by Boullam Noureddine.[m
\ No newline at end of file[m
[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/README.md b/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/README.md[m
[1mnew file mode 100644[m
[1mindex 0000000..5bf153b[m
[1m--- /dev/null[m
[1m+++ b/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/README.md[m	
[36m@@ -0,0 +1,7 @@[m
[32m+[m[32m# API FLASK + STREAMLIT[m
[32m+[m
[32m+[m[32mLe fichier frontend contient le code streamlit pour l'interface graphique[m
[32m+[m
[32m+[m[32mLe fichier backend_api contient le code de l'api Flask qui renvoit les prédictions du modèle `best_model.pkl`[m
[32m+[m
[32m+[m
[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/backend_api.py b/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/backend_api.py[m
[1mnew file mode 100644[m
[1mindex 0000000..ceca8a8[m
[1m--- /dev/null[m
[1m+++ b/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/backend_api.py[m	
[36m@@ -0,0 +1,59 @@[m
[32m+[m[32mfrom flask import Flask, request, jsonify[m
[32m+[m[32mimport joblib[m
[32m+[m[32mimport numpy as np[m
[32m+[m[32mimport pandas as pd[m
[32m+[m
[32m+[m[32mapp = Flask(__name__)[m
[32m+[m
[32m+[m[32m# Load the model[m
[32m+[m[32mmodel_path = "/Users/noure/OneDrive/Desktop/Boullam/notebook_boullam/best_modelNBA.pkl"[m[41m [m
[32m+[m[32mmodel = joblib.load(model_path)[m
[32m+[m
[32m+[m[32m# Load the RobustScaler[m
[32m+[m[32mscaler_path = "/Users/noure/OneDrive/Desktop/Boullam/notebook_boullam/scaler.pkl"[m
[32m+[m[32mscaler = joblib.load(scaler_path)[m
[32m+[m
[32m+[m[32mdef preprocess_data(data):[m
[32m+[m[32m    # Calculate additional features[m
[32m+[m[32m    data['3P%Q'] = data['3P Made'] / data['3PA'] * 100[m
[32m+[m[32m    data['PPM'] = data['PTS'] / data['MIN'][m
[32m+[m[32m    data['FGA/MIN'] = data['FGA'] / data['MIN'][m
[32m+[m[32m    data['FGM/MIN'] = data['FGM'] / data['MIN'][m
[32m+[m[32m    data['3PA/MIN'] = data['3PA'] / data['MIN'][m
[32m+[m[32m    data['3P Made/MIN'] = data['3P Made'] / data['MIN'][m
[32m+[m[32m    data['FTA/MIN'] = data['FTA'] / data['MIN'][m
[32m+[m[32m    data['FTM/MIN'] = data['FTM'] / data['MIN'][m
[32m+[m[32m    data['AST/MIN'] = data['AST'] / data['MIN'][m
[32m+[m[32m    data['REB/MIN'] = data['REB'] / data['MIN'][m
[32m+[m[32m    data['TOV_Rate'] = data['TOV'] / data['MIN'][m
[32m+[m[32m    data['FTM/FTA'] = data['FTM'] / data['FTA'][m
[32m+[m[32m    data['OREB/DREB'] = data['OREB'] / data['DREB'][m
[32m+[m[32m    data['PER'] = (data['PTS'] + data['REB'] + data['AST'] + data['STL'] + data['BLK'] - data['FGA'] - data['TOV']) / data['MIN'][m
[32m+[m[32m    data['Usage_Rate'] = (data['FGA'] + 0.44 * data['FTA'] + data['TOV']) / (data['MIN'] * data['GP'])[m
[32m+[m[32m    data['AST*PTS'] = data['AST'] * data['PTS'][m
[32m+[m[32m    data['Opponent REB'] = data['REB'] - data['DREB'][m
[32m+[m[32m    data['eFG%'] = (data['FGM'] + 0.5 * data['3P Made']) / data['FGA'][m
[32m+[m[32m    data['FTAR'] = data['FTA'] / data['FGA'][m
[32m+[m[32m    data['Reb%'] = 100 * ((data['REB'] * (data['MIN'] / 5)) / (data['MIN'] * (data['REB'] + data['Opponent REB'])))[m
[32m+[m[32m    data['Steal_Rate'] = data['STL'] / data['MIN'][m
[32m+[m[32m    data['Block_Rate'] = data['BLK'] / data['MIN'][m
[32m+[m
[32m+[m[32m    return data[m
[32m+[m
[32m+[m[32m@app.route('/predict', methods=['POST'])[m
[32m+[m[32mdef predict():[m
[32m+[m[32m    data = request.get_json(force=True)[m
[32m+[m[32m    df = pd.DataFrame(data, index=[0])[m
[32m+[m[32m    df = preprocess_data(df)[m
[32m+[m[32m    input_data = df[['GP','MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PPM', 'FGA/MIN', 'FGM/MIN','FTA/MIN', 'FTM/MIN', 'REB/MIN', 'PER', 'Usage_Rate', 'Opponent REB', 'eFG%', 'FTAR', 'Reb%', 'Block_Rate','3P%Q']][m
[32m+[m[32m    input_data = scaler.transform(input_data)[m
[32m+[m[32m    prediction = model.predict(input_data)[m
[32m+[m[32m    probability = model.predict_proba(input_data)[m
[32m+[m[41m    [m
[32m+[m[32m    return jsonify({[m
[32m+[m[32m        'prediction': int(prediction[0]),[m
[32m+[m[32m        'probability': probability.tolist()[m
[32m+[m[32m    })[m
[32m+[m
[32m+[m[32mif __name__ == '__main__':[m
[32m+[m[32m    app.run(debug=True)[m
[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/frontend.py b/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/frontend.py[m
[1mnew file mode 100644[m
[1mindex 0000000..55547ec[m
[1m--- /dev/null[m
[1m+++ b/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/frontend.py[m	
[36m@@ -0,0 +1,54 @@[m
[32m+[m[32mimport streamlit as st[m
[32m+[m[32mimport requests[m
[32m+[m[32mimport json[m
[32m+[m
[32m+[m[32m# Page Title[m
[32m+[m[32mst.title("Prédiction de Joueur de Basketball")[m
[32m+[m
[32m+[m[32m# Image of the NBA[m
[32m+[m[32mst.image("logo-NBA.jpg", caption="NBA", use_column_width=True)[m
[32m+[m
[32m+[m[32m# Sidebar Title[m
[32m+[m[32mst.sidebar.title("Entrez les données du joueur")[m
[32m+[m
[32m+[m[32m# Define the input features with their descriptions[m
[32m+[m[32mFEATURES = {[m
[32m+[m[32m    'GP': 'Nombre de matchs joués',[m
[32m+[m[32m    'MIN': 'Minutes jouées par match',[m
[32m+[m[32m    'PTS': 'Points marqués par match',[m
[32m+[m[32m    'FGM': 'Tirs réussis par match',[m
[32m+[m[32m    'FGA': 'Tentatives de tir par match',[m
[32m+[m[32m    'FG%': 'Pourcentage de réussite des tirs',[m
[32m+[m[32m    '3P Made': 'Paniers à 3 points réussis par match',[m
[32m+[m[32m    '3PA': 'Tentatives de paniers à 3 points par match',[m
[32m+[m[32m    '3P%': 'Pourcentage de réussite des paniers à 3 points',[m
[32m+[m[32m    'FTM': 'Lancers francs réussis par match',[m
[32m+[m[32m    'FTA': 'Tentatives de lancers francs par match',[m
[32m+[m[32m    'FT%': 'Pourcentage de réussite des lancers francs',[m
[32m+[m[32m    'OREB': 'Rebonds offensifs par match',[m
[32m+[m[32m    'DREB': 'Rebonds défensifs par match',[m
[32m+[m[32m    'REB': 'Total des rebonds par match',[m
[32m+[m[32m    'AST': 'Passes décisives par match',[m
[32m+[m[32m    'STL': 'Interceptions par match',[m
[32m+[m[32m    'BLK': 'Contres par match',[m
[32m+[m[32m    'TOV': 'Balles perdues par match'[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32m# Sidebar Inputs with tooltips[m
[32m+[m[32minput_data = {}[m
[32m+[m[32mfor feature, description in FEATURES.items():[m
[32m+[m[32m    input_data[feature] = st.sidebar.number_input(f"{feature} ({description})", value=0.0)[m
[32m+[m
[32m+[m[32m# Predict Button[m
[32m+[m[32mif st.sidebar.button("Prédire"):[m
[32m+[m
[32m+[m[32m    response = requests.post("http://127.0.0.1:5000/predict", json=input_data)[m
[32m+[m[41m    [m
[32m+[m[32m    if response.status_code == 200:[m
[32m+[m[32m        result = response.json()[m
[32m+[m[32m        prediction_message = "Joueur de basket à fort potentiel pour investissement à long terme. 🏀💰" if result['prediction'] == 1 else "Joueur de basket peu prometteur pour investissement à long terme. 🏀📉"[m
[32m+[m[32m        st.success(f"Prédiction: {prediction_message}")[m
[32m+[m[32m        st.info(f"Probabilité: {result['probability'][0][1]*100:.2f}%")[m
[32m+[m[32m    else:[m
[32m+[m
[32m+[m[32m        st.error("Erreur lors de la prédiction")[m
[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/logo-NBA.jpg b/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/logo-NBA.jpg[m
[1mnew file mode 100644[m
[1mindex 0000000..c12e983[m
Binary files /dev/null and b/Boullam_API_FLASK _STREAMLIT/flask_streamlit-main/logo-NBA.jpg differ
[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/notebook_boullam/best_modelNBA.pkl b/Boullam_API_FLASK _STREAMLIT/notebook_boullam/best_modelNBA.pkl[m
[1mnew file mode 100644[m
[1mindex 0000000..24fe301[m
Binary files /dev/null and b/Boullam_API_FLASK _STREAMLIT/notebook_boullam/best_modelNBA.pkl differ
[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/notebook_boullam/scaler.pkl b/Boullam_API_FLASK _STREAMLIT/notebook_boullam/scaler.pkl[m
[1mnew file mode 100644[m
[1mindex 0000000..1ad90bb[m
Binary files /dev/null and b/Boullam_API_FLASK _STREAMLIT/notebook_boullam/scaler.pkl differ
[1mdiff --git a/Boullam_API_FLASK _STREAMLIT/requirements.txt b/Boullam_API_FLASK _STREAMLIT/requirements.txt[m
[1mnew file mode 100644[m
[1mindex 0000000..19d904f[m
[1m--- /dev/null[m
[1m+++ b/Boullam_API_FLASK _STREAMLIT/requirements.txt[m	
[36m@@ -0,0 +1,95 @@[m
[32m+[m[32mabsl-py==2.1.0[m
[32m+[m[32maltair==5.3.0[m
[32m+[m[32mannotated-types==0.6.0[m
[32m+[m[32manyio==4.3.0[m
[32m+[m[32mastunparse==1.6.3[m
[32m+[m[32mattrs==23.2.0[m
[32m+[m[32mawscli==1.32.69[m
[32m+[m[32mblinker==1.8.2[m
[32m+[m[32mbotocore==1.34.69[m
[32m+[m[32mcachetools==5.3.3[m
[32m+[m[32mcertifi==2024.2.2[m
[32m+[m[32mcharset-normalizer==3.3.2[m
[32m+[m[32mclick==8.1.7[m
[32m+[m[32mcolorama==0.4.4[m
[32m+[m[32mcontourpy==1.2.1[m
[32m+[m[32mcycler==0.12.1[m
[32m+[m[32mdistro==1.9.0[m
[32m+[m[32mdocutils==0.16[m
[32m+[m[32mFlask==3.0.3[m
[32m+[m[32mflatbuffers==24.3.25[m
[32m+[m[32mfonttools==4.53.0[m
[32m+[m[32mgast==0.5.4[m
[32m+[m[32mgitdb==4.0.11[m
[32m+[m[32mGitPython==3.1.43[m
[32m+[m[32mgoogle-pasta==0.2.0[m
[32m+[m[32mgrpcio==1.64.0[m
[32m+[m[32mh11==0.14.0[m
[32m+[m[32mh5py==3.11.0[m
[32m+[m[32mhttpcore==1.0.5[m
[32m+[m[32mhttpx==0.27.0[m
[32m+[m[32midna==3.6[m
[32m+[m[32mitsdangerous==2.2.0[m
[32m+[m[32mJinja2==3.1.4[m
[32m+[m[32mjmespath==1.0.1[m
[32m+[m[32mjoblib==1.4.2[m
[32m+[m[32mjsonschema==4.22.0[m
[32m+[m[32mjsonschema-specifications==2023.12.1[m
[32m+[m[32mkeras==3.3.3[m
[32m+[m[32mkiwisolver==1.4.5[m
[32m+[m[32mlibclang==18.1.1[m
[32m+[m[32mMarkdown==3.6[m
[32m+[m[32mmarkdown-it-py==3.0.0[m
[32m+[m[32mMarkupSafe==2.1.5[m
[32m+[m[32mmatplotlib==3.9.0[m
[32m+[m[32mmdurl==0.1.2[m
[32m+[m[32mml-dtypes==0.3.2[m
[32m+[m[32mnamex==0.0.8[m
[32m+[m[32mnumpy==1.26.4[m
[32m+[m[32mopenai==1.12.0[m
[32m+[m[32mopt-einsum==3.3.0[m
[32m+[m[32moptree==0.11.0[m
[32m+[m[32mpackaging==24.0[m
[32m+[m[32mpandas==2.2.2[m
[32m+[m[32mpillow==10.3.0[m
[32m+[m[32mprotobuf==4.25.3[m
[32m+[m[32mpyarrow==16.1.0[m
[32m+[m[32mpyasn1==0.5.1[m
[32m+[m[32mpydantic==2.6.4[m
[32m+[m[32mpydantic_core==2.16.3[m
[32m+[m[32mpydeck==0.9.1[m
[32m+[m[32mpydub==0.25.1[m
[32m+[m[32mPygments==2.18.0[m
[32m+[m[32mpyparsing==3.1.2[m
[32m+[m[32mpython-dateutil==2.9.0.post0[m
[32m+[m[32mpytz==2024.1[m
[32m+[m[32mPyYAML==6.0.1[m
[32m+[m[32mreferencing==0.35.1[m
[32m+[m[32mrequests==2.32.3[m
[32m+[m[32mrich==13.7.1[m
[32m+[m[32mrpds-py==0.18.1[m
[32m+[m[32mrsa==4.7.2[m
[32m+[m[32ms3transfer==0.10.1[m
[32m+[m[32mscikit-learn==1.5.0[m
[32m+[m[32mscipy==1.13.1[m
[32m+[m[32msetuptools==70.0.0[m
[32m+[m[32msix==1.16.0[m
[32m+[m[32msmmap==5.0.1[m
[32m+[m[32msniffio==1.3.1[m
[32m+[m[32mstreamlit==1.35.0[m
[32m+[m[32mtenacity==8.3.0[m
[32m+[m[32mtensorboard==2.16.2[m
[32m+[m[32mtensorboard-data-server==0.7.2[m
[32m+[m[32mtensorflow==2.16.1[m
[32m+[m[32mtermcolor==2.4.0[m
[32m+[m[32mthreadpoolctl==3.5.0[m
[32m+[m[32mtoml==0.10.2[m
[32m+[m[32mtoolz==0.12.1[m
[32m+[m[32mtornado==6.4[m
[32m+[m[32mtqdm==4.66.2[m
[32m+[m[32mtyping_extensions==4.10.0[m
[32m+[m[32mtzdata==2024.1[m
[32m+[m[32murllib3==2.2.1[m
[32m+[m[32mWerkzeug==3.0.3[m
[32m+[m[32mwheel==0.43.0[m
[32m+[m[32mwrapt==1.16.0[m
