from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model_path = "/Users/noure/OneDrive/Desktop/Boullam/notebook_boullam/best_modelNBA.pkl" 
model = joblib.load(model_path)

# Load the RobustScaler
scaler_path = "/Users/noure/OneDrive/Desktop/Boullam/notebook_boullam/scaler.pkl"
scaler = joblib.load(scaler_path)

def preprocess_data(data):
    # Calculate additional features
    data['3P%Q'] = data['3P Made'] / data['3PA'] * 100
    data['PPM'] = data['PTS'] / data['MIN']
    data['FGA/MIN'] = data['FGA'] / data['MIN']
    data['FGM/MIN'] = data['FGM'] / data['MIN']
    data['3PA/MIN'] = data['3PA'] / data['MIN']
    data['3P Made/MIN'] = data['3P Made'] / data['MIN']
    data['FTA/MIN'] = data['FTA'] / data['MIN']
    data['FTM/MIN'] = data['FTM'] / data['MIN']
    data['AST/MIN'] = data['AST'] / data['MIN']
    data['REB/MIN'] = data['REB'] / data['MIN']
    data['TOV_Rate'] = data['TOV'] / data['MIN']
    data['FTM/FTA'] = data['FTM'] / data['FTA']
    data['OREB/DREB'] = data['OREB'] / data['DREB']
    data['PER'] = (data['PTS'] + data['REB'] + data['AST'] + data['STL'] + data['BLK'] - data['FGA'] - data['TOV']) / data['MIN']
    data['Usage_Rate'] = (data['FGA'] + 0.44 * data['FTA'] + data['TOV']) / (data['MIN'] * data['GP'])
    data['AST*PTS'] = data['AST'] * data['PTS']
    data['Opponent REB'] = data['REB'] - data['DREB']
    data['eFG%'] = (data['FGM'] + 0.5 * data['3P Made']) / data['FGA']
    data['FTAR'] = data['FTA'] / data['FGA']
    data['Reb%'] = 100 * ((data['REB'] * (data['MIN'] / 5)) / (data['MIN'] * (data['REB'] + data['Opponent REB'])))
    data['Steal_Rate'] = data['STL'] / data['MIN']
    data['Block_Rate'] = data['BLK'] / data['MIN']

    return data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])
    df = preprocess_data(df)
    input_data = df[['GP','MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PPM', 'FGA/MIN', 'FGM/MIN','FTA/MIN', 'FTM/MIN', 'REB/MIN', 'PER', 'Usage_Rate', 'Opponent REB', 'eFG%', 'FTAR', 'Reb%', 'Block_Rate','3P%Q']]
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': probability.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
