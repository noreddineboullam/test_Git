# NBA Player Prediction

## Overview
This project aims to predict the future performance of NBA players based on various statistical features. It utilizes a machine learning model trained on historical NBA player data to make predictions about whether a player will continue their career successfully or not.

## Features
The following features are used as inputs to the prediction model:
- GP: Number of games played by the player
- MIN: Average number of minutes played per game
- PTS: Average number of points scored per game
- FGM: Average number of field goals made per game
- FGA: Average number of field goal attempts per game
- FG%: Field goal percentage (percentage of successful field goals)
- 3P Made: Average number of three-pointers made per game
- 3PA: Average number of three-point attempts per game
- 3P%: Three-point percentage (percentage of successful three-point attempts)
- FTM: Average number of free throws made per game
- FTA: Average number of free throw attempts per game
- FT%: Free throw percentage (percentage of successful free throws)
- OREB: Average number of offensive rebounds per game
- DREB: Average number of defensive rebounds per game
- REB: Average number of total rebounds per game
- AST: Average number of assists per game
- STL: Average number of steals per game
- BLK: Average number of blocks per game
- TOV: Average number of turnovers per game

## Prediction
The model predicts whether a player will continue their career successfully or not based on the input features. It provides a probability score along with the prediction.

- Successful Prediction: If the model predicts that the player will continue their career successfully, it means the player is likely to have a successful long-term career in the NBA.
- Unsuccessful Prediction: If the model predicts that the player will not continue their career successfully, it means the player's future performance may not meet the criteria for a successful long-term career in the NBA.

## API FLASK + STREAMLIT
in the File notebook_boullam you can find : 

### 1. Frontend 
The file `frontend.py` contains the Streamlit code for the graphical user interface.

### 2. Backend API
The file `backend_api.py` contains the code for the Flask API, which returns predictions from the `best_model.h5` model.

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Turn Flask API on: `python3 backend_api.py`
3. Execute Streamlit: `streamlit run frontend.py`
4. Fill in variables, and click submit.

## Model and Scaler
The files `best_model.pkl` and `scaler.pkl` contain the trained model and scaler, respectively, in pkl format.

## Credits
This project was developed by Boullam Noureddine.