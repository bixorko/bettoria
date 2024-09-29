import requests
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder

API_KEY = 'INSERT_YOUR_API_KEY' 
HEADERS = {'X-Auth-Token': API_KEY}

# Fetch La Liga matches data
def fetch_match_data(league_id='PD', season='2023'):
    url = f'https://api.football-data.org/v4/competitions/{league_id}/matches?season={season}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")
    return response.json()['matches']

# Prepare the fetched data into a DataFrame
def prepare_data(matches):
    data = []
    for match in matches:
        if match['status'] == 'FINISHED':
            full_time = match.get('score', {}).get('fullTime', {})
            half_time = match.get('score', {}).get('halfTime', {})

            home_team_goals = full_time.get('home', 0)
            away_team_goals = full_time.get('away', 0)
            ht_home_team_goals = half_time.get('home', 0)
            ht_away_team_goals = half_time.get('away', 0)

            # Outcome scaled to [-1, 1]
            outcome = 1 if match['score']['winner'] == 'HOME_TEAM' else -1 if match['score']['winner'] == 'AWAY_TEAM' else 0
            ht_winner = 1 if ht_home_team_goals > ht_away_team_goals else -1 if ht_home_team_goals < ht_away_team_goals else 0

            data.append({
                'Home Team': match['homeTeam']['name'],
                'Away Team': match['awayTeam']['name'],
                'Full Time Home Goals': home_team_goals,
                'Full Time Away Goals': away_team_goals,
                'Half Time Home Goals': ht_home_team_goals,
                'Half Time Away Goals': ht_away_team_goals,
                'Outcome': outcome,
                'Half Time Winner': ht_winner,
                'Total Goals': home_team_goals + away_team_goals,
                'Half Time Total Goals': ht_home_team_goals + ht_away_team_goals,
            })

    df = pd.DataFrame(data)
    
    # Adjust Half Time Winner class labels to [0, 1, 2] instead of [-1, 0, 1]
    df['Half Time Winner'] = df['Half Time Winner'].map({-1: 0, 0: 1, 1: 2})

    return df

# Perform one-hot encoding on team names
def one_hot_encode_teams(df):
    encoder = OneHotEncoder(handle_unknown='ignore')  # Set handle_unknown to ignore new categories
    team_names = df[['Home Team', 'Away Team']]

    # Fit and transform the team names to one-hot encoding
    one_hot_encoded_teams = encoder.fit_transform(team_names).toarray()

    # Create a DataFrame from the one-hot encoded array
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_teams, columns=encoder.get_feature_names_out(['Home Team', 'Away Team']))

    return one_hot_encoded_df, encoder

# Save the model to a file
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Load the model from a file
def load_model(filename):
    return joblib.load(filename)

# Perform hyperparameter tuning with cross-validation
def hyperparameter_tuning_xgboost(X, y, task_type='regression'):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42) if task_type == 'regression' else xgb.XGBClassifier(random_state=42)
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy')
    grid_search.fit(X, y)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Evaluate models using k-fold cross-validation and print actual vs. predicted
def evaluate_model(model, X, y, task_type='regression'):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    actuals = []
    predictions = []

    for train_idx, test_idx in kfold.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])

        # Store actual and predicted values for comparison
        actuals.extend(y.iloc[test_idx])
        predictions.extend(preds)

        if task_type == 'regression':
            scores.append(mean_squared_error(y.iloc[test_idx], preds))
        else:
            scores.append(accuracy_score(y.iloc[test_idx], preds))

    print(f"\nCross-validated {'MSE' if task_type == 'regression' else 'accuracy'}: {sum(scores) / len(scores)}")

def main():
    # Fetch and prepare La Liga match data for 2023
    matches = fetch_match_data()
    df = prepare_data(matches)

    # One-Hot Encode Team Names
    X, encoder = one_hot_encode_teams(df)

    # Save the encoder
    joblib.dump(encoder, 'team_name_encoder.pkl')

    # Target variables for different tasks
    y_outcome = df['Outcome']  # Regression task (-1 to 1)
    y_ht_winner = df['Half Time Winner']  # Classification task (remapped to 0, 1, 2)
    y_total_goals = df['Total Goals']  # Regression task (continuous total goals)

    # Train XGBoost for match outcome (regression) and save the model
    print("Tuning hyperparameters for match outcome prediction...")
    xgb_outcome = hyperparameter_tuning_xgboost(X, y_outcome, task_type='regression')
    save_model(xgb_outcome, "xgb_outcome_2023.pkl")

    print("Evaluating match outcome model...")
    evaluate_model(xgb_outcome, X, y_outcome, task_type='regression')

    # Train XGBoost for half-time winner (classification) and save the model
    print("Tuning hyperparameters for half-time winner prediction...")
    xgb_ht_winner = hyperparameter_tuning_xgboost(X, y_ht_winner, task_type='classification')
    save_model(xgb_ht_winner, "xgb_ht_winner_2023.pkl")

    print("Evaluating half-time winner model...")
    evaluate_model(xgb_ht_winner, X, y_ht_winner, task_type='classification')

    # Train XGBoost for total goals prediction (regression) and save the model
    print("Tuning hyperparameters for total goals prediction...")
    xgb_total_goals = hyperparameter_tuning_xgboost(X, y_total_goals, task_type='regression')
    save_model(xgb_total_goals, "xgb_total_goals_2023.pkl")

    print("Evaluating total goals model...")
    evaluate_model(xgb_total_goals, X, y_total_goals, task_type='regression')

# Backtest on the 2024 season and save predictions as CSV
def backtest_on_2024_season():
    # Fetch 2024 season data
    matches_2024 = fetch_match_data(season='2024')
    df_2024 = prepare_data(matches_2024)

    # Load the encoder for team names
    encoder = joblib.load('team_name_encoder.pkl')

    # Apply the encoder to the 2024 season data
    X_2024 = encoder.transform(df_2024[['Home Team', 'Away Team']]).toarray()

    # Target variables for backtesting
    y_outcome_2024 = df_2024['Outcome']

    # Load trained model for match outcome from 2023
    xgb_outcome_2023 = load_model("xgb_outcome_2023.pkl")

    # Predict match outcomes for 2024 season
    predictions = xgb_outcome_2023.predict(X_2024)

    # Create a DataFrame with actual outcomes, predicted outcomes, and team names
    results = pd.DataFrame({
        'Home Team': df_2024['Home Team'],
        'Away Team': df_2024['Away Team'],
        'Actual Outcome': y_outcome_2024,
        'Predicted Outcome': predictions
    })

    # Save the DataFrame to CSV
    results.to_csv('match_outcome_predictions_2024.csv', index=False)
    print("Predictions saved to match_outcome_predictions_2024.csv")

if __name__ == '__main__':
    main()
    backtest_on_2024_season()
