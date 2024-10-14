import argparse
import requests
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

API_KEY = '57f3688019a0433fb6c52744fafee577' 
HEADERS = {'X-Auth-Token': API_KEY}

LEAGUES = {
    'La_Liga': 'PD',
    'Premier_League': 'PL',
    'Serie_A': 'SA',
}

def fetch_match_data(league_id, season):
    url = f'https://api.football-data.org/v4/competitions/{league_id}/matches?season={season}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")
    return response.json()['matches']

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
                'utcDate': match['utcDate']
            })

    df = pd.DataFrame(data)
    df['Half Time Winner'] = df['Half Time Winner'].map({-1: 0, 0: 1, 1: 2})
    return df

def one_hot_encode_teams(df):
    encoder = OneHotEncoder(handle_unknown='ignore')
    team_names = df[['Home Team', 'Away Team']]
    one_hot_encoded_teams = encoder.fit_transform(team_names).toarray()
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_teams, columns=encoder.get_feature_names_out(['Home Team', 'Away Team']))
    return one_hot_encoded_df, encoder

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    return joblib.load(filename)

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

def backtest_daily_for_league(league_id, league_name, season_base='2023'):
    # Fetch and prepare the base training data from 2023
    matches = fetch_match_data(league_id, season=season_base)
    df = prepare_data(matches)
    df['utcDate'] = pd.to_datetime(df['utcDate']).dt.date  # Convert to date only
    X_base, encoder = one_hot_encode_teams(df)
    y_outcome_base = df['Outcome']
    
    # Initial training with 2023 data
    model = hyperparameter_tuning_xgboost(X_base, y_outcome_base, task_type='regression')
    save_model(model, f'xgb_outcome_base_{league_name}.pkl')
    
    # Fetch and prepare matches for 2024 season
    matches_2024 = fetch_match_data(league_id, season='2024')
    df_2024 = prepare_data(matches_2024)
    df_2024['utcDate'] = pd.to_datetime(df_2024['utcDate']).dt.date  # Convert to date only
    total_units = 0
    
    # Filter daily and backtest day by day
    with pd.ExcelWriter(f'{league_name}_daily_predictions.xlsx', engine='openpyxl') as writer:
        unique_dates = sorted(df_2024['utcDate'].unique())
        
        for i, current_date in enumerate(unique_dates):
            daily_df = df_2024[df_2024['utcDate'] == current_date]
            if daily_df.empty:
                print(f"No matches found for {league_name} on {current_date}")
                continue

            X_day = encoder.transform(daily_df[['Home Team', 'Away Team']]).toarray()
            y_day_outcome = daily_df['Outcome']
            
            # Combine 2023 data + previous days in 2024 as training data
            all_train_data = pd.concat([df] + [df_2024[df_2024['utcDate'] <= unique_dates[i-1]]], ignore_index=True)
            X_all_train = encoder.transform(all_train_data[['Home Team', 'Away Team']]).toarray()
            y_all_train_outcome = all_train_data['Outcome']
            
            # Retrain model with all available data
            model.fit(X_all_train, y_all_train_outcome)
            predictions = model.predict(X_day)
            
            # Prepare daily results
            daily_results = pd.DataFrame({
                'Home Team': daily_df['Home Team'],
                'Away Team': daily_df['Away Team'],
                'Actual Outcome': y_day_outcome,
                'Predicted Outcome': predictions
            })
            
            daily_results['Betting Result'] = daily_results.apply(lambda row: 
                2.2 if abs(row['Predicted Outcome']) < 0.2 and row['Actual Outcome'] == 0 else
                -1 if abs(row['Predicted Outcome']) < 0.2 and row['Actual Outcome'] != 0 else
                0, axis=1)
            
            total_units += daily_results['Betting Result'].sum()
            daily_results['Cumulative Units'] = total_units
            
            # Format date for sheet name and save to Excel
            sheet_name = current_date.strftime('%Y-%m-%d')
            daily_results.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"{league_name} predictions for {sheet_name} added to Excel file")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest football predictions by league.")
    parser.add_argument('--leagues', nargs='+', choices=LEAGUES.keys(), required=True, help="Leagues to backtest (La_Liga, Premier_League, Serie_A)")
    args = parser.parse_args()
    
    for league_name in args.leagues:
        league_id = LEAGUES[league_name]
        backtest_daily_for_league(league_id, league_name, season_base='2023')
