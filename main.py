import requests
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

API_KEY = 'INSERT_YOUR_API_KEY' 
HEADERS = {'X-Auth-Token': API_KEY}

weekly_ranges = [
    ('2024-08-15', '2024-08-19'),  
    ('2024-08-23', '2024-08-25'),  
    ('2024-08-26', '2024-08-29'),  
    ('2024-08-31', '2024-09-01'),  
    ('2024-09-13', '2024-09-16'),  
    ('2024-09-20', '2024-09-23'),  
    ('2024-09-24', '2024-09-26'),  
    ('2024-09-27', '2024-09-30'),  
    ('2024-10-04', '2024-10-06')   
]

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

def filter_matches_by_date(df, start_date, end_date):
    df['Match Date'] = pd.to_datetime(df['utcDate']).dt.tz_convert('UTC')
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    return df[(df['Match Date'] >= start_date) & (df['Match Date'] <= end_date)]

def backtest_weekly_for_league(league_id, league_name, season_base='2023'):
    matches = fetch_match_data(league_id, season=season_base)
    df = prepare_data(matches)
    X_base, encoder = one_hot_encode_teams(df)
    y_outcome_base = df['Outcome']
    
    model = hyperparameter_tuning_xgboost(X_base, y_outcome_base, task_type='regression')
    save_model(model, f'xgb_outcome_base_{league_name}.pkl')
    
    matches_2024 = fetch_match_data(league_id, season='2024')
    df_2024 = prepare_data(matches_2024)
    total_units = 0

    for i, (start_date, end_date) in enumerate(weekly_ranges):
        weekly_df = filter_matches_by_date(df_2024, start_date, end_date)
        if weekly_df.empty:
            print(f"No matches found for {league_name} week {i+1} ({start_date} to {end_date})")
            continue
        
        X_week = encoder.transform(weekly_df[['Home Team', 'Away Team']]).toarray()
        y_week_outcome = weekly_df['Outcome']
        
        all_train_data = pd.concat([df] + [filter_matches_by_date(df_2024, weekly_ranges[j][0], weekly_ranges[j][1]) for j in range(i)], ignore_index=True)
        X_all_train = encoder.transform(all_train_data[['Home Team', 'Away Team']]).toarray()
        y_all_train_outcome = all_train_data['Outcome']
        
        model.fit(X_all_train, y_all_train_outcome)
        predictions = model.predict(X_week)
        
        weekly_results = pd.DataFrame({
            'Home Team': weekly_df['Home Team'],
            'Away Team': weekly_df['Away Team'],
            'Actual Outcome': y_week_outcome,
            'Predicted Outcome': predictions
        })
        
        weekly_results['Betting Result'] = weekly_results.apply(lambda row: 
            2.2 if abs(row['Predicted Outcome']) < 0.2 and row['Actual Outcome'] == 0 else
            -1 if abs(row['Predicted Outcome']) < 0.2 and row['Actual Outcome'] != 0 else
            0, axis=1)
        
        total_units += weekly_results['Betting Result'].sum()
        weekly_results['Cumulative Units'] = total_units
        
        weekly_results.to_csv(f'{league_name}_predictions_week_{i+1}.csv', index=False)
        print(f"{league_name} Week {i+1} predictions saved to {league_name}_predictions_week_{i+1}.csv")

if __name__ == '__main__':
    backtest_weekly_for_league('PD', 'La_Liga', season_base='2023')
    backtest_weekly_for_league('PL', 'Premier_League', season_base='2023')
