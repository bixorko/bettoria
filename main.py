import argparse
import requests
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight

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
    df['Outcome'] = df['Outcome'].map({-1: 0, 0: 1, 1: 2})
    return df

def prepare_features(df, encoder=None, is_training=True):
    # One-hot encode the teams
    team_names = df[['Home Team', 'Away Team']]
    if is_training:
        encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoded_teams = encoder.fit_transform(team_names).toarray()
    else:
        one_hot_encoded_teams = encoder.transform(team_names).toarray()
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_teams, columns=encoder.get_feature_names_out(['Home Team', 'Away Team']))

    # Include other features
    other_features = df[['Half Time Winner', 'Half Time Total Goals']].reset_index(drop=True)

    # Combine
    X = pd.concat([one_hot_encoded_df.reset_index(drop=True), other_features], axis=1)

    return X, encoder

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    return joblib.load(filename)

def hyperparameter_tuning_xgboost(X, y, task_type='classification'):
    if task_type == 'classification':
        model = xgb.XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1],
        }

        def f1_draw(y_true, y_pred):
            return f1_score(y_true, y_pred, labels=[1], average='micro')

        scorer = make_scorer(f1_draw)

        # Compute sample weights to handle class imbalance
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer)
        grid_search.fit(X, y, sample_weight=sample_weights)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        # Regression code (if needed)
        pass

def backtest_daily_for_league(league_id, league_name, season_base='2023'):
    # Fetch and prepare the base training data from 2023
    matches = fetch_match_data(league_id, season=season_base)
    df = prepare_data(matches)
    df['utcDate'] = pd.to_datetime(df['utcDate']).dt.date  # Convert to date only
    X_base, encoder = prepare_features(df)
    y_outcome_base = df['Outcome']

    # Initial training with 2023 data
    model = hyperparameter_tuning_xgboost(X_base, y_outcome_base, task_type='classification')
    save_model(model, f'xgb_outcome_base_{league_name}.pkl')

    # Fetch and prepare matches for 2024 season
    matches_2024 = fetch_match_data(league_id, season='2023')
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

            X_day, _ = prepare_features(daily_df, encoder=encoder, is_training=False)
            y_day_outcome = daily_df['Outcome']

            # Combine 2023 data + previous days in 2024 as training data
            if i == 0:
                all_train_data = df
            else:
                all_train_data = pd.concat([df, df_2024[df_2024['utcDate'] <= unique_dates[i-1]]], ignore_index=True)
            X_all_train, _ = prepare_features(all_train_data, encoder=encoder, is_training=False)
            y_all_train_outcome = all_train_data['Outcome']

            # Retrain model with all available data
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_all_train_outcome)
            model.fit(X_all_train, y_all_train_outcome, sample_weight=sample_weights)
            predictions_proba = model.predict_proba(X_day)
            draw_probs = predictions_proba[:, 1]

            # Prepare daily results
            daily_results = pd.DataFrame({
                'Home Team': daily_df['Home Team'],
                'Away Team': daily_df['Away Team'],
                'Actual Outcome': y_day_outcome,
                'Predicted Draw Probability': draw_probs
            })

            # Betting logic
            draw_threshold = 0.4  # You may adjust this threshold
            daily_results['Bet on Draw'] = (draw_probs >= draw_threshold).astype(int)
            daily_results['Actual Draw'] = (y_day_outcome == 1).astype(int)

            daily_results['Betting Result'] = daily_results.apply(lambda row: 
                2.2 if row['Bet on Draw'] == 1 and row['Actual Draw'] == 1 else
                -1 if row['Bet on Draw'] == 1 and row['Actual Draw'] == 0 else
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
        backtest_daily_for_league(league_id, league_name, season_base='2022')
