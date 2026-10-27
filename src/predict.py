import pandas as pd
import numpy as np
from src.data_loader import F1DataLoader
from src.features import F1FeatureEngineer
from src.model import F1Predictor

def predict_race(year, event):
    print(f"--- Predicting Winner for {year} {event} ---")
    
    # 1. Initialize modules
    loader = F1DataLoader()
    engineer = F1FeatureEngineer()
    predictor = F1Predictor()
    
    # 2. Load Data
    data = loader.load_csv_data()
    hist_race = data['hist_race']
    hist_qual = data['hist_qual']
    race_2026 = data['race_2026']
    qual_2026 = data['qual_2026']
    
    # 3. Train the model adaptively for the specific race
    # We use the Adaptive Loop logic here
    weights = {2022: 0.5, 2023: 0.75, 2024: 1.0, 2025: 1.5}
    
    # Build Training Set
    train_data = []
    for year_val, w in weights.items():
        year_df = hist_race[hist_race['Year'] == year_val].copy()
        year_df['weight'] = w
        train_data.append(year_df)
    train_df = pd.concat(train_data)
    
    # Add current season races before this specific race
    # First, we need the round number for the target event
    target_race = race_2026[race_2026['EventName'] == event]
    if target_race.empty:
        print(f"Error: {event} not found in 2026 data.")
        return
    
    round_num = target_race['Round'].iloc[0]
    current_season_prev = race_2026[race_2026['Round'] < round_num].copy()
    current_season_prev['weight'] = 2.0
    train_df = pd.concat([train_df, current_season_prev])
    
    # Prepare training features
    # For training, we'll calculate weighted form using all available data
    all_available = pd.concat([hist_race, race_2026[race_2026['Round'] < round_num]])
    train_df['WeightedForm'] = train_df['DriverId'].apply(lambda x: engineer.calculate_weighted_form(x, all_available))
    
    # We need QualiGapPct for training. Merge with historical qual.
    train_df = train_df.merge(hist_qual[['DriverId', 'Year', 'Round', 'QualiGapPct']], 
                             on=['DriverId', 'Year', 'Round'], how='left')
    
    features = ['GridPosition', 'QualiGapPct', 'WeightedForm']
    X_train = train_df[features].fillna(train_df[features].median())
    y_train = (train_df['Position'] == 1).astype(int)
    
    predictor.train(X_train, y_train)
    
    # 4. Prepare Prediction Set for the target race
    test_group = race_2026[race_2026['Round'] == round_num].copy()
    test_group = test_group.merge(qual_2026[['DriverId', 'Round', 'QualiGapPct']], on=['DriverId', 'Round'], how='left')
    test_group['WeightedForm'] = test_group['DriverId'].apply(lambda x: engineer.calculate_weighted_form(x, all_available))
    
    X_test = test_group[features].fillna(test_group[features].median())
    
    # 5. Generate Probabilities
    probs = predictor.predict_probabilities(X_test)
    test_group['WinProbability'] = probs
    
    # Rank
    predictions = test_group[['FullName', 'TeamName', 'GridPosition', 'WinProbability']].sort_values('WinProbability', ascending=False)
    
    print("\n🏆 TOP 5 PREDICTIONS:")
    print(predictions.head(5).to_string(index=False))
    
    actual_winner = test_group[test_group['Position'] == 1]['FullName'].values[0]
    predicted_winner = predictions.iloc[0]['FullName']
    
    print(f"\nPredicted Winner: {predicted_winner}")
    print(f"Actual Winner: {actual_winner}")
    
    if predicted_winner == actual_winner:
        print("\n✅ SUCCESS: The model predicted the winner correctly!")
    else:
        print("\n❌ FAILURE: The model missed the winner.")

if __name__ == "__main__":
    # Example: Predict Austin 2026
    predict_race(2026, 'United States Grand Prix')
