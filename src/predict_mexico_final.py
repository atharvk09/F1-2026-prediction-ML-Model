import fastf1
import pandas as pd
import numpy as np
from src.data_loader import F1DataLoader
from src.features import F1FeatureEngineer
from src.model import F1Predictor

fastf1.Cache.enable_cache('cache')

def predict_mexico():
    print("Step 1: Fetching Mexico 2026 Qualifying Grid...")
    try:
        # Use the correct event name
        session = fastf1.get_session(2026, 'Mexico City Grand Prix', 'Q')
        session.load()
        grid = session.results[['FullName', 'TeamName', 'GridPosition', 'DriverId']].copy()
    except Exception as e:
        print(f"Error fetching FastF1 data: {e}")
        print("\nIt seems the 2026 Mexico Qualifying data is not yet available in FastF1.")
        print("To predict this, please provide the starting grid (Driver & Position).")
        return

    print("Step 2: Calculating Form and Training Model...")
    loader = F1DataLoader()
    engineer = F1FeatureEngineer()
    predictor = F1Predictor()
    
    data = loader.load_csv_data()
    # Use historical data + 2026 data up to now
    all_available = pd.concat([data['hist_race'], data['race_2026']])
    
    # Training setup (Approximate Model C logic)
    weights = {2022: 0.5, 2023: 0.75, 2024: 1.0, 2025: 1.5}
    train_data = []
    for year, w in weights.items():
        year_df = data['hist_race'][data['hist_race']['Year'] == year].copy()
        year_df['weight'] = w
        train_data.append(year_df)
    train_df = pd.concat(train_data)
    train_df = pd.concat([train_df, data['race_2026']])
    
    # Train features
    train_df['WeightedForm'] = train_df['DriverId'].apply(lambda x: engineer.calculate_weighted_form(x, all_available))
    train_df = train_df.merge(data['hist_qual'][['DriverId', 'Year', 'Round', 'QualiGapPct']], 
                             on=['DriverId', 'Year', 'Round'], how='left')
    
    features = ['GridPosition', 'QualiGapPct', 'WeightedForm']
    X_train = train_df[features].fillna(train_df[features].median())
    y_train = (train_df['Position'] == 1).astype(int)
    
    predictor.train(X_train, y_train)
    
    # Predict for Mexico
    grid['WeightedForm'] = grid['DriverId'].apply(lambda x: engineer.calculate_weighted_form(x, all_available))
    grid['QualiGapPct'] = 0.0 # Baseline gap
    
    X_mexico = grid[features].fillna(grid[features].median())
    probs = predictor.predict_probabilities(X_mexico)
    grid['WinProbability'] = probs
    
    predictions = grid[['FullName', 'TeamName', 'GridPosition', 'WinProbability']].sort_values('WinProbability', ascending=False)
    
    print("\n--- 🔮 MODEL C PREDICTION: MEXICO GP 2026 ---")
    print(predictions.head(5).to_string(index=False))
    print(f"\nPredicted Winner: {predictions.iloc[0]['FullName']}")

if __name__ == "__main__":
    predict_mexico()
