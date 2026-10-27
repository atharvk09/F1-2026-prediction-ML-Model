import fastf1
import pandas as pd
import numpy as np
from src.data_loader import F1DataLoader
from src.features import F1FeatureEngineer
from src.model import F1Predictor

# Setup Cache
fastf1.Cache.enable_cache('cache')

def predict_mexico():
    print("Fetching Mexico 2026 Qualifying/Grid data...")
    try:
        # Try to load the Qualifying session for Mexico 2026
        session = fastf1.get_session(2026, 'Mexico', 'Q')
        session.load()
        # The results from the Quali session give us the GridPosition
        results = session.results
    except Exception as e:
        print(f"Error fetching Mexico data: {e}")
        print("\nNote: If the race hasn't happened or Quali isn't uploaded yet, FastF1 might not have the data.")
        return

    loader = F1DataLoader()
    engineer = F1FeatureEngineer()
    predictor = F1Predictor()
    
    # 1. Load history for training and form calculation
    data = loader.load_csv_data()
    hist_race = data['hist_race']
    hist_qual = data['hist_qual']
    race_2026 = data['race_2026']
    
    # 2. Train the Model adaptively (up to the current date)
    weights = {2022: 0.5, 2023: 0.75, 2024: 1.0, 2025: 1.5}
    train_data = []
    for year, w in weights.items():
        year_df = hist_race[hist_race['Year'] == year].copy()
        year_df['weight'] = w
        train_data.append(year_df)
    train_df = pd.concat(train_data)
    train_df = pd.concat([train_df, race_2026]) # Add all 2026 races completed so far
    
    # Prepare training features
    all_hist = pd.concat([hist_race, race_2026])
    train_df['WeightedForm'] = train_df['DriverId'].apply(lambda x: engineer.calculate_weighted_form(x, all_hist))
    train_df = train_df.merge(hist_qual[['DriverId', 'Year', 'Round', 'QualiGapPct']], on=['DriverId', 'Year', 'Round'], how='left')
    
    features = ['GridPosition', 'QualiGapPct', 'WeightedForm']
    X_train = train_df[features].fillna(train_df[features].median())
    y_train = (train_df['Position'] == 1).astype(int)
    
    predictor.train(X_train, y_train)
    
    # 3. Predict for Mexico Grid
    # Use the qualifying results as the "Test Group"
    mexico_grid = results[['FullName', 'TeamName', 'GridPosition', 'DriverId']].copy()
    
    # Calculate QualiGapPct for Mexico (relative to pole)
    # FastF1 session results usually have the timing
    # For this script, we'll approximate the gap or use 0 for pole
    mexico_grid['QualiGapPct'] = 0.0 # Placeholder; in a full version we'd calc this from session.laps
    
    # Calculate Weighted Form for Mexico Grid
    mexico_grid['WeightedForm'] = mexico_grid['DriverId'].apply(lambda x: engineer.calculate_weighted_form(x, all_hist))
    
    X_mexico = mexico_grid[features].fillna(mexico_grid[features].median())
    probs = predictor.predict_probabilities(X_mexico)
    mexico_grid['WinProbability'] = probs
    
    predictions = mexico_grid[['FullName', 'TeamName', 'GridPosition', 'WinProbability']].sort_values('WinProbability', ascending=False)
    
    print("\n--- 🔮 MODEL C PREDICTION: MEXICO GP 2026 ---")
    print(predictions.head(5).to_string(index=False))
    print(f"\nPredicted Winner: {predictions.iloc[0]['FullName']}")

if __name__ == "__main__":
    predict_mexico()
