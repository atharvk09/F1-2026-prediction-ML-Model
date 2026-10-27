import fastf1
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Setup Cache
fastf1.Cache.enable_cache('cache')

def get_weighted_form(driver_id, race_results_2026):
    # Simplified Weighted Form for the Austin prediction
    # We look at the most recent 3 races in 2026
    driver_data = race_results_2026[race_results_2026['DriverId'] == driver_id].sort_values('Round', ascending=False)
    last_3 = driver_data.head(3)
    
    score = 0
    for pos in last_3['Position']:
        if pos <= 3: score += 3
        elif pos <= 5: score += 1
    return score

def predict_austin():
    print("Fetching Austin 2026 data from FastF1...")
    try:
        session = fastf1.get_session(2026, 'United States', 'R')
        session.load()
        results = session.results
    except Exception as e:
        print(f"Error fetching session: {e}")
        return

    # Correct columns for FastF1 results dataframe
    # Driver name is usually 'FullName', not 'Driver'
    cols = ['FullName', 'TeamName', 'GridPosition', 'Position', 'DriverId']
    df = results[cols].copy()
    
    # To make this a real Model C prediction, we need the previous 2026 data
    # Let's load the 2026 CSV to calculate Weighted Form
    try:
        race_2026 = pd.read_csv('data/race_results_2026.csv')
    except:
        print("Could not load data/race_results_2026.csv for form calculation.")
        return

    # Calculate Weighted Form for every driver on the Austin grid
    df['WeightedForm'] = df['DriverId'].apply(lambda x: get_weighted_form(x, race_2026))
    
    # To simulate the Model C prediction:
    # We'll use a simple scoring heuristic based on the Model C findings:
    # Score = (10 - GridPosition) + (WeightedForm * 0.5)
    df['PredictionScore'] = (10 - df['GridPosition']) + (df['WeightedForm'] * 0.5)
    
    # Rank them
    df = df.sort_values('PredictionScore', ascending=False)
    
    print("\n--- Model C Predictions for Austin 2026 ---")
    print(df[['FullName', 'TeamName', 'GridPosition', 'WeightedForm', 'PredictionScore']].head(5))
    
    actual_winner = df[df['Position'] == 1]['FullName'].values[0]
    predicted_winner = df.iloc[0]['FullName']
    
    print(f"\nPredicted Winner: {predicted_winner}")
    print(f"Actual Winner: {actual_winner}")
    
    if predicted_winner == actual_winner:
        print("\n✅ SUCCESS: Model C correctly predicted the winner!")
    else:
        print("\n❌ FAILURE: Model C missed the winner.")

if __name__ == "__main__":
    predict_austin()
