import pandas as pd
import numpy as np
from src.data_loader import F1DataLoader
from src.features import F1FeatureEngineer

def run_prequal_prediction():
    print("--- 🔮 PRE-QUALIFYING PREDICTION: MEXICO GP 2026 ---")
    
    loader = F1DataLoader()
    engineer = F1FeatureEngineer()
    
    # 1. Load Data
    data = loader.load_csv_data()
    hist_race = data['hist_race']
    race_2026 = data['race_2026']
    
    # Combine for form calculation
    all_available = pd.concat([hist_race, race_2026])
    
    # 2. Identify all drivers competing in 2026
    drivers = race_2026['DriverId'].unique()
    
    predictions = []
    
    # 3. Calculate Score for each driver
    # Since we have no Grid/Quali, we use:
    # Score = (Weighted Form * 0.7) + (Team Season Avg * 0.3)
    
    # Pre-calculate team average points
    team_avg = hist_race.groupby('TeamName')['Points'].mean().to_dict()
    
    for driver_id in drivers:
        # Get driver name and team
        driver_info = race_2026[race_2026['DriverId'] == driver_id].iloc[0]
        name = driver_info['FullName']
        team = driver_info['TeamName']
        
        # Feature 1: Weighted Form (The Model C Secret Sauce)
        form = engineer.calculate_weighted_form(driver_id, all_available)
        
        # Feature 2: Team Strength
        team_score = team_avg.get(team, 0)
        
        # Normalize team score (approximate scaling to match form 0-9)
        norm_team_score = (team_score / 20) * 10
        
        # Final Pre-Quali Score
        final_score = (form * 0.7) + (norm_team_score * 0.3)
        
        predictions.append({
            'Driver': name,
            'Team': team,
            'FormScore': form,
            'TeamScore': round(norm_team_score, 2),
            'WinProbScore': round(final_score, 2)
        })
    
    # Rank and display
    res_df = pd.DataFrame(predictions).sort_values('WinProbScore', ascending=False)
    
    print("\nTOP 5 FAVORITES (Pre-Qualifying):")
    print(res_df.head(5).to_string(index=False))
    print(f"\nPredicted Favorite: {res_df.iloc[0]['Driver']}")

if __name__ == "__main__":
    run_prequal_prediction()
