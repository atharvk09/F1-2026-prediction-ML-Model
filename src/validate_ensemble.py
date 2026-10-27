import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from src.data_loader import F1DataLoader
from src.features import F1FeatureEngineer
from src.model import F1Predictor
from src.ensemble_model import F1EnsemblePredictor

def run_comparison():
    loader = F1DataLoader()
    engineer = F1FeatureEngineer()
    
    # Models
    model_c = F1Predictor()
    model_d = F1EnsemblePredictor()
    
    data = loader.load_csv_data()
    hist_race = data['hist_race']
    hist_qual = data['hist_qual']
    race_2026 = data['race_2026']
    qual_2026 = data['qual_2026']
    
    weights = {2022: 0.5, 2023: 0.75, 2024: 1.0, 2025: 1.5}
    results_c = []
    results_d = []
    
    hist_full = hist_race.merge(hist_qual[['DriverId', 'Year', 'Round', 'QualiGapPct']], 
                                on=['DriverId', 'Year', 'Round'], how='left')
    
    for round_num in range(1, 18):
        # Build Adaptive Training Set
        train_data = []
        for year, w in weights.items():
            year_df = hist_full[hist_full['Year'] == year].copy()
            year_df['weight'] = w
            train_data.append(year_df)
        train_df = pd.concat(train_data)
        
        if round_num > 1:
            current_season_race = race_2026[race_2026['Round'] < round_num].copy()
            current_season_qual = qual_2026[qual_2026['Round'] < round_num][['DriverId', 'Round', 'QualiGapPct']]
            current_season_full = current_season_race.merge(current_season_qual, on=['DriverId', 'Round'], how='left')
            current_season_full['weight'] = 2.0
            train_df = pd.concat([train_df, current_season_full])
        
        # Features
        all_available = pd.concat([hist_race, race_2026[race_2026['Round'] < round_num]])
        train_df['WeightedForm'] = train_df['DriverId'].apply(lambda x: engineer.calculate_weighted_form(x, all_available))
        
        # Ensure QualiGapPct exists
        if 'QualiGapPct' not in train_df.columns:
            train_df = train_df.merge(hist_qual[['DriverId', 'Year', 'Round', 'QualiGapPct']], on=['DriverId', 'Year', 'Round'], how='left')
        
        features = ['GridPosition', 'QualiGapPct', 'WeightedForm']
        X_train = train_df[features].fillna(train_df[features].median())
        y_train = (train_df['Position'] == 1).astype(int)
        
        # Train both
        model_c.train(X_train, y_train)
        model_d.train(X_train, y_train)
        
        # Test Set
        test_group = race_2026[race_2026['Round'] == round_num].copy()
        test_group = test_group.merge(qual_2026[['DriverId', 'Round', 'QualiGapPct']], on=['DriverId', 'Round'], how='left')
        test_group['WeightedForm'] = test_group['DriverId'].apply(lambda x: engineer.calculate_weighted_form(x, all_available))
        X_test = test_group[features].fillna(test_group[features].median())
        
        # Predict
        probs_c = model_c.predict_probabilities(X_test)
        probs_d = model_d.predict_probabilities(X_test)
        
        y_test = (test_group['Position'] == 1).astype(int)
        
        # Fixed log_loss call
        results_c.append({'Correct': 1 if np.argmax(probs_c) == np.argmax(y_test) else 0, 'LogLoss': log_loss(y_test, probs_c/100.0)})
        results_d.append({'Correct': 1 if np.argmax(probs_d) == np.argmax(y_test) else 0, 'LogLoss': log_loss(y_test, probs_d/100.0)})
        
    df_c = pd.DataFrame(results_c)
    df_d = pd.DataFrame(results_d)
    
    print("\n--- FINAL SHOWDOWN: Model C vs Model D ---")
    print(f"Model C (LR) Accuracy: {df_c['Correct'].mean()*100:.2f}% | LogLoss: {df_c['LogLoss'].mean():.4f}")
    print(f"Model D (Ensemble) Accuracy: {df_d['Correct'].mean()*100:.2f}% | LogLoss: {df_d['LogLoss'].mean():.4f}")

if __name__ == "__main__":
    run_comparison()
