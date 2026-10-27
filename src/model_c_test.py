import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

def load_data():
    hist_race = pd.read_csv('data/race_results_2022_2025.csv')
    hist_qual = pd.read_csv('data/qualifying_2022_2025.csv')
    race_2026 = pd.read_csv('data/race_results_2026.csv')
    qual_2026 = pd.read_csv('data/qualifying_2026.csv')
    return hist_race, hist_qual, race_2026, qual_2026

def create_circuit_feature(hist_race):
    circuit_perf = hist_race.groupby(['EventName', 'DriverId'])['Position'].mean().reset_index()
    circuit_perf.rename(columns={'Position': 'CircuitAvgPos'}, inplace=True)
    return circuit_perf

def prepare_features(race_df, qual_df, circuit_perf, historical_race_df):
    # Join race and qual on DriverId and Round
    # The merge on ['DriverId', 'Round'] should be fine. 
    # The issue was TeamName. Both race_df and qual_df have TeamName.
    # Let's drop TeamName from one to avoid suffixes.
    qual_subset = qual_df[['DriverId', 'Round', 'QualiGapPct', 'QualiMissing']]
    df = race_df.merge(qual_subset, on=['DriverId', 'Round'], how='left')
    
    df['MissingQual'] = df['QualiMissing'].fillna(1).astype(int)
    
    team_points = historical_race_df.groupby('TeamName')['Points'].mean().reset_index()
    team_points.rename(columns={'Points': 'TeamAvgPoints'}, inplace=True)
    df = df.merge(team_points, on='TeamName', how='left')
    
    driver_form = historical_race_df.groupby('DriverId')['Position'].mean().reset_index()
    driver_form.rename(columns={'Position': 'DriverAvgPos'}, inplace=True)
    df = df.merge(driver_form, on='DriverId', how='left')
    
    if circuit_perf is not None:
        df = df.merge(circuit_perf, on=['EventName', 'DriverId'], how='left')
        df['CircuitAvgPos'] = df['CircuitAvgPos'].fillna(df['DriverAvgPos'])
    
    return df

def evaluate_model(df, features, target_col='Position'):
    results = []
    for round_id, group in df.groupby('Round'):
        y = (group[target_col] == 1).astype(int)
        X = group[features]
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_processed = imputer.fit_transform(X)
        X_processed = scaler.fit_transform(X_processed)
        
        model = LogisticRegression()
        model.fit(X_processed, y)
        
        probs = model.predict_proba(X_processed)[:, 1]
        if probs.sum() > 0:
            probs = (probs / probs.sum()) * 100
        else:
            probs = np.full(len(probs), 100.0 / len(probs))
            
        pred_winner = np.argmax(probs)
        actual_winner = np.argmax(y)
        
        top_3_pred = np.argsort(probs)[-3:]
        actual_positions = group[target_col].values
        actual_podium_indices = np.where(actual_positions <= 3)[0]
        overlap = len(set(top_3_pred).intersection(set(actual_podium_indices)))
        
        p_sum = probs.sum()
        norm_probs = probs / p_sum if p_sum > 0 else np.full(len(probs), 1.0/len(probs))
        
        results.append({
            'Round': round_id,
            'Correct': 1 if pred_winner == actual_winner else 0,
            'Overlap': overlap,
            'LogLoss': log_loss(y, norm_probs)
        })
        
    res_df = pd.DataFrame(results)
    return {
        'Accuracy': res_df['Correct'].mean() * 100,
        'AvgOverlap': res_df['Overlap'].mean(),
        'MeanLogLoss': res_df['LogLoss'].mean(),
        'Details': res_df
    }

if __name__ == "__main__":
    hist_race, hist_qual, race_2026, qual_2026 = load_data()
    circuit_perf = create_circuit_feature(hist_race)
    
    feat_b = ['GridPosition', 'QualiGapPct', 'MissingQual', 'TeamAvgPoints', 'DriverAvgPos']
    df_b = prepare_features(race_2026, qual_2026, None, hist_race)
    
    feat_c = ['GridPosition', 'QualiGapPct', 'MissingQual', 'TeamAvgPoints', 'DriverAvgPos', 'CircuitAvgPos']
    df_c = prepare_features(race_2026, qual_2026, circuit_perf, hist_race)
    
    res_b = evaluate_model(df_b, feat_b)
    res_c = evaluate_model(df_c, feat_c)
    
    print("--- Model B (Reconstructed Baseline) ---")
    print(f"Accuracy: {res_b['Accuracy']:.2f}%")
    print(f"Avg Podium Overlap: {res_b['AvgOverlap']:.3f}")
    print(f"Log Loss: {res_b['MeanLogLoss']:.4f}")
    
    print("\n--- Model C (Circuit-Specific) ---")
    print(f"Accuracy: {res_c['Accuracy']:.2f}%")
    print(f"Avg Podium Overlap: {res_c['AvgOverlap']:.3f}")
    print(f"Log Loss: {res_c['MeanLogLoss']:.4f}")
    
    comparison = res_b['Details'].copy()
    comparison['ModelC_Correct'] = res_c['Details']['Correct']
    comparison['ModelB_Correct'] = res_b['Details']['Correct']
    print("\n--- Race-by-Race Comparison ---")
    print(comparison[['Round', 'ModelB_Correct', 'ModelC_Correct']])
