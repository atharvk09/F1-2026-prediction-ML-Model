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

def get_circuit_experience(hist_race):
    return hist_race.groupby(['EventName', 'DriverId'])['Year'].nunique().reset_index().rename(columns={'Year': 'CircuitExperience'})

def calc_weighted_form(driver_id, all_races_df):
    d_races = all_races_df[all_races_df['DriverId'] == driver_id].sort_values(['Year', 'Round'], ascending=False)
    last_3 = d_races.head(3)
    score = 0
    for pos in last_3['Position']:
        if pos <= 3: score += 3
        elif pos <= 5: score += 1
    return score

def run_adaptive_eval(use_elite_features=False):
    hist_race, hist_qual, race_2026, qual_2026 = load_data()
    circuit_exp = get_circuit_experience(hist_race)
    weights = {2022: 0.5, 2023: 0.75, 2024: 1.0, 2025: 1.5}
    results = []
    
    hist_full = hist_race.merge(hist_qual[['DriverId', 'Year', 'Round', 'QualiGapPct']], 
                                on=['DriverId', 'Year', 'Round'], how='left')
    
    for round_num in range(1, 18):
        # 1. Build Adaptive Training Set
        train_data = []
        for year, w in weights.items():
            year_data = hist_full[hist_full['Year'] == year].copy()
            year_data['weight'] = w
            train_data.append(year_data)
        train_df = pd.concat(train_data)
        
        if round_num > 1:
            current_season_race = race_2026[race_2026['Round'] < round_num].copy()
            current_season_qual = qual_2026[qual_2026['Round'] < round_num][['DriverId', 'Round', 'QualiGapPct']]
            current_season_full = current_season_race.merge(current_season_qual, on=['DriverId', 'Round'], how='left')
            current_season_full['weight'] = 2.0
            train_df = pd.concat([train_df, current_season_full])
        
        # 2. Prepare Test Set
        test_group = race_2026[race_2026['Round'] == round_num].copy()
        test_group = test_group.merge(qual_2026[['DriverId', 'Round', 'QualiGapPct']], on=['DriverId', 'Round'], how='left')
        
        # Combine for form calculation
        all_available = pd.concat([hist_race, race_2026[race_2026['Round'] < round_num]])
        
        # Apply Weighted Form to both
        test_group['WeightedForm'] = test_group['DriverId'].apply(lambda x: calc_weighted_form(x, all_available))
        train_df['WeightedForm'] = train_df['DriverId'].apply(lambda x: calc_weighted_form(x, all_available))
        
        if use_elite_features:
            # A. Hot Streak
            def calc_hot_streak(driver_id, all_races_df):
                d_races = all_races_df[all_races_df['DriverId'] == driver_id].sort_values(['Year', 'Round'], ascending=False)
                last_3 = d_races.head(3)
                return 1 if (last_3['Position'] <= 3).sum() >= 2 else 0
            
            test_group['HotStreak'] = test_group['DriverId'].apply(lambda x: calc_hot_streak(x, all_available))
            train_df['HotStreak'] = train_df['DriverId'].apply(lambda x: calc_hot_streak(x, all_available))
            
            # B. Team Recent Podiums
            def calc_team_podiums(team_name, all_races_df):
                t_races = all_races_df[all_races_df['TeamName'] == team_name].sort_values(['Year', 'Round'], ascending=False)
                last_3 = t_races.head(3)
                return (last_3['Position'] <= 3).sum()
            
            test_group['TeamRecentPodiums'] = test_group['TeamName'].apply(lambda x: calc_team_podiums(x, all_available))
            train_df['TeamRecentPodiums'] = train_df['TeamName'].apply(lambda x: calc_team_podiums(x, all_available))
            
            # C. Circuit Veteran
            test_group = test_group.merge(circuit_exp, on=['EventName', 'DriverId'], how='left')
            test_group['CircuitVeteran'] = (test_group['CircuitExperience'] >= 3).astype(int)
            train_df = train_df.merge(circuit_exp, on=['EventName', 'DriverId'], how='left')
            train_df['CircuitVeteran'] = (train_df['CircuitExperience'] >= 3).astype(int)
            
            features = ['GridPosition', 'QualiGapPct', 'WeightedForm', 'HotStreak', 'TeamRecentPodiums', 'CircuitVeteran']
        else:
            features = ['GridPosition', 'QualiGapPct', 'WeightedForm']
            
        y_train = (train_df['Position'] == 1).astype(int)
        X_train = train_df[features].fillna(train_df[features].median())
        y_test = (test_group['Position'] == 1).astype(int)
        X_test = test_group[features].fillna(test_group[features].median())
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        
        probs = model.predict_proba(X_test_scaled)[:, 1]
        if probs.sum() > 0:
            probs = (probs / probs.sum()) * 100
        else:
            probs = np.full(len(probs), 100.0 / len(probs))
            
        pred_winner = np.argmax(probs)
        actual_winner = np.argmax(y_test)
        top_3_pred = np.argsort(probs)[-3:]
        actual_podium = np.where(test_group['Position'] <= 3)[0]
        overlap = len(set(top_3_pred).intersection(set(actual_podium)))
        
        p_sum = probs.sum()
        norm_probs = probs / p_sum if p_sum > 0 else np.full(len(probs), 1.0/len(probs))
        
        results.append({
            'Round': round_num,
            'Correct': 1 if pred_winner == actual_winner else 0,
            'Overlap': overlap,
            'LogLoss': log_loss(y_test, norm_probs)
        })
        
    res_df = pd.DataFrame(results)
    return {
        'Accuracy': res_df['Correct'].mean() * 100,
        'AvgOverlap': res_df['Overlap'].mean(),
        'MeanLogLoss': res_df['LogLoss'].mean(),
        'Details': res_df
    }

if __name__ == "__main__":
    res_b = run_adaptive_eval(use_elite_features=False)
    res_c = run_adaptive_eval(use_elite_features=True)
    
    print("--- Adaptive Model B (Weighted Form Baseline) ---")
    print(f"Accuracy: {res_b['Accuracy']:.2f}%")
    print(f"Avg Podium Overlap: {res_b['AvgOverlap']:.3f}")
    print(f"Log Loss: {res_b['MeanLogLoss']:.4f}")
    
    print("\n--- Adaptive Model C+ (Elite Features) ---")
    print(f"Accuracy: {res_c['Accuracy']:.2f}%")
    print(f"Avg Podium Overlap: {res_c['AvgOverlap']:.3f}")
    print(f"Log Loss: {res_c['MeanLogLoss']:.4f}")
    
    comparison = res_b['Details'].copy()
    comparison['ModelC_Correct'] = res_c['Details']['Correct']
    comparison['ModelB_Correct'] = res_b['Details']['Correct']
    print("\n--- Race-by-Race Comparison ---")
    print(comparison[['Round', 'ModelB_Correct', 'ModelC_Correct']])
