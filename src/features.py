import pandas as pd
import numpy as np

class F1FeatureEngineer:
    def __init__(self):
        pass

    def calculate_weighted_form(self, driver_id, all_races_df):
        """
        Implements the Model C 'Elite Form' logic:
        Top 3 finish = 3 points, 4th-5th = 1 point.
        Analyzes the last 3 races for the driver.
        """
        d_races = all_races_df[all_races_df['DriverId'] == driver_id].sort_values(['Year', 'Round'], ascending=False)
        last_3 = d_races.head(3)
        
        score = 0
        for pos in last_3['Position']:
            if pos <= 3: 
                score += 3
            elif pos <= 5: 
                score += 1
        return score

    def prepare_adaptive_features(self, race_df, qual_df, historical_race_df, round_num):
        """
        Prepares the feature set for a specific round, 
        calculating form based on all races prior to this round.
        """
        # Join race and qual
        df = race_df.merge(qual_df[['DriverId', 'Round', 'QualiGapPct']], on=['DriverId', 'Round'], how='left')
        
        # Create the pool of all available data before this race
        # Combine historical data + current season races before round_num
        # Note: In a real pipeline, this df would be passed in to avoid redundant concatenation
        all_available = pd.concat([
            historical_race_df, 
            race_df[race_df['Round'] < round_num]
        ])
        
        # Apply the weighted form calculation
        df['WeightedForm'] = df['DriverId'].apply(lambda x: self.calculate_weighted_form(x, all_available))
        
        # Final feature set for Model C
        features = ['GridPosition', 'QualiGapPct', 'WeightedForm']
        return df, features
