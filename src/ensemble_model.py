import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class F1EnsemblePredictor:
    def __init__(self):
        # Define the three base models
        self.lr = LogisticRegression()
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        
        # Create the Voting Classifier (Soft voting averages probabilities)
        self.ensemble = VotingClassifier(
            estimators=[
                ('lr', self.lr),
                ('rf', self.rf),
                ('xgb', self.xgb)
            ],
            voting='soft'
        )
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_trained = False

    def train(self, X, y):
        """Trains the ensemble of models."""
        # Preprocessing
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.ensemble.fit(X_scaled, y)
        self.is_trained = True
        print("Ensemble Model trained successfully.")

    def predict_probabilities(self, X):
        """Predicts win probabilities using the ensemble."""
        if not self.is_trained:
            raise Exception("Ensemble must be trained before making predictions.")
            
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Probability of class 1 (winner)
        probs = self.ensemble.predict_proba(X_scaled)[:, 1]
        
        # Normalize probabilities to sum to 100% within the race
        if probs.sum() > 0:
            probs = (probs / probs.sum()) * 100
        else:
            probs = np.full(len(probs), 100.0 / len(probs))
            
        return probs

    def save_ensemble(self, path='models/ensemble_model.pkl'):
        import joblib
        state = {
            'ensemble': self.ensemble,
            'scaler': self.scaler,
            'imputer': self.imputer
        }
        joblib.dump(state, path)
        print(f"Ensemble saved to {path}")

    def load_ensemble(self, path='models/ensemble_model.pkl'):
        import joblib
        state = joblib.load(path)
        self.ensemble = state['ensemble']
        self.scaler = state['scaler']
        self.imputer = state['imputer']
        self.is_trained = True
        print(f"Ensemble loaded from {path}")
