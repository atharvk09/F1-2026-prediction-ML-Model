import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

class F1Predictor:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_trained = False

    def train(self, X, y):
        """
        Trains the model on the provided features and target.
        """
        # Impute and Scale
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("Model trained successfully.")

    def predict_probabilities(self, X):
        """
        Predicts win probabilities for a given set of drivers.
        """
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions.")
            
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Probability of class 1 (winner)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Normalize probabilities to sum to 100% within the race
        if probs.sum() > 0:
            probs = (probs / probs.sum()) * 100
        else:
            probs = np.full(len(probs), 100.0 / len(probs))
            
        return probs

    def save_model(self, path='models/model_c_final.pkl'):
        """Saves the model and scaler for production use."""
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'model_type': self.model_type
        }
        joblib.dump(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path='models/model_c_final.pkl'):
        """Loads a saved model and its preprocessing state."""
        state = joblib.load(path)
        self.model = state['model']
        self.scaler = state['scaler']
        self.imputer = state['imputer']
        self.model_type = state['model_type']
        self.is_trained = True
        print(f"Model loaded from {path}")
