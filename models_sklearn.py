import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Dict, Any

class BaselineModel:
    """
    Simple baseline models for price movement prediction.
    """
    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        if model_type == "logistic":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten sequences for traditional ML models.
        """
        # Reshape from (samples, timesteps, features) to (samples, timesteps*features)
        X_flat = X.reshape(X.shape[0], -1)
        return X_flat, y
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the baseline model.
        """
        X_flat, y_flat = self.prepare_data(X, y)
        self.model.fit(X_flat, y_flat)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        """
        X_flat, _ = self.prepare_data(X, None)
        return self.model.predict(X_flat)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        """
        X_flat, _ = self.prepare_data(X, None)
        return self.model.predict_proba(X_flat)

class NeuralNetworkModel:
    """
    Multi-layer perceptron as an alternative to LSTM for sequence modeling.
    """
    def __init__(self, hidden_layer_sizes: Tuple = (100, 50), max_iter: int = 500):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten sequences for neural network.
        """
        # Reshape from (samples, timesteps, features) to (samples, timesteps*features)
        X_flat = X.reshape(X.shape[0], -1)
        return X_flat, y
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the neural network.
        """
        X_flat, y_flat = self.prepare_data(X, y)
        self.model.fit(X_flat, y_flat)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        """
        X_flat, _ = self.prepare_data(X, None)
        return self.model.predict(X_flat)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        """
        X_flat, _ = self.prepare_data(X, None)
        return self.model.predict_proba(X_flat)

class SimpleTimeSeriesModel:
    """
    A simple approach that uses recent values and technical indicators.
    """
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract meaningful features from time series sequences.
        """
        # Extract features: last values, means, trends
        features = []
        for i in range(X.shape[0]):
            sequence = X[i]  # Shape: (timesteps, features)
            
            # Last values (most recent)
            last_values = sequence[-1, :]
            
            # Mean values over the sequence
            mean_values = np.mean(sequence, axis=0)
            
            # Trend (difference between last and first)
            trend_values = sequence[-1, :] - sequence[0, :]
            
            # Volatility (std over the sequence)
            volatility_values = np.std(sequence, axis=0)
            
            # Combine all features
            combined_features = np.concatenate([
                last_values, mean_values, trend_values, volatility_values
            ])
            features.append(combined_features)
        
        X_processed = np.array(features)
        return X_processed, y
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.
        """
        X_processed, y_processed = self.prepare_data(X, y)
        self.model.fit(X_processed, y_processed)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        """
        X_processed, _ = self.prepare_data(X, None)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        """
        X_processed, _ = self.prepare_data(X, None)
        return self.model.predict_proba(X_processed)

# Test functions

def test_baseline_model():
    """Test baseline model with dummy data."""
    # Create dummy data
    X = np.random.rand(100, 10, 5)  # 100 samples, 10 timesteps, 5 features
    y = np.random.randint(0, 2, 100)
    
    # Test logistic regression
    model = BaselineModel("logistic")
    model.fit(X, y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    assert predictions.shape == (100,), f"Prediction shape incorrect: {predictions.shape}"
    assert probabilities.shape == (100, 2), f"Probability shape incorrect: {probabilities.shape}"
    print("✓ Baseline model test passed")

def test_neural_network_model():
    """Test neural network model with dummy data."""
    # Create dummy data
    X = np.random.rand(100, 10, 5)
    y = np.random.randint(0, 2, 100)
    
    # Test neural network
    model = NeuralNetworkModel(hidden_layer_sizes=(50, 25), max_iter=100)
    model.fit(X, y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    assert predictions.shape == (100,), f"Prediction shape incorrect: {predictions.shape}"
    assert probabilities.shape == (100, 2), f"Probability shape incorrect: {probabilities.shape}"
    print("✓ Neural network model test passed")

def test_simple_time_series_model():
    """Test simple time series model with dummy data."""
    # Create dummy data
    X = np.random.rand(100, 10, 5)
    y = np.random.randint(0, 2, 100)
    
    # Test simple time series model
    model = SimpleTimeSeriesModel()
    model.fit(X, y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    assert predictions.shape == (100,), f"Prediction shape incorrect: {predictions.shape}"
    assert probabilities.shape == (100, 2), f"Probability shape incorrect: {probabilities.shape}"
    print("✓ Simple time series model test passed")

if __name__ == "__main__":
    test_baseline_model()
    test_neural_network_model()
    test_simple_time_series_model()
    print("\n✅ All sklearn model tests passed!") 