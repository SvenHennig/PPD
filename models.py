import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Dict, Any

class BaselineModel:
    """
    Simple baseline models for price movement prediction.
    """
    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        if model_type == "logistic":
            self.model = LogisticRegression(random_state=42)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

class LSTMModel:
    """
    LSTM model for time series price movement prediction.
    """
    def __init__(self, input_shape: Tuple[int, int], lstm_units: int = 50, dropout_rate: float = 0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build LSTM architecture.
        """
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=self.input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 50, batch_size: int = 32, verbose: int = 1) -> tf.keras.callbacks.History:
        """
        Train the LSTM model.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                         patience=10, restore_best_weights=True)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        """
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        """
        return self.model.predict(X).flatten()

class GRUModel:
    """
    GRU model as an alternative to LSTM.
    """
    def __init__(self, input_shape: Tuple[int, int], gru_units: int = 50, dropout_rate: float = 0.2):
        self.input_shape = input_shape
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build GRU architecture.
        """
        model = Sequential([
            GRU(self.gru_units, return_sequences=True, input_shape=self.input_shape),
            Dropout(self.dropout_rate),
            GRU(self.gru_units, return_sequences=True),
            Dropout(self.dropout_rate),
            GRU(self.gru_units),
            Dropout(self.dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 50, batch_size: int = 32, verbose: int = 1) -> tf.keras.callbacks.History:
        """
        Train the GRU model.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', 
                         patience=10, restore_best_weights=True)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        """
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        """
        return self.model.predict(X).flatten()

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

def test_lstm_model():
    """Test LSTM model with dummy data."""
    # Create dummy data
    X_train = np.random.rand(80, 10, 5)
    y_train = np.random.randint(0, 2, 80)
    X_test = np.random.rand(20, 10, 5)
    
    # Test LSTM
    model = LSTMModel(input_shape=(10, 5), lstm_units=32)
    
    # Check model architecture
    assert model.model.input_shape == (None, 10, 5), "Input shape incorrect"
    assert model.model.output_shape == (None, 1), "Output shape incorrect"
    
    # Test prediction without training (just architecture)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    assert predictions.shape == (20,), f"Prediction shape incorrect: {predictions.shape}"
    assert probabilities.shape == (20,), f"Probability shape incorrect: {probabilities.shape}"
    print("✓ LSTM model test passed")

def test_gru_model():
    """Test GRU model with dummy data."""
    # Create dummy data
    X_test = np.random.rand(20, 10, 5)
    
    # Test GRU
    model = GRUModel(input_shape=(10, 5), gru_units=32)
    
    # Check model architecture
    assert model.model.input_shape == (None, 10, 5), "Input shape incorrect"
    assert model.model.output_shape == (None, 1), "Output shape incorrect"
    
    # Test prediction without training (just architecture)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    assert predictions.shape == (20,), f"Prediction shape incorrect: {predictions.shape}"
    assert probabilities.shape == (20,), f"Probability shape incorrect: {probabilities.shape}"
    print("✓ GRU model test passed")

if __name__ == "__main__":
    test_baseline_model()
    test_lstm_model()
    test_gru_model()
    print("\n✅ All model tests passed!") 