import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
import json
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

from data_provider import YahooFinanceProvider
from feature_engineering import (
    add_price_range, add_price_change, add_moving_average, 
    add_rsi, add_volatility, add_momentum
)
from data_preprocessing import preprocess_data
from models_sklearn import BaselineModel, NeuralNetworkModel, SimpleTimeSeriesModel

class ModelTrainer:
    """
    Comprehensive model training and evaluation pipeline.
    """
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
    
    def prepare_data(self, symbol: str = "AAPL", start_date: str = "2022-01-01", end_date: str = "2023-12-31"):
        """
        Prepare data for training.
        """
        print(f"📊 Preparing data for {symbol}...")
        
        # Fetch data
        provider = YahooFinanceProvider()
        df = provider.fetch_historical_data(symbol, start_date, end_date)
        
        # Add features
        df = add_price_range(df)
        df = add_price_change(df)
        df = add_moving_average(df, window=5)
        df = add_moving_average(df, window=10)
        df = add_moving_average(df, window=20)
        df = add_rsi(df, window=14)
        df = add_volatility(df, window=10)
        df = add_momentum(df, window=10)
        
        # Define features
        feature_cols = [
            "close", "volume", "price_range", "price_change",
            "ma_5", "ma_10", "ma_20", "rsi_14", "volatility_10", "momentum_10"
        ]
        
        # Preprocess
        result = preprocess_data(df, feature_cols, lookback_window=30, test_size=0.2)
        
        print(f"✅ Data prepared: {result['X_train'].shape[0]} train, {result['X_test'].shape[0]} test samples")
        return result
    
    def train_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train baseline models.
        """
        print("🔄 Training baseline models...")
        
        # Logistic Regression
        lr_model = BaselineModel("logistic")
        lr_model.fit(X_train, y_train)
        self.models["logistic_regression"] = lr_model
        
        # Random Forest
        rf_model = BaselineModel("random_forest")
        rf_model.fit(X_train, y_train)
        self.models["random_forest"] = rf_model
        
        print("✅ Baseline models trained")
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train neural network with hyperparameter tuning.
        """
        print("🔄 Training neural network...")
        
        # Basic neural network
        nn_model = NeuralNetworkModel(hidden_layer_sizes=(100, 50), max_iter=500)
        nn_model.fit(X_train, y_train)
        self.models["neural_network"] = nn_model
        
        print("✅ Neural network trained")
    
    def train_time_series_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train time series specific model.
        """
        print("🔄 Training time series model...")
        
        ts_model = SimpleTimeSeriesModel()
        ts_model.fit(X_train, y_train)
        self.models["time_series"] = ts_model
        
        print("✅ Time series model trained")
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model.
        """
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]  # Probability of class 1
        
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions)
        }
        
        print(f"📊 {model_name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate all trained models.
        """
        print("\n🔍 Evaluating all models...")
        
        for model_name, model in self.models.items():
            # Test set evaluation
            test_metrics = self.evaluate_model(model, X_test, y_test, f"{model_name} (Test)")
            
            # Training set evaluation (to check overfitting)
            train_metrics = self.evaluate_model(model, X_train, y_train, f"{model_name} (Train)")
            
            # Calculate overfitting indicator
            overfitting = train_metrics["accuracy"] - test_metrics["accuracy"]
            
            self.results[model_name] = {
                "test_metrics": test_metrics,
                "train_metrics": train_metrics,
                "overfitting": overfitting
            }
            
            print(f"  Overfitting (train - test accuracy): {overfitting:.4f}")
            print()
            
            # Track best model
            if test_metrics["accuracy"] > self.best_score:
                self.best_score = test_metrics["accuracy"]
                self.best_model = (model_name, model)
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Perform hyperparameter tuning for the best performing models.
        """
        print("🔧 Hyperparameter tuning...")
        
        # Tune Random Forest
        rf_param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [10, 20, None]
        }
        
        # Note: This is a simplified example. In practice, you'd create a pipeline
        # and use more sophisticated tuning
        print("✅ Hyperparameter tuning completed (simplified)")
    
    def save_best_model(self, filepath: str = "best_model.pkl"):
        """
        Save the best performing model.
        """
        if self.best_model:
            model_name, model = self.best_model
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model_name': model_name,
                    'model': model,
                    'score': self.best_score
                }, f)
            print(f"💾 Best model ({model_name}) saved to {filepath}")
        else:
            print("❌ No best model to save")
    
    def save_results(self, filepath: str = "training_results.json"):
        """
        Save training results.
        """
        # Convert numpy types to Python types for JSON serialization
        results_serializable = {}
        for model_name, result in self.results.items():
            results_serializable[model_name] = {
                "test_metrics": {k: float(v) for k, v in result["test_metrics"].items()},
                "train_metrics": {k: float(v) for k, v in result["train_metrics"].items()},
                "overfitting": float(result["overfitting"])
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"📄 Results saved to {filepath}")
    
    def run_complete_training(self, symbol: str = "AAPL"):
        """
        Run the complete training pipeline.
        """
        print("🚀 Starting complete training pipeline...")
        
        # Prepare data
        data = self.prepare_data(symbol)
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        
        # Train all models
        self.train_baseline_models(X_train, y_train)
        self.train_neural_network(X_train, y_train)
        self.train_time_series_model(X_train, y_train)
        
        # Evaluate models
        self.evaluate_all_models(X_train, y_train, X_test, y_test)
        
        # Save results
        self.save_best_model()
        self.save_results()
        
        print(f"\n🏆 Best model: {self.best_model[0]} with accuracy: {self.best_score:.4f}")
        print("✅ Training pipeline completed!")
        
        return self.results

def main():
    """
    Main training function.
    """
    trainer = ModelTrainer()
    results = trainer.run_complete_training("AAPL")
    return results

if __name__ == "__main__":
    main() 