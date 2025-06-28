import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, List
import pickle

class FinancialMetrics:
    """
    Calculate financial performance metrics for trading strategies.
    """
    
    @staticmethod
    def calculate_returns(predictions: np.ndarray, actual_returns: np.ndarray) -> np.ndarray:
        """
        Calculate strategy returns based on predictions.
        predictions: 1 for buy (long), 0 for sell/hold
        actual_returns: actual price returns for each period
        """
        # Strategy: go long when prediction is 1, stay out when 0
        strategy_returns = predictions * actual_returns
        return strategy_returns
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    
    @staticmethod
    def max_drawdown(cumulative_returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        """
        if len(cumulative_returns) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Calculate win rate (percentage of positive returns).
        """
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        """
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0 or np.sum(np.abs(losses)) == 0:
            return np.inf if len(profits) > 0 else 0.0
        
        return np.sum(profits) / np.sum(np.abs(losses))

class ModelEvaluator:
    """
    Comprehensive model evaluation and backtesting.
    """
    
    def __init__(self):
        self.financial_metrics = FinancialMetrics()
    
    def evaluate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate standard classification metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            metrics['auc_roc'] = auc(fpr, tpr)
        
        return metrics
    
    def backtest_strategy(self, predictions: np.ndarray, prices: np.ndarray, 
                         dates: pd.DatetimeIndex = None) -> Dict[str, float]:
        """
        Backtest trading strategy based on model predictions.
        """
        # Calculate actual returns
        actual_returns = np.diff(prices) / prices[:-1]
        
        # Align predictions with returns (predictions are for next day)
        if len(predictions) > len(actual_returns):
            predictions = predictions[:len(actual_returns)]
        elif len(predictions) < len(actual_returns):
            actual_returns = actual_returns[:len(predictions)]
        
        # Calculate strategy returns
        strategy_returns = self.financial_metrics.calculate_returns(predictions, actual_returns)
        cumulative_returns = np.cumprod(1 + strategy_returns)
        
        # Calculate metrics
        metrics = {
            'total_return': cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0,
            'sharpe_ratio': self.financial_metrics.sharpe_ratio(strategy_returns),
            'max_drawdown': self.financial_metrics.max_drawdown(cumulative_returns),
            'win_rate': self.financial_metrics.win_rate(strategy_returns),
            'profit_factor': self.financial_metrics.profit_factor(strategy_returns),
            'total_trades': np.sum(predictions),
            'avg_return_per_trade': np.mean(strategy_returns[predictions == 1]) if np.sum(predictions) > 0 else 0
        }
        
        return metrics, strategy_returns, cumulative_returns
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Confusion Matrix") -> go.Figure:
        """
        Create confusion matrix heatmap.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Down', 'Predicted Up'],
            y=['Actual Down', 'Actual Up'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      title: str = "ROC Curve") -> go.Figure:
        """
        Create ROC curve plot.
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True
        )
        
        return fig
    
    def plot_cumulative_returns(self, cumulative_returns: np.ndarray, 
                               dates: pd.DatetimeIndex = None,
                               title: str = "Cumulative Returns") -> go.Figure:
        """
        Plot cumulative returns over time.
        """
        if dates is None:
            dates = pd.date_range(start='2023-01-01', periods=len(cumulative_returns), freq='D')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns,
            mode='lines',
            name='Strategy Returns',
            line=dict(color='green', width=2)
        ))
        
        # Add buy and hold benchmark
        buy_hold_returns = np.linspace(1, cumulative_returns[-1], len(cumulative_returns))
        fig.add_trace(go.Scatter(
            x=dates,
            y=buy_hold_returns,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            showlegend=True
        )
        
        return fig

def comprehensive_model_evaluation():
    """
    Run comprehensive evaluation of the best trained model.
    """
    print("🔍 Starting comprehensive model evaluation...")
    
    # Load the best model
    try:
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        best_model = model_data['model']
        model_name = model_data['model_name']
        print(f"📂 Loaded best model: {model_name}")
    except FileNotFoundError:
        print("❌ Best model file not found. Please run training first.")
        return
    
    # Prepare fresh data for evaluation
    from data_provider import YahooFinanceProvider
    from feature_engineering import (
        add_price_range, add_price_change, add_moving_average, 
        add_rsi, add_volatility, add_momentum
    )
    from data_preprocessing import preprocess_data
    
    provider = YahooFinanceProvider()
    df = provider.fetch_historical_data("AAPL", "2022-01-01", "2023-12-31")
    
    # Add features
    df = add_price_range(df)
    df = add_price_change(df)
    df = add_moving_average(df, window=5)
    df = add_moving_average(df, window=10)
    df = add_moving_average(df, window=20)
    df = add_rsi(df, window=14)
    df = add_volatility(df, window=10)
    df = add_momentum(df, window=10)
    
    feature_cols = [
        "close", "volume", "price_range", "price_change",
        "ma_5", "ma_10", "ma_20", "rsi_14", "volatility_10", "momentum_10"
    ]
    
    result = preprocess_data(df, feature_cols, lookback_window=30, test_size=0.2)
    X_test, y_test = result['X_test'], result['y_test']
    
    # Get test data prices and dates for backtesting
    test_start_idx = len(result['X_train']) + 30  # Account for lookback window
    test_prices = df['close'].iloc[test_start_idx:test_start_idx + len(X_test) + 1].values
    test_dates = df['timestamp'].iloc[test_start_idx:test_start_idx + len(X_test)].values
    
    # Make predictions
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Classification metrics
    print("\n📊 Classification Metrics:")
    class_metrics = evaluator.evaluate_classification_metrics(y_test, predictions, probabilities)
    for metric, value in class_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Financial metrics and backtesting
    print("\n💰 Financial Metrics:")
    financial_metrics, strategy_returns, cumulative_returns = evaluator.backtest_strategy(
        predictions, test_prices, pd.to_datetime(test_dates)
    )
    
    for metric, value in financial_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    
    # Confusion Matrix
    cm_fig = evaluator.plot_confusion_matrix(y_test, predictions, f"{model_name} - Confusion Matrix")
    cm_fig.write_html("confusion_matrix.html")
    
    # ROC Curve
    roc_fig = evaluator.plot_roc_curve(y_test, probabilities, f"{model_name} - ROC Curve")
    roc_fig.write_html("roc_curve.html")
    
    # Cumulative Returns
    returns_fig = evaluator.plot_cumulative_returns(
        cumulative_returns, pd.to_datetime(test_dates), f"{model_name} - Cumulative Returns"
    )
    returns_fig.write_html("cumulative_returns.html")
    
    print("✅ Visualizations saved as HTML files")
    print("✅ Comprehensive evaluation completed!")
    
    return {
        'classification_metrics': class_metrics,
        'financial_metrics': financial_metrics,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns
    }

if __name__ == "__main__":
    comprehensive_model_evaluation() 