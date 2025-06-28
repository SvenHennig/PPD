import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from data_provider import YahooFinanceProvider
from feature_engineering import (
    add_price_range, add_price_change, add_moving_average, 
    add_rsi, add_volatility, add_momentum
)
from data_preprocessing import preprocess_data
from evaluation_metrics import ModelEvaluator

class FinancialDashboard:
    """
    Interactive dashboard for financial prediction model visualization.
    """
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.evaluator = ModelEvaluator()
        self.setup_layout()
        self.setup_callbacks()
    
    def load_model_and_data(self, symbol="AAPL", start_date="2022-01-01", end_date="2023-12-31"):
        """Load model and prepare data for visualization."""
        # Load best model
        with open('best_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Load training results
        with open('training_results.json', 'r') as f:
            training_results = json.load(f)
        
        # Fetch and prepare data
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
        
        feature_cols = [
            "close", "volume", "price_range", "price_change",
            "ma_5", "ma_10", "ma_20", "rsi_14", "volatility_10", "momentum_10"
        ]
        
        result = preprocess_data(df, feature_cols, lookback_window=30, test_size=0.2)
        
        return model_data, training_results, df, result
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Financial Price Prediction Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Stock Symbol:"),
                    dcc.Input(id='symbol-input', value='AAPL', type='text'),
                ], style={'width': '20%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Start Date:"),
                    dcc.DatePickerSingle(
                        id='start-date-picker',
                        date='2022-01-01'
                    ),
                ], style={'width': '20%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("End Date:"),
                    dcc.DatePickerSingle(
                        id='end-date-picker',
                        date='2023-12-31'
                    ),
                ], style={'width': '20%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Button('Update Dashboard', id='update-button', n_clicks=0,
                               style={'marginTop': 25})
                ], style={'width': '20%', 'display': 'inline-block'}),
            ], style={'marginBottom': 30}),
            
            # Model Performance Overview
            html.Div([
                html.H2("Model Performance Overview"),
                html.Div(id='performance-cards')
            ], style={'marginBottom': 30}),
            
            # Charts
            html.Div([
                # Price and Predictions
                html.Div([
                    dcc.Graph(id='price-predictions-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # Cumulative Returns
                html.Div([
                    dcc.Graph(id='cumulative-returns-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                # Feature Importance
                html.Div([
                    dcc.Graph(id='feature-importance-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # Confusion Matrix
                html.Div([
                    dcc.Graph(id='confusion-matrix-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                # Training History
                html.Div([
                    dcc.Graph(id='training-history-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # ROC Curve
                html.Div([
                    dcc.Graph(id='roc-curve-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output('performance-cards', 'children'),
             Output('price-predictions-chart', 'figure'),
             Output('cumulative-returns-chart', 'figure'),
             Output('feature-importance-chart', 'figure'),
             Output('confusion-matrix-chart', 'figure'),
             Output('training-history-chart', 'figure'),
             Output('roc-curve-chart', 'figure')],
            [Input('update-button', 'n_clicks')],
            [dash.dependencies.State('symbol-input', 'value'),
             dash.dependencies.State('start-date-picker', 'date'),
             dash.dependencies.State('end-date-picker', 'date')]
        )
        def update_dashboard(n_clicks, symbol, start_date, end_date):
            try:
                # Load data
                model_data, training_results, df, result = self.load_model_and_data(
                    symbol, start_date, end_date
                )
                
                best_model = model_data['model']
                model_name = model_data['model_name']
                X_test, y_test = result['X_test'], result['y_test']
                
                # Make predictions
                predictions = best_model.predict(X_test)
                probabilities = best_model.predict_proba(X_test)[:, 1]
                
                # Get test data for visualization
                test_start_idx = len(result['X_train']) + 30
                test_prices = df['close'].iloc[test_start_idx:test_start_idx + len(X_test) + 1].values
                test_dates = df['timestamp'].iloc[test_start_idx:test_start_idx + len(X_test)].values
                
                # Calculate metrics
                class_metrics = self.evaluator.evaluate_classification_metrics(
                    y_test, predictions, probabilities
                )
                financial_metrics, strategy_returns, cumulative_returns = self.evaluator.backtest_strategy(
                    predictions, test_prices, pd.to_datetime(test_dates)
                )
                
                # Create components
                performance_cards = self.create_performance_cards(class_metrics, financial_metrics)
                price_chart = self.create_price_predictions_chart(test_dates, test_prices, predictions, y_test)
                returns_chart = self.create_cumulative_returns_chart(test_dates, cumulative_returns)
                feature_chart = self.create_feature_importance_chart(best_model, model_name)
                confusion_chart = self.evaluator.plot_confusion_matrix(y_test, predictions, f"{model_name} - Confusion Matrix")
                training_chart = self.create_training_history_chart(training_results)
                roc_chart = self.evaluator.plot_roc_curve(y_test, probabilities, f"{model_name} - ROC Curve")
                
                return (performance_cards, price_chart, returns_chart, feature_chart, 
                       confusion_chart, training_chart, roc_chart)
                
            except Exception as e:
                # Return empty charts on error
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
                return (html.Div("Error loading data"), empty_fig, empty_fig, empty_fig, 
                       empty_fig, empty_fig, empty_fig)
    
    def create_performance_cards(self, class_metrics, financial_metrics):
        """Create performance metric cards."""
        cards = []
        
        # Classification metrics
        cards.append(html.Div([
            html.H4("Accuracy"),
            html.H2(f"{class_metrics['accuracy']:.1%}")
        ], style={'textAlign': 'center', 'backgroundColor': '#f0f0f0', 'padding': '20px', 
                 'margin': '10px', 'borderRadius': '5px', 'width': '15%', 'display': 'inline-block'}))
        
        cards.append(html.Div([
            html.H4("F1 Score"),
            html.H2(f"{class_metrics['f1_score']:.3f}")
        ], style={'textAlign': 'center', 'backgroundColor': '#f0f0f0', 'padding': '20px', 
                 'margin': '10px', 'borderRadius': '5px', 'width': '15%', 'display': 'inline-block'}))
        
        # Financial metrics
        cards.append(html.Div([
            html.H4("Total Return"),
            html.H2(f"{financial_metrics['total_return']:.1%}")
        ], style={'textAlign': 'center', 'backgroundColor': '#e8f5e8', 'padding': '20px', 
                 'margin': '10px', 'borderRadius': '5px', 'width': '15%', 'display': 'inline-block'}))
        
        cards.append(html.Div([
            html.H4("Sharpe Ratio"),
            html.H2(f"{financial_metrics['sharpe_ratio']:.2f}")
        ], style={'textAlign': 'center', 'backgroundColor': '#e8f5e8', 'padding': '20px', 
                 'margin': '10px', 'borderRadius': '5px', 'width': '15%', 'display': 'inline-block'}))
        
        cards.append(html.Div([
            html.H4("Max Drawdown"),
            html.H2(f"{financial_metrics['max_drawdown']:.1%}")
        ], style={'textAlign': 'center', 'backgroundColor': '#ffe8e8', 'padding': '20px', 
                 'margin': '10px', 'borderRadius': '5px', 'width': '15%', 'display': 'inline-block'}))
        
        cards.append(html.Div([
            html.H4("Win Rate"),
            html.H2(f"{financial_metrics['win_rate']:.1%}")
        ], style={'textAlign': 'center', 'backgroundColor': '#e8f0ff', 'padding': '20px', 
                 'margin': '10px', 'borderRadius': '5px', 'width': '15%', 'display': 'inline-block'}))
        
        return html.Div(cards)
    
    def create_price_predictions_chart(self, dates, prices, predictions, actual):
        """Create price chart with predictions."""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dates),
            y=prices[:-1],  # Exclude last price (no prediction for it)
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ))
        
        # Prediction markers
        buy_signals = pd.to_datetime(dates)[predictions == 1]
        buy_prices = prices[:-1][predictions == 1]
        
        fig.add_trace(go.Scatter(
            x=buy_signals,
            y=buy_prices,
            mode='markers',
            name='Buy Signals',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))
        
        # Actual direction markers
        up_days = pd.to_datetime(dates)[actual == 1]
        up_prices = prices[:-1][actual == 1]
        
        fig.add_trace(go.Scatter(
            x=up_days,
            y=up_prices,
            mode='markers',
            name='Actual Up Days',
            marker=dict(color='lightgreen', size=4, symbol='circle')
        ))
        
        fig.update_layout(
            title="Price Chart with Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True
        )
        
        return fig
    
    def create_cumulative_returns_chart(self, dates, cumulative_returns):
        """Create cumulative returns chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dates),
            y=cumulative_returns,
            mode='lines',
            name='Strategy Returns',
            line=dict(color='green', width=2)
        ))
        
        # Add buy and hold benchmark
        buy_hold_returns = np.linspace(1, cumulative_returns[-1], len(cumulative_returns))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dates),
            y=buy_hold_returns,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            showlegend=True
        )
        
        return fig
    
    def create_feature_importance_chart(self, model, model_name):
        """Create feature importance chart."""
        fig = go.Figure()
        
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                "close", "volume", "price_range", "price_change",
                "ma_5", "ma_10", "ma_20", "rsi_14", "volatility_10", "momentum_10"
            ]
            importances = model.feature_importances_
            
            fig.add_trace(go.Bar(
                x=feature_names,
                y=importances,
                name='Feature Importance'
            ))
            
            fig.update_layout(
                title=f"{model_name} - Feature Importance",
                xaxis_title="Features",
                yaxis_title="Importance",
                xaxis_tickangle=-45
            )
        else:
            fig.add_annotation(text="Feature importance not available for this model", 
                             x=0.5, y=0.5)
        
        return fig
    
    def create_training_history_chart(self, training_results):
        """Create training history comparison chart."""
        fig = go.Figure()
        
        models = list(training_results.keys())
        test_accuracies = [training_results[model]['test_accuracy'] for model in models]
        train_accuracies = [training_results[model]['train_accuracy'] for model in models]
        
        fig.add_trace(go.Bar(
            x=models,
            y=test_accuracies,
            name='Test Accuracy',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=models,
            y=train_accuracies,
            name='Train Accuracy',
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Model Comparison - Train vs Test Accuracy",
            xaxis_title="Models",
            yaxis_title="Accuracy",
            barmode='group'
        )
        
        return fig
    
    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        self.app.run(debug=debug, port=port, host='0.0.0.0')

def main():
    """Run the dashboard application."""
    dashboard = FinancialDashboard()
    print("🚀 Starting Financial Prediction Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    dashboard.run()

if __name__ == "__main__":
    main() 