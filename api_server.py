from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback
from data_provider import YahooFinanceProvider
from feature_engineering import (
    add_price_range, add_price_change, add_moving_average, 
    add_rsi, add_volatility, add_momentum
)
from data_preprocessing import preprocess_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PredictionAPI:
    """
    REST API for financial price prediction model.
    """
    
    def __init__(self):
        self.model = None
        self.model_info = None
        self.feature_columns = [
            "close", "volume", "price_range", "price_change",
            "ma_5", "ma_10", "ma_20", "rsi_14", "volatility_10", "momentum_10"
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            with open('best_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_info = {
                'model_name': model_data['model_name'],
                'loaded_at': datetime.now().isoformat(),
                'features': self.feature_columns
            }
            logger.info(f"Model loaded successfully: {model_data['model_name']}")
            
        except FileNotFoundError:
            logger.error("Model file not found. Please train a model first.")
            self.model = None
            self.model_info = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.model_info = None
    
    def prepare_features(self, symbol, days_back=60):
        """
        Fetch and prepare features for prediction.
        """
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back + 30)).strftime('%Y-%m-%d')
            
            # Fetch data
            provider = YahooFinanceProvider()
            df = provider.fetch_historical_data(symbol, start_date, end_date)
            
            if df.empty:
                raise ValueError(f"No data available for symbol {symbol}")
            
            # Add features
            df = add_price_range(df)
            df = add_price_change(df)
            df = add_moving_average(df, window=5)
            df = add_moving_average(df, window=10)
            df = add_moving_average(df, window=20)
            df = add_rsi(df, window=14)
            df = add_volatility(df, window=10)
            df = add_momentum(df, window=10)
            
            # Preprocess for model input
            result = preprocess_data(df, self.feature_columns, lookback_window=30, test_size=0.0)
            
            if len(result['X_train']) == 0:
                raise ValueError("Insufficient data for prediction")
            
            # Get the most recent sequence for prediction
            latest_sequence = result['X_train'][-1:] if len(result['X_train']) > 0 else None
            current_price = df['close'].iloc[-1]
            
            return latest_sequence, current_price, df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def predict(self, symbol):
        """
        Make prediction for a given symbol.
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Prepare features
            X, current_price, df = self.prepare_features(symbol)
            
            if X is None:
                raise ValueError("Could not prepare features for prediction")
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Get additional context
            latest_data = df.iloc[-1]
            
            result = {
                'symbol': symbol,
                'prediction': int(prediction),
                'prediction_label': 'UP' if prediction == 1 else 'DOWN',
                'probability_up': float(probability[1]),
                'probability_down': float(probability[0]),
                'confidence': float(max(probability)),
                'current_price': float(current_price),
                'prediction_timestamp': datetime.now().isoformat(),
                'model_name': self.model_info['model_name'],
                'latest_data': {
                    'date': latest_data['timestamp'].isoformat() if pd.notnull(latest_data['timestamp']) else None,
                    'close': float(latest_data['close']),
                    'volume': float(latest_data['volume']),
                    'rsi_14': float(latest_data['rsi_14']) if pd.notnull(latest_data['rsi_14']) else None,
                    'ma_20': float(latest_data['ma_20']) if pd.notnull(latest_data['ma_20']) else None
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

# Initialize API
prediction_api = PredictionAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': prediction_api.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    if prediction_api.model_info is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(prediction_api.model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction for a stock symbol.
    
    Request body:
    {
        "symbol": "AAPL"
    }
    """
    try:
        # Validate request
        if not request.json or 'symbol' not in request.json:
            return jsonify({'error': 'Symbol is required'}), 400
        
        symbol = request.json['symbol'].upper()
        
        # Make prediction
        result = prediction_api.predict(symbol)
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/<symbol>', methods=['GET'])
def predict_get(symbol):
    """
    Make prediction for a stock symbol (GET endpoint).
    """
    try:
        symbol = symbol.upper()
        result = prediction_api.predict(symbol)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple stock symbols.
    
    Request body:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"]
    }
    """
    try:
        # Validate request
        if not request.json or 'symbols' not in request.json:
            return jsonify({'error': 'Symbols list is required'}), 400
        
        symbols = [s.upper() for s in request.json['symbols']]
        
        if len(symbols) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 symbols allowed per batch'}), 400
        
        results = []
        errors = []
        
        for symbol in symbols:
            try:
                result = prediction_api.predict(symbol)
                results.append(result)
            except Exception as e:
                errors.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': results,
            'errors': errors,
            'total_requested': len(symbols),
            'successful': len(results),
            'failed': len(errors)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload the model."""
    try:
        prediction_api.load_model()
        if prediction_api.model is None:
            return jsonify({'error': 'Failed to reload model'}), 500
        
        return jsonify({
            'message': 'Model reloaded successfully',
            'model_info': prediction_api.model_info
        })
        
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Financial Prediction API Server...")
    print("📡 API endpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /model/info          - Model information")
    print("  POST /predict             - Single prediction")
    print("  GET  /predict/<symbol>    - Single prediction (GET)")
    print("  POST /batch_predict       - Batch predictions")
    print("  POST /model/reload        - Reload model")
    print("\n📊 Server will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 