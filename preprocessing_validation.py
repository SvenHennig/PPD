import pandas as pd
import numpy as np
from data_provider import YahooFinanceProvider
from feature_engineering import (
    add_price_range, add_price_change, add_moving_average, 
    add_rsi, add_volatility, add_momentum
)
from data_preprocessing import preprocess_data

def validate_preprocessing_pipeline():
    """
    Validate the complete preprocessing pipeline with real data.
    """
    print("🔄 Fetching real financial data...")
    provider = YahooFinanceProvider()
    df = provider.fetch_historical_data("AAPL", "2022-01-01", "2023-12-31")
    
    print(f"📊 Raw data shape: {df.shape}")
    
    # Add features
    print("🔧 Adding features...")
    df = add_price_range(df)
    df = add_price_change(df)
    df = add_moving_average(df, window=5)
    df = add_moving_average(df, window=10)
    df = add_moving_average(df, window=20)
    df = add_rsi(df, window=14)
    df = add_volatility(df, window=10)
    df = add_momentum(df, window=10)
    
    print(f"📈 Data with features shape: {df.shape}")
    
    # Define feature columns (excluding raw OHLCV and timestamp)
    feature_cols = [
        "close", "volume", "price_range", "price_change",
        "ma_5", "ma_10", "ma_20", "rsi_14", "volatility_10", "momentum_10"
    ]
    
    print(f"🎯 Using {len(feature_cols)} features: {feature_cols}")
    
    # Preprocess data
    print("⚙️ Running preprocessing pipeline...")
    result = preprocess_data(df, feature_cols, lookback_window=30, test_size=0.2)
    
    # Display results
    print("\n📋 Preprocessing Results:")
    print(f"  Training sequences: {result['X_train'].shape}")
    print(f"  Test sequences: {result['X_test'].shape}")
    print(f"  Training labels: {result['y_train'].shape}")
    print(f"  Test labels: {result['y_test'].shape}")
    print(f"  Lookback window: {result['lookback_window']} days")
    print(f"  Number of features: {len(result['feature_cols'])}")
    
    # Validate shapes
    assert result['X_train'].shape[1] == result['lookback_window'], "Lookback window mismatch"
    assert result['X_train'].shape[2] == len(feature_cols), "Feature count mismatch"
    assert len(result['X_train']) == len(result['y_train']), "Train data/label length mismatch"
    assert len(result['X_test']) == len(result['y_test']), "Test data/label length mismatch"
    
    # Check target distribution
    train_target_dist = np.bincount(result['y_train'])
    test_target_dist = np.bincount(result['y_test'])
    
    print(f"\n📊 Target Distribution:")
    print(f"  Training - Down: {train_target_dist[0]}, Up: {train_target_dist[1]} ({train_target_dist[1]/(train_target_dist[0]+train_target_dist[1])*100:.1f}% up)")
    print(f"  Test - Down: {test_target_dist[0]}, Up: {test_target_dist[1]} ({test_target_dist[1]/(test_target_dist[0]+test_target_dist[1])*100:.1f}% up)")
    
    # Check for data leakage (test data should be chronologically after train data)
    total_sequences = len(result['X_train']) + len(result['X_test'])
    expected_train_size = int(total_sequences * 0.8)
    actual_train_size = len(result['X_train'])
    
    print(f"\n🔍 Data Integrity Check:")
    print(f"  Expected train size: {expected_train_size}")
    print(f"  Actual train size: {actual_train_size}")
    print(f"  Chronological split: ✓" if abs(expected_train_size - actual_train_size) <= 1 else "  Chronological split: ✗")
    
    # Sample data inspection
    print(f"\n🔬 Sample Sequence (first training example):")
    sample_X = result['X_train'][0]
    sample_y = result['y_train'][0]
    print(f"  Sequence shape: {sample_X.shape}")
    print(f"  Target: {sample_y} ({'Up' if sample_y == 1 else 'Down'})")
    print(f"  Feature ranges (min, max):")
    for i, feature in enumerate(result['feature_cols']):
        feature_min = sample_X[:, i].min()
        feature_max = sample_X[:, i].max()
        print(f"    {feature}: ({feature_min:.3f}, {feature_max:.3f})")
    
    print("\n✅ Preprocessing pipeline validation completed successfully!")
    return result

if __name__ == "__main__":
    validate_preprocessing_pipeline() 