import pandas as pd
import numpy as np
from data_provider import YahooFinanceProvider
from feature_engineering import (
    add_price_range, add_price_change, add_moving_average, 
    add_rsi, add_volatility, add_momentum
)

def validate_features_with_sample_data():
    """
    Validate feature calculations with real sample data.
    """
    print("Fetching sample data for feature validation...")
    provider = YahooFinanceProvider()
    df = provider.fetch_historical_data("AAPL", "2023-01-01", "2023-03-01")
    
    print(f"Original data shape: {df.shape}")
    print("Original columns:", df.columns.tolist())
    
    # Add all features
    df = add_price_range(df)
    df = add_price_change(df)
    df = add_moving_average(df, window=5)
    df = add_moving_average(df, window=10)
    df = add_moving_average(df, window=20)
    df = add_rsi(df, window=14)
    df = add_volatility(df, window=10)
    df = add_momentum(df, window=10)
    
    print(f"Data shape after features: {df.shape}")
    print("New columns:", [col for col in df.columns if col not in ["timestamp", "open", "high", "low", "close", "volume"]])
    
    # Validate calculations
    print("\n--- Feature Validation ---")
    
    # Check price range
    manual_range = df["high"] - df["low"]
    assert np.allclose(df["price_range"], manual_range, equal_nan=True), "Price range calculation error"
    print("✓ Price range calculation verified")
    
    # Check price change
    manual_change = df["close"] - df["open"]
    assert np.allclose(df["price_change"], manual_change, equal_nan=True), "Price change calculation error"
    print("✓ Price change calculation verified")
    
    # Check moving average
    manual_ma5 = df["close"].rolling(window=5).mean()
    assert np.allclose(df["ma_5"], manual_ma5, equal_nan=True), "MA5 calculation error"
    print("✓ Moving average calculation verified")
    
    # Check RSI bounds
    rsi_values = df["rsi_14"].dropna()
    assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI values out of bounds"
    print("✓ RSI values within valid range [0, 100]")
    
    # Check for NaN handling
    print(f"\nNaN counts per feature:")
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaNs")
    
    # Sample output
    print(f"\nSample data (last 5 rows):")
    print(df[["timestamp", "close", "ma_5", "ma_10", "rsi_14", "volatility_10"]].tail())
    
    print("\n✅ All feature validations passed!")
    return df

if __name__ == "__main__":
    validate_features_with_sample_data() 