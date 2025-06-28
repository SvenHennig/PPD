import pandas as pd
import numpy as np

def add_price_range(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_range"] = df["high"] - df["low"]
    return df

def add_price_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_change"] = df["close"] - df["open"]
    return df

def add_moving_average(df: pd.DataFrame, window: int = 5, col: str = "close") -> pd.DataFrame:
    df = df.copy()
    df[f"ma_{window}"] = df[col].rolling(window=window).mean()
    return df

def add_rsi(df: pd.DataFrame, window: int = 14, col: str = "close") -> pd.DataFrame:
    df = df.copy()
    delta = df[col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return df

def add_volatility(df: pd.DataFrame, window: int = 10, col: str = "close") -> pd.DataFrame:
    df = df.copy()
    df[f"volatility_{window}"] = df[col].rolling(window=window).std()
    return df

def add_momentum(df: pd.DataFrame, window: int = 10, col: str = "close") -> pd.DataFrame:
    df = df.copy()
    df[f"momentum_{window}"] = df[col] - df[col].shift(window)
    return df

# Test functions

def test_feature_engineering():
    df = pd.DataFrame({
        "open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "high": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "low": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    })
    df = add_price_range(df)
    assert "price_range" in df.columns, "price_range not added"
    df = add_price_change(df)
    assert "price_change" in df.columns, "price_change not added"
    df = add_moving_average(df, window=3)
    assert "ma_3" in df.columns, "ma_3 not added"
    df = add_rsi(df, window=3)
    assert "rsi_3" in df.columns, "rsi_3 not added"
    df = add_volatility(df, window=3)
    assert "volatility_3" in df.columns, "volatility_3 not added"
    df = add_momentum(df, window=3)
    assert "momentum_3" in df.columns, "momentum_3 not added"
    print("Feature engineering tests passed.")

if __name__ == "__main__":
    test_feature_engineering() 