import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

def create_target_variable(df: pd.DataFrame, target_col: str = "close") -> pd.DataFrame:
    """
    Create binary target variable: 1 if next period's close > current close, else 0
    """
    df = df.copy()
    df["target"] = (df[target_col].shift(-1) > df[target_col]).astype(int)
    # Remove last row as it has no future data
    df = df[:-1]
    return df

def normalize_features(df: pd.DataFrame, feature_cols: List[str], scaler=None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize features using MinMaxScaler
    """
    df = df.copy()
    if scaler is None:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler

def create_sequences(df: pd.DataFrame, feature_cols: List[str], target_col: str, lookback_window: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time series sequences for LSTM training
    """
    # Remove rows with NaN values
    df_clean = df[feature_cols + [target_col]].dropna()
    
    X, y = [], []
    for i in range(lookback_window, len(df_clean)):
        X.append(df_clean[feature_cols].iloc[i-lookback_window:i].values)
        y.append(df_clean[target_col].iloc[i])
    
    return np.array(X), np.array(y)

def chronological_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data chronologically (no shuffling)
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

def preprocess_data(df: pd.DataFrame, feature_cols: List[str], lookback_window: int = 30, test_size: float = 0.2) -> dict:
    """
    Complete preprocessing pipeline
    """
    # Create target variable
    df_with_target = create_target_variable(df)
    
    # Normalize features
    df_normalized, scaler = normalize_features(df_with_target, feature_cols)
    
    # Create sequences
    X, y = create_sequences(df_normalized, feature_cols, "target", lookback_window)
    
    # Split data
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "lookback_window": lookback_window
    }

# Test functions

def test_target_variable_creation():
    df = pd.DataFrame({
        "close": [1, 2, 3, 4, 5],
        "volume": [100, 200, 300, 400, 500]
    })
    df_target = create_target_variable(df)
    expected_targets = [1, 1, 1, 1]  # Each next close is higher
    assert list(df_target["target"]) == expected_targets, "Target variable creation failed"
    assert len(df_target) == len(df) - 1, "Target dataframe length incorrect"
    print("✓ Target variable creation test passed")

def test_normalization():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    })
    feature_cols = ["feature1", "feature2"]
    df_norm, scaler = normalize_features(df, feature_cols)
    
    # Check if values are between 0 and 1
    assert df_norm[feature_cols].min().min() >= 0, "Normalized values below 0"
    assert df_norm[feature_cols].max().max() <= 1, "Normalized values above 1"
    print("✓ Normalization test passed")

def test_sequence_creation():
    df = pd.DataFrame({
        "feature1": range(10),
        "feature2": range(10, 20),
        "target": [0, 1] * 5
    })
    feature_cols = ["feature1", "feature2"]
    X, y = create_sequences(df, feature_cols, "target", lookback_window=3)
    
    assert X.shape == (7, 3, 2), f"X shape incorrect: {X.shape}"
    assert y.shape == (7,), f"y shape incorrect: {y.shape}"
    print("✓ Sequence creation test passed")

def test_chronological_split():
    X = np.random.rand(100, 10, 5)
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=0.2)
    
    assert len(X_train) == 80, f"Train set size incorrect: {len(X_train)}"
    assert len(X_test) == 20, f"Test set size incorrect: {len(X_test)}"
    assert len(y_train) == 80, f"Train labels size incorrect: {len(y_train)}"
    assert len(y_test) == 20, f"Test labels size incorrect: {len(y_test)}"
    print("✓ Chronological split test passed")

def test_full_preprocessing_pipeline():
    # Create sample data with features
    df = pd.DataFrame({
        "close": np.random.rand(100) * 100 + 50,
        "volume": np.random.rand(100) * 1000000,
        "ma_5": np.random.rand(100) * 100 + 50,
        "rsi_14": np.random.rand(100) * 100,
        "volatility_10": np.random.rand(100) * 10
    })
    
    feature_cols = ["close", "volume", "ma_5", "rsi_14", "volatility_10"]
    result = preprocess_data(df, feature_cols, lookback_window=10, test_size=0.2)
    
    assert "X_train" in result, "X_train missing from result"
    assert "X_test" in result, "X_test missing from result"
    assert "y_train" in result, "y_train missing from result"
    assert "y_test" in result, "y_test missing from result"
    assert "scaler" in result, "scaler missing from result"
    
    print("✓ Full preprocessing pipeline test passed")

if __name__ == "__main__":
    test_target_variable_creation()
    test_normalization()
    test_sequence_creation()
    test_chronological_split()
    test_full_preprocessing_pipeline()
    print("\n✅ All preprocessing tests passed!") 