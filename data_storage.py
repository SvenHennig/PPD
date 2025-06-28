import pandas as pd
from pathlib import Path
from typing import Optional

class DataStorage:
    """
    Handles saving and loading of raw OHLCV data.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def save_to_csv(self, df: pd.DataFrame, symbol: str, suffix: str = "raw") -> str:
        file_path = self.data_dir / f"{symbol}_{suffix}.csv"
        df.to_csv(file_path, index=False)
        return str(file_path)

    def load_from_csv(self, symbol: str, suffix: str = "raw") -> Optional[pd.DataFrame]:
        file_path = self.data_dir / f"{symbol}_{suffix}.csv"
        if not file_path.exists():
            return None
        return pd.read_csv(file_path, parse_dates=["timestamp"])

# Basic test functions

def test_data_storage():
    import numpy as np
    # Create dummy data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=3),
        "open": [1, 2, 3],
        "high": [2, 3, 4],
        "low": [0, 1, 2],
        "close": [1.5, 2.5, 3.5],
        "volume": [100, 200, 300]
    })
    storage = DataStorage("test_data")
    path = storage.save_to_csv(df, "TEST")
    loaded = storage.load_from_csv("TEST")
    assert loaded is not None, "Failed to load CSV."
    assert loaded.shape == df.shape, "Loaded data shape mismatch."
    assert (loaded["close"] == df["close"]).all(), "Data mismatch."
    print("DataStorage test passed.")

if __name__ == "__main__":
    test_data_storage() 