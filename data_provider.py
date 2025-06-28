import pandas as pd
import requests
from datetime import datetime
from typing import Optional

class DataProvider:
    """
    Abstract base class for market data providers.
    """
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> bool:
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        return all(col in df.columns for col in required_columns)

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

class YahooFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider using yfinance library.
    """
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        import yfinance as yf
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        data = data.reset_index()
        
        # Handle multi-level columns by flattening them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
        
        data = data.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df = data[["timestamp", "open", "high", "low", "close", "volume"]]
        df = self.handle_missing_values(df)
        assert self.validate_schema(df), "Data schema validation failed."
        return df

# Basic test functions

def test_yahoo_finance_provider():
    provider = YahooFinanceProvider()
    df = provider.fetch_historical_data("AAPL", "2023-01-01", "2023-01-31")
    assert not df.empty, "DataFrame is empty."
    assert YahooFinanceProvider.validate_schema(df), "Schema validation failed."
    print("YahooFinanceProvider test passed.")

if __name__ == "__main__":
    test_yahoo_finance_provider() 