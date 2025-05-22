import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
sys.path.append('..')

# Import strategy and data loading utilities
from code.strategies.enhanced_1h_strategy import Strategy
from code.utilities.data_manager import DataManager

def load_ohlcv_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load OHLCV data for a given symbol and date range.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data
    """
    # Convert dates to datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Load data from CSV file
    data_path = Path(__file__).parent.parent.parent / 'data' / f'{symbol}_1h.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}. Please download the data first.")
    
    # Read the CSV file
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    
    # Filter by date range
    df = df.loc[start_dt:end_dt]
    
    return df 

# Initialize data manager
data = DataManager(name="binance")

# Load data
symbol = "BTC/USDT:USDT"
start_date = "2023-01-01 00:00:00"
end_date = "2023-12-31 00:00:00"

# Load data
df = data.load(symbol, timeframe="1h", start_date=start_date, end_date=end_date)

print(f"Data shape: {df.shape}")
df.head() 