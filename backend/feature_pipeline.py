from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import ta

FEATURE_COLS = [
    'Close_norm', 'High_norm', 'Low_norm', 'Open_norm',
    'Volume_log', 'SMA_7', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7',
    'Daily_Return', 'Close_to_Open', 'std_5',
    'Price_ROC', 'Relative_Volume'
]

TARGET_COLS = [f'target_day_{i}' for i in range(1, 8)]


def compute_features(df: pd.DataFrame, include_targets: bool = False) -> pd.DataFrame:
    """
    Compute rolling-normalized features and technical indicators as established in the notebook.
    """
    if df.empty or len(df) < 10:
        return pd.DataFrame()

    df = df.copy()

    # Ensure required columns exist
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            raise ValueError(f"Missing required price column: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 252-day Rolling Normalization (1 trading year)
    # Use min_periods=1 so shorter histories are still handled gracefully
    rolling_mean = df['Close'].rolling(window=252, min_periods=1).mean()
    df['rolling_mean'] = rolling_mean

    # Price normalization
    df['Close_norm'] = (df['Close'] / rolling_mean) * 100.0
    df['High_norm']  = (df['High'] / rolling_mean) * 100.0
    df['Low_norm']   = (df['Low'] / rolling_mean) * 100.0
    df['Open_norm']  = (df['Open'] / rolling_mean) * 100.0

    # Moving averages on normalized price
    df['SMA_7']  = df['Close_norm'].rolling(7, min_periods=1).mean()
    df['EMA_12'] = df['Close_norm'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close_norm'].ewm(span=26, adjust=False).mean()

    # RSI indicator (using original close price)
    try:
        rsi_series = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    except Exception:
        rsi_series = pd.Series(50.0, index=df.index)
    df['RSI'] = rsi_series.fillna(50.0)

    # MACD on normalized price
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # Lag features
    df['Lag_1'] = df['Close_norm'].shift(1)
    df['Lag_2'] = df['Close_norm'].shift(2)
    df['Lag_3'] = df['Close_norm'].shift(3)
    df['Lag_7'] = df['Close_norm'].shift(7)

    # Return & Momentum
    df['Daily_Return']  = df['Close'].pct_change()
    df['Close_to_Open'] = df['Close_norm'] - df['Open_norm']
    df['std_5']         = df['Close_norm'].rolling(5, min_periods=1).std()
    df['Price_ROC']     = df['Close_norm'].pct_change(5)

    # Volume features
    df['Volume_log'] = np.log1p(np.maximum(df['Volume'], 0))
    vol_20_mean = df['Volume'].rolling(20, min_periods=1).mean()
    df['Relative_Volume'] = np.where(vol_20_mean > 0, df['Volume'] / vol_20_mean, 1.0)

    # Clean forward/back fill for feature columns
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill().fillna(0.0)

    # Optional targets for training
    if include_targets:
        for day in range(1, 8):
            df[f'target_day_{day}'] = df['Close_norm'].shift(-day)
        df.dropna(subset=TARGET_COLS, inplace=True)

    return df


def add_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add full suite of technical indicators for frontend visual charting."""
    if df.empty:
        return df

    df = df.copy()

    # Moving averages (20, 50, 200)
    df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['SMA_200'] = df['Close'].rolling(200, min_periods=1).mean()

    # Bollinger Bands
    try:
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
    except Exception:
        df['BB_upper'] = df['SMA_20'] * 1.05
        df['BB_middle'] = df['SMA_20']
        df['BB_lower'] = df['SMA_20'] * 0.95

    # MACD standard
    try:
        macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD_line'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
    except Exception:
        df['MACD_line'] = 0.0
        df['MACD_signal'] = 0.0
        df['MACD_hist'] = 0.0

    # RSI
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    except Exception:
        df['RSI'] = 50.0

    return df
