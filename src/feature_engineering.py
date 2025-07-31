# src/feature_engineering.py

import pandas as pd
import numpy as np
import ta

def create_features(df):
    """
    Engineers a comprehensive set of features for the BTCUSDT dataset.

    Args:
        df (pandas.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume', 'timestamp'.

    Returns:
        pandas.DataFrame: The DataFrame with all new features.
    """
    # Basic price and volatility features
    df['Price Change'] = df['close'] - df['open']
    df['Volatility'] = (df['high'] - df['low']) / df['open']
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_diff'] = df['close'] - df['open']

    # Rolling window features
    df['Rolling_Mean_Close'] = df['close'].rolling(window=5).mean()
    df['Rolling_Std_Close'] = df['close'].rolling(window=5).std()
    df['rolling_max_6h'] = df['close'].rolling(window=6).max()
    df['rolling_min_6h'] = df['close'].rolling(window=6).min()

    # Target variables for prediction
    df['Target_Close'] = df['close'].shift(-1)
    df['Target_Movement'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Volatility and return features
    window = 6
    df['vol_1h'] = df['high'] - df['low']
    df['vol_mean_6h'] = df['vol_1h'].rolling(window).mean()
    df['vol_std_6h']  = df['vol_1h'].rolling(window).std()
    df['vol_max_6h']  = df['vol_1h'].rolling(window).max()
    df['vol_min_6h']  = df['vol_1h'].rolling(window).min()
    df['return_1h'] = df['close'].pct_change().shift(-1)
    df['return_mean_6h'] = df['return_1h'].rolling(window).mean()
    df['return_std_6h'] = df['return_1h'].rolling(window).std()

    # Time-based and cyclical features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['day'] = df['timestamp'].dt.day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Technical indicators using the 'ta' library
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    bb = ta.volatility.BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_30'] = df['close'].ewm(span=30).mean()

    # Lag and interaction features
    df['close_lag_1'] = df['close'].shift(1)
    df['volume_lag_1'] = df['volume'].shift(1)
    df['price_volatility_interaction'] = df['close'] * df['Volatility']
    
    # Drop rows with NaN values created by feature engineering
    df.dropna(inplace=True)

    return df