import pandas as pd

def calculate_sma(data: pd.DataFrame, window: int, price_column: str = 'Close') -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA) for a given price series.

    Args:
        data (pd.DataFrame): DataFrame containing the price data.
        window (int): The rolling window period for the SMA.
        price_column (str): The name of the column containing the price data (default: 'Close').

    Returns:
        pd.Series: A Series containing the SMA values.
                   Returns an empty Series if the price_column is not found or data is insufficient.
    """
    if price_column not in data.columns:
        print(f"Error: Price column '{price_column}' not found in DataFrame.")
        return pd.Series(dtype=float)
    
    if len(data) < window:
        print(f"Error: Data length ({len(data)}) is less than SMA window ({window}).")
        return pd.Series(dtype=float)
        
    return data[price_column].rolling(window=window).mean()

def calculate_ema(data: pd.DataFrame, window: int, price_column: str = 'Close') -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA) for a given price series.

    Args:
        data (pd.DataFrame): DataFrame containing the price data.
        window (int): The smoothing period for the EMA.
        price_column (str): The name of the column containing the price data (default: 'Close').

    Returns:
        pd.Series: A Series containing the EMA values.
                   Returns an empty Series if the price_column is not found or data is insufficient.
    """
    if price_column not in data.columns:
        print(f"Error: Price column '{price_column}' not found in DataFrame.")
        return pd.Series(dtype=float)

    if len(data) < window: # EMA calculation might need more data points for stability depending on adjust parameter
        print(f"Warning: Data length ({len(data)}) might be insufficient for a stable EMA ({window}).")
        # EMA can technically be calculated with len(data) >= 1, but it's often less meaningful with very short series.
        # yfinance itself needs at least window periods.
        # For simplicity, we'll allow it but with a warning, or enforce len(data) >= window if stricter.
        # Let's enforce it for now to match SMA's behavior for this project.
        return pd.Series(dtype=float)

    return data[price_column].ewm(span=window, adjust=False).mean()

if __name__ == '__main__':
    # Create sample data for testing
    sample_dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                   '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'])
    sample_prices = [10, 12, 11, 13, 14, 15, 16, 17, 18, 19]
    sample_df = pd.DataFrame({'Date': sample_dates, 'Close': sample_prices})
    sample_df.set_index('Date', inplace=True)

    # Test SMA
    sma_5 = calculate_sma(sample_df, window=5)
    print("SMA (5-day):\n", sma_5)

    sma_3 = calculate_sma(sample_df, window=3)
    print("\nSMA (3-day):\n", sma_3)
    
    # Test with a non-existent column for SMA
    sma_error_col = calculate_sma(sample_df, window=3, price_column='NonExistent')
    print("\nSMA (error column):\n", sma_error_col)

    # Test with insufficient data for SMA
    sma_error_len = calculate_sma(sample_df.head(2), window=3)
    print("\nSMA (insufficient data):\n", sma_error_len)

    # Test EMA
    ema_5 = calculate_ema(sample_df, window=5)
    print("\nEMA (5-period):\n", ema_5)

    ema_3 = calculate_ema(sample_df, window=3)
    print("\nEMA (3-period):\n", ema_3)

    # Test with a non-existent column for EMA
    ema_error_col = calculate_ema(sample_df, window=3, price_column='NonExistent')
    print("\nEMA (error column):\n", ema_error_col)

    # Test with insufficient data for EMA
    ema_error_len = calculate_ema(sample_df.head(2), window=3) # Using window 3 for a 2-row df
    print("\nEMA (insufficient data):\n", ema_error_len)

def calculate_rsi(data: pd.DataFrame, window: int = 14, price_column: str = 'Close') -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        data (pd.DataFrame): DataFrame containing the price data.
        window (int): The period for RSI calculation (default: 14).
        price_column (str): The name of the column containing the price data (default: 'Close').

    Returns:
        pd.Series: A Series containing the RSI values.
                   Returns an empty Series if the price_column is not found or data is insufficient.
    """
    if price_column not in data.columns:
        print(f"Error: Price column '{price_column}' not found in DataFrame.")
        return pd.Series(dtype=float)

    if len(data) <= window: # RSI needs at least 'window' periods of differences, so 'window + 1' data points.
        print(f"Error: Data length ({len(data)}) is insufficient for RSI window ({window}). Needs > window entries.")
        return pd.Series(dtype=float)

    delta = data[price_column].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

if __name__ == '__main__':
    # ... (previous test code for SMA and EMA remains the same)
    # Create sample data for testing
    sample_dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', 
                                   '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                   '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'])
    sample_prices = [10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 19, 18, 17, 16] # Extended for RSI
    sample_df = pd.DataFrame({'Date': sample_dates, 'Close': sample_prices})
    sample_df.set_index('Date', inplace=True)

    # Test SMA
    sma_5 = calculate_sma(sample_df, window=5)
    print("SMA (5-day):\n", sma_5)

    sma_3 = calculate_sma(sample_df, window=3)
    print("\nSMA (3-day):\n", sma_3)
    
    # Test with a non-existent column for SMA
    sma_error_col = calculate_sma(sample_df, window=3, price_column='NonExistent')
    print("\nSMA (error column):\n", sma_error_col)

    # Test with insufficient data for SMA
    sma_error_len = calculate_sma(sample_df.head(2), window=3)
    print("\nSMA (insufficient data):\n", sma_error_len)

    # Test EMA
    ema_5 = calculate_ema(sample_df, window=5)
    print("\nEMA (5-period):\n", ema_5)

    ema_3 = calculate_ema(sample_df, window=3)
    print("\nEMA (3-period):\n", ema_3)

    # Test with a non-existent column for EMA
    ema_error_col = calculate_ema(sample_df, window=3, price_column='NonExistent')
    print("\nEMA (error column):\n", ema_error_col)

    # Test with insufficient data for EMA
    ema_error_len = calculate_ema(sample_df.head(2), window=3) # Using window 3 for a 2-row df
    print("\nEMA (insufficient data):\n", ema_error_len)
    
    # Test RSI
    rsi_14 = calculate_rsi(sample_df, window=5) # Using a smaller window for sample data
    print("\nRSI (5-period):\n", rsi_14)

    rsi_default = calculate_rsi(sample_df.head(10)) # Test with default window 14, needs more data
    print("\nRSI (14-period with limited data - expecting error or NaNs):\n", rsi_default)
    
    rsi_full_data_14 = calculate_rsi(sample_df, window=14)
    print("\nRSI (14-period with full data):\n", rsi_full_data_14)


    # Test RSI with non-existent column
    rsi_error_col = calculate_rsi(sample_df, window=5, price_column='NonExistent')
    print("\nRSI (error column):\n", rsi_error_col)

    # Test RSI with insufficient data
    rsi_error_len = calculate_rsi(sample_df.head(5), window=10) # 5 data points, window 10
    print("\nRSI (insufficient data):\n", rsi_error_len)
