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

    price_data_source = data[price_column]
    if isinstance(price_data_source, pd.DataFrame):
        if not price_data_source.empty and price_data_source.shape[1] > 0:
            price_series = price_data_source.iloc[:, 0] # Use first column
        else: # Empty DataFrame or DataFrame with no columns
            print(f"Warning in SMA: Price data for '{price_column}' is an empty DataFrame.")
            return pd.Series(dtype=float)
    elif isinstance(price_data_source, pd.Series):
        price_series = price_data_source
    else:
        print(f"Error in SMA: Price data for '{price_column}' is not a Series or DataFrame. Type: {type(price_data_source)}")
        return pd.Series(dtype=float)

    if len(price_series) < window:
        print(f"Error: Data length ({len(price_series)}) is less than SMA window ({window}).")
        return pd.Series(dtype=float)

    return price_series.rolling(window=window).mean()

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

    price_data_source = data[price_column]
    if isinstance(price_data_source, pd.DataFrame):
        if not price_data_source.empty and price_data_source.shape[1] > 0:
            price_series = price_data_source.iloc[:, 0]
        else:
            print(f"Warning in EMA: Price data for '{price_column}' is an empty DataFrame.")
            return pd.Series(dtype=float)
    elif isinstance(price_data_source, pd.Series):
        price_series = price_data_source
    else:
        print(f"Error in EMA: Price data for '{price_column}' is not a Series or DataFrame. Type: {type(price_data_source)}")
        return pd.Series(dtype=float)

    if len(price_series) < window:
        print(f"Error: Data length ({len(price_series)}) is less than EMA window ({window}).") # Matched SMA's strictness
        return pd.Series(dtype=float)

    return price_series.ewm(span=window, adjust=False).mean()

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

    price_data_source = data[price_column]
    if isinstance(price_data_source, pd.DataFrame):
        if not price_data_source.empty and price_data_source.shape[1] > 0:
            price_series = price_data_source.iloc[:, 0]
        else:
            print(f"Warning in RSI: Price data for '{price_column}' is an empty DataFrame.")
            return pd.Series(dtype=float)
    elif isinstance(price_data_source, pd.Series):
        price_series = price_data_source
    else:
        print(f"Error in RSI: Price data for '{price_column}' is not a Series or DataFrame. Type: {type(price_data_source)}")
        return pd.Series(dtype=float)

    if len(price_series) <= window:
        print(f"Error: Data length ({len(price_series)}) is insufficient for RSI window ({window}). Needs > window entries.")
        return pd.Series(dtype=float)

    delta = price_series.diff() # Use price_series here

    # Calculate gain and loss using .ewm for a typical Wilder's RSI smoothing, or .rolling for simple mean
    # Using .rolling().mean() as per the original code's structure for gain/loss before RS calc.
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Avoid division by zero for rs
    # If loss is 0, rs is undefined or infinite. Common practice: if loss is 0, rsi is 100 if gain > 0, else 50 (or other neutral).
    # For simplicity, if loss is zero, and gain is also zero, RSI is undefined (NaN). If gain > 0 and loss is 0, RSI is 100.
    rs = gain / loss
    rs.replace([float('inf'), -float('inf')], float('nan'), inplace=True) # Handle division by zero leading to inf

    rsi = 100 - (100 / (1 + rs))

    # Fill specific RSI conditions
    rsi.loc[(gain > 0) & (loss == 0)] = 100 # If gain and no loss, RSI is 100
    rsi.loc[(gain == 0) & (loss == 0)] = 50 # Or some other neutral value if both are zero (can also be NaN then ffill)
                                          # This case might be covered if rs results in NaN and then RSI calc with NaN is NaN.
                                          # Let's ensure a defined behavior for no gain/no loss.
                                          # A common initial RSI value before it's truly calculable is 50.
    # If after all calculations, RSI is still NaN (e.g., rs was NaN and not covered by above specific conditions),
    # it might be due to insufficient data points for the rolling means to produce non-NaN values early in the series.
    # These NaNs are usually handled by `dropna()` where features are prepared or used.

    return rsi

if __name__ == '__main__':
    # Ensure the __main__ block is defined only once at the end of the file.
    # The prompt split it, so I will use the latter definition.
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
