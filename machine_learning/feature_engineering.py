import pandas as pd
# Assuming technical_analysis.indicators is in a reachable path.
# If running this file standalone, ensure PYTHONPATH is set or use relative imports if structured as a package.
# For this project structure, we might need to adjust imports if running standalone vs as part of the app.
# For now, let's assume it can be found.
from technical_analysis.indicators import calculate_sma, calculate_rsi

def create_lagged_features(df: pd.DataFrame, column_name: str, num_lags: int) -> pd.DataFrame:
    """
    Creates lagged features for a specified column.

    Args:
        df (pd.DataFrame): Input DataFrame with a DateTimeIndex.
        column_name (str): The name of the column to lag.
        num_lags (int): The number of lags to create.

    Returns:
        pd.DataFrame: DataFrame with new lagged columns.
    """
    df_lagged = df.copy()
    for i in range(1, num_lags + 1):
        df_lagged[f'{column_name}_lag_{i}'] = df_lagged[column_name].shift(i)
    return df_lagged

def create_percentage_change_feature(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Creates a feature for the percentage change from the previous day.

    Args:
        df (pd.DataFrame): Input DataFrame with a DateTimeIndex.
        column_name (str): The name of the column for pct change calculation.

    Returns:
        pd.DataFrame: DataFrame with the new percentage change column.
    """
    df_pct_change = df.copy()
    df_pct_change[f'{column_name}_pct_change'] = df_pct_change[column_name].pct_change() * 100
    return df_pct_change

def create_target_variable(df: pd.DataFrame, column_name: str = 'Close') -> pd.Series:
    """
    Creates the target variable: 1 if next day's price is higher, 0 otherwise.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): The column to base the target on (default: 'Close').

    Returns:
        pd.Series: Series containing the target variable (0 or 1).
    """
    # Shift the 'Close' price to get the next day's price
    df['next_day_price'] = df[column_name].shift(-1)
    # Create target: 1 if next_day_price > current_price, else 0
    df['target'] = (df['next_day_price'] > df[column_name]).astype(int)
    return df['target']

def prepare_features_for_ml(
    historical_data: pd.DataFrame,
    price_column: str = 'Close',
    sma_window: int = 10,
    rsi_window: int = 14,
    num_lags: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares features (X) and target (y) for machine learning.

    Args:
        historical_data (pd.DataFrame): DataFrame from get_historical_data.
        price_column (str): The main price column to use for features (default 'Close').
        sma_window (int): Window for SMA calculation.
        rsi_window (int): Window for RSI calculation.
        num_lags (int): Number of lagged price features to create.

    Returns:
        tuple[pd.DataFrame, pd.Series]:
            - X: DataFrame of engineered features, NaN rows dropped.
            - y: Series for the target variable, aligned with X.
    """
    if price_column not in historical_data.columns:
        raise ValueError(f"Price column '{price_column}' not found in historical_data.")

    df = historical_data.copy()

    # 1. Create basic features
    df = create_lagged_features(df, price_column, num_lags)
    df = create_percentage_change_feature(df, price_column)

    # 2. Add technical indicators
    df[f'SMA_{sma_window}'] = calculate_sma(df, window=sma_window, price_column=price_column)
    df[f'RSI_{rsi_window}'] = calculate_rsi(df, window=rsi_window, price_column=price_column)

    # 3. Create target variable
    # Important: Create target before dropping NaNs that might be introduced by target creation itself (due to shift(-1))
    # However, target variable depends on future values, so it is best to create it based on the original price column
    # before some rows get dropped due to feature engineering.
    y = create_target_variable(df, price_column) # df here will have next_day_price and target columns added

    # 4. Define feature columns (all columns except original OHLCV, target, and next_day_price)
    # Let's be explicit about features.
    # We want lags, pct_change, SMA, RSI.
    # Original 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits' could also be features if desired.
    # For this task, sticking to the requested features.
    feature_names = [f'{price_column}_lag_{i}' for i in range(1, num_lags + 1)]
    feature_names.append(f'{price_column}_pct_change')
    feature_names.append(f'SMA_{sma_window}')
    feature_names.append(f'RSI_{rsi_window}')

    # Select only the defined features for X
    X = df[feature_names].copy()

    # 5. Handle NaNs
    # Drop rows with any NaN values in X or y (aligns X and y)
    # NaNs can come from lags, pct_change, indicators, or target variable's last row.
    combined = X.join(y, lsuffix='_X', rsuffix='_y') # Use suffixes if 'target' name collision
    combined.dropna(inplace=True)

    X_cleaned = combined[X.columns] # Get back the feature columns
    y_cleaned = combined[y.name]    # Get back the target column (original name)

    return X_cleaned, y_cleaned


if __name__ == '__main__':
    # Example Usage (requires yfinance and data_ingestion module)
    # To run this standalone, you might need to adjust paths or ensure yfinance is installed
    # and that technical_analysis.indicators can be imported.
    raw_data_example = pd.DataFrame()
    try:
        # Attempt to import from the project structure
        from data_ingestion.stock_data import get_historical_data

        # Fetch sample data using yfinance if successful
        sample_ticker_ml = "MSFT" # Using a different ticker to avoid conflict if app.py is running
        sample_start_ml = "2022-01-01"
        sample_end_ml = "2023-01-01" # Get a year's worth of data
        raw_data_example = get_historical_data(sample_ticker_ml, sample_start_ml, sample_end_ml)
        if raw_data_example.empty:
            print(f"Warning: get_historical_data for {sample_ticker_ml} returned empty DataFrame.")
    except ImportError:
        print("Warning: Could not import get_historical_data. Using fallback sample data for feature engineering test.")

    if raw_data_example.empty: # Fallback if yfinance fetch failed or import error
        num_days_fallback = 60 # Enough days for lags, indicators, and some data left
        sample_dates_fallback = pd.to_datetime([f'2023-01-{d:02d}' for d in range(1, num_days_fallback + 1)])
        # Create more realistic price data for RSI calculation (needs some ups and downs)
        prices = [150]
        for i in range(1, num_days_fallback):
            change = (i % 5 - 2) * 0.5 # Small daily changes, some up, some down
            prices.append(prices[-1] + change + 0.1 * (i%2) ) # Add some noise
        raw_data_example = pd.DataFrame({'Close': prices,
                                         'Open': prices, 'High': prices, 'Low': prices, 'Volume': [1000]*num_days_fallback},
                                        index=sample_dates_fallback)
        print("Using generated fallback data for testing feature engineering.")


    if not raw_data_example.empty and 'Close' in raw_data_example.columns:
        print("\nOriginal Data for ML (tail):")
        print(raw_data_example.tail())

        # Test prepare_features_for_ml
        X_ml, y_ml = prepare_features_for_ml(raw_data_example, sma_window=10, rsi_window=14, num_lags=5)

        print("\nFeatures for ML (X) (head to see effects of NaN drop):")
        print(X_ml.head())
        print("\nFeatures for ML (X) (tail):")
        print(X_ml.tail())

        print("\nTarget for ML (y) (head to see effects of NaN drop):")
        print(y_ml.head())
        print("\nTarget for ML (y) (tail):")
        print(y_ml.tail())

        print(f"\nShape of X: {X_ml.shape}, Shape of y: {y_ml.shape}")
        print(f"Number of NaNs in X: {X_ml.isnull().sum().sum()}")
        print(f"Number of NaNs in y: {y_ml.isnull().sum().sum()}")

        if X_ml.empty or y_ml.empty:
            print("Warning: Resulting X or y DataFrame is empty. Check data length and NaN handling.")
        elif len(X_ml) != len(y_ml):
            print(f"CRITICAL WARNING: X and y have different lengths after processing! X:{len(X_ml)}, y:{len(y_ml)}")

    else:
        print("Could not fetch or generate sample data for testing ML features.")
