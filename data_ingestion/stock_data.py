import yfinance as yf
import pandas as pd

def get_historical_data(ticker: str, start_date: str = None, end_date: str = None, period: str = None, interval: str = "1d", prepost: bool = False) -> pd.DataFrame:
    """
    Fetches historical stock data using yfinance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. Used if 'period' is None.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Used if 'period' is None.
        period (str, optional): Data period to download (e.g., "1mo", "6mo", "ytd", "1y", "max").
                               If provided, start_date and end_date are ignored by yfinance.
        interval (str, optional): Data interval (e.g., "1d", "1wk", "1mo"). Defaults to "1d".
        prepost (bool, optional): Set to True to include Pre and Post market data. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with historical data (OHLC, Volume), indexed by Date.
                      Returns an empty DataFrame on error or if no data is found.
    """
    try:
        params = {
            "progress": False,
            "auto_adjust": True,
            "interval": interval,
            "prepost": prepost # Added prepost
        }

        if period:
            data = yf.download(ticker, period=period, **params)
        elif start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, **params)
        else:
            print("Error in get_historical_data: Provide either 'period' or both 'start_date' and 'end_date'.")
            return pd.DataFrame()

        if data.empty:
            # yfinance often prints its own messages for "No data found"
            # print(f"No data found for {ticker} with parameters: period={period}, start={start_date}, end={end_date}, interval={interval}")
            pass

        # Flatten columns if they are MultiIndex (often happens with yf.download for single ticker)
        if not data.empty and isinstance(data.columns, pd.MultiIndex):
            # Keep the first level of column names (e.g., 'Open', 'High', 'Low', 'Close', 'Volume')
            # yfinance might return [('Open', 'TICKER'), ('High', 'TICKER'), ...]
            data.columns = data.columns.get_level_values(0)

        # Ensure the index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex) and not data.empty:
             data.index = pd.to_datetime(data.index) # Convert if not already DatetimeIndex

        return data

    except Exception as e:
        print(f"Error fetching historical data for {ticker} using yfinance: {e}")
        return pd.DataFrame()

def get_company_info(ticker: str) -> dict:
    """
    Fetches basic company information for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        dict: A dictionary containing company information (e.g., name, sector, industry),
              or an empty dictionary if the ticker is invalid or info is unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get('regularMarketPrice') is None: # Check if info is empty or lacks a key field
            print(f"No company information found for ticker {ticker}.")
            return {}
        return info
    except Exception as e:
        print(f"Error fetching company info for {ticker}: {e}")
        return {}

if __name__ == '__main__':
    # Example usage:
    sample_ticker = "MSFT"
    sample_start_date = "2023-01-01"
    sample_end_date = "2023-03-31" # Shorter period for quicker test

    print(f"--- Testing get_historical_data with start/end dates for {sample_ticker} ---")
    historical_data_sd = get_historical_data(sample_ticker, start_date=sample_start_date, end_date=sample_end_date)
    if not historical_data_sd.empty:
        print(f"Columns for {sample_ticker} (start/end): {historical_data_sd.columns}")
        print(f"Shape: {historical_data_sd.shape}")
        print(historical_data_sd.head(2))
        print(historical_data_sd.tail(2))
    else:
        print(f"No data returned for {sample_ticker} with start/end dates.")

    print(f"\n--- Testing get_historical_data with period for {sample_ticker} ---")
    data_period_1mo = get_historical_data(sample_ticker, period="1mo")
    if not data_period_1mo.empty:
        print(f"Columns for {sample_ticker} (period='1mo'): {data_period_1mo.columns}")
        print(f"{sample_ticker} 1 month data (period='1mo'). Shape: {data_period_1mo.shape}")
        print(data_period_1mo.head(2))
        print(data_period_1mo.tail(2))
    else:
        print(f"No data returned for {sample_ticker} with period='1mo'.")


    data_period_ytd_weekly = get_historical_data("TSLA", period="ytd", interval="1wk")
    if not data_period_ytd_weekly.empty:
        print(f"\nColumns for TSLA (period='ytd', interval='1wk'): {data_period_ytd_weekly.columns}")
        print(f"TSLA YTD data (period='ytd', interval='1wk'). Shape: {data_period_ytd_weekly.shape}")
        print(data_period_ytd_weekly.tail(2))
    else:
        print(f"\nNo data returned for TSLA with period='ytd', interval='1wk'.")

    print(f"\n--- Testing get_historical_data with no date/period (should fail) ---")
    no_date_data = get_historical_data("GE")
    if no_date_data.empty:
        print("Correctly returned empty DataFrame when no date/period is specified.")
    else:
        print("Test FAILED: Data returned even when no date/period specified.")


    print(f"\n--- Testing get_company_info for {sample_ticker} ---")
    company_info = get_company_info(sample_ticker)
    if company_info and company_info.get('longName'): # Check for a common field
        print(f"Name: {company_info.get('longName')}")
        print(f"Sector: {company_info.get('sector')}")
        print(f"Industry: {company_info.get('industry')}")
    else:
        print(f"Could not retrieve full company info for {sample_ticker}.")

    sample_invalid_ticker = "INVALIDTICKERXYZ123"
    print(f"\n--- Testing with invalid ticker: {sample_invalid_ticker} ---")
    invalid_data = get_historical_data(sample_invalid_ticker, start_date="2023-01-01", end_date="2023-01-05")
    if invalid_data.empty:
        print(f"Correctly received empty DataFrame for invalid ticker historical data.")

    invalid_info = get_company_info(sample_invalid_ticker)
    if not invalid_info or invalid_info.get('regularMarketPrice') is None :
        print(f"Correctly received minimal/empty info for invalid ticker company data.")

    print("\n--- Testing get_historical_data with prepost=True ---")
    # For prepost data, a short period and small interval are best.
    # yfinance provides 1m data for last 7 days, other intervals for last 60 days.
    # Using 15m for a 2-day period to likely see some effect if run during extended hours or for active stocks.
    data_prepost_test = get_historical_data("AAPL", period="2d", interval="15m", prepost=True)
    print(f"AAPL data for 2 days, 15min interval, prepost=True. Shape: {data_prepost_test.shape}")
    if not data_prepost_test.empty:
        print("Sample of data with prepost=True (first 3 rows):")
        print(data_prepost_test.head(3))
        print("Sample of data with prepost=True (last 3 rows):")
        print(data_prepost_test.tail(3))
        if isinstance(data_prepost_test.index, pd.DatetimeIndex):
            print(f"Index type is DatetimeIndex. First timestamp: {data_prepost_test.index[0]}")
    else:
        print("No data returned for AAPL with prepost=True (this might be normal if market is closed and no recent extended hours data).")
