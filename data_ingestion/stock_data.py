import yfinance as yf
import pandas as pd

def get_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker symbol between specified dates.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        start_date (str): The start date for the historical data (YYYY-MM-DD).
        end_date (str): The end date for the historical data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data (Date, Open, High, Low, Close, Volume),
                      or an empty DataFrame if the ticker is invalid or data is unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for ticker {ticker} between {start_date} and {end_date}.")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
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
    sample_end_date = "2023-12-31"

    historical_data = get_historical_data(sample_ticker, sample_start_date, sample_end_date)

    if not historical_data.empty:
        print(f"\nHistorical Data for {sample_ticker}:")
        print(historical_data.head())

    company_info = get_company_info(sample_ticker)
    if company_info:
        print(f"\nCompany Info for {sample_ticker}:")
        print(f"Name: {company_info.get('longName')}")
        print(f"Sector: {company_info.get('sector')}")
        print(f"Industry: {company_info.get('industry')}")

    sample_invalid_ticker = "INVALIDTICKER"
    invalid_data = get_historical_data(sample_invalid_ticker, sample_start_date, sample_end_date)
    if invalid_data.empty:
        print(f"\nAttempted to fetch data for invalid ticker {sample_invalid_ticker}, and received an empty DataFrame as expected.")

    invalid_info = get_company_info(sample_invalid_ticker)
    if not invalid_info:
        print(f"Attempted to fetch info for invalid ticker {sample_invalid_ticker}, and received an empty dict as expected.")
