import streamlit as st

# Basic Streamlit layout
st.title("Stock Analysis App")

st.sidebar.header("User Input Features")

# Import core Streamlit library
import streamlit as st
# Import datetime for date input handling
import datetime
# Import pandas for DataFrame manipulation (though mostly handled in other modules now)
import pandas as pd

# Import custom modules for application functionality
from data_ingestion.stock_data import get_historical_data, get_company_info
from technical_analysis.indicators import calculate_sma, calculate_ema, calculate_rsi
from visualization.charts import plot_stock_prices

# --- Streamlit Page Configuration (Optional) ---
# st.set_page_config(layout="wide") # Example: Use wide layout

# --- Application Title ---
st.title("Stock Analysis App")

# --- Sidebar for User Inputs ---
st.sidebar.header("User Input Features")

# Ticker symbol input
ticker_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()

# Date range selection
# Default start date is Jan 1st of the previous year, default end date is today
# Note: Using datetime.date for date inputs, converted to string for yfinance
today = datetime.date.today()
default_start_date = datetime.date(today.year - 1, 1, 1)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", today)

# Validate date range
if start_date > end_date:
    st.sidebar.error("Error: Start date must be before end date.")
    st.stop() # Stop execution if date range is invalid

# --- Technical Indicator Selection ---
st.sidebar.subheader("Technical Indicators")
selected_indicators = st.sidebar.multiselect(
    "Select indicators to overlay:",
    options=["SMA", "EMA", "RSI"],
    # default=[] # Optionally, set default selected indicators
)

# Initialize variables for indicator data and parameters
sma_window, ema_window, rsi_window = 20, 20, 14 # Default window values
sma_series, ema_series, rsi_series = None, None, None # To store calculated indicator series

# Sliders for indicator parameters (only show if indicator is selected)
if "SMA" in selected_indicators:
    sma_window = st.sidebar.slider(
        "SMA Window", 
        min_value=5, max_value=100, value=sma_window, key="sma_window_slider",
        help="Number of periods for Simple Moving Average."
    )
if "EMA" in selected_indicators:
    ema_window = st.sidebar.slider(
        "EMA Window", 
        min_value=5, max_value=100, value=ema_window, key="ema_window_slider",
        help="Number of periods for Exponential Moving Average."
    )
if "RSI" in selected_indicators:
    rsi_window = st.sidebar.slider(
        "RSI Window", 
        min_value=7, max_value=30, value=rsi_window, key="rsi_window_slider",
        help="Number of periods for Relative Strength Index."
    )

# --- Main Content Area ---

# Display header with the selected ticker symbol
st.header(f"Stock Analysis for {ticker_symbol}")

# Main logic: Proceed only if a ticker symbol is entered
if ticker_symbol:
    # --- Company Information Section ---
    st.subheader("Company Information")
    company_info = get_company_info(ticker_symbol)
    if company_info and company_info.get('regularMarketPrice') is not None: # Check for valid info
        st.write(f"**Name:** {company_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
        st.write(f"**Website:** {company_info.get('website', 'N/A')}")
        # Business summary in an expander
        with st.expander("About Company (Business Summary)"):
            st.write(company_info.get('longBusinessSummary', 'No summary available.'))
    else:
        st.warning(f"Could not retrieve valid company information for '{ticker_symbol}'. Please check the ticker symbol.")

    # --- Historical Data and Chart Section ---
    st.subheader("Price Chart & Technical Indicators")
    # Fetch historical stock data
    # Convert date objects to string format "YYYY-MM-DD" for yfinance
    stock_data_df = get_historical_data(ticker_symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if not stock_data_df.empty:
        # --- Calculate Technical Indicators ---
        # Ensure 'Close' column exists before trying to calculate indicators
        if 'Close' in stock_data_df.columns:
            if "SMA" in selected_indicators and sma_window:
                sma_series = calculate_sma(stock_data_df, window=sma_window, price_column='Close')
            if "EMA" in selected_indicators and ema_window:
                ema_series = calculate_ema(stock_data_df, window=ema_window, price_column='Close')
            if "RSI" in selected_indicators and rsi_window:
                rsi_series = calculate_rsi(stock_data_df, window=rsi_window, price_column='Close')
        else:
            st.warning("Required 'Close' price column not found in the data. Cannot calculate indicators.")

        # --- Display Stock Price Chart with Indicators ---
        price_chart_fig = plot_stock_prices(
            data=stock_data_df, 
            ticker_symbol=ticker_symbol, 
            price_column='Close', # Base plot on 'Close' price
            sma_series=sma_series, sma_window=sma_window,
            ema_series=ema_series, ema_window=ema_window,
            rsi_series=rsi_series, rsi_window=rsi_window
        )
        st.plotly_chart(price_chart_fig, use_container_width=True)
        
        # --- Display Historical Data Table ---
        st.subheader("Historical Data")
        # Show limited columns for brevity or allow user to select
        # Displaying with Date as index, sorted descending
        st.dataframe(stock_data_df.sort_index(ascending=False).head(20)) # Show most recent 20 days
        
        # Optional: Allow downloading all data
        # csv = stock_data_df.to_csv().encode('utf-8')
        # st.download_button(
        #     label="Download Full Data as CSV",
        #     data=csv,
        #     file_name=f'{ticker_symbol}_historical_data.csv',
        #     mime='text/csv',
        # )
    else:
        # This message now also covers cases where yfinance returns an empty df for a valid-looking ticker (e.g. delisted)
        st.warning(f"No historical stock data found for '{ticker_symbol}' for the selected date range. It might be an invalid ticker, delisted, or no data available for the period.")
else:
    # Initial instruction message when no ticker is entered
    st.info("Please enter a stock ticker symbol in the sidebar to get started.")
