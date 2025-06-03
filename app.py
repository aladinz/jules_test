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
from machine_learning.feature_engineering import prepare_features_for_ml
from machine_learning.model import train_model, make_prediction
from sentiment_analysis.news_fetcher import fetch_news
from sentiment_analysis.sentiment_analyzer import analyze_sentiment_vader, get_average_sentiment_score

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

# --- ML Prediction Toggle ---
st.sidebar.subheader("Machine Learning Prediction")
enable_ml_prediction = st.sidebar.checkbox("Enable ML Price Prediction", value=False)

# --- Sentiment Analysis Toggle ---
st.sidebar.subheader("Sentiment Analysis")
enable_sentiment_analysis = st.sidebar.checkbox("Enable Sentiment Analysis", value=False)


# --- Main Content Area ---

# Display header with the selected ticker symbol
st.header(f"Stock Analysis for {ticker_symbol}")

# Main logic: Proceed only if a ticker symbol is entered
if ticker_symbol:
    # --- Company Information Section ---
    st.subheader("Company Information")
    try:
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
    except Exception as e:
        st.error(f"An error occurred while fetching company information: {e}")

    # --- Historical Data and Chart Section ---
    st.subheader("Price Chart & Technical Indicators")
    stock_data_df = pd.DataFrame() # Initialize as empty
    try:
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
            st.subheader("Historical Data Preview (Last 20 days)")
            st.dataframe(stock_data_df.sort_index(ascending=False).head(20))

        else:
            st.warning(f"No historical stock data found for '{ticker_symbol}' for the selected date range. It might be an invalid ticker, delisted, or no data available for the period.")
    except Exception as e:
        st.error(f"An error occurred while fetching or processing historical data: {e}")


    # --- ML Prediction Section (conditional) ---
    if enable_ml_prediction and not stock_data_df.empty:
        st.subheader("ML-Based Price Prediction (Experimental)")
        try:
            # 1. Prepare features for ML
            # Using default windows for SMA/RSI in ML features, could be made configurable
            X_ml, y_ml = prepare_features_for_ml(stock_data_df.copy(), sma_window=10, rsi_window=14, num_lags=5)

            min_data_for_ml = 30 # Arbitrary minimum number of samples after feature engineering
            if X_ml.empty or y_ml.empty or len(X_ml) < min_data_for_ml :
                st.warning(f"Not enough data to train an ML model or make a prediction. Need at least {min_data_for_ml} data points after feature engineering, got {len(X_ml)}.")
            else:
                # 2. Train the model (this happens on each run if enabled, could be cached/persisted in a real app)
                # Capture print output from train_model
                from io import StringIO
                import sys
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                trained_model = train_model(X_ml, y_ml) # Uses default test_size=0.2

                sys.stdout = old_stdout # Restore stdout
                model_train_log = captured_output.getvalue()

                if trained_model:
                    st.text("Model Training Log:")
                    st.text_area("Log", model_train_log, height=100) # Display accuracy from training

                    # 3. Prepare the latest features for prediction
                    # The last row of X_ml corresponds to the features for the *last date in stock_data_df*
                    # for which a target could be generated. The actual prediction is for the *next* day.
                    latest_features = X_ml.iloc[[-1]]

                    # 4. Make prediction
                    prediction, probability = make_prediction(trained_model, latest_features)

                    if prediction is not None and probability is not None:
                        prediction_text = "**UP**" if prediction == 1 else "**DOWN/SAME**"
                        st.markdown(f"**ML Prediction for Next Trading Day:** {prediction_text}")
                        st.write(f"**Confidence (Probability of UP):** {probability*100:.2f}%")
                    else:
                        st.error("Could not make a prediction with the trained model.")
                    st.caption("Note: This ML model is experimental and for informational purposes only. Do not use for financial decisions.")
                else:
                    st.error("Failed to train the ML model. Check logs above if any, or data quality.")
                    if model_train_log: # Show log even if model training returned None
                         st.text("Model Training Log (Attempt):")
                         st.text_area("Log", model_train_log, height=100)

        except ValueError as ve: # Catch specific errors from feature engineering or model
             st.error(f"ML Error: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred during the ML prediction process: {e}")
    elif enable_ml_prediction and stock_data_df.empty:
        st.warning("ML Prediction: Cannot perform ML prediction as no historical data is available for the ticker.")

    # --- Sentiment Analysis Section (conditional) ---
    if enable_sentiment_analysis and ticker_symbol:
        st.subheader("Sentiment Analysis (Simulated News)")
        try:
            news_items = fetch_news(ticker_symbol)
            if news_items:
                headlines = [item['headline'] for item in news_items]
                sentiment_scores_detailed = analyze_sentiment_vader(headlines)
                average_compound_score = get_average_sentiment_score(sentiment_scores_detailed)

                # Interpret average score
                avg_sentiment_text = "Neutral"
                if average_compound_score > 0.05:
                    avg_sentiment_text = "Positive"
                elif average_compound_score < -0.05:
                    avg_sentiment_text = "Negative"

                st.write(f"**Average News Sentiment (Compound Score):** {average_compound_score:.4f} ({avg_sentiment_text})")

                with st.expander("View Individual News Headlines & Sentiments", expanded=False):
                    for item, scores_data in zip(news_items, sentiment_scores_detailed):
                        compound = scores_data['sentiment']['compound']
                        item_sentiment_text = "Neutral"
                        if compound > 0.05:
                            item_sentiment_text = "Positive"
                        elif compound < -0.05:
                            item_sentiment_text = "Negative"

                        st.markdown(f"""
                        **Headline:** {item['headline']}
                        *Source:* {item['source']} | *Date:* {item['date']}
                        *Sentiment:* {item_sentiment_text} (Compound: {compound:.4f})
                        """)
                        st.divider()
                st.caption("Note: Sentiment data is based on simulated news and VADER analysis.")
            else:
                st.info(f"No simulated news found for '{ticker_symbol}'.")
        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {e}")
    elif enable_sentiment_analysis and not ticker_symbol:
        st.warning("Sentiment Analysis: Please enter a ticker symbol.")


else:
    # Initial instruction message when no ticker is entered
    st.info("Please enter a stock ticker symbol in the sidebar to get started.")
