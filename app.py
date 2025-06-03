import streamlit as st

# Basic Streamlit layout
st.title("Stock Analysis App")

st.sidebar.header("User Input Features")

# Import core Streamlit library
import streamlit as st
# Import datetime for date input handling
import datetime
# Import pandas for DataFrame manipulation
import pandas as pd
# Import io and sys for capturing print output from model training
from io import StringIO
import sys

# Import custom modules for application functionality
from data_ingestion.stock_data import get_historical_data, get_company_info
from technical_analysis.indicators import calculate_sma, calculate_ema, calculate_rsi
from visualization.charts import plot_stock_prices
from machine_learning.feature_engineering import prepare_features_for_ml
from machine_learning.model import train_model, make_prediction
from sentiment_analysis.news_fetcher import fetch_news
from sentiment_analysis.sentiment_analyzer import analyze_sentiment_vader, get_average_sentiment_score
from backtesting.engine import run_backtest
from backtesting.metrics import (
    calculate_total_return, calculate_annualized_return,
    calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_win_rate, summarize_trades
)


# --- Streamlit Page Configuration (Optional) ---
# st.set_page_config(layout="wide")

# --- Main App Structure ---
st.sidebar.title("Jules Swing Trade Pro")
app_mode = st.sidebar.selectbox("Choose App Mode",
                                ["Trading Analysis", "Backtesting", "About"])

if app_mode == "Trading Analysis":
    st.title("Trading Analysis Dashboard")

    # --- Sidebar for User Inputs (Trading Analysis) ---
    st.sidebar.header("User Input Features")
    ticker_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL", key="ta_ticker").upper()

    today = datetime.date.today()
    default_start_date = datetime.date(today.year - 1, 1, 1)
    start_date = st.sidebar.date_input("Start Date", default_start_date, key="ta_start_date")
    end_date = st.sidebar.date_input("End Date", today, key="ta_end_date")

    if start_date > end_date:
        st.sidebar.error("Error: Start date must be before end date.")
        st.stop()

    st.sidebar.subheader("Technical Indicators")
    selected_indicators = st.sidebar.multiselect(
        "Select indicators to overlay:",
        options=["SMA", "EMA", "RSI"],
        key="ta_selected_indicators"
    )
    sma_window, ema_window, rsi_window = 20, 20, 14
    sma_series, ema_series, rsi_series = None, None, None
    if "SMA" in selected_indicators:
        sma_window = st.sidebar.slider("SMA Window", 5, 100, sma_window, key="ta_sma_window")
    if "EMA" in selected_indicators:
        ema_window = st.sidebar.slider("EMA Window", 5, 100, ema_window, key="ta_ema_window")
    if "RSI" in selected_indicators:
        rsi_window = st.sidebar.slider("RSI Window", 7, 30, rsi_window, key="ta_rsi_window")

    enable_ml_prediction = st.sidebar.checkbox("Enable ML Price Prediction", value=False, key="ta_enable_ml")
    enable_sentiment_analysis = st.sidebar.checkbox("Enable Sentiment Analysis", value=False, key="ta_enable_sentiment")

    # --- Main Content Area (Trading Analysis) ---
    if not ticker_symbol:
        st.info("Please enter a stock ticker symbol in the sidebar to get started for Trading Analysis.")
        st.stop()

    st.header(f"Stock Analysis for {ticker_symbol}")

    # Company Information
    st.subheader("Company Information")
    try:
        company_info = get_company_info(ticker_symbol)
        if company_info and company_info.get('regularMarketPrice') is not None:
            st.write(f"**Name:** {company_info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
            # ... (other company info fields) ...
            with st.expander("About Company (Business Summary)"):
                st.write(company_info.get('longBusinessSummary', 'No summary available.'))
        else:
            st.warning(f"Could not retrieve valid company information for '{ticker_symbol}'.")
    except Exception as e:
        st.error(f"Error fetching company info: {e}")

    # Historical Data and Chart
    st.subheader("Price Chart & Technical Indicators")
    stock_data_df = pd.DataFrame()
    try:
        stock_data_df = get_historical_data(ticker_symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if not stock_data_df.empty:
            if 'Close' in stock_data_df.columns:
                if "SMA" in selected_indicators: sma_series = calculate_sma(stock_data_df, sma_window)
                if "EMA" in selected_indicators: ema_series = calculate_ema(stock_data_df, ema_window)
                if "RSI" in selected_indicators: rsi_series = calculate_rsi(stock_data_df, rsi_window)

            price_chart_fig = plot_stock_prices(stock_data_df, ticker_symbol,
                                                sma_series=sma_series, sma_window=sma_window,
                                                ema_series=ema_series, ema_window=ema_window,
                                                rsi_series=rsi_series, rsi_window=rsi_window)
            st.plotly_chart(price_chart_fig, use_container_width=True)
            st.subheader("Historical Data Preview (Last 20 days)")
            st.dataframe(stock_data_df.sort_index(ascending=False).head(20))
        else:
            st.warning(f"No historical data for '{ticker_symbol}' in selected range.")
    except Exception as e:
        st.error(f"Error fetching/processing historical data: {e}")

    # ML Prediction
    if enable_ml_prediction and not stock_data_df.empty:
        st.subheader("ML-Based Price Prediction (Experimental)")
        # ... (ML prediction logic as before, ensuring it uses stock_data_df) ...
        try:
            X_ml, y_ml = prepare_features_for_ml(stock_data_df.copy()) # Use defaults or make configurable
            min_data_for_ml = 30
            if X_ml.empty or y_ml.empty or len(X_ml) < min_data_for_ml:
                st.warning(f"Not enough data for ML (need {min_data_for_ml} points, got {len(X_ml)}).")
            else:
                old_stdout_ml = sys.stdout
                sys.stdout = captured_output_ml = StringIO()
                trained_model = train_model(X_ml, y_ml)
                sys.stdout = old_stdout_ml
                model_train_log = captured_output_ml.getvalue()

                if trained_model:
                    st.text("Model Training Log:")
                    st.text_area("Log_ml", model_train_log, height=100)
                    latest_features = X_ml.iloc[[-1]]
                    prediction, probability = make_prediction(trained_model, latest_features)
                    if prediction is not None:
                        pred_text = "**UP**" if prediction == 1 else "**DOWN/SAME**"
                        st.markdown(f"**Prediction for Next Day:** {pred_text} (Confidence: {probability*100:.2f}%)")
                    st.caption("ML model is experimental. For informational purposes only.")
                else:
                    st.error("Failed to train ML model.")
                    if model_train_log: st.text_area("Log_ml_fail", model_train_log, height=100)
        except Exception as e:
            st.error(f"ML Error: {e}")
    elif enable_ml_prediction and stock_data_df.empty:
        st.warning("ML: No historical data for prediction.")

    # Sentiment Analysis
    if enable_sentiment_analysis: # ticker_symbol is already checked for Trading Analysis mode
        st.subheader("Sentiment Analysis (Simulated News)")
        # ... (Sentiment analysis logic as before) ...
        try:
            news_items = fetch_news(ticker_symbol)
            if news_items:
                headlines = [item['headline'] for item in news_items]
                sentiment_scores_detailed = analyze_sentiment_vader(headlines)
                avg_score = get_average_sentiment_score(sentiment_scores_detailed)
                avg_text = "Neutral"
                if avg_score > 0.05: avg_text = "Positive"
                elif avg_score < -0.05: avg_text = "Negative"
                st.write(f"**Avg News Sentiment:** {avg_score:.4f} ({avg_text})")
                # ... (display individual headlines in expander) ...
                st.caption("Sentiment based on simulated news & VADER.")
            else:
                st.info(f"No simulated news for '{ticker_symbol}'.")
        except Exception as e:
            st.error(f"Sentiment Analysis Error: {e}")


elif app_mode == "Backtesting":
    st.title("Strategy Backtester")

    # 1. Inputs for Backtesting
    st.sidebar.header("Backtest Settings")
    bt_ticker = st.sidebar.text_input("Ticker for Backtest", value="AAPL", key="bt_ticker").upper()

    # Use pd.to_datetime for default dates for Streamlit's date_input compatibility if needed,
    # but direct datetime.date objects are usually fine.
    bt_default_start = datetime.date(2022, 1, 1)
    bt_default_end = datetime.date(2023, 1, 1)
    bt_start_date = st.sidebar.date_input("Backtest Start Date", bt_default_start, key="bt_start")
    bt_end_date = st.sidebar.date_input("Backtest End Date", bt_default_end, key="bt_end")

    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=100000, key="bt_capital")

    if bt_start_date > bt_end_date:
        st.sidebar.error("Error: Backtest start date must be before end date.")
        st.stop()

    # Strategy Selection
    strategy_type_display = st.sidebar.selectbox("Choose Strategy",
                                         ["SMA Crossover", "RSI Mean Reversion"], key="bt_strategy_display")

    strategy_params = {} # Initialize
    if strategy_type_display == "SMA Crossover":
        strategy_params['type'] = 'sma_crossover'
        strategy_params['short_window'] = st.sidebar.slider("Short SMA Window", 5, 50, 10, key="bt_sma_short")
        strategy_params['long_window'] = st.sidebar.slider("Long SMA Window", 20, 200, 50, key="bt_sma_long")
        strategy_params['short_window_col'] = f'SMA_{strategy_params["short_window"]}'
        strategy_params['long_window_col'] = f'SMA_{strategy_params["long_window"]}'

    elif strategy_type_display == "RSI Mean Reversion":
        strategy_params['type'] = 'rsi_reversion'
        strategy_params['rsi_window'] = st.sidebar.slider("RSI Window", 5, 30, 14, key="bt_rsi_window")
        strategy_params['oversold_threshold'] = st.sidebar.slider("RSI Oversold Threshold", 10, 40, 30, key="bt_rsi_oversold")
        strategy_params['overbought_threshold'] = st.sidebar.slider("RSI Overbought Threshold", 60, 90, 70, key="bt_rsi_overbought")
        strategy_params['rsi_col'] = f'RSI_{strategy_params["rsi_window"]}'

    if st.sidebar.button("Run Backtest", key="bt_run"):
        if not bt_ticker:
            st.error("Please enter a ticker symbol for backtesting.")
        else:
            st.subheader(f"Backtest Results for {bt_ticker} ({strategy_type_display})")
            try:
                # 2. Fetch Data & Calculate Indicators
                # Convert date inputs to string for get_historical_data
                hist_data = get_historical_data(bt_ticker,
                                                bt_start_date.strftime("%Y-%m-%d"),
                                                bt_end_date.strftime("%Y-%m-%d"))

                if hist_data.empty or len(hist_data) < max(strategy_params.get('long_window', 0), strategy_params.get('rsi_window',0), 20): # Ensure enough data for indicators and backtest
                    st.error("Could not fetch sufficient historical data for the given ticker and date range to run the backtest with these indicator settings.")
                else:
                    # Add necessary indicators to hist_data
                    if strategy_params['type'] == 'sma_crossover':
                        hist_data[strategy_params['short_window_col']] = calculate_sma(hist_data, strategy_params['short_window'])
                        hist_data[strategy_params['long_window_col']] = calculate_sma(hist_data, strategy_params['long_window'])
                    elif strategy_params['type'] == 'rsi_reversion':
                        hist_data[strategy_params['rsi_col']] = calculate_rsi(hist_data, strategy_params['rsi_window'])

                    # Drop rows with NaNs from indicator calculation before sending to backtester
                    hist_data.dropna(inplace=True)
                    if hist_data.empty:
                         st.error("Data became empty after indicator calculation and NaN removal. Try a longer date range or different indicator parameters.")
                         st.stop()


                    # 3. Run Backtest
                    portfolio_df, trades_list = run_backtest(hist_data, strategy_params, initial_capital)

                    if portfolio_df is None or portfolio_df.empty:
                        st.warning("Backtest did not generate results. This might happen if no trades were made or data was insufficient after indicator processing.")
                    else:
                        # 4. Calculate and Display Metrics
                        st.subheader("Backtest Performance Metrics")

                        col1, col2, col3, col4 = st.columns(4)

                        total_return = calculate_total_return(portfolio_df)
                        annualized_return = calculate_annualized_return(portfolio_df)
                        sharpe = calculate_sharpe_ratio(portfolio_df) # Default risk_free_rate_annual=0.0
                        max_dd = calculate_max_drawdown(portfolio_df)

                        with col1:
                            st.metric("Total Return", f"{total_return*100:.2f}%")
                        with col2:
                            st.metric("Annualized Return", f"{annualized_return*100:.2f}%")
                        with col3:
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        with col4:
                            st.metric("Max Drawdown", f"{max_dd*100:.2f}%")

                        trade_summary_stats = summarize_trades(trades_list)
                        win_rate_val = calculate_win_rate(trades_list)

                        col_trades1, col_trades2, col_trades3 = st.columns(3)
                        with col_trades1:
                            st.metric("Number of Trades", trade_summary_stats['num_trades'])
                        with col_trades2:
                            st.metric("Win Rate", f"{win_rate_val*100:.2f}%")
                        with col_trades3:
                             profit_factor_val = trade_summary_stats['profit_factor']
                             st.metric("Profit Factor", f"{profit_factor_val:.2f}" if profit_factor_val != float('inf') else "inf")

                        st.markdown("---")
                        st.subheader("Trade Statistics Details")
                        st.json(trade_summary_stats) # Display all trade summary stats

                        # 5. Display Portfolio Chart
                        st.subheader("Portfolio Value Over Time")
                        st.line_chart(portfolio_df['portfolio_value'])

                        # Optional: Overlay with stock price (requires normalization or dual axis)
                        # from plotly.subplots import make_subplots
                        # import plotly.graph_objects as go
                        # fig_bt_portfolio = make_subplots(specs=[[{"secondary_y": True}]])
                        # fig_bt_portfolio.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df['portfolio_value'], name="Portfolio Value"), secondary_y=False)
                        # fig_bt_portfolio.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], name=f"{bt_ticker} Close Price"), secondary_y=True)
                        # st.plotly_chart(fig_bt_portfolio, use_container_width=True)


                        # 6. Display Trades List (Optional)
                        if st.checkbox("Show Trades List", key="bt_show_trades"):
                            st.subheader("Trades Log")
                            if trades_list:
                                st.dataframe(pd.DataFrame(trades_list))
                            else:
                                st.info("No trades were executed during this backtest.")

                        st.info("Disclaimer: Backtesting results are based on historical data and do not guarantee future performance. Slippage, commissions, and other transaction costs are not considered in this simulation.")

            except Exception as e:
                st.error(f"An error occurred during backtesting: {e}")
                st.error("Make sure the selected date range provides enough data for indicator calculations (e.g., more days than the longest window).")


elif app_mode == "About":
    st.title("About Jules Swing Trade Pro")
    st.write("""
    This application provides tools for stock trading analysis, including:
    - Historical price visualization with technical indicators.
    - Experimental ML-based price movement prediction.
    - Simulated news sentiment analysis.
    - Strategy backtesting capabilities.

    Developed by Jules the AI Agent.
    **Disclaimer:** All information and tools provided are for educational and informational purposes only.
    They do not constitute financial advice. Trading and investing involve risk of loss.
    """)

# else: (If more modes are added)
#    st.info("Select an application mode from the sidebar.")
