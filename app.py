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
import plotly.graph_objects as go

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


# --- Constants and Global Configs (if any, none for now) ---

# --- Helper Functions / Render Functions ---

# --- About Page Render Function ---
def render_about_page():
    """Renders the 'About' page content."""
    st.title("About Jules Swing Trade Pro")
    st.markdown("---")

    st.markdown("""
    Welcome to **Jules Swing Trade Pro**! This application is designed to assist swing traders and
    market enthusiasts by providing a suite of tools for stock analysis, strategy backtesting,
    and exploring potential market insights. It's intended for users who want to combine technical
    analysis with experimental machine learning and sentiment data to inform their trading decisions.
    """)
    st.markdown("---")

    st.subheader("Key Features")
    st.markdown("""
    -   **Trading Analysis Mode**:
        -   Interactive price charts (Line or Candlestick) with integrated volume display.
        -   Overlay popular technical indicators: Simple Moving Average (SMA), Exponential Moving Average (EMA), and Relative Strength Index (RSI).
        -   Dynamic adjustment of indicator parameters.
    -   **Machine Learning Insights (Experimental)**:
        -   Next-day price direction prediction (UP/DOWN/SAME) using a Logistic Regression model.
        -   Display of the ML model's training accuracy.
        -   Option to enable/disable ML predictions.
    -   **Sentiment Analysis (Simulated)**:
        -   VADER-based sentiment scoring (Positive, Negative, Neutral) of simulated news headlines for the selected ticker.
        -   Display of average sentiment and individual headline scores.
        -   Option to enable/disable sentiment analysis.
    -   **Backtesting Engine**:
        -   Test predefined trading strategies: SMA Crossover and RSI Mean Reversion.
        -   Customize strategy parameters, date ranges, and initial capital.
        -   View detailed performance metrics: Total Return, Annualized Return, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, and more.
        -   Visualize portfolio performance against a benchmark (stock's close price) with trade markers.
    """)
    st.markdown("---")

    st.subheader("How to Use")
    st.markdown("""
    1.  **Select an Application Mode**: Use the "Choose App Mode" selectbox in the sidebar to navigate between:
        *   `Trading Analysis`: For interactive charting and current stock analysis.
        *   `Backtesting`: To test trading strategies on historical data.
        *   `About`: To learn more about this application.

    2.  **Trading Analysis Mode**:
        *   Enter a stock ticker symbol (e.g., "AAPL", "MSFT").
        *   Select the date range for analysis.
        *   Choose technical indicators to overlay and adjust their parameters.
        *   Select your preferred chart type (Line or Candlestick).
        *   Optionally, enable "ML Price Prediction" and/or "Sentiment Analysis" sections using the checkboxes in the sidebar. Results will appear in their respective tabs.

    3.  **Backtesting Mode**:
        *   Go to the "Backtest Settings" section in the sidebar.
        *   Enter a ticker symbol, select a date range, and set the initial capital.
        *   Choose a strategy ("SMA Crossover" or "RSI Mean Reversion").
        *   Adjust the parameters for the selected strategy.
        *   Click the "Run Backtest" button. Results (metrics, charts, trades log) will be displayed in the main area.
    """)
    st.markdown("---")

    st.subheader("Data Sources & Technologies")
    st.markdown("""
    -   **Stock Data**: Fetched from Yahoo Finance via the `yfinance` library.
    -   **Technical Indicators**: Calculated using `pandas` and standard financial formulas.
    -   **Machine Learning Model**: Implemented using `scikit-learn` (Logistic Regression).
    -   **Sentiment Analysis**: Utilizes the `vaderSentiment` library for sentiment scoring (Note: News headlines are currently simulated within the app).
    -   **Charting**: Powered by `Plotly`.
    -   **Application Framework**: Built with `Streamlit`.
    """)
    st.markdown("---")

    st.subheader("Important Disclaimers")
    st.warning("""
    -   **For Educational & Informational Purposes Only**: This application and all its content are provided strictly for educational and informational purposes. They do not constitute financial, investment, or trading advice.
    -   **Trading Involves Risk**: Trading and investing in financial markets carry a substantial risk of loss. Decisions should be made carefully and ideally with the guidance of a qualified financial advisor.
    -   **Past Performance is Not Indicative of Future Results**: Any historical performance, whether actual or backtested, does not guarantee future outcomes. Market conditions change, and strategies that worked in the past may not work in the future.
    -   **Backtesting Limitations**: The backtesting engine simulates trading based on historical data. It does not account for real-world factors such as slippage (difference between expected and actual trade execution price), commissions, taxes, or the impact of trades on market liquidity. These factors can significantly affect actual trading results.
    -   **Simulated News**: The sentiment analysis feature currently uses simulated news headlines. Therefore, the sentiment scores generated are illustrative and not based on real-time news.
    -   **No Warranty**: This application is provided "as-is" without any warranties of accuracy, completeness, or reliability.
    """)
    st.markdown("---")
    st.markdown(f"*Jules Swing Trade Pro - Version 1.0 (Conceptual Build)*")

# --- Backtesting Mode Handler Function ---
def handle_backtest_execution(bt_ticker, bt_start_date, bt_end_date, initial_capital, strategy_params, strategy_type_display):
    """
    Handles the execution of the backtest and displays results.
    """
    st.subheader(f"Backtest Results for {bt_ticker} ({strategy_type_display})")
    with st.spinner(f"Running backtest for {bt_ticker}... This may take a moment."):
        try:
            hist_data = get_historical_data(bt_ticker,
                                            bt_start_date.strftime("%Y-%m-%d"),
                                            bt_end_date.strftime("%Y-%m-%d"))

            required_data_length = 20
            if 'long_window' in strategy_params:
                required_data_length = max(required_data_length, strategy_params.get('long_window', 0) + 5)
            if 'rsi_window' in strategy_params:
                 required_data_length = max(required_data_length, strategy_params.get('rsi_window', 0) + 5)

            if hist_data.empty or len(hist_data) < required_data_length:
                st.error(f"Could not fetch sufficient historical data. Need at least {required_data_length} data points for selected settings and date range.")
                st.stop()

            if strategy_params['type'] == 'sma_crossover':
                hist_data[strategy_params['short_window_col']] = calculate_sma(hist_data, strategy_params['short_window'])
                hist_data[strategy_params['long_window_col']] = calculate_sma(hist_data, strategy_params['long_window'])
            elif strategy_params['type'] == 'rsi_reversion':
                hist_data[strategy_params['rsi_col']] = calculate_rsi(hist_data, strategy_params['rsi_window'])

            hist_data.dropna(inplace=True)
            if hist_data.empty or len(hist_data) < 2:
                 st.error("Data became empty or too short after indicator calculation and NaN removal. Try a longer date range or different indicator parameters.")
                 st.stop()

            portfolio_df, trades_list = run_backtest(hist_data, strategy_params, initial_capital)

            if portfolio_df is None or portfolio_df.empty:
                st.warning("Backtest did not generate results. This might happen if no trades were made or data was insufficient after indicator processing.")
            else:
                st.subheader("Backtest Performance Metrics")
                st.markdown("#### Key Performance Indicators")
                kpi_cols = st.columns(3)
                total_return = calculate_total_return(portfolio_df)
                annualized_return = calculate_annualized_return(portfolio_df)
                sharpe = calculate_sharpe_ratio(portfolio_df)

                kpi_cols[0].metric("Total Return", f"{total_return*100:.2f}%")
                kpi_cols[1].metric("Annualized Return", f"{annualized_return*100:.2f}%")
                kpi_cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}")

                kpi_cols2 = st.columns(3)
                max_dd = calculate_max_drawdown(portfolio_df)
                trade_summary_stats = summarize_trades(trades_list)
                win_rate_val = calculate_win_rate(trades_list)
                profit_factor_val = trade_summary_stats['profit_factor']

                kpi_cols2[0].metric("Maximum Drawdown", f"{max_dd*100:.2f}%")
                kpi_cols2[1].metric("Win Rate", f"{win_rate_val*100:.2f}%")
                kpi_cols2[2].metric("Profit Factor", f"{profit_factor_val:.2f}" if profit_factor_val != float('inf') else "Infinity")

                st.markdown("#### Trade Statistics")
                trade_stat_cols = st.columns(2)
                trade_stat_cols[0].metric("Number of Trades", trade_summary_stats['num_trades'])
                trade_stat_cols[0].metric("Winning Trades", trade_summary_stats['num_winning_trades'])
                trade_stat_cols[0].metric("Losing Trades", trade_summary_stats['num_losing_trades'])

                trade_stat_cols[1].metric("Avg. Gain per Trade", f"{trade_summary_stats['average_gain_per_trade']:.2f}")
                trade_stat_cols[1].metric("Avg. Loss per Trade", f"{trade_summary_stats['average_loss_per_trade']:.2f}")
                trade_stat_cols[1].metric("Gross Profit", f"{trade_summary_stats['total_gross_profit']:.2f}")
                trade_stat_cols[1].metric("Gross Loss", f"{trade_summary_stats['total_gross_loss']:.2f}")

                st.subheader("Portfolio Performance vs. Benchmark")
                aligned_hist_data_close = hist_data['Close'].reindex(portfolio_df.index)
                normalized_portfolio = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0]) * 100
                if not aligned_hist_data_close.empty and not pd.isna(aligned_hist_data_close.iloc[0]) and aligned_hist_data_close.iloc[0] != 0:
                    normalized_benchmark = (aligned_hist_data_close / aligned_hist_data_close.iloc[0]) * 100
                else:
                    normalized_benchmark = pd.Series(100, index=portfolio_df.index)
                    st.caption("Benchmark normalization failed; showing flat line for benchmark.")

                fig_bt_perf = go.Figure()
                fig_bt_perf.add_trace(go.Scatter(x=normalized_portfolio.index, y=normalized_portfolio,
                                                 mode='lines', name='Portfolio Value (Normalized)'))
                fig_bt_perf.add_trace(go.Scatter(x=normalized_benchmark.index, y=normalized_benchmark,
                                                 mode='lines', name=f'{bt_ticker} Benchmark (Normalized)',
                                                 line=dict(dash='dot')))

                if st.checkbox("Show Trade Markers on Portfolio Chart", key="bt_show_trade_markers"):
                    buy_dates, buy_values_portfolio, sell_dates, sell_values_portfolio = [], [], [], []
                    for trade in trades_list:
                        trade_date = pd.to_datetime(trade['date'])
                        if trade_date in normalized_portfolio.index:
                            portfolio_val_at_trade = normalized_portfolio.loc[trade_date]
                            if trade['type'] == 'BUY':
                                buy_dates.append(trade_date)
                                buy_values_portfolio.append(portfolio_val_at_trade)
                            elif trade['type'] == 'SELL':
                                sell_dates.append(trade_date)
                                sell_values_portfolio.append(portfolio_val_at_trade)

                    fig_bt_perf.add_trace(go.Scatter(x=buy_dates, y=buy_values_portfolio, mode='markers', name='Buy Orders',
                                        marker_symbol='triangle-up', marker_color='green', marker_size=10))
                    fig_bt_perf.add_trace(go.Scatter(x=sell_dates, y=sell_values_portfolio, mode='markers', name='Sell Orders',
                                        marker_symbol='triangle-down', marker_color='red', marker_size=10))

                fig_bt_perf.update_layout(title='Portfolio Value vs. Benchmark (Normalized to 100)',
                                          xaxis_title='Date', yaxis_title='Normalized Value',
                                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_bt_perf, use_container_width=True)

                if st.checkbox("Show Trades List", key="bt_show_trades_df"):
                    st.subheader("Trades Log")
                    if trades_list: st.dataframe(pd.DataFrame(trades_list))
                    else: st.info("No trades were executed during this backtest.")

                st.info("Disclaimer: Backtesting results are based on historical data and do not guarantee future performance. Slippage, commissions, and other transaction costs are not considered in this simulation.")

        except Exception as e:
            st.error(f"An error occurred during backtesting: {e}")
            st.error("Make sure the selected date range provides enough data for indicator calculations (e.g., more days than the longest window).")

# --- Trading Analysis Mode Render Functions ---
def render_price_chart_tab(stock_data_df, ticker_symbol, chart_type,
                           selected_indicators_list, sma_window_val, ema_window_val, rsi_window_val):
    """
    Renders the Price Chart tab content for the Trading Analysis mode.

    Args:
        stock_data_df (pd.DataFrame): DataFrame containing historical stock data.
        ticker_symbol (str): The stock ticker symbol.
        chart_type (str): Type of chart ('line' or 'candlestick').
        selected_indicators_list (list): List of selected indicators (e.g., ["SMA", "RSI"]).
        sma_window_val (int): Window for SMA.
        ema_window_val (int): Window for EMA.
        rsi_window_val (int): Window for RSI.
    """
    st.subheader("Price Chart & Technical Indicators")
    if not stock_data_df.empty:
        sma_series_data, ema_series_data, rsi_series_data = None, None, None
        if 'Close' in stock_data_df.columns:
            if "SMA" in selected_indicators_list: sma_series_data = calculate_sma(stock_data_df, sma_window_val)
            if "EMA" in selected_indicators_list: ema_series_data = calculate_ema(stock_data_df, ema_window_val)
            if "RSI" in selected_indicators_list: rsi_series_data = calculate_rsi(stock_data_df, rsi_window_val)

        price_chart_fig = plot_stock_prices(data=stock_data_df,
                                            ticker_symbol=ticker_symbol,
                                            chart_type=chart_type.lower(),
                                            sma_series=sma_series_data, sma_window=sma_window_val,
                                            ema_series=ema_series_data, ema_window=ema_window_val,
                                            rsi_series=rsi_series_data, rsi_window=rsi_window_val)
        st.plotly_chart(price_chart_fig, use_container_width=True)
    else:
        st.warning(f"No historical data for '{ticker_symbol}' in selected range to display chart.")

def render_historical_data_tab(stock_data_df):
    """
    Renders the Historical Data tab content for the Trading Analysis mode.

    Args:
        stock_data_df (pd.DataFrame): DataFrame containing historical stock data.
    """
    st.subheader("Historical Data")
    if not stock_data_df.empty:
        st.dataframe(stock_data_df.sort_index(ascending=False))
    else:
        st.warning("No historical data to display.")

def render_ml_predictions_tab(stock_data_df, is_ml_enabled):
    """
    Renders the ML Predictions tab content for the Trading Analysis mode.

    Args:
        stock_data_df (pd.DataFrame): DataFrame containing historical stock data.
        is_ml_enabled (bool): Flag indicating if ML predictions are enabled.
    """
    st.subheader("ML-Based Price Prediction (Experimental)")
    if not is_ml_enabled:
        st.info("ML Prediction is disabled. Enable it from the sidebar to see predictions.")
        return

    if not stock_data_df.empty:
        try:
            X_ml, y_ml = prepare_features_for_ml(stock_data_df.copy())
            min_data_for_ml = 30
            if X_ml.empty or y_ml.empty or len(X_ml) < min_data_for_ml:
                st.warning(f"Not enough data for ML (need {min_data_for_ml} points, got {len(X_ml)}).")
            else:
                with st.spinner("Training ML model..."):
                    old_stdout_ml = sys.stdout; sys.stdout = captured_output_ml = StringIO()
                    trained_model = train_model(X_ml, y_ml)
                    sys.stdout = old_stdout_ml; model_train_log = captured_output_ml.getvalue()

                if trained_model:
                    st.text("Model Training Log:"); st.text_area("Log_ml", model_train_log, height=100)
                    latest_features = X_ml.iloc[[-1]]
                    prediction, probability = make_prediction(trained_model, latest_features)
                    if prediction is not None:
                        pred_text = "**UP**" if prediction == 1 else "**DOWN/SAME**"
                        st.markdown(f"**Prediction for Next Day:** {pred_text} (Confidence: {probability*100:.2f}%)")
                    st.caption("ML model is experimental. For informational purposes only.")
                else:
                    st.error("Failed to train ML model.")
                    if model_train_log: st.text_area("Log_ml_fail", model_train_log, height=100)
        except Exception as e: st.error(f"ML Error: {e}")
    else:
        st.warning("ML: No historical data available for prediction.")

def render_sentiment_analysis_tab(ticker_symbol, is_sentiment_enabled):
    """
    Renders the Sentiment Analysis tab content for the Trading Analysis mode.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        is_sentiment_enabled (bool): Flag indicating if sentiment analysis is enabled.
    """
    st.subheader("Sentiment Analysis (Simulated News)")
    if not is_sentiment_enabled:
        st.info("Sentiment Analysis is disabled. Enable it from the sidebar to see sentiment data.")
        return

    with st.spinner(f"Fetching and analyzing news sentiment for {ticker_symbol}..."):
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

                with st.expander("View Individual News Headlines & Sentiments"):
                    for item, scores_data in zip(news_items, sentiment_scores_detailed):
                        compound = scores_data['sentiment']['compound']
                        item_sentiment_text = "Neutral"
                        if compound > 0.05: item_sentiment_text = "Positive"
                        elif compound < -0.05: item_sentiment_text = "Negative"
                        st.markdown(f"**Headline:** {item['headline']}\n\n*Source:* {item['source']} | *Date:* {item['date']}\n\n*Sentiment:* {item_sentiment_text} (Compound: {compound:.4f})")
                        st.divider()
                st.caption("Sentiment based on simulated news & VADER.")
            else: st.info(f"No simulated news for '{ticker_symbol}'.")
        except Exception as e: st.error(f"Sentiment Analysis Error: {e}")

def render_company_info_tab(company_info, ticker_symbol):
    """
    Renders the Company Info tab content for the Trading Analysis mode.

    Args:
        company_info (dict): Dictionary containing company information.
        ticker_symbol (str): The stock ticker symbol.
    """
    st.subheader("Company Information")
    if company_info and company_info.get('regularMarketPrice') is not None:
        st.write(f"**Name:** {company_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
        st.write(f"**Website:** {company_info.get('website', 'N/A')}")
        with st.expander("About Company (Business Summary)"):
            st.write(company_info.get('longBusinessSummary', 'No summary available.'))
    elif not company_info :
         st.warning(f"Could not retrieve company information for '{ticker_symbol}' due to earlier error.")
    else:
        st.warning(f"No detailed company information found for '{ticker_symbol}'.")

# --- Main App Structure ---
st.sidebar.title("Jules Swing Trade Pro")
app_mode = st.sidebar.selectbox("Choose App Mode",
                                ["Trading Analysis", "Backtesting", "About"])

if app_mode == "Trading Analysis":
    st.title("Trading Analysis Dashboard")

    # --- Sidebar for User Inputs (Trading Analysis) ---
    # These inputs are now part of the main_trading_analysis_page function
    # ticker_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL", key="ta_ticker").upper()

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

    # Chart Type Selection for Trading Analysis
    st.sidebar.subheader("Chart Settings")
    ta_chart_type = st.sidebar.radio("Select Chart Type:", ('Line', 'Candlestick'), key="ta_chart_type")

    enable_ml_prediction = st.sidebar.checkbox("Enable ML Price Prediction", value=False, key="ta_enable_ml")
    enable_sentiment_analysis = st.sidebar.checkbox("Enable Sentiment Analysis", value=False, key="ta_enable_sentiment")

    # --- Main Content Area (Trading Analysis) ---
    if not ticker_symbol:
        st.info("Please enter a stock ticker symbol in the sidebar to get started for Trading Analysis.")
        st.stop()

    st.header(f"Stock Analysis for {ticker_symbol}")

    # Fetch data once, with a spinner
    company_info = None
    stock_data_df = pd.DataFrame()

    with st.spinner(f"Fetching market data for {ticker_symbol}..."):
        try:
            company_info = get_company_info(ticker_symbol)
        except Exception as e:
            st.error(f"Error fetching company info: {e}")
            company_info = {} # Ensure it's a dict

        try:
            stock_data_df = get_historical_data(ticker_symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            # stock_data_df remains empty

    # Define tab structure
    tab_titles = ["Price Chart", "Historical Data", "Company Info"]
    if enable_ml_prediction:
        tab_titles.insert(2, "ML Predictions") # Insert before Sentiment
    if enable_sentiment_analysis:
        # Insert after ML or at position 2 if ML is not enabled
        ml_offset = 1 if enable_ml_prediction else 0
        tab_titles.insert(2 + ml_offset, "Sentiment Analysis")

    tabs = st.tabs(tab_titles)

    with tabs[0]: # Price Chart Tab
        st.subheader("Price Chart & Technical Indicators")
        if not stock_data_df.empty:
            if 'Close' in stock_data_df.columns:
                # Calculate indicators only if selected
                current_sma_series, current_ema_series, current_rsi_series = None, None, None
                if "SMA" in selected_indicators: current_sma_series = calculate_sma(stock_data_df, sma_window)
                if "EMA" in selected_indicators: current_ema_series = calculate_ema(stock_data_df, ema_window)
                if "RSI" in selected_indicators: current_rsi_series = calculate_rsi(stock_data_df, rsi_window)

            price_chart_fig = plot_stock_prices(data=stock_data_df,
                                                ticker_symbol=ticker_symbol,
                                                chart_type=ta_chart_type.lower(),
                                                sma_series=current_sma_series, sma_window=sma_window,
                                                ema_series=current_ema_series, ema_window=ema_window,
                                                rsi_series=current_rsi_series, rsi_window=rsi_window)
            st.plotly_chart(price_chart_fig, use_container_width=True)
        else:
            st.warning(f"No historical data for '{ticker_symbol}' in selected range to display chart.")

    with tabs[1]: # Historical Data Tab
        st.subheader("Historical Data Preview")
        if not stock_data_df.empty:
            st.dataframe(stock_data_df.sort_index(ascending=False)) # Show all data, not just head(20)
        else:
            st.warning("No historical data to display.")

    # Handle optional tabs based on their presence in tab_titles
    current_tab_index = 2 # Starts after "Price Chart" and "Historical Data"

    if enable_ml_prediction:
        with tabs[current_tab_index]:
            st.subheader("ML-Based Price Prediction (Experimental)")
            if not stock_data_df.empty:
                try:
                    X_ml, y_ml = prepare_features_for_ml(stock_data_df.copy())
                    min_data_for_ml = 30
                    if X_ml.empty or y_ml.empty or len(X_ml) < min_data_for_ml:
                        st.warning(f"Not enough data for ML (need {min_data_for_ml} points, got {len(X_ml)}).")
                    else:
                        with st.spinner("Training ML model..."):
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
            else:
                st.warning("ML: No historical data available for prediction.")
        current_tab_index +=1

    if enable_sentiment_analysis:
        with tabs[current_tab_index]:
            st.subheader("Sentiment Analysis (Simulated News)")
            with st.spinner(f"Fetching and analyzing news sentiment for {ticker_symbol}..."):
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

                        with st.expander("View Individual News Headlines & Sentiments"):
                            for item, scores_data in zip(news_items, sentiment_scores_detailed):
                                compound = scores_data['sentiment']['compound']
                                item_sentiment_text = "Neutral"
                                if compound > 0.05: item_sentiment_text = "Positive"
                                elif compound < -0.05: item_sentiment_text = "Negative"
                                st.markdown(f"**Headline:** {item['headline']}\n\n*Source:* {item['source']} | *Date:* {item['date']}\n\n*Sentiment:* {item_sentiment_text} (Compound: {compound:.4f})")
                                st.divider()
                        st.caption("Sentiment based on simulated news & VADER.")
                    else:
                        st.info(f"No simulated news for '{ticker_symbol}'.")
                except Exception as e:
                    st.error(f"Sentiment Analysis Error: {e}")
        current_tab_index += 1

    # Company Info Tab (always the last one based on initial tab_titles)
    # Need to adjust its index if optional tabs are not shown
    company_info_tab_idx = tab_titles.index("Company Info")
    with tabs[company_info_tab_idx]:
        st.subheader("Company Information")
        if company_info and company_info.get('regularMarketPrice') is not None:
            st.write(f"**Name:** {company_info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
            st.write(f"**Website:** {company_info.get('website', 'N/A')}")
            with st.expander("About Company (Business Summary)"):
                st.write(company_info.get('longBusinessSummary', 'No summary available.'))
        elif not company_info : # If error occurred during fetch and it's None
             st.warning(f"Could not retrieve company information for '{ticker_symbol}' due to earlier error.")
        else: # If info is empty dict but no error
            st.warning(f"No detailed company information found for '{ticker_symbol}'.")


elif app_mode == "Backtesting":
    st.title("Strategy Backtester")

    # --- Sidebar Inputs for Backtesting ---
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

    st.sidebar.header("Backtest Settings") # Ensure this header is here for context
    bt_ticker = st.sidebar.text_input("Ticker for Backtest", value="AAPL", key="bt_ticker_input").upper() # Changed key
    bt_default_start = datetime.date(2022, 1, 1)
    bt_default_end = datetime.date(2023, 1, 1)
    bt_start_date = st.sidebar.date_input("Backtest Start Date", bt_default_start, key="bt_start_date_input") # Changed key
    bt_end_date = st.sidebar.date_input("Backtest End Date", bt_default_end, key="bt_end_date_input") # Changed key
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=100000, key="bt_capital_input") # Changed key

    if bt_start_date > bt_end_date:
        st.sidebar.error("Error: Backtest start date must be before end date.")
        st.stop()

    strategy_type_display = st.sidebar.selectbox("Choose Strategy",
                                         ["SMA Crossover", "RSI Mean Reversion"], key="bt_strategy_select") # Changed key

    strategy_params = {}
    if strategy_type_display == "SMA Crossover":
        strategy_params['type'] = 'sma_crossover'
        strategy_params['short_window'] = st.sidebar.slider("Short SMA Window", 5, 50, 10, key="bt_sma_short_slider") # Changed key
        strategy_params['long_window'] = st.sidebar.slider("Long SMA Window", 20, 200, 50, key="bt_sma_long_slider") # Changed key
        strategy_params['short_window_col'] = f'SMA_{strategy_params["short_window"]}'
        strategy_params['long_window_col'] = f'SMA_{strategy_params["long_window"]}'
    elif strategy_type_display == "RSI Mean Reversion":
        strategy_params['type'] = 'rsi_reversion'
        strategy_params['rsi_window'] = st.sidebar.slider("RSI Window", 5, 30, 14, key="bt_rsi_window_slider") # Changed key
        strategy_params['oversold_threshold'] = st.sidebar.slider("RSI Oversold Threshold", 10, 40, 30, key="bt_rsi_oversold_slider") # Changed key
        strategy_params['overbought_threshold'] = st.sidebar.slider("RSI Overbought Threshold", 60, 90, 70, key="bt_rsi_overbought_slider") # Changed key
        strategy_params['rsi_col'] = f'RSI_{strategy_params["rsi_window"]}'

    if st.sidebar.button("Run Backtest", key="bt_run_button"): # Changed key
        if not bt_ticker:
            st.error("Please enter a ticker symbol for backtesting.")
        else:
            handle_backtest_execution(bt_ticker, bt_start_date, bt_end_date, initial_capital, strategy_params, strategy_type_display)

elif app_mode == "About":
    render_about_page()

# else: (If more modes are added)
#    st.info("Select an application mode from the sidebar.")
