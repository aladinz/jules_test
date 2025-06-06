import streamlit as st
import datetime
import pandas as pd
from io import StringIO
import sys
import plotly.graph_objects as go

import yfinance as yf # Ensure yfinance is imported as yf
from sklearn.linear_model import LogisticRegression # Added for debug expander
# Import custom modules
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

# --- Helper Functions / Render Functions ---

def render_about_page():
    """Renders the 'About' page content."""
    st.title("About Jules Swing Trade Pro")
    st.markdown("---")
    st.markdown("Welcome to **Jules Swing Trade Pro**! ... (rest of About content from previous steps)")
    st.markdown("""
    Welcome to **Jules Swing Trade Pro**! This application is designed to assist swing traders and
    market enthusiasts by providing a suite of tools for stock analysis, strategy backtesting,
    and exploring potential market insights. It's intended for users who want to combine technical
    analysis with experimental machine learning and sentiment data to inform their trading decisions.
    """)
    st.markdown("---")
    st.subheader("Key Features")
    st.markdown("""
    -   **Trading Analysis Mode**: Interactive charts, technical indicators.
    -   **Machine Learning Insights**: Next-day price direction prediction.
    -   **Sentiment Analysis**: VADER-based sentiment scoring.
    -   **Backtesting Engine**: Test strategies with performance metrics.
    """)
    st.markdown("---")
    st.subheader("How to Use")
    st.markdown("Select modes from the sidebar. Input data as prompted.")
    st.markdown("---")
    st.subheader("Data Sources & Technologies")
    st.markdown("Yahoo Finance, scikit-learn, VADER, Plotly, Streamlit.")
    st.markdown("---")
    st.subheader("Important Disclaimers")
    st.warning("Educational purposes only. Not financial advice. Trading involves risk.")
    st.markdown("---")
    st.markdown(f"*Jules Swing Trade Pro - Version 1.0 (Conceptual Build)*")

# --- Market Overview Page Render Function ---
def render_market_overview_page():
    st.title("U.S. Market Overview")

    indices = {
        "S&P 500": "^GSPC",
        "Dow 30": "^DJI",
        "Nasdaq Comp.": "^IXIC"  # Changed key here
    }

    period_options = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y"}
    selected_period_label = st.radio(
        "Chart Period:",
        list(period_options.keys()),
        index=2,
        horizontal=True,
        key="market_overview_period_radio"
    )
    yf_period = period_options[selected_period_label]

    st.markdown("---")

    cols = st.columns(len(indices))

    for i, (name, symbol) in enumerate(indices.items()):
        with cols[i]:
            st.subheader(name)
            try:
                with st.spinner(f"Fetching {name}..."):
                    ticker_obj = yf.Ticker(symbol)
                    info = ticker_obj.info

                    current_price = info.get('regularMarketPrice', info.get('currentPrice'))
                    prev_close = info.get('regularMarketPreviousClose')
                    price_change = None
                    percent_change = None

                    if current_price is not None and prev_close is not None:
                        price_change = current_price - prev_close
                        if prev_close != 0:
                            percent_change = (price_change / prev_close) * 100
                        else:
                            percent_change = float('inf') if price_change > 0 else float('-inf') if price_change < 0 else 0.0

                        delta_text = f"{price_change:,.2f} ({percent_change:.2f}%)" if percent_change is not None else None
                        st.metric(
                            label="Current Level",
                            value=f"{current_price:,.2f}" if current_price is not None else "N/A",
                            delta=delta_text,
                            delta_color=("inverse" if price_change is not None and price_change < 0 else "normal")
                        )
                    else:
                        st.metric(label="Current Level", value="Fetching...", delta="Fetching...")
                        # Fallback to historical data for last close if info lacks current price
                        hist_for_latest = get_historical_data(symbol, period="5d", interval="1d")
                        if not hist_for_latest.empty and len(hist_for_latest) >= 2:
                            latest_close = hist_for_latest['Close'].iloc[-1]
                            prev_day_close = hist_for_latest['Close'].iloc[-2]
                            price_change = latest_close - prev_day_close
                            percent_change = (price_change / prev_day_close) * 100 if prev_day_close != 0 else 0.0
                            delta_text = f"{price_change:,.2f} ({percent_change:.2f}%)"
                            st.metric(
                                label="Last Close",
                                value=f"{latest_close:,.2f}",
                                delta=delta_text,
                                delta_color=("inverse" if price_change < 0 else "normal")
                            )
                        elif not hist_for_latest.empty:
                             st.metric(label="Last Close", value=f"{hist_for_latest['Close'].iloc[-1]:,.2f}", delta="N/A")
                        else:
                            st.metric(label="Current Level", value="N/A", delta="N/A") # Final fallback

                    # Historical Data for Chart
                    hist_df = get_historical_data(symbol, period=yf_period, interval="1d")
                    if not hist_df.empty and 'Close' in hist_df.columns:
                        st.line_chart(hist_df['Close'], use_container_width=True)
                    else:
                        st.warning("Chart data unavailable.")
            except Exception as e:
                st.error(f"Data error for {name}")
                print(f"Error fetching data for {name} ({symbol}): {e}") # For server log


def handle_backtest_execution(bt_ticker, bt_start_date, bt_end_date, initial_capital, strategy_params, strategy_type_display):
    """Handles the execution of the backtest and displays results."""
    st.subheader(f"Backtest Results for {bt_ticker} ({strategy_type_display})")
    with st.spinner(f"Running backtest for {bt_ticker}... This may take a moment."):
        try:
            hist_data = get_historical_data(bt_ticker, bt_start_date.strftime("%Y-%m-%d"), bt_end_date.strftime("%Y-%m-%d"))
            required_data_length = 20
            if 'long_window' in strategy_params: required_data_length = max(required_data_length, strategy_params.get('long_window', 0) + 5)
            if 'rsi_window' in strategy_params: required_data_length = max(required_data_length, strategy_params.get('rsi_window', 0) + 5)

            if hist_data.empty or len(hist_data) < required_data_length:
                st.error(f"Could not fetch sufficient historical data. Need at least {required_data_length} data points."); st.stop()

            if strategy_params['type'] == 'sma_crossover':
                hist_data[strategy_params['short_window_col']] = calculate_sma(hist_data, strategy_params['short_window'])
                hist_data[strategy_params['long_window_col']] = calculate_sma(hist_data, strategy_params['long_window'])
            elif strategy_params['type'] == 'rsi_reversion':
                hist_data[strategy_params['rsi_col']] = calculate_rsi(hist_data, strategy_params['rsi_window'])

            hist_data.dropna(inplace=True)
            if hist_data.empty or len(hist_data) < 2: st.error("Data empty/too short after indicators. Adjust params/dates."); st.stop()

            portfolio_df, trades_list = run_backtest(hist_data, strategy_params, initial_capital)

            if portfolio_df is None or portfolio_df.empty: st.warning("Backtest generated no results."); return

            st.subheader("Backtest Performance Metrics")
            st.markdown("#### Key Performance Indicators")
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("Total Return", f"{calculate_total_return(portfolio_df)*100:.2f}%")
            kpi_cols[1].metric("Annualized Return", f"{calculate_annualized_return(portfolio_df)*100:.2f}%")
            kpi_cols[2].metric("Sharpe Ratio", f"{calculate_sharpe_ratio(portfolio_df):.2f}")

            kpi_cols2 = st.columns(3)
            trade_summary = summarize_trades(trades_list) # Calculate once
            kpi_cols2[0].metric("Maximum Drawdown", f"{calculate_max_drawdown(portfolio_df)*100:.2f}%")
            kpi_cols2[1].metric("Win Rate", f"{calculate_win_rate(trades_list)*100:.2f}%")
            profit_factor = trade_summary['profit_factor']
            kpi_cols2[2].metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "Infinity")

            st.markdown("#### Trade Statistics")
            ts_cols = st.columns(2)
            ts_cols[0].metric("Number of Trades", trade_summary['num_trades'])
            ts_cols[0].metric("Winning Trades", trade_summary['num_winning_trades'])
            ts_cols[0].metric("Losing Trades", trade_summary['num_losing_trades'])
            ts_cols[1].metric("Avg. Gain / Trade", f"{trade_summary['average_gain_per_trade']:.2f}")
            ts_cols[1].metric("Avg. Loss / Trade", f"{trade_summary['average_loss_per_trade']:.2f}")
            ts_cols[1].metric("Gross Profit", f"{trade_summary['total_gross_profit']:.2f}")
            ts_cols[1].metric("Gross Loss", f"{trade_summary['total_gross_loss']:.2f}")

            st.subheader("Portfolio Performance vs. Benchmark")
            aligned_close = hist_data['Close'].reindex(portfolio_df.index)
            norm_portfolio = (portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0]) * 100
            if not aligned_close.empty and not pd.isna(aligned_close.iloc[0]) and aligned_close.iloc[0] != 0:
                norm_benchmark = (aligned_close / aligned_close.iloc[0]) * 100
            else:
                norm_benchmark = pd.Series(100, index=portfolio_df.index); st.caption("Benchmark normalization failed.")

            bt_fig = go.Figure()
            bt_fig.add_trace(go.Scatter(x=norm_portfolio.index, y=norm_portfolio, name='Portfolio'))
            bt_fig.add_trace(go.Scatter(x=norm_benchmark.index, y=norm_benchmark, name=f'{bt_ticker} Benchmark', line=dict(dash='dot')))
            if st.checkbox("Show Trade Markers", key="bt_trade_markers"):
                buys_x, buys_y, sells_x, sells_y = [],[],[],[]
                for trade in trades_list:
                    t_date = pd.to_datetime(trade['date'])
                    if t_date in norm_portfolio.index:
                        val = norm_portfolio.loc[t_date]
                        if trade['type'] == 'BUY': buys_x.append(t_date); buys_y.append(val)
                        else: sells_x.append(t_date); sells_y.append(val)
                bt_fig.add_trace(go.Scatter(x=buys_x, y=buys_y, mode='markers', name='Buys', marker=dict(symbol='triangle-up', color='green', size=10)))
                bt_fig.add_trace(go.Scatter(x=sells_x, y=sells_y, mode='markers', name='Sells', marker=dict(symbol='triangle-down', color='red', size=10)))
            bt_fig.update_layout(title='Portfolio vs. Benchmark (Normalized)', xaxis_title='Date', yaxis_title='Normalized Value', legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(bt_fig, use_container_width=True)

            if st.checkbox("Show Trades Log", key="bt_show_trades_log"):
                st.subheader("Trades Log")
                if trades_list:
                    trades_df = pd.DataFrame(trades_list)
                    desired_columns = ['date', 'type', 'price', 'shares', 'pnl', 'reason', 'cash_change', 'cash_remaining']
                    ordered_columns = [col for col in desired_columns if col in trades_df.columns]
                    for col in trades_df.columns:
                        if col not in ordered_columns: ordered_columns.append(col)
                    trades_df_display = trades_df[ordered_columns].copy()
                    if 'date' in trades_df_display.columns:
                        try: trades_df_display['date'] = pd.to_datetime(trades_df_display['date']).dt.strftime('%Y-%m-%d')
                        except Exception: pass
                    for col_format in ['price', 'pnl', 'cash_change', 'cash_remaining']:
                        if col_format in trades_df_display.columns:
                            trades_df_display[col_format] = trades_df_display[col_format].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
                    if 'shares' in trades_df_display.columns:
                         trades_df_display['shares'] = trades_df_display['shares'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                    st.dataframe(trades_df_display)
                else: st.info("No trades were executed during this backtest.")
            st.info("Disclaimer: Backtesting is based on historical data and does not guarantee future results. Costs like slippage/commission are not included.")
        except Exception as e:
            st.error(f"Backtesting Error: {e}. Ensure date range allows for indicator calculations.")

def render_price_chart_tab(stock_data_df, ticker_symbol, chart_type, selected_indicators, sma_w, ema_w, rsi_w):
    st.subheader("Price Chart & Technical Indicators")
    if stock_data_df.empty: st.warning(f"No data for '{ticker_symbol}' to display chart."); return

    sma, ema, rsi = None, None, None
    if "SMA" in selected_indicators: sma = calculate_sma(stock_data_df, sma_w)
    if "EMA" in selected_indicators: ema = calculate_ema(stock_data_df, ema_w)
    if "RSI" in selected_indicators: rsi = calculate_rsi(stock_data_df, rsi_w)

    fig = plot_stock_prices(
        data=stock_data_df,
        ticker_symbol=ticker_symbol,
        price_column='Close',
        chart_type=chart_type.lower(),
        sma_series=sma, sma_window=sma_w,
        ema_series=ema, ema_window=ema_w,
        rsi_series=rsi, rsi_window=rsi_w
    )
    st.plotly_chart(fig, use_container_width=True)

def render_historical_data_tab(stock_data_df):
    st.subheader("Historical Data")
    st.dataframe(stock_data_df.sort_index(ascending=False) if not stock_data_df.empty else "No historical data.")

def render_ml_predictions_tab(stock_data_df, is_enabled):
    st.subheader("ML-Based Price Prediction (Experimental)")
    if not is_enabled: st.info("Enable ML from sidebar."); return
    if stock_data_df.empty: st.warning("No data for ML."); return
    accuracy_val, log_content, model = None, "", None
    try:
        X_ml, y_ml = prepare_features_for_ml(stock_data_df.copy())
        if X_ml.empty or len(X_ml) < 30: st.warning(f"Not enough data for ML (need 30 pts, got {len(X_ml)})."); return

        with st.expander("Debug: ML Training Data Info", expanded=True):
            st.write("Shape of X_ml (features):", X_ml.shape)
            st.write("Shape of y_ml (target):", y_ml.shape)
            if not y_ml.empty:
                st.write("Target Variable (y_ml) Distribution (1=UP, 0=DOWN/SAME):")
                st.dataframe(y_ml.value_counts(normalize=True).rename("percentage").to_frame())
            else:
                st.write("Target Variable (y_ml) is empty.")
            st.write("Features (X_ml) head (first 5 rows):")
            st.dataframe(X_ml.head())
            st.write("Features (X_ml) describe:")
            st.dataframe(X_ml.describe())

        with st.spinner("Training ML model..."):
            old_stdout = sys.stdout; sys.stdout = captured_output = StringIO()
            model, accuracy_val = train_model(X_ml, y_ml)
            sys.stdout = old_stdout; log_content = captured_output.getvalue()

        if model:
            pred, prob = make_prediction(model, X_ml.iloc[[-1]])

            with st.expander("Debug: ML Prediction Details", expanded=True):
                st.write(f"Raw probability for class 0 (DOWN/SAME): {1 - prob:.4f}" if prob is not None else "Prob N/A")
                st.write(f"Raw probability for class 1 (UP): {prob:.4f}" if prob is not None else "Prob N/A")
                st.write(f"Prediction made by model's .predict() (0 or 1): {pred}" if pred is not None else "Pred N/A")
                st.write(f"Threshold for .predict() is typically: 0.5")

                if isinstance(model, LogisticRegression) and hasattr(model, 'coef_') and hasattr(X_ml, 'columns'):
                    if X_ml.shape[1] == model.coef_[0].shape[0]:
                        st.write("Model Coefficients (Logistic Regression):")
                        try:
                            feature_names = [str(col) for col in X_ml.columns]
                            coeffs_df = pd.DataFrame(model.coef_[0], index=feature_names, columns=['Coefficient'])
                            st.dataframe(coeffs_df.sort_values(by='Coefficient', ascending=False))
                        except Exception as e_coeff:
                            st.write(f"Could not display coefficients due to error: {e_coeff}")
                            st.write(f"Model coef shape: {model.coef_[0].shape}, X_ml columns: {len(X_ml.columns)}")
                    else:
                        st.write(f"Could not display coefficients: Mismatch between number of feature columns in X_ml ({X_ml.shape[1]}) and model coefficients ({model.coef_[0].shape[0]}).")
                elif not isinstance(model, LogisticRegression):
                    st.write(f"Model is of type {type(model)}, not LogisticRegression. Cannot display coefficients directly.")

            if pred is not None: st.markdown(f"**Prediction for Next Day:** {'**UP**' if pred==1 else '**DOWN/SAME**'}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Test Accuracy", f"{accuracy_val*100:.2f}%" if accuracy_val is not None else "N/A")
                if accuracy_val is not None: st.progress(int(accuracy_val*100))
            with col2:
                if pred is not None and prob is not None:
                    st.metric("Prediction Confidence", f"{prob*100:.2f}%"); st.progress(int(prob*100))
                else: st.info("Prediction could not be made.")
            if log_content: st.text("Model Training Log (stdout):"); st.text_area("ml_log", log_content, height=100)
            st.caption("ML model is experimental. For informational use only.")
        else:
            st.error("Failed to train ML model.")
            if log_content: st.text("Model Training Log (stdout - Attempt):"); st.text_area("ml_log_fail", log_content, height=100)
    except Exception as e: st.error(f"ML Error: {e}")

def render_sentiment_analysis_tab(ticker_symbol, is_enabled):
    st.subheader("Sentiment Analysis (Simulated News)")
    if not is_enabled: st.info("Enable Sentiment from sidebar."); return
    with st.spinner(f"Fetching news for {ticker_symbol}..."):
        try:
            news = fetch_news(ticker_symbol)
            if not news: st.info(f"No simulated news for '{ticker_symbol}'."); return

            headlines = [n['headline'] for n in news]
            scores = analyze_sentiment_vader(headlines)
            avg_score = get_average_sentiment_score(scores)
            avg_text = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
            st.write(f"**Avg News Sentiment:** {avg_score:.4f} ({avg_text})")

            # Bar chart logic
            positive_count, neutral_count, negative_count = 0,0,0
            for score_data in scores:
                compound = score_data['sentiment']['compound']
                if compound > 0.05: positive_count += 1
                elif compound < -0.05: negative_count += 1
                else: neutral_count += 1

            chart_df = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Headlines': [positive_count, neutral_count, negative_count]
            })

            if chart_df['Headlines'].sum() > 0:
                st.subheader("News Sentiment Distribution")
                st.bar_chart(chart_df.set_index('Sentiment'))
            elif news:
                st.info("No distinct positive/negative sentiment categories to plot (e.g., all items were neutral).")

            with st.expander("View Individual News & Sentiments"):
                for item, score_data in zip(news, scores):
                    s = score_data['sentiment']; c = s['compound']
                    it = "Positive" if c > 0.05 else "Negative" if c < -0.05 else "Neutral"
                    st.markdown(f"**H:** {item['headline']}\n\n*S:* {item['source']} | *D:* {item['date']} | *Sent:* {it} ({c:.4f})"); st.divider()
            st.caption("Sentiment based on simulated news & VADER.")
        except Exception as e: st.error(f"Sentiment Error: {e}")

def render_company_info_tab(company_info, ticker_symbol):
    st.subheader("Company Information")
    if company_info and company_info.get('regularMarketPrice'):
        st.write(f"**Name:** {company_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
        st.write(f"**Website:** {company_info.get('website', 'N/A')}")
        with st.expander("About Company (Business Summary)"): st.write(company_info.get('longBusinessSummary', 'N/A'))
    else: st.warning(f"No detailed company info for '{ticker_symbol}'.")

def main_trading_analysis_page():
    """Main function to render the Trading Analysis mode page structure and logic."""
    st.title("Trading Analysis Dashboard")

    st.sidebar.header("User Input Features")
    # ticker_symbol is now defined inside main_trading_analysis_page
    ticker_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL", key="ta_ticker").upper()

    today = datetime.date.today(); default_start = datetime.date(today.year - 1, 1, 1)
    start_date = st.sidebar.date_input("Start Date", default_start, key="ta_start_date")
    end_date = st.sidebar.date_input("End Date", today, key="ta_end_date")
    if start_date > end_date: st.sidebar.error("Start date must be before end date."); st.stop()

    st.sidebar.subheader("Technical Indicators")
    selected_indicators = st.sidebar.multiselect("Indicators:", ["SMA", "EMA", "RSI"], key="ta_selected_indicators")
    sma_w = st.sidebar.slider("SMA Window", 5,100,20, key="ta_sma_window") if "SMA" in selected_indicators else 20
    ema_w = st.sidebar.slider("EMA Window", 5,100,20, key="ta_ema_window") if "EMA" in selected_indicators else 20
    rsi_w = st.sidebar.slider("RSI Window", 7,30,14, key="ta_rsi_window") if "RSI" in selected_indicators else 14

    st.sidebar.subheader("Chart Settings")
    ta_chart_type = st.sidebar.radio("Chart Type:", ('Line', 'Candlestick'), key="ta_chart_type")

    enable_ml = st.sidebar.checkbox("Enable ML Prediction", value=False, key="ta_enable_ml")
    enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis", value=False, key="ta_enable_sentiment")

    if not ticker_symbol: st.info("Enter ticker for Trading Analysis."); st.stop()
    st.header(f"Stock Analysis for {ticker_symbol}") # This needs ticker_symbol

    company_info_data, stock_data_df_main = None, pd.DataFrame()
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        try: company_info_data = get_company_info(ticker_symbol)
        except Exception as e: st.error(f"Company Info Error: {e}"); company_info_data={}
        try: stock_data_df_main = get_historical_data(ticker_symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        except Exception as e: st.error(f"Historical Data Error: {e}") # stock_data_df_main will remain empty

    if stock_data_df_main.empty:
        st.error(
            f"Failed to fetch valid historical data for '{ticker_symbol}' for the selected date range. \n"
            "Possible reasons:\n"
            "- Incorrect ticker symbol.\n"
            "- No data available for the specified dates (e.g., market holidays, before listing, future dates).\n"
            "- Network connectivity issues or temporary problems with the data provider (Yahoo Finance).\n\n"
            "Please check your inputs or try again later. No chart or further analysis can be displayed without this core data."
        )
        st.stop() # Stop execution for this page/view

    tab_titles = ["Price Chart", "Historical Data", "Company Info"]
    if enable_ml: tab_titles.insert(2, "ML Predictions")
    if enable_sentiment: tab_titles.insert(2 + (1 if enable_ml else 0), "Sentiment Analysis")

    tabs = st.tabs(tab_titles)
    with tabs[0]: render_price_chart_tab(stock_data_df_main, ticker_symbol, ta_chart_type, selected_indicators, sma_w, ema_w, rsi_w)
    with tabs[1]: render_historical_data_tab(stock_data_df_main)

    idx_offset = 2
    if enable_ml:
        with tabs[idx_offset]: render_ml_predictions_tab(stock_data_df_main, enable_ml)
        idx_offset +=1
    if enable_sentiment:
        with tabs[idx_offset]: render_sentiment_analysis_tab(ticker_symbol, enable_sentiment)

    company_info_tab_actual_idx = tab_titles.index("Company Info")
    with tabs[company_info_tab_actual_idx]: render_company_info_tab(company_info_data, ticker_symbol)

# --- Main App Structure ---
st.sidebar.title("Jules Swing Trade Pro")
app_mode = st.sidebar.selectbox("Choose App Mode",
                                ["Trading Analysis", "U.S. Market Overview", "Backtesting", "About"])

if app_mode == "Trading Analysis":
    main_trading_analysis_page()
elif app_mode == "U.S. Market Overview":
    render_market_overview_page()
elif app_mode == "Backtesting":
    st.title("Strategy Backtester")
    st.sidebar.header("Backtest Settings")
    bt_ticker = st.sidebar.text_input("Ticker for Backtest", "AAPL", key="bt_ticker_input").upper()
    bt_start = st.sidebar.date_input("Backtest Start", datetime.date(2022,1,1), key="bt_start_date_input")
    bt_end = st.sidebar.date_input("Backtest End", datetime.date(2023,1,1), key="bt_end_date_input")
    bt_capital = st.sidebar.number_input("Initial Capital", 1000, 1000000, 100000, key="bt_capital_input")
    if bt_start > bt_end: st.sidebar.error("Start date must be before end."); st.stop()

    bt_strategy_disp = st.sidebar.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"], key="bt_strategy_select")

    bt_params = {}
    st.sidebar.markdown("---"); st.sidebar.subheader("Risk Management")
    bt_params['stop_loss_pct'] = st.sidebar.number_input("Stop-Loss %", 0.0, 50.0, 0.0, 0.5, "%.1f", key="bt_sl")
    bt_params['take_profit_pct'] = st.sidebar.number_input("Take-Profit %", 0.0, 100.0, 0.0, 0.5, "%.1f", key="bt_tp")

    if bt_strategy_disp == "SMA Crossover":
        bt_params['type'] = 'sma_crossover'
        bt_params['short_window'] = st.sidebar.slider("Short SMA", 5,50,10, key="bt_sma_s")
        bt_params['long_window'] = st.sidebar.slider("Long SMA", 20,200,50, key="bt_sma_l")
        bt_params['short_window_col'] = f"SMA_{bt_params['short_window']}"
        bt_params['long_window_col'] = f"SMA_{bt_params['long_window']}"
    elif bt_strategy_disp == "RSI Mean Reversion":
        bt_params['type'] = 'rsi_reversion'
        bt_params['rsi_window'] = st.sidebar.slider("RSI Window", 5,30,14, key="bt_rsi_w")
        bt_params['oversold_threshold'] = st.sidebar.slider("Oversold",10,40,30, key="bt_rsi_os")
        bt_params['overbought_threshold'] = st.sidebar.slider("Overbought",60,90,70, key="bt_rsi_ob")
        bt_params['rsi_col'] = f"RSI_{bt_params['rsi_window']}"

    if st.sidebar.button("Run Backtest", key="bt_run_button"):
        if not bt_ticker: st.error("Enter ticker for backtest.")
        else: handle_backtest_execution(bt_ticker, bt_start, bt_end, bt_capital, bt_params, bt_strategy_disp)
elif app_mode == "About":
    render_about_page()
