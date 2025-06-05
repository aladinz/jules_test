# Jules Swing Trade Pro

## Overview

Jules Swing Trade Pro is a comprehensive stock trading analysis and backtesting platform built with Streamlit. It's designed for traders and market enthusiasts who want to leverage technical analysis, machine learning-based predictions, sentiment analysis, and robust strategy backtesting to inform their swing trading decisions. The application provides a user-friendly interface to explore stock data, visualize trends, and evaluate trading strategies.

## Key Features

*   **Trading Analysis Dashboard**:
    *   Interactive price charts (Line and Candlestick options).
    *   Volume display.
    *   Common technical indicators: Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI) with configurable window periods.
    *   Experimental Machine Learning predictions for next-day price direction (UP/DOWN/SAME) with confidence scores and model accuracy display.
    *   Sentiment analysis based on simulated news headlines, using VADER for scoring, including an average sentiment display and a distribution chart of positive/neutral/negative news items.
    *   Detailed company information and historical data tables.
    *   Organized into tabs for clarity: Price Chart, Historical Data, ML Predictions, Sentiment Analysis, Company Info.

*   **U.S. Market Overview**:
    *   Provides a snapshot of major U.S. market indices: S&P 500 (`^GSPC`), Dow 30 (`^DJI`), and Nasdaq Composite (`^IXIC`).
    *   Displays current levels, price changes, and percentage changes for each index.
    *   Includes trend charts for each index with selectable periods (1M, 3M, 6M, YTD, 1Y).

*   **Strategy Backtesting Engine**:
    *   Test predefined trading strategies:
        *   Simple Moving Average (SMA) Crossover.
        *   Relative Strength Index (RSI) Mean Reversion.
    *   Configure strategy parameters (indicator windows, RSI thresholds).
    *   Set Stop-Loss and Take-Profit percentages for risk management.
    *   View detailed performance metrics: Total Return, Annualized Return, Sharpe Ratio, Maximum Drawdown, Win Rate, Profit Factor, and more trade statistics.
    *   Visualize portfolio performance over time compared to a benchmark (the stock itself).
    *   Optional detailed log of all simulated trades, including exit reasons (strategy signal, stop-loss, take-profit).

## Technologies Used

*   **Frontend**: Streamlit
*   **Data Manipulation**: Pandas
*   **Financial Data Source**: Yahoo Finance (via `yfinance`)
*   **Charting**: Plotly (directly and via Streamlit's native charts)
*   **Machine Learning**: scikit-learn (Logistic Regression)
*   **Sentiment Analysis**: VADER (via `vaderSentiment`)
*   **Numerical Operations**: NumPy

## Setup and Installation

1.  **Prerequisites**:
    *   Python 3.9 or higher.
    *   `pip` (Python package installer).
    *   Git (for cloning the repository).

2.  **Clone the Repository**:
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd jules-swing-trade-pro # Or your project directory name
    ```

3.  **Create a Virtual Environment (Recommended)**:
    *   It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scriptsctivate
    ```

4.  **Install Dependencies**:
    *   Ensure you have the `requirements.txt` file from the project.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  Navigate to the root directory of the project where `app.py` is located.
2.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
3.  The application should open in your default web browser.

## Brief Usage Guide

*   **Mode Selection**: Use the "Choose App Mode" selectbox in the sidebar to navigate between "Trading Analysis", "U.S. Market Overview", "Backtesting", and "About".

*   **Trading Analysis**:
    *   Enter a stock ticker symbol (e.g., AAPL, MSFT).
    *   Select a date range.
    *   Choose technical indicators (SMA, EMA, RSI) and adjust their window parameters from the sidebar.
    *   Select chart type (Line/Candlestick).
    *   Optionally, enable ML Predictions and Sentiment Analysis using the checkboxes in the sidebar.
    *   View results and information in the respective tabs on the main panel.

*   **U.S. Market Overview**:
    *   View snapshots of S&P 500, Dow 30, and Nasdaq Composite.
    *   Select different chart periods (1M, 3M, 6M, YTD, 1Y) using the radio buttons to see index trends.

*   **Backtesting**:
    *   Enter a stock ticker and date range for the backtest.
    *   Set the initial capital for the simulation.
    *   Choose a strategy (SMA Crossover or RSI Mean Reversion) from the dropdown.
    *   Configure strategy-specific parameters (e.g., SMA windows, RSI thresholds).
    *   Set optional Stop-Loss and Take-Profit percentages.
    *   Click "Run Backtest" in the sidebar.
    *   View performance metrics, portfolio value chart, and optionally the detailed trades log.

*   **About**:
    *   Contains information about the application, its features, technologies, and disclaimers.

## Important Disclaimers

*   **Educational Use Only**: This application is intended for educational and informational purposes only. It is NOT financial advice.
*   **Trading Risks**: All trading and investment activities involve substantial risk. Any decisions based on information from this tool are at your own risk.
*   **Past Performance**: Past performance, whether actual or backtested via this tool, is not indicative of future results.
*   **Backtesting Limitations**: Simulated backtesting results have inherent limitations. They do not account for real-world factors such as market liquidity, slippage, commissions, or the psychological impact of trading.
*   **Data Accuracy**: Stock market data is sourced from Yahoo Finance and may be subject to inaccuracies, omissions, or delays. Always verify data from multiple sources before making financial decisions.
*   **Simulated News**: The sentiment analysis feature currently uses simulated news data. It does not reflect real-time news sentiment for any specific stock.
*   **No Warranty**: This software is provided "as-is" without any warranties of any kind.

## To Do / Future Enhancements (Examples)
*   Integration with live news APIs for real-time sentiment analysis.
*   More advanced backtesting features (e.g., commissions, slippage, different position sizing models).
*   Additional technical indicators and drawing tools on charts.
*   User accounts for saving preferences and results.
