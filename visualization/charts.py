import plotly.graph_objects as go
import pandas as pd

def plot_stock_prices(data: pd.DataFrame, ticker_symbol: str, price_column: str = 'Close') -> go.Figure:
    """
    Generates a line chart for stock prices.

    Args:
        data (pd.DataFrame): DataFrame containing stock data with a DateTimeIndex.
        ticker_symbol (str): The stock ticker symbol for chart title.
        price_column (str): The column to plot (e.g., 'Close', 'Open'). Defaults to 'Close'.

    Returns:
        go.Figure: A Plotly figure object. Returns an empty figure if data is empty or column not found.
    """
    fig = go.Figure()

    if price_column not in data.columns:
        print(f"Error: Price column '{price_column}' not found in data.")
        return fig # Return empty figure

    if data.empty:
        print("Error: Data for plotting is empty.")
        return fig # Return empty figure

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_stock_prices(data: pd.DataFrame,
                      ticker_symbol: str,
                      price_column: str = 'Close', # Still relevant for line chart if OHLC not available
                      chart_type: str = 'line', # 'line' or 'candlestick'
                      sma_series: pd.Series = None,
                      sma_window: int = None,
                      ema_series: pd.Series = None,
                      ema_window: int = None,
                      rsi_series: pd.Series = None,
                      rsi_window: int = None) -> go.Figure:
    """
    Generates a chart for stock prices (line or candlestick) with optional indicators and volume.
    - Price (Line/Candlestick), SMA, EMA on the top subplot.
    - Volume on the middle subplot.
    - RSI on the bottom subplot (if provided).

    Args:
        data (pd.DataFrame): DataFrame containing stock data. Must include DateTimeIndex.
                             For 'candlestick', requires 'Open', 'High', 'Low', 'Close'.
                             For 'volume', requires 'Volume'.
        ticker_symbol (str): The stock ticker symbol for chart title.
        price_column (str): The column for line chart if candlestick fails or is not chosen.
        chart_type (str): 'line' or 'candlestick'.
        sma_series (pd.Series, optional): Series containing SMA values.
        sma_window (int, optional): Window used for SMA calculation.
        ema_series (pd.Series, optional): Series containing EMA values.
        ema_window (int, optional): Window used for EMA calculation.
        rsi_series (pd.Series, optional): Series containing RSI values.
        rsi_window (int, optional): Window used for RSI calculation.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if data.empty:
        print("Error: Data for plotting is empty.")
        return go.Figure() # Return empty figure

    has_ohlc = all(col in data.columns for col in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in data.columns

    # Determine number of rows and row heights
    rows = 2
    row_heights = [0.7, 0.3]
    specs = [[{"secondary_y": False}], [{"secondary_y": False}]] # Specs for subplots

    if rsi_series is not None and not rsi_series.empty:
        rows = 3
        row_heights = [0.6, 0.2, 0.2] # Adjust heights for 3 plots
        specs = [[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]


    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=row_heights,
                        specs=specs)

    # Subplot 1: Price (Line or Candlestick) and Overlays (SMA, EMA)
    actual_chart_type = chart_type
    if chart_type == 'candlestick' and not has_ohlc:
        print("Warning: Candlestick chart selected, but OHLC data is missing. Falling back to line chart.")
        actual_chart_type = 'line'

    if actual_chart_type == 'candlestick':
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name=f'{ticker_symbol} Price'),
                      row=1, col=1)
    else: # Line chart
        if price_column not in data.columns:
            print(f"Error: Price column '{price_column}' for line chart not found in data.")
            # Add a dummy trace to avoid error if no price data at all
            fig.add_trace(go.Scatter(x=[None],y=[None], name="No Price Data"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name=data[price_column].name), # Use series name
                          row=1, col=1)

    if sma_series is not None and not sma_series.empty:
        fig.add_trace(go.Scatter(x=sma_series.index, y=sma_series, mode='lines', name=f'SMA ({sma_window})',
                                 line=dict(width=1)), row=1, col=1)
    if ema_series is not None and not ema_series.empty:
        fig.add_trace(go.Scatter(x=ema_series.index, y=ema_series, mode='lines', name=f'EMA ({ema_window})',
                                 line=dict(width=1)), row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)


    # Subplot 2: Volume
    if has_volume:
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgba(100,100,150,0.5)'),
                      row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=[None],y=[None], name="No Volume Data"), row=2, col=1) # Placeholder
    fig.update_yaxes(title_text="Volume", row=2, col=1)


    # Subplot 3: RSI (if applicable)
    if rsi_series is not None and not rsi_series.empty:
        fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, mode='lines', name=f'RSI ({rsi_window})',
                                 line=dict(color='purple', width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1)
        fig.update_yaxes(title_text="RSI", range=[0,100], row=3, col=1) # RSI range 0-100
        fig.update_xaxes(title_text="Date", row=3, col=1) # Date axis label on the bottom-most plot
    else: # If no RSI, date axis label goes on the volume plot
        fig.update_xaxes(title_text="Date", row=2, col=1)


    fig.update_layout(
        title_text=f"{ticker_symbol} Analysis",
        height=700 if rows==3 else 500, # Adjust height based on number of subplots
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False # Turn off rangeslider for all x-axes
    )

    # Ensure x-axis title is only on the very bottom plot
    if rows == 2:
         fig.update_xaxes(title_text="Date", row=2, col=1)
         fig.update_xaxes(title_text=None, row=1, col=1) # Remove title from price subplot x-axis
    elif rows == 3:
         fig.update_xaxes(title_text="Date", row=3, col=1)
         fig.update_xaxes(title_text=None, row=1, col=1)
         fig.update_xaxes(title_text=None, row=2, col=1)


    return fig

if __name__ == '__main__':
    # Sample Data for testing (more comprehensive)
    num_days = 60
    sample_dates_idx = pd.to_datetime([pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(num_days)])
    close_prices = 150 + np.cumsum(np.random.randn(num_days) * 2 + 0.1) # Random walk with slight upward drift
    open_prices = close_prices - np.random.uniform(-1, 1, num_days)
    high_prices = np.maximum(close_prices, open_prices) + np.random.uniform(0, 2, num_days)
    low_prices = np.minimum(close_prices, open_prices) - np.random.uniform(0, 2, num_days)
    volume_data = np.random.randint(100000, 5000000, num_days)

    sample_df_full = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume_data
    }, index=sample_dates_idx)

    sma_10 = sample_df_full['Close'].rolling(window=10).mean()
    ema_20 = sample_df_full['Close'].ewm(span=20, adjust=False).mean()

    delta_rsi = sample_df_full['Close'].diff()
    gain_rsi = (delta_rsi.where(delta_rsi > 0, 0)).fillna(0).rolling(window=14).mean()
    loss_rsi = (-delta_rsi.where(delta_rsi < 0, 0)).fillna(0).rolling(window=14).mean()
    rs_rsi = gain_rsi / loss_rsi
    rs_rsi.replace([np.inf, -np.inf], np.nan, inplace=True)
    rs_rsi.fillna(method='ffill', inplace=True) # Or some other handling for NaNs from division by zero
    rsi_14 = 100 - (100 / (1 + rs_rsi))
    rsi_14.fillna(50, inplace=True) # Fill initial NaNs in RSI with neutral 50


    print("Generating sample charts (if running locally, uncomment .show()):")

    # Test 1: Line chart, Volume, No indicators
    fig1 = plot_stock_prices(sample_df_full, "TEST1", chart_type='line')
    # fig1.show()
    print("Fig1: Line chart with Volume created.")

    # Test 2: Candlestick chart, Volume, SMA
    fig2 = plot_stock_prices(sample_df_full, "TEST2", chart_type='candlestick',
                             sma_series=sma_10, sma_window=10)
    # fig2.show()
    print("Fig2: Candlestick chart with Volume and SMA created.")

    # Test 3: Line chart, Volume, SMA, EMA
    fig3 = plot_stock_prices(sample_df_full, "TEST3", chart_type='line',
                             sma_series=sma_10, sma_window=10,
                             ema_series=ema_20, ema_window=20)
    # fig3.show()
    print("Fig3: Line chart with Volume, SMA, EMA created.")

    # Test 4: Candlestick, Volume, SMA, EMA, RSI
    fig4 = plot_stock_prices(sample_df_full, "TEST4", chart_type='candlestick',
                             sma_series=sma_10, sma_window=10,
                             ema_series=ema_20, ema_window=20,
                             rsi_series=rsi_14, rsi_window=14)
    # fig4.show()
    print("Fig4: Candlestick chart with Volume, SMA, EMA, RSI created.")

    # Test 5: Line chart, Volume, RSI only
    fig5 = plot_stock_prices(sample_df_full, "TEST5", chart_type='line',
                             rsi_series=rsi_14, rsi_window=14)
    # fig5.show()
    print("Fig5: Line chart with Volume and RSI created.")

    # Test 6: Data missing OHLC for candlestick (should fallback to line)
    sample_df_no_ohlc = sample_df_full[['Close', 'Volume']].copy()
    fig6 = plot_stock_prices(sample_df_no_ohlc, "TEST6_NO_OHLC", chart_type='candlestick')
    # fig6.show()
    print("Fig6: Candlestick requested with missing OHLC (fallback to line) created.")

    # Test 7: Data missing Volume
    sample_df_no_volume = sample_df_full[['Open', 'High', 'Low', 'Close']].copy()
    fig7 = plot_stock_prices(sample_df_no_volume, "TEST7_NO_VOL", chart_type='line', rsi_series=rsi_14, rsi_window=14)
    # fig7.show()
    print("Fig7: Chart with missing Volume data created.")

    print("\nSample charts generation complete.")
