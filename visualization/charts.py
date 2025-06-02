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

from plotly.subplots import make_subplots

def plot_stock_prices(data: pd.DataFrame, 
                      ticker_symbol: str, 
                      price_column: str = 'Close', 
                      sma_series: pd.Series = None, 
                      sma_window: int = None,
                      ema_series: pd.Series = None,
                      ema_window: int = None,
                      rsi_series: pd.Series = None,
                      rsi_window: int = None) -> go.Figure:
    """
    Generates a line chart for stock prices, optionally including SMA, EMA, and RSI.
    RSI will be plotted on a separate subplot below the price chart.

    Args:
        data (pd.DataFrame): DataFrame containing stock data with a DateTimeIndex.
        ticker_symbol (str): The stock ticker symbol for chart title.
        price_column (str): The column to plot (e.g., 'Close', 'Open'). Defaults to 'Close'.
        sma_series (pd.Series, optional): Series containing SMA values.
        sma_window (int, optional): Window used for SMA calculation.
        ema_series (pd.Series, optional): Series containing EMA values.
        ema_window (int, optional): Window used for EMA calculation.
        rsi_series (pd.Series, optional): Series containing RSI values.
        rsi_window (int, optional): Window used for RSI calculation.

    Returns:
        go.Figure: A Plotly figure object. Returns an empty figure if data is empty or column not found.
    """
    if price_column not in data.columns:
        print(f"Error: Price column '{price_column}' not found in data.")
        return go.Figure()

    if data.empty:
        print("Error: Data for plotting is empty.")
        return go.Figure()

    if rsi_series is not None and not rsi_series.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])
        price_row, rsi_row = 1, 2
    else:
        fig = make_subplots(rows=1, cols=1)
        price_row, rsi_row = 1, None

    # Price Plot
    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name=price_column), 
                  row=price_row, col=1)

    # SMA Plot
    if sma_series is not None and not sma_series.empty:
        fig.add_trace(go.Scatter(x=sma_series.index, y=sma_series, mode='lines', name=f'SMA ({sma_window})'), 
                      row=price_row, col=1)

    # EMA Plot
    if ema_series is not None and not ema_series.empty:
        fig.add_trace(go.Scatter(x=ema_series.index, y=ema_series, mode='lines', name=f'EMA ({ema_window})'), 
                      row=price_row, col=1)
    
    # RSI Plot
    if rsi_row and rsi_series is not None and not rsi_series.empty:
        fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, mode='lines', name=f'RSI ({rsi_window})'), 
                      row=rsi_row, col=1)
        fig.update_yaxes(title_text="RSI", row=rsi_row, col=1)
        # Add overbought/oversold lines for RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=rsi_row, col=1)


    fig.update_layout(
        title_text=f"{ticker_symbol} Analysis ({price_column})",
        xaxis_title="Date" if rsi_row is None else None, # Hide x-axis title for price plot if RSI is shown
        yaxis_title=f"Price (USD)", # Assuming USD
        legend_title_text="Indicators",
        showlegend=True
    )
    fig.update_xaxes(title_text="Date", row=rsi_row if rsi_row else price_row, col=1) # Set x-axis title on the bottom plot

    return fig

if __name__ == '__main__':
    # Sample Data for testing
    sample_dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                   '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                   '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15'])
    sample_prices_close = [150, 152, 151, 155, 156, 154, 158, 160, 159, 162, 165, 163, 160, 158, 162]
    sample_df = pd.DataFrame({'Close': sample_prices_close}, index=sample_dates)

    # Dummy indicator data
    sma_data = sample_df['Close'].rolling(window=5).mean()
    ema_data = sample_df['Close'].ewm(span=5, adjust=False).mean()
    # Simplified RSI calculation for testing plot function
    delta = sample_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss
    rsi_data = 100 - (100 / (1 + rs))


    # Test plotting 'Close' price only
    fig_close_only = plot_stock_prices(sample_df, "SAMPLE", price_column='Close')
    # fig_close_only.show()

    # Test plotting with SMA
    fig_sma = plot_stock_prices(sample_df, "SAMPLE", price_column='Close', sma_series=sma_data, sma_window=5)
    # fig_sma.show()

    # Test plotting with EMA
    fig_ema = plot_stock_prices(sample_df, "SAMPLE", price_column='Close', ema_series=ema_data, ema_window=5)
    # fig_ema.show()

    # Test plotting with SMA and EMA
    fig_sma_ema = plot_stock_prices(sample_df, "SAMPLE", price_column='Close', 
                                    sma_series=sma_data, sma_window=5,
                                    ema_series=ema_data, ema_window=5)
    # fig_sma_ema.show()
    
    # Test plotting with RSI
    fig_rsi = plot_stock_prices(sample_df, "SAMPLE", price_column='Close', rsi_series=rsi_data, rsi_window=5)
    # fig_rsi.show()

    # Test plotting with All indicators
    fig_all = plot_stock_prices(sample_df, "SAMPLE", price_column='Close', 
                                sma_series=sma_data, sma_window=5,
                                ema_series=ema_data, ema_window=5,
                                rsi_series=rsi_data, rsi_window=5)
    # fig_all.show()

    print("Sample plots created. If running locally, uncomment .show() to view them.")
