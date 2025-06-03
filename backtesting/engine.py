import pandas as pd
import numpy as np

def run_backtest(historical_data_df: pd.DataFrame,
                 strategy_params: dict,
                 initial_capital: float = 100000.0) -> tuple[pd.DataFrame | None, list | None]:
    """
    Runs a backtest for a given strategy.

    Args:
        historical_data_df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' prices
                                           and pre-calculated indicator columns. Indexed by date.
        strategy_params (dict): Dictionary defining the strategy. Must include 'type'
                                (e.g., 'sma_crossover', 'rsi_reversion') and other
                                necessary parameters for that strategy.
        initial_capital (float): Starting capital for the backtest.

    Returns:
        tuple[pd.DataFrame | None, list | None]:
            - DataFrame with portfolio metrics over time (e.g., 'portfolio_value', 'cash', 'shares_held', 'position').
            - List of trade dictionaries.
            Returns (None, None) if critical errors occur (e.g., missing required columns).
    """
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in historical_data_df.columns:
            print(f"Error: Missing required column '{col}' in historical_data_df.")
            return None, None

    data = historical_data_df.copy()

    # Initialize portfolio columns
    data['signal'] = 0
    data['position'] = 0      # 0 for Flat, 1 for Long
    data['cash'] = initial_capital
    data['shares_held'] = 0.0
    data['portfolio_value'] = initial_capital
    data['cost_basis_per_share'] = 0.0 # To track entry price for P&L calculation

    trades_list = []

    # --- 1. Strategy-Specific Signal Generation ---
    strategy_type = strategy_params.get('type')

    if strategy_type == 'sma_crossover':
        short_col = strategy_params.get('short_window_col')
        long_col = strategy_params.get('long_window_col')
        if not short_col or not long_col or short_col not in data.columns or long_col not in data.columns:
            print(f"Error: SMA Crossover strategy requires '{short_col}' and '{long_col}' in data.")
            return None, None

        data['signal'] = np.where(data[short_col] > data[long_col], 1, 0)
        data['signal'] = np.where(data[short_col] < data[long_col], -1, data['signal'])

    elif strategy_type == 'rsi_reversion':
        rsi_col = strategy_params.get('rsi_col')
        oversold = strategy_params.get('oversold_threshold')
        overbought = strategy_params.get('overbought_threshold')
        if not rsi_col or pd.isna(oversold) or pd.isna(overbought) or rsi_col not in data.columns:
            print("Error: RSI Reversion strategy requires rsi_col, oversold_threshold, and overbought_threshold params and column in data.")
            return None, None

        data['signal'] = np.where(data[rsi_col] < oversold, 1, 0)
        data['signal'] = np.where(data[rsi_col] > overbought, -1, data['signal'])

    else:
        print(f"Error: Unknown strategy type '{strategy_type}'.")
        return None, None

    if data.empty:
        print("Error: Data is empty after initial processing (e.g. after indicator calculation and NaN drop) or before loop.")
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'signal', 'position', 'cash', 'shares_held', 'portfolio_value']), []

    # --- 2. Main Backtesting Loop ---
    for i in range(1, len(data)):
        current_idx = data.index[i]
        prev_idx = data.index[i-1]

        data.loc[current_idx, 'cash'] = data.loc[prev_idx, 'cash']
        data.loc[current_idx, 'shares_held'] = data.loc[prev_idx, 'shares_held']
        data.loc[current_idx, 'position'] = data.loc[prev_idx, 'position']
        data.loc[current_idx, 'portfolio_value'] = data.loc[prev_idx, 'portfolio_value']

        trade_execution_price = data['Close'].iloc[i-1]

        if pd.isna(trade_execution_price) or trade_execution_price <= 0:
            current_day_close_price_for_mtm = data['Close'].iloc[i]
            if pd.isna(current_day_close_price_for_mtm):
                 data.loc[current_idx, 'portfolio_value'] = data.loc[prev_idx, 'portfolio_value']
            else:
                 data.loc[current_idx, 'portfolio_value'] = data.loc[prev_idx, 'cash'] + \
                                                          (data.loc[prev_idx, 'shares_held'] * current_day_close_price_for_mtm)
            continue

        current_signal = data['signal'].iloc[i-1]

        if current_signal == 1:
            if data.loc[prev_idx, 'position'] == 0:
                shares_to_buy = data.loc[prev_idx, 'cash'] // trade_execution_price
                if shares_to_buy > 0:
                    data.loc[current_idx, 'shares_held'] = shares_to_buy
                    data.loc[current_idx, 'cash'] = data.loc[prev_idx, 'cash'] - (shares_to_buy * trade_execution_price)
                    data.loc[current_idx, 'position'] = 1
                    trades_list.append({
                        'date': prev_idx,
                        'type': 'BUY',
                        'price': trade_execution_price,
                        'shares': shares_to_buy,
                        'cash_change': -(shares_to_buy * trade_execution_price),
                        'cash_remaining': data.loc[current_idx, 'cash']
                    })
                    # Update cost basis (average cost if adding to existing, but here it's all-in from flat)
                    data.loc[current_idx, 'cost_basis_per_share'] = trade_execution_price
                else: # Not enough cash to buy even 1 share, or price is invalid
                    data.loc[current_idx, 'cost_basis_per_share'] = data.loc[prev_idx, 'cost_basis_per_share'] # carry forward
            else: # Already in a position or other condition
                 data.loc[current_idx, 'cost_basis_per_share'] = data.loc[prev_idx, 'cost_basis_per_share'] # carry forward

        elif current_signal == -1:
            if data.loc[prev_idx, 'position'] == 1:
                if data.loc[prev_idx, 'shares_held'] > 0:
                    entry_cost_basis = data.loc[prev_idx, 'cost_basis_per_share'] # Get from when shares were bought
                    pnl_per_share = trade_execution_price - entry_cost_basis
                    total_pnl = pnl_per_share * data.loc[prev_idx, 'shares_held']

                    cash_from_sale = data.loc[prev_idx, 'shares_held'] * trade_execution_price
                    data.loc[current_idx, 'cash'] = data.loc[prev_idx, 'cash'] + cash_from_sale
                    shares_sold = data.loc[prev_idx, 'shares_held']
                    data.loc[current_idx, 'shares_held'] = 0
                    data.loc[current_idx, 'position'] = 0
                    data.loc[current_idx, 'cost_basis_per_share'] = 0.0 # Reset cost basis
                    trades_list.append({
                        'date': prev_idx,
                        'type': 'SELL',
                        'price': trade_execution_price,
                        'shares': shares_sold,
                        'pnl': total_pnl, # Profit and Loss for this trade
                        'cash_change': cash_from_sale,
                        'cash_remaining': data.loc[current_idx, 'cash']
                    })
                else: # In a long position but somehow 0 shares held (should not happen with current logic)
                    data.loc[current_idx, 'cost_basis_per_share'] = 0.0
            else: # Not in a position to sell
                data.loc[current_idx, 'cost_basis_per_share'] = data.loc[prev_idx, 'cost_basis_per_share'] # carry forward
        else: # No trade signal or hold signal
            data.loc[current_idx, 'cost_basis_per_share'] = data.loc[prev_idx, 'cost_basis_per_share'] # carry forward cost_basis


        current_day_close_price = data['Close'].iloc[i]
        if pd.isna(current_day_close_price):
            data.loc[current_idx, 'portfolio_value'] = data.loc[prev_idx, 'portfolio_value']
        else:
            data.loc[current_idx, 'portfolio_value'] = data.loc[current_idx, 'cash'] + \
                                                     (data.loc[current_idx, 'shares_held'] * current_day_close_price)

    portfolio_df = data[['Open', 'High', 'Low', 'Close', 'signal', 'position', 'cash', 'shares_held', 'portfolio_value', 'cost_basis_per_share']].copy()
    return portfolio_df, trades_list


if __name__ == '__main__':
    initial_capital_main = 100000.0
    # --- Sample Data Generation (Consistent with previous tests) ---
    dates = pd.to_datetime([f'2023-01-{d:02d}' for d in range(1, 31)])
    prices_close = np.array([
        150, 152, 151, 153, 155, 154, 156, 158, 157, 160,
        159, 157, 155, 156, 154, 152, 150, 148, 149, 151,
        150, 152, 153, 155, 156, 158, 160, 162, 161, 163
    ])
    sample_ohlc_data = pd.DataFrame({
        'Open': prices_close - np.random.uniform(-0.5, 0.5, size=len(prices_close)),
        'High': prices_close + np.random.uniform(0, 1, size=len(prices_close)),
        'Low': prices_close - np.random.uniform(0, 1, size=len(prices_close)),
        'Close': prices_close
    }, index=dates)

    sample_ohlc_data['Low'] = sample_ohlc_data[['Open', 'Close', 'Low']].min(axis=1)
    sample_ohlc_data['High'] = sample_ohlc_data[['Open', 'Close', 'High']].max(axis=1)

    print("--- Testing SMA Crossover Strategy ---")
    sma_data_test = sample_ohlc_data.copy()
    sma_data_test['SMA_short'] = sma_data_test['Close'].rolling(window=5).mean()
    sma_data_test['SMA_long'] = sma_data_test['Close'].rolling(window=10).mean()
    sma_data_test.dropna(inplace=True)

    if not sma_data_test.empty:
        sma_strategy_params = {
            'type': 'sma_crossover',
            'short_window_col': 'SMA_short',
            'long_window_col': 'SMA_long'
        }
        portfolio_sma, trades_sma = run_backtest(sma_data_test, sma_strategy_params, initial_capital=initial_capital_main)

        if portfolio_sma is not None:
            print("\nSMA Crossover Portfolio Metrics (Tail):")
            print(portfolio_sma.tail())
            print("\nSMA Crossover Trades:")
            if trades_sma:
                for trade in trades_sma[-5:]:
                    print(trade)
            else:
                print("No trades executed for SMA Crossover.")
        else:
            print("SMA Crossover backtest failed to produce results.")
    else:
        print("Not enough data for SMA Crossover test after SMA calculation.")

    print("\n--- Testing RSI Reversion Strategy ---")
    rsi_data_test = sample_ohlc_data.copy()
    delta = rsi_data_test['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).fillna(0).rolling(window=14).mean()
    rs = gain / loss
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    rs.fillna(method='ffill', inplace=True)
    rsi_data_test['RSI_value'] = 100 - (100 / (1 + rs))
    rsi_data_test['RSI_value'].fillna(50, inplace=True)
    rsi_data_test.dropna(subset=['Close'], inplace=True)

    if not rsi_data_test.empty:
        rsi_strategy_params = {
            'type': 'rsi_reversion',
            'rsi_col': 'RSI_value',
            'oversold_threshold': 30,
            'overbought_threshold': 70
        }
        portfolio_rsi, trades_rsi = run_backtest(rsi_data_test, rsi_strategy_params, initial_capital=initial_capital_main)

        if portfolio_rsi is not None:
            print("\nRSI Reversion Portfolio Metrics (Tail):")
            print(portfolio_rsi.tail())
            print("\nRSI Reversion Trades:")
            if trades_rsi:
                for trade in trades_rsi[-5:]:
                    print(trade)
            else:
                print("No trades executed for RSI Reversion.")
        else:
            print("RSI Reversion backtest failed to produce results.")
    else:
        print("Not enough data for RSI Reversion test after RSI calculation.")

    print("\n--- Testing Strategy with Missing Indicator Columns ---")
    missing_col_params = {'type': 'sma_crossover', 'short_window_col': 'SMA_NONEXISTENT', 'long_window_col': 'SMA_long'}
    temp_df_for_missing_test = sample_ohlc_data.copy()
    temp_df_for_missing_test['SMA_long'] = temp_df_for_missing_test['Close'].rolling(window=10).mean()

    portfolio_missing, trades_missing = run_backtest(temp_df_for_missing_test, missing_col_params, initial_capital=initial_capital_main)
    if portfolio_missing is None:
        print("Test passed: Backtest returned None for missing SMA columns as expected.")
    else:
        print("Test failed: Backtest should have returned None for missing SMA columns.")

    print("\n--- Testing with Empty Input DataFrame (with columns) ---")
    empty_df_with_cols = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'SMA_short', 'SMA_long'])
    portfolio_empty, trades_empty = run_backtest(empty_df_with_cols, sma_strategy_params, initial_capital=initial_capital_main)

    if portfolio_empty is not None and portfolio_empty.empty :
        print("Test passed: Backtest with empty DataFrame correctly resulted in an empty portfolio DataFrame (loop did not run).")
    elif portfolio_empty is None: # This case might be hit if a check before creating `data` fails
         print("Test passed: Backtest with empty DataFrame returned None (e.g. caught by an early check).")
    else:
        print("Test failed for empty DataFrame. Output:")
        print(portfolio_empty)

    print("\n--- Testing with insufficient data for loop (1 row) ---")
    short_df = sample_ohlc_data.head(1).copy()
    short_df['SMA_short'] = 100
    short_df['SMA_long'] = 90
    portfolio_short, trades_short = run_backtest(short_df, sma_strategy_params, initial_capital=initial_capital_main)
    if portfolio_short is not None and len(portfolio_short) == 1 and \
       portfolio_short['portfolio_value'].iloc[0] == initial_capital_main and not trades_short:
        print("Test passed: Backtest with 1 row data correctly resulted in initial state and no trades.")
    else:
        print("Test failed: Backtest with 1 row data produced unexpected result.")
        if portfolio_short is not None:
            print(portfolio_short)
        if trades_short:
            print("Trades:", trades_short)

    print("\nBacktesting engine core logic tests completed.")
