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
    data['cost_basis_per_share'] = 0.0

    trades_list = []
    current_position_entry_price = 0.0

    # Retrieve SL/TP parameters
    stop_loss_pct = strategy_params.get('stop_loss_pct', 0.0) / 100.0
    take_profit_pct = strategy_params.get('take_profit_pct', 0.0) / 100.0

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

        # Carry forward state from previous day
        data.loc[current_idx, 'cash'] = data.loc[prev_idx, 'cash']
        data.loc[current_idx, 'shares_held'] = data.loc[prev_idx, 'shares_held']
        data.loc[current_idx, 'position'] = data.loc[prev_idx, 'position']
        data.loc[current_idx, 'cost_basis_per_share'] = data.loc[prev_idx, 'cost_basis_per_share']
        # current_position_entry_price is a variable, carried by the loop scope from previous iteration or BUY action

        active_signal = data.loc[prev_idx, 'signal'] # Signal from previous day's close
        strategy_execution_price = data.loc[prev_idx, 'Close'] # Strategy trades execute based on previous close

        exited_today = False
        exit_price = 0.0
        exit_reason = ""

        # --- Stop-Loss / Take-Profit Checks (only if in a position) ---
        if data.loc[current_idx, 'position'] == 1: # Check current position state (copied from i-1, will be updated if exit)
            # Stop-Loss Check: uses current day's Low price
            if stop_loss_pct > 0 and current_position_entry_price > 0:
                sl_level = current_position_entry_price * (1 - stop_loss_pct)
                if data.loc[current_idx, 'Low'] <= sl_level:
                    exit_price = sl_level
                    exit_reason = "Stop-Loss Hit"
                    exited_today = True

            # Take-Profit Check (if not already stopped out): uses current day's High price
            if not exited_today and take_profit_pct > 0 and current_position_entry_price > 0:
                tp_level = current_position_entry_price * (1 + take_profit_pct)
                if data.loc[current_idx, 'High'] >= tp_level:
                    exit_price = tp_level
                    exit_reason = "Take-Profit Hit"
                    exited_today = True

            # Strategy-based Sell Signal Check (if not SL/TP exited today)
            # Signal is from prev_idx, execution price is also from prev_idx ('Close')
            if not exited_today and active_signal == -1:
                if pd.notna(strategy_execution_price) and strategy_execution_price > 0:
                    exit_price = strategy_execution_price
                    exit_reason = "Strategy Signal"
                    exited_today = True
                # If strategy_execution_price is NaN/invalid, this exit path is skipped.

            if exited_today and exit_price > 0:
                shares_to_sell = data.loc[current_idx, 'shares_held'] # Shares held at start of day i (copied from prev_idx)
                cost_basis = data.loc[current_idx, 'cost_basis_per_share']
                pnl = (exit_price - cost_basis) * shares_to_sell

                data.loc[current_idx, 'cash'] += shares_to_sell * exit_price
                data.loc[current_idx, 'shares_held'] = 0
                data.loc[current_idx, 'position'] = 0 # Now Flat
                data.loc[current_idx, 'cost_basis_per_share'] = 0.0
                current_position_entry_price = 0.0 # Reset as we've exited

                trades_list.append({
                    'date': current_idx, # Trade executed on day i
                    'type': 'SELL',
                    'price': exit_price,
                    'shares': shares_to_sell,
                    'pnl': pnl,
                    'reason': exit_reason,
                    'cash_change': shares_to_sell * exit_price,
                    'cash_remaining': data.loc[current_idx, 'cash']
                })

        # --- Buy Signal Logic (only if currently flat *after* any SL/TP/Strategy exits on day i) ---
        # Signal is from prev_idx, execution price is also from prev_idx ('Close')
        if data.loc[current_idx, 'position'] == 0 and active_signal == 1:
            if pd.notna(strategy_execution_price) and strategy_execution_price > 0:
                buy_price = strategy_execution_price
                # Use cash available at this point in day i (after potential morning SL/TP sales)
                shares_to_buy = data.loc[current_idx, 'cash'] // buy_price

                if shares_to_buy > 0:
                    data.loc[current_idx, 'shares_held'] = shares_to_buy
                    data.loc[current_idx, 'cash'] -= shares_to_buy * buy_price
                    data.loc[current_idx, 'position'] = 1 # Now Long
                    data.loc[current_idx, 'cost_basis_per_share'] = buy_price
                    current_position_entry_price = buy_price

                    trades_list.append({
                        'date': current_idx, # Buy action taken on day i
                        'type': 'BUY',
                        'price': buy_price,
                        'shares': shares_to_buy,
                        'reason': 'Strategy Signal',
                        'cash_change': -(shares_to_buy * buy_price),
                        'cash_remaining': data.loc[current_idx, 'cash']
                    })

        # Update Daily Portfolio Value using Close price of day i
        current_day_close_price = data.loc[current_idx, 'Close']
        if pd.isna(current_day_close_price):
            data.loc[current_idx, 'portfolio_value'] = data.loc[prev_idx, 'portfolio_value'] # Carry forward if current close is NaN
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
        if portfolio_short is not None: print(portfolio_short)
        if trades_short: print("Trades:", trades_short)

    # --- 5. Test SL/TP Functionality ---
    print("\n--- Testing Stop-Loss and Take-Profit ---")
    sl_tp_dates = pd.to_datetime([f'2023-02-{d:02d}' for d in range(1, 15)])
    sl_tp_data_base = {
        'Open':  [100, 101, 102, 95,  96,  97,  105, 106, 107, 103, 102, 100, 98, 99], # Day i Open
        'High':  [102, 103, 103, 97,  98,  108, 108, 109, 108, 105, 104, 102, 99, 100],# Day i High (for TP)
        'Low':   [99,  100, 90,  94,  95,  96,  104, 105, 102, 101, 98, 97, 96, 97], # Day i Low (for SL)
        'Close': [101, 102, 92,  96,  97,  107, 106, 107, 103, 102, 100, 98, 97, 98]  # Day i Close (for MTM and strategy signal)
    }
    sl_tp_df = pd.DataFrame(sl_tp_data_base, index=sl_tp_dates)

    # Strategy: Always signal buy (1) to enter, then let SL/TP manage exit.
    # We need an indicator column for the strategy part, even if it's simple.
    sl_tp_df['SMA_short'] = 110 # Always above long SMA
    sl_tp_df['SMA_long'] = 100  # Constant SMAs to force a buy signal initially

    # Test Case 5.1: Stop-Loss Hit
    print("\nTest 5.1: Stop-Loss Hit")
    params_sl = {
        'type': 'sma_crossover', 'short_window_col': 'SMA_short', 'long_window_col': 'SMA_long',
        'stop_loss_pct': 5.0, 'take_profit_pct': 0.0 # 5% SL, no TP
    }
    # For this test, entry price will be Close of day 0 (101). SL = 101 * 0.95 = 95.95
    # Day 2 Low is 90. So SL should hit on Day 2.
    portfolio_sl, trades_sl = run_backtest(sl_tp_df.copy(), params_sl, initial_capital_main)
    if portfolio_sl is not None and trades_sl:
        print("Trades for SL test:")
        for trade in trades_sl: print(trade)
        if any(t['reason'] == 'Stop-Loss Hit' for t in trades_sl):
            print("Stop-Loss Hit Test: PASSED")
        else:
            print("Stop-Loss Hit Test: FAILED - No SL trade recorded.")
            print(portfolio_sl.tail())
    else:
        print("Stop-Loss Hit Test: FAILED - No portfolio or trades.")

    # Test Case 5.2: Take-Profit Hit
    print("\nTest 5.2: Take-Profit Hit")
    params_tp = {
        'type': 'sma_crossover', 'short_window_col': 'SMA_short', 'long_window_col': 'SMA_long',
        'stop_loss_pct': 0.0, 'take_profit_pct': 5.0 # 5% TP, no SL
    }
    # Entry at 101. TP = 101 * 1.05 = 106.05
    # Day 5 High is 108. TP should hit on Day 5.
    # (Note: The sample data's 'SMA_short' always > 'SMA_long', so initial buy signal is always on.)
    # We need to ensure the buy happens, then TP.
    # The data needs to be long enough for the initial SMA calculation if we didn't use constants.
    # For this specific test, we are giving constant SMA values to ensure a BUY signal.

    # Let's use a fresh copy of sl_tp_df for each specific test if it modifies it (run_backtest makes a copy)
    portfolio_tp, trades_tp = run_backtest(sl_tp_df.copy(), params_tp, initial_capital_main)
    if portfolio_tp is not None and trades_tp:
        print("Trades for TP test:")
        for trade in trades_tp: print(trade)
        if any(t['reason'] == 'Take-Profit Hit' for t in trades_tp):
            print("Take-Profit Hit Test: PASSED")
        else:
            print("Take-Profit Hit Test: FAILED - No TP trade recorded.")
            print(portfolio_tp.tail())
    else:
        print("Take-Profit Hit Test: FAILED - No portfolio or trades.")

    # Test Case 5.3: Strategy Exit before SL/TP
    print("\nTest 5.3: Strategy Exit before SL/TP")
    sl_tp_df_strat = sl_tp_df.copy()
    # Modify signal to cause a strategy exit before SL/TP might hit
    # Buy on day 1 (signal from day 0). Let's say signal turns to sell on day 3 (Close).
    # Entry at 101 (Close of day 0).
    # SL = 101 * 0.9 = 90.9 (10% SL). TP = 101 * 1.1 = 111.1 (10% TP)
    # Day 2 Low = 90 (SL would hit). Day 5 High = 108 (TP not hit)
    # If signal at Close of day 2 is Sell (-1), then exit at Close of day 2 (92).
    # sl_tp_df_strat.loc[sl_tp_df_strat.index[2], 'SMA_short'] = 90 # Force sell signal (short < long) for day 3 execution

    sl_tp_df_strat_exit = sl_tp_df.copy()
    # Modify SMAs for sl_tp_df_strat_exit.index[1] (second day of data)
    # This signal will be active when processing sl_tp_df_strat_exit.index[2] (third day)
    sl_tp_df_strat_exit.loc[sl_tp_df_strat_exit.index[1], 'SMA_short'] = 90  # short < long
    sl_tp_df_strat_exit.loc[sl_tp_df_strat_exit.index[1], 'SMA_long'] = 100

    # Ensure Day 2's Low/High do not trigger SL/TP for an entry price of 101
    # Data for index[2] (third day): Open=102, High=103, Low=90, Close=92
    # current_position_entry_price = 101 (from BUY based on index[0]'s signal, executed on index[1])
    # SL = 101 * 0.9 = 90.9. Low[index[2]] = 90. This WILL hit SL.
    # To test STRATEGY exit, the SL must NOT be hit on the day of the strategy exit.
    sl_tp_df_strat_exit.loc[sl_tp_df_strat_exit.index[2], 'Low'] = 92 # Was 90. New Low is above SL (90.9).
                                                                    # High[index[2]] is 103. TP is 111.1. No TP.
    params_strat_exit = {
        'type': 'sma_crossover', 'short_window_col': 'SMA_short', 'long_window_col': 'SMA_long',
        'stop_loss_pct': 10.0, 'take_profit_pct': 10.0
    }
    portfolio_strat, trades_strat = run_backtest(sl_tp_df_strat_exit, params_strat_exit, initial_capital_main)
    # Expected:
    # 1. BUY on index[1] (price 101, signal from index[0])
    # 2. Active signal for index[2] (from data at index[1]) is now SELL (-1)
    # 3. On index[2], SL/TP not hit with Low=92/High=103. Strategy SELL occurs at Close of index[1] (price 102).

    if portfolio_strat is not None and trades_strat:
        print("Trades for Strategy Exit test (Revised):")
        for trade in trades_strat: print(trade)
        if len(trades_strat) >= 2 and \
           trades_strat[1]['reason'] == 'Strategy Signal' and \
           trades_strat[1]['price'] == sl_tp_df_strat_exit.loc[sl_tp_df_strat_exit.index[1],'Close']: # Executed at Close of signal day
            print("Strategy Exit Test (Revised): PASSED")
        else:
            print("Strategy Exit Test (Revised): FAILED")
            print(portfolio_strat.tail()) # Print tail for debugging
    else:
        print("Strategy Exit Test (Revised): FAILED - No portfolio or trades.")

    print("\nBacktesting engine core logic tests completed.")
