import pandas as pd
import numpy as np

def calculate_total_return(portfolio_df: pd.DataFrame) -> float:
    """
    Calculates the total return over the entire backtesting period.
    Assumes 'portfolio_value' column exists.
    """
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns:
        return 0.0
    if len(portfolio_df['portfolio_value']) < 1: # Needs at least one value to avoid index error
        return 0.0

    start_value = portfolio_df['portfolio_value'].iloc[0]
    end_value = portfolio_df['portfolio_value'].iloc[-1]

    if start_value == 0: # Avoid division by zero
        return 0.0 if end_value == 0 else np.inf # Or handle as appropriate

    return (end_value - start_value) / start_value


def calculate_annualized_return(portfolio_df: pd.DataFrame, num_trading_days_per_year: int = 252) -> float:
    """
    Calculates the annualized return.
    Assumes 'portfolio_value' column exists and DataFrame index is DatetimeIndex.
    """
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns or len(portfolio_df) < 2:
        return 0.0 # Not enough data

    start_value = portfolio_df['portfolio_value'].iloc[0]
    end_value = portfolio_df['portfolio_value'].iloc[-1]

    if start_value == 0:
        return 0.0

    start_date = portfolio_df.index[0]
    end_date = portfolio_df.index[-1]

    time_delta_days = (end_date - start_date).days
    if time_delta_days <= 0: # No time passed or invalid range
        return calculate_total_return(portfolio_df) # Or 0.0, total return is more informative here

    total_time_in_years = time_delta_days / 365.25

    if total_time_in_years == 0: # Should be caught by time_delta_days <=0, but as safeguard
        return calculate_total_return(portfolio_df)

    annualized_return = ((end_value / start_value) ** (1 / total_time_in_years)) - 1
    return annualized_return


def calculate_sharpe_ratio(portfolio_df: pd.DataFrame,
                           risk_free_rate_annual: float = 0.0,
                           num_trading_days_per_year: int = 252) -> float:
    """
    Calculates the Sharpe Ratio.
    Assumes 'portfolio_value' column exists.
    """
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns or len(portfolio_df) < 2: # Need at least 2 to calc returns
        return 0.0

    portfolio_daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()

    if portfolio_daily_returns.empty: # Handles case where only 1 data point in returns after dropna
        return 0.0

    daily_risk_free_rate = risk_free_rate_annual / num_trading_days_per_year
    excess_daily_returns = portfolio_daily_returns - daily_risk_free_rate

    mean_excess_return = excess_daily_returns.mean()
    std_dev_excess_return = excess_daily_returns.std()

    if std_dev_excess_return == 0 or pd.isna(std_dev_excess_return): # Avoid division by zero or if std is NaN
        return 0.0 if mean_excess_return == 0 else np.sign(mean_excess_return) * np.inf # Or handle as appropriate

    sharpe_ratio = (mean_excess_return / std_dev_excess_return) * np.sqrt(num_trading_days_per_year)
    return sharpe_ratio


def calculate_max_drawdown(portfolio_df: pd.DataFrame) -> float:
    """
    Calculates the Maximum Drawdown.
    Assumes 'portfolio_value' column exists.
    """
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns:
        return 0.0
    if len(portfolio_df['portfolio_value']) < 1:
        return 0.0

    rolling_max = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() # This will be negative or zero

    return abs(max_drawdown) # Return as a positive percentage (e.g., 0.10 for 10%)


def calculate_win_rate(trades_list: list[dict]) -> float:
    """
    Calculates the percentage of trades that were profitable.
    Assumes 'pnl' field exists for SELL trades in trades_list.
    """
    if not trades_list:
        return 0.0

    completed_trades_pnl = [trade['pnl'] for trade in trades_list if isinstance(trade, dict) and trade.get('type') == 'SELL' and 'pnl' in trade]

    if not completed_trades_pnl:
        return 0.0 # No completed (SELL) trades with P&L found

    winning_trades = [pnl for pnl in completed_trades_pnl if pnl > 0]

    return len(winning_trades) / len(completed_trades_pnl)


def summarize_trades(trades_list: list[dict]) -> dict:
    """
    Calculates various trade statistics.
    Assumes 'pnl' field exists for SELL trades in trades_list.
    """
    summary = {
        'num_trades': 0,
        'num_winning_trades': 0,
        'num_losing_trades': 0,
        'total_gross_profit': 0.0,
        'total_gross_loss': 0.0,
        'average_gain_per_trade': 0.0,
        'average_loss_per_trade': 0.0,
        'profit_factor': 0.0
    }

    if not trades_list:
        return summary

    completed_trades_pnl = [trade['pnl'] for trade in trades_list if isinstance(trade, dict) and trade.get('type') == 'SELL' and 'pnl' in trade]

    if not completed_trades_pnl:
        return summary # No completed (SELL) trades with P&L found

    summary['num_trades'] = len(completed_trades_pnl)

    winning_pnls = [pnl for pnl in completed_trades_pnl if pnl > 0]
    losing_pnls = [pnl for pnl in completed_trades_pnl if pnl <= 0] # Includes zero P&L as losing/non-winning

    summary['num_winning_trades'] = len(winning_pnls)
    summary['num_losing_trades'] = len(losing_pnls)

    summary['total_gross_profit'] = sum(winning_pnls)
    summary['total_gross_loss'] = abs(sum(losing_pnls)) # Gross loss is positive value

    if summary['num_winning_trades'] > 0:
        summary['average_gain_per_trade'] = summary['total_gross_profit'] / summary['num_winning_trades']

    if summary['num_losing_trades'] > 0:
        summary['average_loss_per_trade'] = sum(losing_pnls) / summary['num_losing_trades'] # Will be negative or zero

    if summary['total_gross_loss'] > 0:
        summary['profit_factor'] = summary['total_gross_profit'] / summary['total_gross_loss']
    elif summary['total_gross_profit'] > 0: # Gross loss is 0, but profit is > 0
        summary['profit_factor'] = np.inf
    # else profit_factor remains 0 if both are 0

    return summary


if __name__ == '__main__':
    # Sample portfolio_df DataFrame
    dates_range = pd.to_datetime([f'2023-01-{d:02d}' for d in range(1, 11)])
    portfolio_values_sample = [
        100000, 101000, 100500, 102000, 101500,
        103000, 102500, 104000, 103500, 105000
    ]
    sample_portfolio_df = pd.DataFrame({'portfolio_value': portfolio_values_sample}, index=dates_range)

    print("--- Testing Performance Metrics ---")

    total_return = calculate_total_return(sample_portfolio_df)
    print(f"Total Return: {total_return:.4%}")

    annualized_return = calculate_annualized_return(sample_portfolio_df)
    print(f"Annualized Return: {annualized_return:.4%}")

    sharpe = calculate_sharpe_ratio(sample_portfolio_df, risk_free_rate_annual=0.01)
    print(f"Sharpe Ratio (rf=1%): {sharpe:.4f}")

    sharpe_zero_rf = calculate_sharpe_ratio(sample_portfolio_df, risk_free_rate_annual=0.0)
    print(f"Sharpe Ratio (rf=0%): {sharpe_zero_rf:.4f}")

    max_dd = calculate_max_drawdown(sample_portfolio_df)
    print(f"Max Drawdown: {max_dd:.4%}")

    flat_portfolio_values = [100000] * 10
    flat_portfolio_df = pd.DataFrame({'portfolio_value': flat_portfolio_values}, index=dates_range)
    sharpe_flat = calculate_sharpe_ratio(flat_portfolio_df)
    print(f"Sharpe Ratio (Flat Portfolio): {sharpe_flat:.4f}")

    empty_portfolio_df = pd.DataFrame(columns=['portfolio_value'])
    print(f"Total Return (Empty): {calculate_total_return(empty_portfolio_df):.4%}")
    print(f"Annualized Return (Empty): {calculate_annualized_return(empty_portfolio_df):.4%}")
    print(f"Sharpe Ratio (Empty): {calculate_sharpe_ratio(empty_portfolio_df):.4f}")
    print(f"Max Drawdown (Empty): {calculate_max_drawdown(empty_portfolio_df):.4%}")

    single_row_df = pd.DataFrame({'portfolio_value': [100000]}, index=[dates_range[0]])
    print(f"Total Return (1 row): {calculate_total_return(single_row_df):.4%}")
    print(f"Annualized Return (1 row): {calculate_annualized_return(single_row_df):.4%}")
    print(f"Sharpe Ratio (1 row): {calculate_sharpe_ratio(single_row_df):.4f}")
    print(f"Max Drawdown (1 row): {calculate_max_drawdown(single_row_df):.4%}")

    print("\n--- Testing Trade-Related Metrics ---")
    sample_trades_list_metrics = [
        {'type': 'SELL', 'pnl': 100.0},
        {'type': 'SELL', 'pnl': -50.0},
        {'type': 'SELL', 'pnl': 200.0},
        {'type': 'BUY', 'pnl': np.nan},
        {'type': 'SELL', 'pnl': -20.0},
        {'type': 'SELL', 'pnl': 0.0},
    ]

    win_rate = calculate_win_rate(sample_trades_list_metrics)
    print(f"Win Rate: {win_rate:.2%}")

    trade_summary = summarize_trades(sample_trades_list_metrics)
    print(f"Trade Summary: {trade_summary}")

    empty_trades = []
    print(f"Win Rate (Empty Trades): {calculate_win_rate(empty_trades):.2%}")
    print(f"Trade Summary (Empty Trades): {summarize_trades(empty_trades)}")

    all_wins_trades = [{'type': 'SELL', 'pnl': 10}, {'type': 'SELL', 'pnl': 20}]
    print(f"Win Rate (All Wins): {calculate_win_rate(all_wins_trades):.2%}")
    print(f"Trade Summary (All Wins): {summarize_trades(all_wins_trades)}")

    all_losses_trades = [{'type': 'SELL', 'pnl': -10}, {'type': 'SELL', 'pnl': -20}]
    print(f"Win Rate (All Losses): {calculate_win_rate(all_losses_trades):.2%}")
    print(f"Trade Summary (All Losses): {summarize_trades(all_losses_trades)}")

    mixed_trades_profit_factor_inf = [{'type': 'SELL', 'pnl': 10}]
    print(f"Trade Summary (Profit, No Loss): {summarize_trades(mixed_trades_profit_factor_inf)}")
