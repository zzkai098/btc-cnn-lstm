import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Trading Signal Generation using Tanh Mapping
def get_tanh_signal(predictions, scale_factor=None):
    """
    Map predictions to [-1, 1] range as trading signals.

    Args:
        predictions (np.array): inverse transformed true return predictions
        scale_factor (float): scaling factor. 
                              
    Returns:
        signals (np.array): signals in the range [-1, 1]
    """
    # If no scale factor is provided, compute it automatically
    if scale_factor is None:
        pred_std = np.std(predictions)
        if pred_std == 0:
            scale_factor = 1.0
        else:
            scale_factor = 1.0 / pred_std
        print(f" Scale Factor: {scale_factor:.2f} (based on data stddev)")
    else:
        print(f" Scale Factor: {scale_factor}" )

    # Tanh( x * scale )
    signals = np.tanh(predictions * scale_factor)

    print("-" * 30)
    print(f"Signal Range: [{signals.min():.4f}, {signals.max():.4f}]")
    print(f"Signal Mean:  {signals.mean():.4f}")
    print(f"Signal Std:   {signals.std():.4f}")
    print("-" * 30)

    return signals

def backtest_strategy(signals, returns, threshold=0.0, cost_rate=0.0005):
    """
    Args:
        signals: model predicted signals (continuous values -1 ~ 1)
        returns: true market returns (1-min Log Return)
        threshold: entry threshold (trade only if abs(signal) > threshold)
        cost_rate: trading cost rate (default 0.01%)
    """
    # 1. Generate positions
    position = np.zeros_like(signals)

    # 2. Create mask for signals exceeding threshold
    mask = np.abs(signals) > threshold

    # 3. Assign positions based on signal strength    
    position[mask] = signals[mask] # retain original signal strength as position size
    # for example: threshold 0.3, signal 0.5 -> position 0.5; signal -0.8 -> position -0.8; signal 0.2 -> position 0

    """ --- Calculate strategy returns: position * returns ---"""
    strategy_raw_returns = position * returns

    """  --- Calculate transaction costs --- """
    pos_diff = np.diff(position, prepend=0)
    turnover = np.abs(pos_diff)
    costs = turnover * cost_rate
        
    strategy_net_returns = strategy_raw_returns - costs

    equity_bh = np.exp(np.cumsum(returns))       # Buy & Hold btc equity
    equity_strat = np.exp(np.cumsum(strategy_net_returns)) # Strategy equity

    # Total Return
    total_ret_bh = equity_bh[-1] - 1
    total_ret_strat = equity_strat[-1] - 1
    
    # Sharpe Ratio
    periods_per_year = 365 * 24 * 60
    std_strat = np.std(strategy_net_returns)
    if std_strat == 0: std_strat = 1e-9
    sharpe = (np.mean(strategy_net_returns) / std_strat) * np.sqrt(periods_per_year)

    # Sortino Ratio
    downside_returns = strategy_net_returns[strategy_net_returns < 0]
    downside_std = np.std(downside_returns)
    if downside_std == 0: downside_std = 1e-9
    sortino = (np.mean(strategy_net_returns) / downside_std) * np.sqrt(periods_per_year)

    # Max Drawdown
    peak = np.maximum.accumulate(equity_strat)
    drawdown = (equity_strat - peak) / peak
    max_dd = np.min(drawdown)

    # Trade Count
    trade_count = np.sum(turnover > 0)

    return {
        'equity_bh': equity_bh,
        'equity_strat': equity_strat,
        'metrics': {
            'Total Return (B&H)': f"{total_ret_bh:.2%}",
            'Total Return (Strat)': f"{total_ret_strat:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Sortino Ratio': f"{sortino:.2f}",
            'Max Drawdown': f"{max_dd:.2%}",
            'Trade Count': int(trade_count),
            'Cost Rate': f"{cost_rate:.2%}"
        }
    }

def plot_trade_signals(signals, last_n=500):
    """
    Visualize trading signals:
    1. Signal over time (showing the last `last_n` points)
    2. Histogram distribution of the signals

    Parameters:
    - signals: array-like, trading signals, typically in the range [-1, 1]
    - last_n: int, optional, number of recent points to display in the time plot
    """
    plt.figure(figsize=(12, 5))

    # Subplot 1: Signal over time
    plt.subplot(1, 2, 1)
    plt.plot(signals[-last_n:], label='Trade Signal', color='dodgerblue', alpha=0.8)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Trading Signals (Last {last_n} steps)")
    plt.ylabel("Signal Strength (-1 to 1)")
    plt.xlabel("Time")
    plt.legend()

    # Subplot 2: Histogram distribution of signals
    plt.subplot(1, 2, 2)
    sns.histplot(signals, bins=50, kde=True, color='green')
    plt.title("Signal Distribution")
    plt.xlabel("Signal Value")
    plt.xlim(-1.1, 1.1)

    plt.tight_layout()
    plt.show()


def plot_backtest_strategy(signals, returns, thresholds=[0.0, 0.6], cost_rates=[0.0, 0.0001], test_days='Last 14 Days'):
    """
    Compare backtest results of different strategy thresholds and costs, print metrics, and plot equity curves.

    Parameters:
    - signals: array-like, trading signals (e.g., from model prediction)
    - returns: array-like, real step returns
    - thresholds: list of float, thresholds for strategy execution
    - cost_rates: list of float, transaction cost rates corresponding to thresholds
    - test_days: str, label for the test period (used in plot title)
    """
    
    # Run backtests
    results = []
    for thresh, cost in zip(thresholds, cost_rates):
        res = backtest_strategy(signals, returns, threshold=thresh, cost_rate=cost)
        results.append(res)
    
    # Print metrics comparison
    print(f"\n📊 Strategy Performance Comparison ({test_days})")
    print("=" * 85)
    header = ['Metric'] + [f'Thresh={t}, Cost={c}' for t, c in zip(thresholds, cost_rates)]
    print(f"{header[0]:<22} | " + " | ".join(f"{h:<18}" for h in header[1:]))
    print("-" * 85)
    
    metrics_keys = results[0]['metrics'].keys()
    for k in metrics_keys:
        row = [k] + [res['metrics'][k] for res in results]
        print(f"{row[0]:<22} | " + " | ".join(f"{v:<18}" for v in row[1:]))
    print("=" * 85)
    
    # Plot equity curves
    plt.figure(figsize=(15, 7))
    plt.plot(results[0]['equity_bh'], label='Buy & Hold (BTC)', color='black', alpha=0.3, linewidth=2, linestyle='--')
    
    for i, res in enumerate(results):
        plt.plot(res['equity_strat'], label=f'LSTM (Thresh={thresholds[i]}, Cost={cost_rates[i]})', 
                 linewidth=2 if thresholds[i] > 0 else 1, alpha=0.7 if thresholds[i] == 0 else 1,
                 color='green' if thresholds[i] == 0 else 'blue')
    
    plt.title(f'LSTM Strategy Backtest (Threshold & Cost Analysis) - Test Days: {test_days}')
    plt.ylabel('Normalized Equity')
    plt.xlabel('Time (Minutes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    pass
    