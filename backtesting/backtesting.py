"""
===============================================================================
Backtesting & Performance Evaluation Module
===============================================================================

Description:
    This module provides a comprehensive suite of tools for quantitative trading 
    strategy evaluation. It bridges the gap between machine learning model 
    predictions and financial performance metrics.

    Key functionalities include:
    1. Signal Generation: Converting raw model predictions (returns) into 
       bounded trading signals (tanh mapping).
    2. Vectorized Backtesting: Simulating trading logic with transaction costs, 
       calculating equity curves, and position turnover.
    3. Risk Metrics: Calculation of Sharpe, Sortino, Max Drawdown, VaR/CVaR, 
       Information Ratio, and Treynor Ratio.
    4. Visualization: Professional-grade plotting for:
       - Equity curves (Strategy vs Benchmark)
       - Signal distribution and time-series analysis
       - Metric heatmaps and bar comparison charts
       - "Red/Green" dashboard for signal regime analysis

Dependencies:
    - numpy, pandas, math
    - matplotlib.pyplot, seaborn

Usage:
    Import this module to evaluate model predictions against market data.
    >>> from backtesting import backtest_strategy, plot_metric_comparison

Author: BTC SATOSHI LAB
Last Updated: 2025-12
===============================================================================
"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates




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

def set_academic_style():
    """ """
    sns.set_theme(style="whitegrid")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300
    })

def plot_market_scenarios_vertical(df):
    """
    Row 1: Ideal Market Scenarios
    Row 2: Real Market Performance
    """
    set_academic_style()

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    C_BENCH = '#4D4D4D'
    C_LSTM  = '#2E86C1'
    C_XGB   = '#00235B'  #00235B
    C_ARIMA = '#117A65'
    C_RIDGE = '#F39C12'

    # Subplot 1: Ideal Market (Top)
    ax1 = axes[0]

    ax1.plot(df.index, df['benchmark'], label='Benchmark', color=C_BENCH, ls='--', alpha=0.7)
    ax1.plot(df.index, df['cum_return_ideal_xgb'], label='Ideal XGB', color=C_XGB, lw=2)
    ax1.plot(df.index, df['cum_return_ideal_lstm'], label='Ideal LSTM', color=C_LSTM, lw=2)
    ax1.plot(df.index, df['cum_return_ideal_arima'], label='Ideal ARIMA', color=C_ARIMA, lw=2)
    ax1.plot(df.index, df['cum_return_ideal_ridge'], label='Ideal RIDGE', color=C_RIDGE, lw=2)

    ax1.set_title("(a) Ideal Market Scenarios: Theoretical Upper Bounds",
                  fontsize=14, fontweight='bold', loc='left', pad=10)
    ax1.set_ylabel("Cumulative Return (Ideal)", fontsize=11)
    ax1.legend(frameon=False, loc='upper left')
    ax1.margins(x=0.01)

    # Subplot 2: Real Market (Bottom)
    ax2 = axes[1]

    ax2.plot(df.index, df['benchmark'], label='Benchmark', color=C_BENCH, ls='--', alpha=0.7)
    ax2.plot(df.index, df['cum_return_real_xgb'], label='Real XGB', color=C_XGB, lw=2)
    ax2.plot(df.index, df['cum_return_real_lstm'], label='Real LSTM', color=C_LSTM, lw=2)
    ax2.plot(df.index, df['cum_return_real_arima'], label='Real ARIMA', color=C_ARIMA, lw=2)
    ax2.plot(df.index, df['cum_return_real_ridge'], label='Real RIDGE', color=C_RIDGE, lw=2)

    ax2.set_title("(b) Real Market Performance: Actual Strategy Returns",
                  fontsize=14, fontweight='bold', loc='left', pad=10)
    ax2.set_xlabel("Time Steps (Minutes)", fontsize=11)
    ax2.set_ylabel("Cumulative Return (Real)", fontsize=11)
    ax2.legend(frameon=False, loc='lower left')

    ax2.margins(x=0.01)

    plt.tight_layout()
    return fig

def sharpe_ratio(r, rf=0, periods=252):
    excess = r - rf
    if excess.std() == 0: return np.nan
    return np.sqrt(periods) * excess.mean() / excess.std()

def sortino_ratio(r, rf=0, periods=252):
    excess = r - rf
    downside = excess[excess < 0]
    if downside.std() == 0: return np.nan
    return np.sqrt(periods) * excess.mean() / downside.std()

def max_drawdown(cum):
    roll = cum.cummax()
    dd = (cum - roll) / roll
    return dd.min()

def information_ratio(r, benchmark):
    diff = r - benchmark
    if diff.std() == 0: return np.nan
    return diff.mean() / diff.std()

def treynor_ratio(r, benchmark, rf=0):
    cov = np.cov(r - rf, benchmark - rf)[0][1]
    var_b = np.var(benchmark - rf)
    if var_b == 0: return np.nan
    beta = cov / var_b
    if beta == 0: return np.nan
    return (r.mean() - rf) / beta

def cvar(r, alpha=0.05):
    VaR = r.quantile(alpha)
    tail = r[r <= VaR]
    if len(tail) == 0: return np.nan
    return tail.mean()


def build_metric_tables(df):
    benchmark_cum = df["benchmark"]
    benchmark = benchmark_cum.pct_change().dropna()

    model_names = sorted(
        set(col.split("_")[-1] for col in df.columns if col.startswith("signals_"))
    )

    ideal_rows = {"benchmark": {}}
    real_rows = {"benchmark": {}}

    # ===== Benchmark =====
    ideal_rows["benchmark"] = {
        "Sharpe": sharpe_ratio(benchmark),
        "Sortino": sortino_ratio(benchmark),
        "MaxDD": max_drawdown(benchmark_cum),
        "IR": np.nan,
        "Treynor": np.nan,
        "CVaR(5%)": cvar(benchmark),
    }
    real_rows["benchmark"] = ideal_rows["benchmark"]

    for m in model_names:

        r_ideal = df[f"cum_return_ideal_{m}"].pct_change().dropna()
        r_real = df[f"cum_return_real_{m}"].pct_change().dropna()

        ideal_rows[m] = {
            "Sharpe": sharpe_ratio(r_ideal),
            "Sortino": sortino_ratio(r_ideal),
            "MaxDD": max_drawdown(df[f"cum_return_ideal_{m}"]),
            "IR": information_ratio(r_ideal, benchmark),
            "Treynor": treynor_ratio(r_ideal, benchmark),
            "CVaR(5%)": cvar(r_ideal),
        }

        real_rows[m] = {
            "Sharpe": sharpe_ratio(r_real),
            "Sortino": sortino_ratio(r_real),
            "MaxDD": max_drawdown(df[f"cum_return_real_{m}"]),
            "IR": information_ratio(r_real, benchmark),
            "Treynor": treynor_ratio(r_real, benchmark),
            "CVaR(5%)": cvar(r_real),
        }

    ideal_table = pd.DataFrame(ideal_rows).T
    real_table = pd.DataFrame(real_rows).T

    return ideal_table, real_table


def plot_heatmap(table, title):
    plt.figure(figsize=(10, 5))

    # Use a diverging color map: Red (Bad) -> Yellow -> Green (Good)
    # This matches financial intuition (High Sharpe is good/Green, High Drawdown (neg) is bad/Red)
    cmap = "RdYlGn"

    ax = sns.heatmap(
        table,
        annot=True,
        cmap=cmap,
        center=0.0,      # Center the colormap at 0
        fmt=".3f",
        linewidths=1.2,
        linecolor="#f0f0f0",
        cbar_kws={"shrink": 0.8, "label": "Metric Value"},
        annot_kws={"size": 10, "weight": "bold"}
    )

    # Styling for a professional report look
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left', pad=20)

    # Move X-axis labels to the top for a "Table" feel
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(rotation=0, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_metric_comparison(df, title="Model Performance Evaluation"):
    # 1. Transpose the DataFrame so that Index=Metrics, Columns=Models
    # This ensures we get one subplot per Metric, comparing all Models.
    df = df.T

    # 2. Update color map to match lowercase model names in the dataframe
    color_map = {
        'benchmark': '#95A5A6',
        'arima':     '#117A65',
        'lstm':      '#2E86C1',
        'xgb':       '#2C3E50',
        'ridge':     '#F39C12',
    }
    default_color = '#7F8C8D'

    metrics = df.index.tolist()
    models = df.columns.tolist()

    n_cols = 3
    n_rows = math.ceil(len(metrics) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), dpi=120)
    fig.suptitle(title, fontsize=18, weight='bold', color='#2C3E50', y=0.98)

    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = df.loc[metric].values

        # Match lowercase model names
        current_colors = [color_map.get(m.lower(), default_color) for m in models]

        bars = ax.bar(models, vals, color=current_colors, edgecolor='white', width=0.7, alpha=0.9)

        ax.set_title(metric, fontsize=13, weight='bold', color='#34495E', pad=12)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.margins(y=0.25)

        # Add value labels
        for bar, val in zip(bars, vals):
            if pd.isna(val):
                continue
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2.

            # Smart positioning of labels (above positive bars, below negative bars)
            if height >= 0:
                xy_pos = height
                va_align = 'bottom'
                offset = 3
            else:
                xy_pos = height
                va_align = 'top'
                offset = -3

            # Format numbers
            if abs(height) < 0.001 and height != 0:
                label_text = f'{height:.1e}'
            else:
                label_text = f'{height:.4f}'

            ax.annotate(label_text,
                        xy=(x_pos, xy_pos),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va=va_align,
                        fontsize=9, fontweight='medium', color='#2C3E50')

    # Hide unused subplots
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0, w_pad=2.0)
    plt.show()

def plot_red_green_dashboard(df):

    sns.set_style("white", {"axes.spines.right": False, "axes.spines.top": False})

    models = sorted([col.replace('signals_', '') for col in df.columns if col.startswith('signals_')])

    color_long = '#2ECC71'
    color_short = '#FF6B6B'
    color_line = '#2C3E50'


    n_models = len(models)

    fig = plt.figure(figsize=(16, 4 * n_models), dpi=120)

    gs = fig.add_gridspec(n_models, 2, width_ratios=[3, 1], wspace=0.05, hspace=0.5)

    fig.suptitle('Trading Signals Dashboard (14 Days View)',
                 fontsize=18, weight='bold', y=0.96, color='#2C3E50')

    for i, model in enumerate(models):
        col_name = f'signals_{model}'

        # -------------------------------------------------
        # Time Series Plot
        # -------------------------------------------------
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax_ts.plot(df.index, df[col_name], color=color_line, linewidth=0.6, alpha=0.5)

        ax_ts.fill_between(df.index, df[col_name], 0,
                           where=(df[col_name] >= 0),
                           interpolate=True, color=color_long, alpha=0.5)

        ax_ts.fill_between(df.index, df[col_name], 0,
                           where=(df[col_name] < 0),
                           interpolate=True, color=color_short, alpha=0.5)

        ax_ts.set_title(f"Model: {model.upper()}", fontsize=14, weight='bold', loc='left', color='#34495E', pad=10)

        ax_ts.set_ylabel('Signal')
        ax_ts.grid(axis='y', linestyle=':', alpha=0.4)

        # -------------------------------------------------
        # Histogram
        # -------------------------------------------------
        ax_hist = fig.add_subplot(gs[i, 1], sharey=ax_ts)
        n, bins, patches = ax_hist.hist(df[col_name], bins=40, orientation='horizontal',
                                        color='gray', alpha=0.6, density=True, edgecolor='none')

        for patch, bin_val in zip(patches, bins):
            if bin_val >= 0:
                patch.set_facecolor(color_long)
            else:
                patch.set_facecolor(color_short)
            patch.set_alpha(0.6)

        ax_hist.axhline(0, color='black', linewidth=0.8, alpha=0.3)
        ax_hist.axis('off')
        ax_hist.axvline(0, color='black', linewidth=0.5)

        # -------------------------------------------------
        limit = max(abs(df[col_name].min()), abs(df[col_name].max())) * 1.1
        if limit == 0: limit = 1.0
        ax_ts.set_ylim(-limit, limit)
        if i == len(models) - 1:

            ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax_ts.set_xlabel('Date (Month-Day)', fontsize=11)
        else:
            ax_ts.set_xticklabels([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass
    