import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# Global Style Configuration
# =============================================================================
STYLE_CONFIG = {
    'color_true': '#7399C6',
    'color_pred': '#A50026',   #A50026, #8B0000, , #C0392B, #7399C6
    'color_grid': '#D9D9D9',

    'linewidth': 1.5,
    'alpha_true': 0.7,
    'alpha_pred': 1.0,
    'font_title': 16,
    'font_label': 12,
    'font_family': 'serif'
}

def win_rate(y_true, y_pred):
    hits = (np.sign(y_true) == np.sign(y_pred))
    return np.mean(hits)

def plot_true_pred(y_true, y_pred, title= 'Model Forecast vs Actual (test set))', save_path=None):
    """ plot true vs predicted values"""
    with plt.style.context('seaborn-v0_8-white'):
        plt.rcParams['font.family'] = STYLE_CONFIG['font_family']

        plt.figure(figsize=(12, 6), dpi=300)

        plt.plot(
            y_true,
            label='Actual',
            color=STYLE_CONFIG['color_true'],
            linewidth=STYLE_CONFIG['linewidth'],
            linestyle='-',
            alpha=STYLE_CONFIG['alpha_true']
        )

        plt.plot(
            y_pred,
            label='Forecast',
            color=STYLE_CONFIG['color_pred'],
            linewidth=STYLE_CONFIG['linewidth'],
            linestyle='-',
            alpha=STYLE_CONFIG['alpha_pred']
        )

        plt.title(title, fontsize=STYLE_CONFIG['font_title'], fontweight='bold', loc='left', pad=20)

        plt.xlabel('Time Steps', fontsize=STYLE_CONFIG['font_label'])
        plt.ylabel('Return / Value', fontsize=STYLE_CONFIG['font_label'])

        plt.legend(loc='upper right', frameon=False, fontsize=11)

        plt.grid(True, axis='y', linestyle='-', color=STYLE_CONFIG['color_grid'], alpha=1)
        plt.grid(False, axis='x')

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_linewidth(1.2)
        plt.gca().spines['left'].set_linewidth(1.2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

def plot_loss(train_losses, val_losses=None, title="Training & Validation Loss"):
    """
    """
    with plt.style.context('seaborn-v0_8-white'):
        plt.rcParams['font.family'] = STYLE_CONFIG['font_family']
        plt.figure(figsize=(10, 5), dpi=300)

        plt.plot(
            train_losses,
            label='Train Loss',
            color=STYLE_CONFIG['color_true'],
            linewidth=STYLE_CONFIG['linewidth']
        )

        if val_losses is not None:
            plt.plot(
                val_losses,
                label='Val Loss',
                color=STYLE_CONFIG['color_pred'],
                linewidth=STYLE_CONFIG['linewidth']
            )

        plt.title(title, fontsize=STYLE_CONFIG['font_title'], fontweight='bold', loc='left', pad=15)
        plt.xlabel("Epoch", fontsize=STYLE_CONFIG['font_label'])
        plt.ylabel("Loss (RMSE)", fontsize=STYLE_CONFIG['font_label'])

        plt.legend(frameon=False, fontsize=11)

        plt.grid(True, axis='y', linestyle='-', color=STYLE_CONFIG['color_grid'])
        plt.grid(False, axis='x')

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

# Regression Metrics
def regression_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    wr   = win_rate(y_true, y_pred)

    print("-" * 40)
    print(f"Model Performance Metrics")
    print("-" * 40)
    print(f"{'MSE':<12} : {mse:.6f}")
    print(f"{'MAE':<12} : {mae:.6f}")
    print(f"{'RMSE':<12} : {rmse:.6f}")
    print(f"{'R² Score':<12} : {r2:.4f}")
    print(f"{'Win Rate':<12} : {wr:.2%}")
    print("-" * 40)

    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Win_Rate': wr
    }