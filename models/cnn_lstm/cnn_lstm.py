import os
import talib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, num_layers = 2, dropout = 0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #1 CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = input_dim, out_channels = 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        #2 connnet CNN to LSTM
        self.feature_fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        #3 LSTM
        self.lstm = nn.LSTM(
            input_size = hidden_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout
        )

        #4 layer Norm
        self.norm = nn.LayerNorm(hidden_dim)

        #5 Attention 
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )

        #6    
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # initialization (Xavier/Orthogonal)
        self._init_weights()

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]  (Batch, Sequence, Features)
        """
        # === Step 1: CNN ===
        # Conv1d -> [batch, channels, seq_len]
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn(x_cnn)

        # === Step 2: pre LSTM ===
        cnn_out = cnn_out.transpose(1, 2) # -> [Batch, Seq_Len, 64]
        cnn_out = self.feature_fc(cnn_out) # -> [Batch, Seq_Len, Hidden_Dim] [Batch, Seq_Len, 128]

        #=== Step 3: LSTM ===
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.norm(lstm_out) # lstm_out: [Batch, Seq_Len, Hidden_Dim]

        #=== Step 4: Attention ===
        attn_score = self.attention(lstm_out)
        attention_weights = torch.softmax(attn_score, dim = 1) #attn_weights: [Batch, Seq_Len, 1]

        context = torch.sum(attention_weights * lstm_out, dim = 1)   # [Batch, Seq_Len, 1] * [Batch, Seq_Len, 128] -> [Batch, 128] [Batch, Hidden_Dim]

        # === Step 5: Output ===
        output = self.fc_out(context)
        return output

    def _init_weights(self):
        ''' Initialize weights for LSTM and other layers '''
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)
                n = param.size(0)
                param.data[n // 4:n // 2] = 1.0

        for module in list(self.cnn) + list(self.feature_fc) + list(self.attention) + list(self.fc_out):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

class RMSELoss(nn.Module):
    """ Loss function: RMSE """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        return torch.sqrt(mse + self.eps)

def create_sequences(X, y, seq_len=30):
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    y_values = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
    y_values = y_values.ravel()

    num_samples = len(X_values) - seq_len
    X_seq = np.zeros((num_samples, seq_len, X_values.shape[1]), dtype=np.float32)
    y_seq = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        X_seq[i] = X_values[i:i+seq_len]
        y_seq[i, 0] = y_values[i + seq_len]  # predict next step

    return X_seq, y_seq

class LstmTargetProcessor:
    def __init__(self, price_col='Close', window=30):
        self.price_col = price_col
        self.window = window

        # Scaler for normalized target
        self.scaler = None
        # rolling volatility (inverse)
        self.volatility = None

    # log return
    def get_log_return(self, df, lag=1, col_name='target_log_return'):
        df[col_name] = np.log(df[self.price_col].shift(-lag) / df[self.price_col])
        return df

    # cumlative log return
    def get_cum_log_return(self, df, window=30):
        col = f'target_cum_return_{window}m'
        return self.get_log_return(df, lag=window, col_name=col)
    
    # 3. binary target
    def get_binary_target(self, df, lag=1, col_name='target_binary'):
        future_price = df[self.price_col].shift(-lag)
        df[col_name] = (future_price > df[self.price_col]).astype(float)
        df.loc[future_price.isna(), col_name] = np.nan
        return df

    # normalize by volatility
    def normalize_by_volatility(self, df, target_cols):
        log_ret = np.log(df[self.price_col] / df[self.price_col].shift(1))
        vol = log_ret.rolling(self.window).std().shift(1)

        eps = 1e-8
        vol = vol.fillna(vol.median()).clip(lower=eps)
        self.volatility = vol  # use for inverse transform later

        new_cols = []
        for col in target_cols:
            new_col = f"{col}_norm"
            df[new_col] = df[col] / vol
            new_cols.append(new_col)

        return df, new_cols

    # clip & RobustScaler    
    def clean_and_scale(self, df, target_cols, fit=True):
        df = df.copy()

        # clip extreme values
        for col in target_cols:
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df[col] = df[col].clip(low, high)

        # scaling
        if fit:
            self.scaler = RobustScaler()
            df[target_cols] = self.scaler.fit_transform(df[target_cols])
        else:
            df[target_cols] = self.scaler.transform(df[target_cols])

        return df

    """ fit_transform (training phase) """
    def fit_transform(self, df):
        df = df.copy()

        # Generate all targets
        df = self.get_log_return(df)
        df = self.get_cum_log_return(df, window=self.window)
        df = self.get_binary_target(df)

        # target normalization
        base_targets = [
            'target_log_return',
            f'target_cum_return_{self.window}m'
        ]

        # Volatility normalization
        df, norm_cols = self.normalize_by_volatility(df, base_targets)

        # Clip + RobustScaler
        df = self.clean_and_scale(df, norm_cols, fit=True)

        return df, norm_cols

    """ transform (evaluation phase) """
    def transform(self, df):
        df = df.copy()

        df['target_log_return'] = np.nan
        df[f'target_cum_return_{self.window}m'] = np.nan

        base_targets = [
            'target_log_return',
            f'target_cum_return_{self.window}m'
        ]

        df, norm_cols = self.normalize_by_volatility(df, base_targets)

        df = self.clean_and_scale(df, norm_cols, fit=False)

        return df, norm_cols


    # inverse transform
    def inverse_transform(self, scaled_pred, volatility=None):
        """
        scaled_pred: preictions (scaled)
        volatility: rolling volatility (for inverse normalization)
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted.")

        if volatility is None:
            volatility = self.volatility

        # inverse RobustScaler
        unscaled = self.scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten()

        # inverse volatility normalization
        return unscaled * volatility.values[-len(unscaled):]

class LstmFeatureProcessor:
    def __init__(self,
                 rolling_smooth=5,
                 lag_steps=[1,2,3,5,10,15,30],
                 sma_windows=[3,7,15,30,50],
                 ema_windows=[3,7,15,30,50],
                 rsi_periods=[7,14,21]):
        self.rolling_smooth = rolling_smooth
        self.lag_steps = lag_steps
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.rsi_periods = rsi_periods
        self.lag_cols = [] 

    # External asset returns
    def add_external_asset_returns(self, df, external_cols=None):
        df = df.copy()
        if external_cols is None:
            external_cols = ['ETH/USD_close','EUR/USD_close','GLD_close','SPY_close','VIXY_close']
        ret_cols = []
        for col in external_cols:
            if col in df.columns:
                ret_col = f'{col}_ret'
                df[ret_col] = np.log(df[col] / df[col].shift(1))
                df.drop(columns=[col], inplace=True)
                ret_cols.append(ret_col)
        return df, ret_cols

    # Lag features
    def add_lag_features(self, df, cols=None):
        df = df.copy()
        if cols is None:
            cols = df.columns
        self.lag_cols = []  
        for col in cols:
            for lag in self.lag_steps:
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = df[col].shift(lag)
                self.lag_cols.append(lag_col)
        return df

    # SMA / EMA
    def add_sma_ema(self, df):
        df = df.copy()
        # SMA
        for w in self.sma_windows:
            df[f"SMA{w}"] = df['Close'].rolling(w).mean().shift(1)
        # EMA
        for w in self.ema_windows:
            ema = talib.EMA(df['Close'].values, timeperiod=w)
            ema = np.roll(ema, 1)  
            ema[0] = df['Close'].iloc[0]  
            df[f"EMA{w}"] = ema
        return df

    # MACD / RSI
    def add_macd_rsi(self, df):
        df = df.copy()
        close_shifted = df['Close'].shift(1).values  
        macd_line, macd_signal, macd_hist = talib.MACD(close_shifted)
        df['MACD_Line'] = macd_line
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        for period in self.rsi_periods:
            df[f"RSI{period}"] = talib.RSI(close_shifted, timeperiod=period)
        return df

    # VWAP / VWSDEV
    @staticmethod
    def calculate_daily_vwap(df):
        df = df.copy()
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TPV'] = df['TP'] * df['Volume']
        df['Day'] = df.index.date
        df['Cum_TPV'] = df.groupby('Day')['TPV'].cumsum()
        df['Cum_Volume'] = df.groupby('Day')['Volume'].cumsum()
        df['VWAP'] = df['Cum_TPV'] / df['Cum_Volume']
        df['VWSDEV'] = np.sqrt(
            (((df['TP'] - df['VWAP'])**2 * df['Volume']).groupby(df['Day']).cumsum())
            / df['Cum_Volume']
        )
        df.drop(columns=['TP','TPV','Day','Cum_TPV','Cum_Volume'], inplace=True)
        return df

    def add_vwap_features(self, df):
        df = df.copy()
        df = self.calculate_daily_vwap(df)
        df['dist_vwap'] = (df['Close'] - df['VWAP']) / df['VWAP']
        df['VWAP_smooth'] = df['VWAP'].rolling(self.rolling_smooth).mean()
        df['VWSDEV_smooth'] = df['VWSDEV'].rolling(self.rolling_smooth).mean()
        return df

    # Candle features
    def add_candle_features(self, df):
        df = df.copy()
        df['shadow_upper'] = ((df['High'] - df[['Open','Close']].max(axis=1)) / df['Open']).rolling(self.rolling_smooth).mean()
        df['shadow_lower'] = ((df[['Open','Close']].min(axis=1) - df['Low']) / df['Open']).rolling(self.rolling_smooth).mean()
        df['body_size'] = (df['Close'] - df['Open']) / df['Open']
        df['candle_range'] = (df['High'] - df['Low']) / df['Open']
        return df

    # Volume dynamics
    def add_volume_features(self, df):
        df = df.copy()
        df['log_volume'] = np.log(df['Volume'] + 1)
        df['vol_ratio'] = (df['Volume'] / df['Volume'].rolling(30).mean()).rolling(self.rolling_smooth).mean()
        df['price_vol_trend'] = df['body_size'] * df['log_volume']
        return df

    # RSI slope
    def add_rsi_slope(self, df):
        df = df.copy()
        if 'RSI7' in df.columns:
            df['RSI7_slope'] = (df['RSI7'] - df['RSI7'].shift(1)).rolling(self.rolling_smooth).mean()
        return df

    # Process all features
    def process_all(self, df):
        df = df.copy()
        # External asset returns
        df, ext_cols = self.add_external_asset_returns(df)
        # Lag features
        lag_cols_base = ['Open','High','Low','Close','Volume'] + ext_cols
        df = self.add_lag_features(df, cols=lag_cols_base)
        df = self.add_sma_ema(df)
        df = self.add_macd_rsi(df)
        df = self.add_vwap_features(df)
        df = self.add_candle_features(df)
        df = self.add_volume_features(df)
        df = self.add_rsi_slope(df)
        df.dropna(inplace=True)
        return df

    # Get feature columns
    def get_feature_columns(self, df):
        all_cols = [c for c in df.columns if c not in ['target_log_return','target_cum_return_30m','target_binary']]
        return all_cols


class LstmFactorAnalyzer:
    def __init__(self, df, feature_cols, target_col='target_cum_return_30m_norm'):
        """
        calculate feature quality: IC, MI, redundancy
        """
        self.clean_df = df[feature_cols + [target_col]].dropna()
        self.features = feature_cols
        self.target = target_col

        print(f"Factor Analyzer Initialized. Samples: {len(self.clean_df)}")

    
    # 1. IC & ICIR analysis (Information Coefficient)
    def calculate_ic_ir(self, method='spearman'):
        """
        calculate IC (Rank IC) & ICIR (stability of IC)        
            - IC: correlation between feature and target
            - Rolling IC: correlation over rolling windows
            - ICIR: Mean(Rolling IC) / Std(Rolling IC)
        """
        
        print("   -> Calculating IC & ICIR...")

        ic_data = []

        rolling_window = 500

        for feature in self.features:
            # 1. IC (Rank IC)
            # Using Spearman correlation for rank-based IC
            g_ic, _ = spearmanr(self.clean_df[feature], self.clean_df[self.target])

            # 2. Roling IC (for calculating IR)
            rolling_ic = self.clean_df[feature].rolling(rolling_window).corr(self.clean_df[self.target])
            rolling_ic = rolling_ic.dropna()

            ic_mean = rolling_ic.mean()
            ic_std = rolling_ic.std()

            # ICIR: Information Coefficient Information Ratio
            ic_ir = ic_mean / (ic_std + 1e-9)

            ic_data.append({
                'Feature': feature,
                'IC_Global': g_ic,      
                'IC_Mean': ic_mean,     
                'IC_Std': ic_std,       
                'ICIR': ic_ir,          
                'Abs_IC_Global': abs(g_ic) 
            })

        return pd.DataFrame(ic_data).sort_values(by='Abs_IC_Global', ascending=False)
 
    # 2. Mutual Information
    def calculate_mutual_info(self, n_neighbors=3):
        """
        calculate Mutual Information (MI) between features and target
        for non-linear dependency detection
        """
        print("   -> Calculating Mutual Information (Non-linear dependency)...")

        # accelalerated sampling for large datasets
        if len(self.clean_df) > 50000:
            sample_df = self.clean_df.sample(20000, random_state=42)
        else:
            sample_df = self.clean_df

        X = sample_df[self.features]
        y = sample_df[self.target]

        # calculate MI
        mi_scores = mutual_info_regression(X, y, discrete_features='auto', n_neighbors=n_neighbors, random_state=42)

        mi_df = pd.DataFrame({
            'Feature': self.features,
            'Mutual_Info': mi_scores
        }).sort_values(by='Mutual_Info', ascending=False)

        return mi_df
    
    # 3. correlated feature removal
    def remove_redundant_features(self, ic_df, threshold=0.90):
        """
        remove redundant features based on correlation
        """
        print(f"   -> Removing redundant features (Threshold > {threshold})...")

        corr_matrix = self.clean_df[self.features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()

        for column in upper.columns:
            # Find features with correlation greater than threshold
            high_corr_cols = upper.index[upper[column] > threshold].tolist()

            for row in high_corr_cols:
                # compare IC values to decide which feature to drop
                if row in to_drop or column in to_drop:
                    continue

                ic_row = ic_df.loc[ic_df['Feature'] == row, 'Abs_IC_Global'].values[0]
                ic_col = ic_df.loc[ic_df['Feature'] == column, 'Abs_IC_Global'].values[0]

                if ic_row > ic_col:
                    to_drop.add(column) # remove column
                    # print(f"      Drop {column} (Corr with {row}: {upper.loc[row, column]:.2f}, Lower IC)")
                else:
                    to_drop.add(row)    # remomve row
                    # print(f"      Drop {row} (Corr with {column}: {upper.loc[row, column]:.2f}, Lower IC)")

        print(f"      Dropped {len(to_drop)} redundant features.")
        return list(set(self.features) - to_drop)

    # 4. Full analysis pipeline
    def run_full_analysis(self, ic_threshold=0.01, mi_threshold=0.005, corr_threshold=0.90, top_n=None):
        """
        Full analysis pipeline to select high-quality features
         - Step 1: Calculate IC & ICIR
         - Step 2: Calculate Mutual Information
         - Step 3: Initial filtering based on IC/MI thresholds
         - Step 4: Remove redundant features based on correlation
         - Step 5: Select top N features based on combined ranking
        """
        # 1. IC/IR
        ic_df = self.calculate_ic_ir()

        # 2. MI
        mi_df = self.calculate_mutual_info()

        # 3. Merge IC & MI
        metrics_df = pd.merge(ic_df, mi_df, on='Feature')

        # 4. Initial filtering
        valid_features = metrics_df[
            (metrics_df['Abs_IC_Global'] > ic_threshold) |
            (metrics_df['Mutual_Info'] > mi_threshold)
        ]['Feature'].tolist()

        print(f"   -> Features passing signal threshold: {len(valid_features)} / {len(self.features)}")

        # update features for next step
        current_features = self.features
        self.features = valid_features 

        # 5. remove redundant features
        final_features = self.remove_redundant_features(metrics_df, threshold=corr_threshold)

        self.features = current_features

        # 6. Select top N features based on combined ranking
        if top_n and len(final_features) > top_n:

            final_metrics = metrics_df[metrics_df['Feature'].isin(final_features)].copy()
            final_metrics['Score'] = final_metrics['Abs_IC_Global'].rank(pct=True) + final_metrics['Mutual_Info'].rank(pct=True)
            final_features = final_metrics.sort_values(by='Score', ascending=False).head(top_n)['Feature'].tolist()

        print(f"Final Selected Features: {len(final_features)}")

        return final_features, metrics_df

    # 5. Visualization
    def plot_analysis(self, metrics_df, final_features):
        """
        Visualize feature quality: IC vs MI
        """
        plt.figure(figsize=(12, 6))
        metrics_df['Selected'] = metrics_df['Feature'].isin(final_features)

        sns.scatterplot(
            data=metrics_df,
            x='Abs_IC_Global',
            y='Mutual_Info',
            hue='Selected',
            alpha=0.7,
            palette={True: 'green', False: 'red'}
        )
        # Annotate top features
        top_feats = metrics_df[metrics_df['Selected']].sort_values(by='Mutual_Info', ascending=False).head(5)
        for _, row in top_feats.iterrows():
            plt.text(row['Abs_IC_Global'], row['Mutual_Info'], row['Feature'], fontsize=9)

        plt.title('Feature Quality: IC (Linear) vs MI (Non-linear)')
        plt.xlabel('Abs Global IC (Spearman)')
        plt.ylabel('Mutual Information')
        plt.grid(True, alpha=0.3)
        plt.show()

def rolling_zscore_matrix(df, feature_cols, window=60, eps=1e-8):
    """
    Rolling Z-Score
    """
    X_vals = df[feature_cols].values

    cumsum = np.cumsum(np.vstack([np.zeros(X_vals.shape[1]), X_vals]), axis=0)
    cumsum2 = np.cumsum(np.vstack([np.zeros(X_vals.shape[1]), X_vals**2]), axis=0)

    mean = (cumsum[window:] - cumsum[:-window]) / window
    mean2 = (cumsum2[window:] - cumsum2[:-window]) / window

    var = np.maximum(mean2 - mean**2, eps)
    std = np.sqrt(var)

    zscore = (X_vals[window-1:] - mean) / std

    index = df.index[window-1:]
    return pd.DataFrame(zscore, index=index, columns=feature_cols)
