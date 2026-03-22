
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import RobustScaler

class FeatureProcessor:
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
        self.lag_cols = []  # to record lag feature names

    # macro assets returns
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
        df, ext_cols = self.add_external_asset_returns(df)
        # Lag features
        lag_cols_base = ['Open','High','Low','Close','Volume'] + ext_cols
        df = self.add_lag_features(df, cols=lag_cols_base)
        # technical / Candle / Volume
        df = self.add_sma_ema(df)
        df = self.add_macd_rsi(df)
        df = self.add_vwap_features(df)
        df = self.add_candle_features(df)
        df = self.add_volume_features(df)
        df = self.add_rsi_slope(df)
        df.dropna(inplace=True)
        return df

    def get_feature_columns(self, df):
        """ Get all feature columns """
        all_cols = [c for c in df.columns if c not in ['target_log_return','target_cum_return_30m','target_binary']]
        return all_cols


class TargetProcessor:
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

    # cumulative log return
    def get_cum_log_return(self, df, window=30):
        col = f'target_cum_return_{window}m'
        return self.get_log_return(df, lag=window, col_name=col)

    # binary target
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

    # clip extreme values & RobustScaler
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

    # fit_transform (training phase)
    def fit_transform(self, df):
        df = df.copy()

        # 1. Generate all targets
        df = self.get_log_return(df)
        df = self.get_cum_log_return(df, window=self.window)
        df = self.get_binary_target(df)

        # target normalization
        base_targets = [
            'target_log_return',
            f'target_cum_return_{self.window}m'
        ]

        # 2. Volatility normalization
        df, norm_cols = self.normalize_by_volatility(df, base_targets)

        # 3. Clip + RobustScaler
        df = self.clean_and_scale(df, norm_cols, fit=True)

        return df, norm_cols

    # transform (evaluation phase)
    def transform(self, df):
        df = df.copy()

        # log return / cum return， but do not compute binary target
        df['target_log_return'] = np.nan
        df[f'target_cum_return_{self.window}m'] = np.nan

        base_targets = [
            'target_log_return',
            f'target_cum_return_{self.window}m'
        ]

        df, norm_cols = self.normalize_by_volatility(df, base_targets)

        df = self.clean_and_scale(df, norm_cols, fit=False)

        return df, norm_cols

    # ============================================================
    # inverse (for predictions)
    # ============================================================
    def inverse_transform(self, scaled_pred, volatility=None):
        """
        scaled_pred: scaled
        volatility: rolling volatility
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted.")

        if volatility is None:
            volatility = self.volatility

        # inverse RobustScaler
        unscaled = self.scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten()

        # inverse volatility scaling
        return unscaled * volatility.values[-len(unscaled):]

