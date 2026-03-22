import pandas as pd
import numpy as np 

# ---------- Data loading functions ----------
def load_btc(file_path, tz = 'America/New_york'):
    """ Load BTC (UNIX timestamp) """
    df = pd.read_csv(file_path)
    time_col = 'Timestamp'
    
    df[time_col] = pd.to_datetime(df[time_col], unit = 's', utc = True, errors = 'coerce')
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()
    df = df.tz_convert(tz)
    
    df = df[df.index >= pd.Timestamp('2025-01-01', tz=tz)] # only keep data from 2025 onwards
    
    return df
    
def load_other(file_path, tz="America/New_York"):
    """ Load other assets (normal datetime) """
    df = pd.read_csv(file_path)
    time_col = 'datetime'

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()
    df = df.tz_convert(tz)
    return df

def mark_us_market_hours(df):
    """
    Adds a column 'US_Trading_Hours' indicating whether the index is within
    US stock market hours (09:30 - 16:00 EST, Mon - Fri).
    """
    df = df.copy()
    # Ensure datetime index is in New York timezone
    if df.index.tzinfo is None:
        df = df.tz_localize("America/New_York")
    else:
        df = df.tz_convert("America/New_York")

    local_index = df.index

    # US market is Mon–Fri, 09:30–16:00
    df['US_Trading_Hours'] = (
        (local_index.weekday < 5) &  # Mon–Fri
        (
            ((local_index.hour == 9) & (local_index.minute >= 30)) |
            ((local_index.hour > 9) & (local_index.hour < 16))
        )
    )

    return df

def build_merged_dataset(
    btc_path="data/btcusd_1-min_data.csv",
    eth_path="data/ETH_USD_1min_2020_2025.csv",
    eur_path="data/EUR_USD_1min_2020_2025.csv",
    gld_path="data/GLD_1min_2020_2025.csv",
    spy_path="data/SPY_1min_2020_2025.csv",
    vixy_path="data/VIXY_1min_2020_2025.csv"
):
    print("Loading CSV files...")

    # Load all assets
    btc  = load_btc(btc_path)
    eth  = load_other(eth_path)
    eur  = load_other(eur_path)
    gld  = load_other(gld_path)
    spy  = load_other(spy_path)
    vixy = load_other(vixy_path)

    print("Aligning to BTC 7x24 timeline...")
    full_index = btc.index

    # Reindex 24/7 assets with forward-fill
    eth  = eth.reindex(full_index).ffill()
    eur  = eur.reindex(full_index).ffill()

    # ----------------- Smart-fill for limited trading assets -----------------
    def smart_fill_us_hours(df):
        df = df.reindex(full_index)
        df['tmp'] = np.nan
        trading_hours = df.between_time("04:00", "11:59") # since these assests only trade in 04:00-12:00 EST
        df.loc[trading_hours.index, :] = trading_hours
        df = df.ffill()
        df = df.drop(columns=['tmp'])
        return df
    
    gld  = smart_fill_us_hours(gld)
    spy  = smart_fill_us_hours(spy)
    vixy = smart_fill_us_hours(vixy)
    # ------------------------------------------------------------------------

    print("Merging all assets...")
    merged = btc.join([eth, eur, gld, spy, vixy], how="left")

    print("Adding US market trading hours flag...")
    merged = mark_us_market_hours(merged)

    print("Done! Final shape:", merged.shape)
    return merged

def preprocess_merged_df(df):
    """
    1. ETH/USD EUR/USD interpolate and forward-fill
    2. mask columns for ETH and EUR
    3. generate mask columns for each asset
    4. conert US_Trading_Hours to int
    """
    # ETH/USD fill
    eth_cols = ['ETH/USD_open', 'ETH/USD_high', 'ETH/USD_low', 'ETH/USD_close']
    df[eth_cols] = df[eth_cols].interpolate(method='time').ffill()

    # EUR/USD fill
    eur_cols = ['EUR/USD_open', 'EUR/USD_high', 'EUR/USD_low', 'EUR/USD_close']
    df[eur_cols] = df[eur_cols].interpolate(method='time').ffill()

    # mask 
    df['ETH_mask'] = df['ETH/USD_close'].notna()
    df['EUR_mask'] = df['EUR/USD_close'].notna()

    # convert US_Trading_Hours to int
    df['US_Trading_Hours'] = df['US_Trading_Hours'].astype(int)

    tradable_assets = ['GLD', 'SPY', 'VIXY']
    for asset in tradable_assets:
        df[f'{asset}_mask'] = df[f'{asset}_close'].notna()

    return df



if __name__ == "__main__":
    df = build_merged_dataset()