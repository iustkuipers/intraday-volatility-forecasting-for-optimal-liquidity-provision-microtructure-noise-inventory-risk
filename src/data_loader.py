import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load TAQ quote data and return cleaned dataframe
    with timestamp index, mid price, and log returns.
    """

    # 1. Load CSV
    df = pd.read_csv(filepath)

    # 2. Standardize column names
    df.columns = df.columns.str.lower()

    # Expected: date, time_m, bid, bidsiz, ask, asksiz
    required_cols = ["date", "time_m", "bid", "bidsiz", "ask", "asksiz"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 3. Parse timestamp
    # time_m is already in HH:MM:SS.nanoseconds format from WRDS
    df["time_m"] = df["time_m"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time_m"]
    )

    # 4. Set index early for time filtering
    df = df.set_index("timestamp")

    # 5. Filter to regular trading hours (09:30-15:59:59.999999999)
    df = df.between_time("09:30", "15:59:59.999999999")

    # 6. Drop bad quotes
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["ask"] >= df["bid"]]

    # 7. Remove extreme spreads (> 1% of mid) - microstructure noise
    spread = df["ask"] - df["bid"]
    mid_price = (df["bid"] + df["ask"]) / 2
    df = df[spread / mid_price < 0.01]

    # 8. Compute mid
    df["mid"] = (df["bid"] + df["ask"]) / 2

    # 9. Sort and compute log returns
    df = df.sort_index()
    df["log_return"] = np.log(df["mid"]).diff()

    return df[["bid", "ask", "bidsiz", "asksiz", "mid", "log_return"]]

def resample_to_1min(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Resample tick data to 1-minute frequency using last quote.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tick-level data with timestamp index
    verbose : bool
        Print progress updates (default True)
    
    Returns
    -------
    pd.DataFrame
        1-minute resampled data
    """
    
    def log(msg):
        if verbose:
            elapsed = (datetime.now() - start_time).total_seconds()
            mins, secs = int(elapsed // 60), int(elapsed % 60)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp} | {mins}m {secs}s] {msg}", flush=True)
    
    start_time = datetime.now()
    log(f"Resampling {len(df):,} tick rows to 1-minute frequency...")
    
    df_1min = df.resample("1min").last()
    log(f"✅ Resampled to {len(df_1min):,} 1-minute bars")

    # Drop empty minutes
    log("Dropping empty minutes...")
    df_1min = df_1min.dropna(subset=["mid"])
    log(f"✅ After removing empty minutes: {len(df_1min):,} bars")

    # Recompute log returns after resampling
    log("Recomputing log returns...")
    df_1min["log_return"] = np.log(df_1min["mid"]).diff()
    log(f"✅ resample_to_1min() complete")

    return df_1min