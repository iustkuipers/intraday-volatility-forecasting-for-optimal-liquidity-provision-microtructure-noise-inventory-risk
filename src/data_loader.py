import pandas as pd
import numpy as np


def load_data(filepath: str, return_threshold: float = 0.01) -> pd.DataFrame:
    """
    Load TAQ quote data and return cleaned dataframe
    with timestamp index, mid price, and quote-based log returns.
    
    This function:
    - Filters to regular trading hours (09:30-15:59:59.999999999)
    - Removes locked/crossed markets (ask <= bid)
    - Removes extreme spreads (>1% of mid)
    - Removes quote-stuffing noise (consecutive identical quotes)
    - Filters outlier returns (microstructure glitches)
    - Returns QUOTE-BASED volatility (not trade-based)
    
    Parameters
    ----------
    filepath : str
        Path to TAQ CSV file
    return_threshold : float
        Maximum allowed log return (default 0.01 = 1% = bad data)
    
    Returns
    -------
    pd.DataFrame
        Cleaned quote data with columns:
        [bid, ask, bidsiz, asksiz, mid, log_return]
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

    # 5. Filter to regular trading hours (09:30 inclusive, 16:00 exclusive)
    #df = df.between_time("09:30", "15:59:59.9999", inclusive="left")
    df = df.between_time("09:30", "12:35", inclusive="left")  # For testing, use shorter window

    # 6. Remove quote-stuffing noise (consecutive identical quotes)
    # These add no information and distort volatility
    df["bid_changed"] = df["bid"].shift() != df["bid"]
    df["ask_changed"] = df["ask"].shift() != df["ask"]
    df = df[df["bid_changed"] | df["ask_changed"]]
    df = df.drop(columns=["bid_changed", "ask_changed"])

    # 7. Drop bad quotes (invalid bid/ask, crossed/locked)
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df[df["ask"] > df["bid"]]  # Exclude locked (ask==bid) and crossed (ask<bid)

    # 8. Remove extreme spreads (> 1% of mid) - microstructure noise
    spread = df["ask"] - df["bid"]
    mid_price = (df["bid"] + df["ask"]) / 2
    df = df[spread / mid_price < 0.01]

    # 9. Compute mid price
    df["mid"] = (df["bid"] + df["ask"]) / 2

    # 10. Sort and compute quote-based log returns
    df = df.sort_index()
    df["log_return"] = np.log(df["mid"]).diff()

    # 11. Remove outlier returns (bad data/glitches)
    # 1% tick-to-tick move in SPY at millisecond freq = almost always bad data
    df = df[df["log_return"].abs() < return_threshold]

    return df[["bid", "ask", "bidsiz", "asksiz", "mid", "log_return"]]

def compute_realized_vol(df: pd.DataFrame, minute: int = 1) -> pd.DataFrame:
    """
    Compute realized volatility from tick-level log returns.
    
    This computes true microstructure-aware volatility:
    RV = sqrt(sum of squared log returns within period)
    
    More accurate than close-to-close returns for high-frequency data.
    Essential for market-making models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tick-level data with 'log_return' column and timestamp index
    minute : int
        Resampling interval in minutes (default 1)
    
    Returns
    -------
    pd.DataFrame
        Data with columns:
        [bid, ask, mid, realized_vol]
        aggregated to minute frequency
    """
    
    # Compute realized variance: sum of squared returns
    rv_squared = (df["log_return"] ** 2).resample(f"{minute}min").sum()
    
    # Take square root to get realized volatility
    realized_vol = np.sqrt(rv_squared)
    
    # Get last quote of each minute for mid price
    df_1min = df[["bid", "ask", "mid", "bidsiz", "asksiz"]].resample(f"{minute}min").last()
    
    # Combine
    df_1min["realized_vol"] = realized_vol
    
    # Drop empty bars
    df_1min = df_1min.dropna(subset=["mid"])
    
    return df_1min[[
        "bid", "ask", "bidsiz", "asksiz", "mid", "realized_vol"
    ]]


def resample_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample tick data to 1-minute frequency using last quote.
    
    This is a simpler approach that uses close-to-close (quote-based)
    returns. For high-frequency analysis, use compute_realized_vol()
    instead, which properly captures intraday volatility.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tick-level data with timestamp index
    
    Returns
    -------
    pd.DataFrame
        1-minute aggregated data
    """

    df_1min = df.resample("1min").last()

    # Drop empty minutes
    df_1min = df_1min.dropna(subset=["mid"])

    # Recompute log returns after resampling (close-to-close 1-min returns)
    df_1min["log_return"] = np.log(df_1min["mid"]).diff()

    return df_1min