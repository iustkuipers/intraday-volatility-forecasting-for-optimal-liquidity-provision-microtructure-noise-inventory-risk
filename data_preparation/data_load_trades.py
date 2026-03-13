"""
data_load_trades.py  —  Clean TAQ consolidated trades and save to parquet.

Output columns: timestamp, ex, sym_root, trade_price, trade_size, trade_direction
    trade_direction is NaN here; computed via Lee-Ready after merging with quotes.

Irregular TR_SCOND codes excluded:
    T  — Form T (extended hours)
    Z  — Out of sequence
    L  — Sold last
    W  — Average price trade
    G  — Bunched trade
    J  — Roll trade
    K  — Roll trade (amended)

Saves:
    ../data/clean/trades.parquet       (full day)
    ../data/clean/trades_5min.parquet  (09:30 – 09:35)
"""

import pandas as pd
import numpy as np

RAW_PATH     = "data/raw/gqmrclkar4e23itg.csv"
OUT_FULL     = "data/clean/trades.parquet"
OUT_5MIN     = "data/clean/trades_5min.parquet"

MARKET_OPEN  = pd.Timestamp("09:30:00").time()
MARKET_CLOSE = pd.Timestamp("16:00:00").time()

# TR_SCOND values (or substrings) that flag irregular / non-regular prints
IRREGULAR_FLAGS = {"T", "Z", "L", "W", "G", "J", "K"}

# ── 1. Load & timestamp ───────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
print(f"Raw rows              : {len(df):,}")

df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME_M"])
df = df.drop(columns=["DATE", "TIME_M", "SYM_SUFFIX"], errors="ignore")
df = df.sort_values("timestamp").reset_index(drop=True)

# ── 2. Filter regular trading hours ──────────────────────────────────────────
t = df["timestamp"].dt.time
df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].reset_index(drop=True)
print(f"After RTH filter      : {len(df):,}")

# ── 3. Remove invalid trades (zero / negative price or size) ─────────────────
df = df[(df["PRICE"] > 0) & (df["SIZE"] > 0)].reset_index(drop=True)
print(f"After price/size check: {len(df):,}")

# ── 4. Remove irregular trade conditions ─────────────────────────────────────
# Keep rows where TR_SCOND is NaN (blank = regular) or contains none of the
# irregular flag letters.
def is_regular(cond):
    if pd.isna(cond):
        return True
    tokens = set(str(cond).split())
    return tokens.isdisjoint(IRREGULAR_FLAGS)

mask = df["TR_SCOND"].apply(is_regular)
df = df[mask].reset_index(drop=True)
print(f"After condition filter : {len(df):,}")

# ── 5. Rename to clean schema ─────────────────────────────────────────────────
df = df.rename(columns={"PRICE": "trade_price", "SIZE": "trade_size",
                        "EX": "ex", "SYM_ROOT": "sym_root"})
df = df.drop(columns=["TR_SCOND"], errors="ignore")

# trade_direction requires Lee-Ready (needs quote mid at trade time).
# Placeholder NaN — assigned in data_merged.py after merging with quotes.
df["trade_direction"] = np.nan

df = df[["timestamp", "ex", "sym_root", "trade_price", "trade_size", "trade_direction"]]

# ── 6. Save ───────────────────────────────────────────────────────────────────
df.to_parquet(OUT_FULL, index=False)
print(f"\nSaved full            : {OUT_FULL}  ({len(df):,} rows)")

window_end = df["timestamp"].iloc[0].normalize() + pd.Timedelta(hours=9, minutes=35)
df_5min = df[df["timestamp"] < window_end].copy()
df_5min.to_parquet(OUT_5MIN, index=False)
print(f"Saved 5-min           : {OUT_5MIN}  ({len(df_5min):,} rows)")
