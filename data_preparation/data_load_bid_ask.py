"""
data_load_bid_ask.py  —  Clean TAQ consolidated quotes and save to parquet.

Output columns: timestamp, ex, sym_root, bid, ask, bid_size, ask_size, mid, spread
Cleaning: RTH filter, invalid/crossed quotes removed, same-timestamp dedup.
Extreme spreads are kept — the MM model filters unfillable quotes naturally.
Saves:
    ../data/clean/quotes.parquet       (full day)
    ../data/clean/quotes_5min.parquet  (09:30 – 09:35)
"""

import pandas as pd

RAW_PATH     = "data/raw/fngctoxmiinkeimk.csv"
OUT_FULL     = "data/clean/quotes.parquet"
OUT_5MIN     = "data/clean/quotes_5min.parquet"

MARKET_OPEN  = pd.Timestamp("09:30:00").time()
MARKET_CLOSE = pd.Timestamp("16:00:00").time()

# Quotes whose spread exceeds this fraction of mid are stale / erroneous exchange
# feeds (e.g. one venue posting a price miles from the NBBO).  For SPY the normal
# NBBO spread is $0.01; this cap allows up to 0.2% of mid (~$0.94 at $472), which
# is ~94× normal — extremely liberal, removing only obvious garbage.
MAX_REL_SPREAD = 0.002

# ── 1. Load & timestamp ───────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
print(f"Raw rows          : {len(df):,}")

df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME_M"])
df = df.drop(columns=["DATE", "TIME_M", "SYM_SUFFIX"], errors="ignore")
df = df.sort_values("timestamp").reset_index(drop=True)

# ── 2. Filter regular trading hours ──────────────────────────────────────────
t = df["timestamp"].dt.time
df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].reset_index(drop=True)
print(f"After RTH filter  : {len(df):,}")

# ── 3. Remove invalid quotes (zero / negative / crossed) ─────────────────────
df = df[(df["BID"] > 0) & (df["ASK"] > 0) & (df["ASK"] > df["BID"])].reset_index(drop=True)
print(f"After quote filter: {len(df):,}")

# ── 4. Compute mid and spread ────────────────────────────────────────────────
df["mid"]    = (df["BID"] + df["ASK"]) / 2
df["spread"] = df["ASK"] - df["BID"]

# ── 4b. Remove extreme spread outliers (stale / erroneous exchange feeds) ────
# Quotes with spread > MAX_REL_SPREAD × mid are data errors: a single exchange
# posting a price far from the consolidated NBBO.  Removing them here prevents
# merge_asof in data_merged from attaching a wide-spread snapshot to a trade,
# which would make our deep-book quote look price-competitive and trigger a
# phantom fill ("NBBO spike artefact").
n_before = len(df)
df = df[df["spread"] / df["mid"] <= MAX_REL_SPREAD].reset_index(drop=True)
print(f"After spread cap      : {len(df):,}  (removed {n_before - len(df):,} spike quotes)")

# ── 5. Drop duplicate quote states at the same timestamp ─────────────────────
# Multiple exchanges can broadcast the same NBBO at the exact same nanosecond.
# Since EX was dropped, these are indistinguishable — keep first occurrence.
df = df.drop_duplicates(subset=["timestamp", "BID", "ASK", "BIDSIZ", "ASKSIZ"]).reset_index(drop=True)
print(f"After dedup       : {len(df):,}")

# ── 6. Rename to clean schema ─────────────────────────────────────────────────
df = df.rename(columns={"BID": "bid", "ASK": "ask", "BIDSIZ": "bid_size", "ASKSIZ": "ask_size",
                        "EX": "ex", "SYM_ROOT": "sym_root"})
df = df[["timestamp", "ex", "sym_root", "bid", "ask", "bid_size", "ask_size", "mid", "spread"]]

# ── 7. Save ───────────────────────────────────────────────────────────────────
df.to_parquet(OUT_FULL, index=False)
print(f"\nSaved full        : {OUT_FULL}  ({len(df):,} rows)")

window_end = df["timestamp"].iloc[0].normalize() + pd.Timedelta(hours=9, minutes=35)
df_5min = df[df["timestamp"] < window_end].copy()
df_5min.to_parquet(OUT_5MIN, index=False)
print(f"Saved 5-min       : {OUT_5MIN}  ({len(df_5min):,} rows)")
