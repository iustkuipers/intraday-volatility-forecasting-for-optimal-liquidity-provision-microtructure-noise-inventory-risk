"""
data_analysis.py  —  EDA on the two raw TAQ files (bid/ask + trades).
"""

import pandas as pd
import numpy as np

BID_ASK_PATH = "data/raw/fngctoxmiinkeimk.csv"
TRADES_PATH  = "data/raw/gqmrclkar4e23itg.csv"

# ── helpers ────────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df["TIMESTAMP"] = pd.to_datetime(df["DATE"] + " " + df["TIME_M"])
    df = df.drop(columns=["DATE", "TIME_M", "SYM_SUFFIX"], errors="ignore")
    return df.sort_values("TIMESTAMP").reset_index(drop=True)

def basic_info(df, label):
    print("=" * 60)
    print(f"BASIC INFO  [{label}]")
    print("=" * 60)
    print(f"  Rows            : {len(df):,}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"  Symbols         : {df['SYM_ROOT'].unique()}")
    print(f"  Date range      : {df['TIMESTAMP'].min()}  →  {df['TIMESTAMP'].max()}")
    print(f"  Unique dates    : {df['TIMESTAMP'].dt.date.nunique()}")
    print(df['TIMESTAMP'].dt.date.value_counts().sort_index().to_string())

def nan_report(df, label):
    print("\n" + "=" * 60)
    print(f"NaN REPORT  [{label}]")
    print("=" * 60)
    nans = df.isna().sum()
    pct  = df.isna().mean() * 100
    report = pd.DataFrame({"null_count": nans, "null_%": pct.round(3)})
    print(report.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# BID / ASK
# ══════════════════════════════════════════════════════════════════════════════
ba = load(BID_ASK_PATH)

basic_info(ba, "BID/ASK")
nan_report(ba, "BID/ASK")

print("\n" + "=" * 60)
print("QUOTE SIDE BREAKDOWN  [BID/ASK]  (raw)")
print("=" * 60)
both    = ((ba["BID"] > 0) & (ba["ASK"] > 0)).sum()
bid_only = ((ba["BID"] > 0) & (ba["ASK"] <= 0)).sum()
ask_only = ((ba["BID"] <= 0) & (ba["ASK"] > 0)).sum()
neither  = ((ba["BID"] <= 0) & (ba["ASK"] <= 0)).sum()
crossed  = ((ba["BID"] > 0) & (ba["ASK"] > 0) & (ba["BID"] >= ba["ASK"])).sum()
print(f"  Both sides valid  : {both:,}")
print(f"  Bid-only          : {bid_only:,}")
print(f"  Ask-only          : {ask_only:,}")
print(f"  Neither           : {neither:,}")
print(f"  Crossed (bid>=ask): {crossed:,}")

print("\n" + "=" * 60)
print("SPREAD DISTRIBUTION  [BID/ASK]  (valid two-sided)")
print("=" * 60)
valid = ba[(ba["BID"] > 0) & (ba["ASK"] > 0) & (ba["ASK"] > ba["BID"])].copy()
valid["mid"]        = (valid["BID"] + valid["ASK"]) / 2
valid["spread"]     = valid["ASK"] - valid["BID"]
valid["spread_pct"] = valid["spread"] / valid["mid"] * 100
print(valid["spread"].describe().rename("spread ($)").to_string())
print()
print(valid["spread_pct"].describe().rename("spread (%)").to_string())
print("\nSpread $ percentiles:")
for p in [50, 90, 95, 99, 99.9]:
    print(f"  p{p:<5}: ${valid['spread'].quantile(p/100):.4f}  ({valid['spread_pct'].quantile(p/100):.4f}%)")

print("\n" + "=" * 60)
print("PRICE & SIZE OVERVIEW  [BID/ASK]  (valid two-sided)")
print("=" * 60)
print(valid[["BID", "ASK", "mid", "BIDSIZ", "ASKSIZ"]].describe().to_string())

print("\n" + "=" * 60)
print("TICK FREQUENCY  [BID/ASK]  (valid two-sided, ms)")
print("=" * 60)
valid["dt_ms"] = valid["TIMESTAMP"].diff().dt.total_seconds() * 1000
print(valid["dt_ms"].describe().rename("inter-tick (ms)").to_string())

print("\n" + "=" * 60)
print("EXCHANGE BREAKDOWN  [BID/ASK]")
print("=" * 60)
print(ba["EX"].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# TRADES
# ══════════════════════════════════════════════════════════════════════════════
tr = load(TRADES_PATH)

basic_info(tr, "TRADES")
nan_report(tr, "TRADES")

print("\n" + "=" * 60)
print("PRICE & SIZE OVERVIEW  [TRADES]")
print("=" * 60)
print(tr[["PRICE", "SIZE"]].describe().to_string())

print("\n" + "=" * 60)
print("TRADE CONDITION (TR_SCOND) BREAKDOWN  [TRADES]")
print("=" * 60)
print(tr["TR_SCOND"].value_counts().to_string())

print("\n" + "=" * 60)
print("EXCHANGE BREAKDOWN  [TRADES]")
print("=" * 60)
print(tr["EX"].value_counts().to_string())

print("\n" + "=" * 60)
print("TICK FREQUENCY  [TRADES]  (ms)")
print("=" * 60)
tr["dt_ms"] = tr["TIMESTAMP"].diff().dt.total_seconds() * 1000
print(tr["dt_ms"].describe().rename("inter-trade (ms)").to_string())
