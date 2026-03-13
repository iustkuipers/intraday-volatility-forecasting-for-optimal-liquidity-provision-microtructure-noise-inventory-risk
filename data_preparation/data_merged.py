"""
data_merged.py  —  Build a unified market event stream (quotes + trades).

Architecture:
    1. Load clean quotes and trades
    2. Enrich trades with prevailing quote state (merge_asof backward)
    3. Compute trade features (Lee-Ready direction, signed volume)
    4. Cast quotes and enriched trades to a shared schema with event_type flag
    5. Concatenate → sort → compute stream-level features (imbalance, returns, vol)
    6. Save

The simulator steps through every row in timestamp order seeing both quote
updates and trade prints — capturing spread dynamics, liquidity changes, and
pre-trade signals that a trade-only dataset would miss.

Saves:
    ../data/merged/events.parquet       (full day,  ~22M rows)
    ../data/merged/events_5min.parquet  (09:30 – 09:35)
"""

import pandas as pd
import numpy as np

QUOTES_PATH  = "data/clean/quotes.parquet"
TRADES_PATH  = "data/clean/trades.parquet"
OUT_FULL     = "data/merged/events.parquet"
OUT_5MIN     = "data/merged/events_5min.parquet"

VOL_WINDOW = 20   # rolling window for realized vol (events, not just trades)

# Any quote row whose spread exceeds this multiple of the rolling median is a
# residual NBBO spike (stale / erroneous exchange snapshot that survived the
# data_load_bid_ask filter).  We forward-fill from the last clean state so the
# simulator never sees a momentarily collapsed best_bid that would trigger a
# phantom fill on an out-of-book quote.
SPREAD_HEAL_MULT = 10    # spike threshold = 10 × rolling-median spread

# ── 1. Load ───────────────────────────────────────────────────────────────────
quotes = pd.read_parquet(QUOTES_PATH)
trades = pd.read_parquet(TRADES_PATH)

print(f"Quotes rows : {len(quotes):,}")
print(f"Trades rows : {len(trades):,}")

quotes = quotes.sort_values("timestamp").reset_index(drop=True)
trades = trades.sort_values("timestamp").reset_index(drop=True)

# ── 2. Attach prevailing quote state to each trade ────────────────────────────
trades_enriched = pd.merge_asof(
    trades,
    quotes,
    on="timestamp",
    direction="backward",
    suffixes=("_trade", "_quote"),
)
# Drop the tiny set of trades with no prior quote (very start of session)
trades_enriched = trades_enriched.dropna(subset=["bid", "ask", "mid"]).reset_index(drop=True)
print(f"Trades after quote-match: {len(trades_enriched):,}")

# ── 3. Trade features ─────────────────────────────────────────────────────────
trades_enriched["trade_direction"] = np.where(
    trades_enriched["trade_price"] > trades_enriched["mid"],  1,
    np.where(trades_enriched["trade_price"] < trades_enriched["mid"], -1, 0)
)
trades_enriched["signed_volume"] = (
    trades_enriched["trade_direction"] * trades_enriched["trade_size"]
)

# ── 4. Build quote event rows ─────────────────────────────────────────────────
quotes_events = quotes.copy()
quotes_events["event_type"]      = "quote"
quotes_events["trade_price"]     = np.nan
quotes_events["trade_size"]      = np.nan
quotes_events["trade_direction"] = 0
quotes_events["signed_volume"]   = 0

# ── 5. Build trade event rows ─────────────────────────────────────────────────
# Consolidate ex/sym_root: trade rows carry ex_trade/ex_quote from the merge.
# Unify into single ex / sym_root columns, then drop the redundant suffixed ones.
trade_events = trades_enriched.copy()
trade_events["event_type"] = "trade"
trade_events["ex"]       = trade_events["ex_trade"]
trade_events["sym_root"] = trade_events["sym_root_trade"]
trade_events = trade_events.drop(columns=["ex_trade", "sym_root_trade",
                                           "ex_quote", "sym_root_quote"], errors="ignore")

# ── 6. Combine into event stream ──────────────────────────────────────────────
events = pd.concat([quotes_events, trade_events], ignore_index=True)
events = events.sort_values("timestamp").reset_index(drop=True)
print(f"Event stream rows   : {len(events):,}")

# ── 7. Stream-level features ──────────────────────────────────────────────────
events["imbalance"] = (
    (events["bid_size"] - events["ask_size"]) /
    (events["bid_size"] + events["ask_size"])
)

# Log return — only non-zero when mid actually moves to avoid inflating vol
# with quote spam that leaves price unchanged.
events["log_return"] = np.where(
    events["mid"].diff() != 0,
    np.log(events["mid"]).diff(),
    0
)

events["realized_vol"] = (
    events["log_return"]
    .rolling(VOL_WINDOW, min_periods=2)
    .std()
)

# ── 7b. Heal residual NBBO spike rows ───────────────────────────────────────
# Compute a trailing median spread (100-event window) and mark rows where
# spread > SPREAD_HEAL_MULT × local median as spikes.  Replace their bid/ask/mid
# with a forward-fill of the last non-spike state.
_rolling_med = events["spread"].rolling(100, min_periods=5).median().bfill()
_spike_mask  = events["spread"] > SPREAD_HEAL_MULT * _rolling_med
_n_spikes    = int(_spike_mask.sum())
if _n_spikes > 0:
    _heal_cols = ["bid", "ask", "mid", "spread"]
    events.loc[_spike_mask, _heal_cols] = float("nan")
    events[_heal_cols] = events[_heal_cols].ffill()
    print(f"Healed NBBO spikes  : {_n_spikes:,} rows replaced by forward-fill")

# Relative spread — normalises for price level (0.01 at $10 ≠ 0.01 at $500)
events["rel_spread"] = events["spread"] / events["mid"]

# Depth — total visible liquidity at best bid/ask
events["depth"] = events["bid_size"] + events["ask_size"]

# Queue fraction — approximates how far back in queue a 1-lot (100 share) order
# would sit. Without L2 data we can't know exact position; this gives the
# fraction of the visible queue that a standard order would represent.
# Used by the simulator to scale fill probability.
events["queue_fraction_bid"] = 100 / events["bid_size"].replace(0, np.nan)
events["queue_fraction_ask"] = 100 / events["ask_size"].replace(0, np.nan)

# Trade intensity — rolling count and rolling volume over last 200 events.
# Both are fill-probability signals: high count = active tape, better fills;
# high volume = large players moving, adverse selection risk rises.
events["trade_flag"]      = (events["event_type"] == "trade").astype(int)
events["trade_volume"]    = events["trade_size"].fillna(0)
events["trade_intensity"] = events["trade_flag"].rolling(200, min_periods=1).sum()
events["trade_vol_intensity"] = events["trade_volume"].rolling(200, min_periods=1).sum()

# ── 8. Save ───────────────────────────────────────────────────────────────────
import os
os.makedirs("data/merged", exist_ok=True)

events.to_parquet(OUT_FULL, index=False)
print(f"\nSaved full  : {OUT_FULL}  ({len(events):,} rows)")

window_end = events["timestamp"].iloc[0].normalize() + pd.Timedelta(hours=9, minutes=35)
events_5min = events[events["timestamp"] < window_end].copy()
events_5min.to_parquet(OUT_5MIN, index=False)
print(f"Saved 5-min : {OUT_5MIN}  ({len(events_5min):,} rows)")

# ── 9. Validation ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

# Event type counts
counts = events["event_type"].value_counts()
print(f"  quote events : {counts.get('quote', 0):,}")
print(f"  trade events : {counts.get('trade', 0):,}")
print(f"  total        : {len(events):,}")

# Trade rows: trade_price / direction should never be NaN / 0
tr = events[events["event_type"] == "trade"]
print(f"\n  Trades with NaN trade_price     : {tr['trade_price'].isna().sum():,}  (expect 0)")
print(f"  Trades with NaN bid             : {tr['bid'].isna().sum():,}           (expect 0)")
print(f"  Trades with direction=0 (at mid): {(tr['trade_direction'] == 0).sum():,}")
print(f"  Trade direction breakdown       : { tr['trade_direction'].value_counts().to_dict() }")

# Quote rows: trade columns should all be NaN / 0
qt = events[events["event_type"] == "quote"]
print(f"\n  Quotes with non-NaN trade_price : {qt['trade_price'].notna().sum():,}  (expect 0)")
print(f"  Quotes with NaN bid             : {qt['bid'].isna().sum():,}           (expect 0)")

# Imbalance range
print(f"\n  Imbalance range : [{events['imbalance'].min():.3f}, {events['imbalance'].max():.3f}]  (expect [-1, 1])")

# NaN summary for key columns
nan_cols = ["bid", "ask", "mid", "spread", "imbalance", "log_return",
            "realized_vol", "rel_spread", "depth",
            "queue_fraction_bid", "queue_fraction_ask",
            "trade_intensity", "trade_vol_intensity"]
print("\n  NaN counts (key columns):")
for col in nan_cols:
    print(f"    {col:<15}: {events[col].isna().sum():,}")

print("\n  Columns:", list(events.columns))
