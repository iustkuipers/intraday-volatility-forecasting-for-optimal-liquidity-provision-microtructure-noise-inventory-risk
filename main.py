"""
main.py — SPY market-making simulator.
Run from project root: python main.py
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

from simulator.simulator import Simulator
from simulator.strategy import ConstantSpreadStrategy
from simulator.fill_model import DeterministicFillModel
from simulator.metrics import Metrics
from simulator.output import OutputManager

# ── Config ────────────────────────────────────────────────────────────────────
# SPY mid ≈ $472, NBBO half-spread ≈ $0.005
# spread_frac = 0.000011  →  half-spread ≈ $0.0052  (NBBO touch)
# max_position_value = 500_000  →  max_inv ≈ 1059 shares (~10 round-lots)
# risk_aversion = 0.00005  →  skew ≈ 45% of half-spread at 100 shares

EVENTS_PATH = "data/merged/events.parquet"

CONFIG = dict(
    spread_frac        = 0.000011,
    order_size         = 100,
    risk_aversion      = 0.00005,
    max_position_value = 500_000,
    vol_cap            = 0.005,
)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading events ...")
events_full  = pd.read_parquet(EVENTS_PATH)
t_start      = events_full["timestamp"].iloc[0]
t_end        = t_start + pd.Timedelta(minutes=5)
events       = events_full[events_full["timestamp"] < t_end].copy()
print(f"Window : {t_start}  →  {t_end}  ({len(events):,} rows)")

# ── Run ───────────────────────────────────────────────────────────────────────
sim = Simulator(
    strategy   = ConstantSpreadStrategy(**CONFIG),
    fill_model = DeterministicFillModel(),
)
results = sim.run(events)

# ── Summary ───────────────────────────────────────────────────────────────────
last      = results.iloc[-1]
n_fills   = int(results["fill_count"].sum())
bid_fills = (results["fill_side"] == "bid").sum()
ask_fills = (results["fill_side"] == "ask").sum()

print(f"\n{'='*55}")
print(f"  Events          : {len(results):,}")
print(f"  Total fills     : {n_fills}  (bid {bid_fills} / ask {ask_fills})")
print(f"  Fill rate       : {n_fills / len(results) * 100:.4f}%")
print(f"  Final inventory : {last['inventory']}")
print(f"  Realized PnL    : {last['realized_pnl']:.4f}")
print(f"  Portfolio value : {last['portfolio_value']:.4f}")
print(f"{'='*55}")

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics = Metrics.compute(results)
print("\nMetrics:")
for k, v in metrics.items():
    print(f"  {k:<20} {v}")

# ── Adverse selection ─────────────────────────────────────────────────────────
fill_rows = results[results["fill_count"] > 0].reset_index(drop=True)
if len(fill_rows) >= 2:
    mids  = results["mid"].to_numpy(dtype=float)
    ts_ns = pd.to_datetime(results["timestamp"]).astype("int64").to_numpy()
    idxs, prices, signs = [], [], []
    for i, row in results[results["fill_count"] > 0].iterrows():
        if row["fill_side"] == "bid":
            idxs.append(i); prices.append(float(row["bid_quote"])); signs.append(+1)
        elif row["fill_side"] == "ask":
            idxs.append(i); prices.append(float(row["ask_quote"])); signs.append(-1)
    if idxs:
        idxs   = np.array(idxs);   prices = np.array(prices); signs = np.array(signs)
        j_1s   = np.searchsorted(ts_ns, ts_ns[idxs] + int(1e9)).clip(0, len(mids)-1)
        moves  = signs * (mids[j_1s] - prices)
        print(f"\n  Adverse selection (1s lag):")
        print(f"  Mean signed move : {moves.mean():>+.4f}")
        print(f"  % adverse        : {(moves < 0).mean()*100:.1f}%")

# ── Quote sanity ──────────────────────────────────────────────────────────────
bad_bid = (results["bid_quote"] > results["mid"]).mean() * 100
bad_ask = (results["ask_quote"] < results["mid"]).mean() * 100
print(f"\n  bid_quote > mid  : {bad_bid:.2f}%  (must be 0%)")
print(f"  ask_quote < mid  : {bad_ask:.2f}%  (must be 0%)")

# ── Save ──────────────────────────────────────────────────────────────────────
run_path = OutputManager.save_run(results=results, metrics=metrics, label="SPY_NBBOtouch")
print(f"\nSaved → {run_path}")
