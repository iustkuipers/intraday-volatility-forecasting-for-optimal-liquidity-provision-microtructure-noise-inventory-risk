"""
Strategy comparison: Baseline vs Vol-adaptive vs Vol+Inventory

Runs 3 strategies on the same half-day session with PROBABILISTIC fills
(fill_model.fill_probability) and a fixed random seed for reproducibility.

Strategies
----------
1. Baseline       : constant delta=0.03,       phi=0.0
2. Vol-adaptive   : delta = EWMA vol series,   phi=0.0
3. Vol+Inv        : delta = EWMA vol series,   phi=0.001

Each strategy gets a fresh engine seeded identically so random draws
are independent but reproducible across runs.

Metrics compared
----------------
Trades | Inv variance | Max |Inv| | Total PnL | PnL std | Ann. Sharpe
"""

import sys
import os
import shutil
import pandas as pd
import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── purge stale bytecode so updated engine.py is always used ─────────────────
_cache = os.path.join(ROOT, "src", "simulator", "__pycache__")
if os.path.isdir(_cache):
    shutil.rmtree(_cache)

from src.data_loader import load_data, compute_realized_vol
from src.volatility.ewma import ewma_volatility_forecast
from src.simulator.engine import MarketMakerEngine

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join(ROOT, "data", "t822bpd5q8g1deky.csv")
BASELINE_DELTA = 0.03
EWMA_LAM       = 0.94
PHI_INVENTORY  = 0.001   # frozen inventory-skew coefficient
ALPHA_AS_SWEEP = [0.003, 0.001]   # alpha values to sweep (reduced from 0.02)
SEED           = 42      # fixed seed → reproducible probabilistic fills

# ── load & prepare data ───────────────────────────────────────────────────────
print("Loading data …")
df_tick = load_data(DATA_PATH)
print(f"  Tick rows : {len(df_tick):,}")

df_1min = compute_realized_vol(df_tick, minute=1)
print(f"  1-min bars: {len(df_1min):,}")
print(f"  Window    : {df_1min.index[0]}  →  {df_1min.index[-1]}")

# ── EWMA vol forecast (aligned to 1-min bars) ─────────────────────────────────
rv_series    = df_1min["realized_vol"] ** 2          # realized variance
ewma_vol     = ewma_volatility_forecast(rv_series, lam=EWMA_LAM)
delta_series = ewma_vol.reindex(df_1min.index)       # same index as df_1min

print(f"\n  EWMA vol  : mean={delta_series.mean():.6f}  "
      f"min={delta_series.min():.6f}  max={delta_series.max():.6f}")

# ── metrics helper ────────────────────────────────────────────────────────────
def compute_metrics(res: pd.DataFrame) -> dict:
    pv        = res["portfolio_value"]
    inv       = res["inventory"]
    pnl_steps = pv.diff().dropna()
    total_pnl = pv.iloc[-1] - pv.iloc[0]
    pnl_std   = pnl_steps.std()
    sharpe    = (pnl_steps.mean() / pnl_std * np.sqrt(252 * 390)
                 if pnl_std > 0 else np.nan)
    return {
        "Trades":        int(res["trade_count"].iloc[-1]),
        "Inv variance":  round(float(inv.var()), 4),
        "Max |Inv|":     int(inv.abs().max()),
        "Mean |Inv|":    round(float(inv.abs().mean()), 2),
        "Total PnL ($)": round(float(total_pnl), 4),
        "PnL std":       round(float(pnl_std), 6),
        "Ann. Sharpe":   round(float(sharpe), 2),
    }


# ── alpha sweep ───────────────────────────────────────────────────────────────
pct = lambda a, ref: f"{(a-ref)/abs(ref)*100:+.1f}%" if ref != 0 else "n/a"

for ALPHA_AS in ALPHA_AS_SWEEP:

    mean_penalty = ALPHA_AS * delta_series.mean() * 470
    print(f"\n{'#'*75}")
    print(f"  α_as = {ALPHA_AS}  |  est. penalty at mean vol ≈ ${mean_penalty:.4f}/fill")
    print(f"{'#'*75}")

    strategies = {
        "Baseline (δ=0.03, φ=0)":      dict(delta=BASELINE_DELTA, phi=0.0,
                                             volatility_series=ewma_vol, alpha_as=ALPHA_AS),
        "Vol-adaptive (φ=0)":           dict(delta=delta_series,    phi=0.0,
                                             volatility_series=ewma_vol, alpha_as=ALPHA_AS),
        f"Vol+Inv (φ={PHI_INVENTORY})": dict(delta=delta_series,    phi=PHI_INVENTORY,
                                             volatility_series=ewma_vol, alpha_as=ALPHA_AS),
    }

    results = {}
    for name, params in strategies.items():
        engine = MarketMakerEngine(initial_cash=0.0, random_state=SEED)
        results[name] = engine.run(df_1min, **params)

    rows    = {name: compute_metrics(res) for name, res in results.items()}
    summary = pd.DataFrame(rows).T

    print("\n" + "=" * 75)
    print(f"STRATEGY COMPARISON  |  seed={SEED}  |  α_as={ALPHA_AS}")
    print("=" * 75)
    print(summary.to_string())
    print("=" * 75)

    b  = rows["Baseline (δ=0.03, φ=0)"]
    va = rows["Vol-adaptive (φ=0)"]
    vi = rows[f"Vol+Inv (φ={PHI_INVENTORY})"]

    print("\nKey insights vs baseline:")
    print(f"  Trades  : baseline={b['Trades']}  vol-adapt={va['Trades']} ({pct(va['Trades'],b['Trades'])})  vol+inv={vi['Trades']} ({pct(vi['Trades'],b['Trades'])})")
    print(f"  Inv var : baseline={b['Inv variance']}  vol-adapt={va['Inv variance']} ({pct(va['Inv variance'],b['Inv variance'])})  vol+inv={vi['Inv variance']} ({pct(vi['Inv variance'],b['Inv variance'])})")
    print(f"  Max|Inv|: baseline={b['Max |Inv|']}  vol-adapt={va['Max |Inv|']}  vol+inv={vi['Max |Inv|']}")
    print(f"  PnL std : baseline={b['PnL std']}  vol-adapt={va['PnL std']} ({pct(va['PnL std'],b['PnL std'])})  vol+inv={vi['PnL std']} ({pct(vi['PnL std'],b['PnL std'])})")
    print(f"  Tot PnL : baseline={b['Total PnL ($)']}  vol-adapt={va['Total PnL ($)']} ({pct(va['Total PnL ($)'],b['Total PnL ($)'])})  vol+inv={vi['Total PnL ($)']} ({pct(vi['Total PnL ($)'],b['Total PnL ($)'])})")
    print(f"  Sharpe  : baseline={b['Ann. Sharpe']}  vol-adapt={va['Ann. Sharpe']}  vol+inv={vi['Ann. Sharpe']}")
