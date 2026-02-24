import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime

from src.data_loader import load_data, compute_realized_vol
from src.volatility.realized_vol import realized_variance_from_returns
from src.volatility.ewma import ewma_variance_forecast
from src.simulator.engine import MarketMakerEngine
from src.market_making.quoting import compute_spread


# ─────────────────────────────────────────────
# HEARTBEAT — prints every 60s in background
# ─────────────────────────────────────────────
_start_time = None
_heartbeat_stop = threading.Event()

def _heartbeat():
    while not _heartbeat_stop.wait(60):
        elapsed = int(time.time() - _start_time)
        mins, secs = elapsed // 60, elapsed % 60
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now} | {mins}m {secs}s] ⏳ Still running...", flush=True)

def start_heartbeat():
    global _start_time
    _start_time = time.time()
    _heartbeat_stop.clear()
    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()

def stop_heartbeat():
    _heartbeat_stop.set()

def log(msg: str):
    elapsed = int(time.time() - _start_time)
    mins, secs = elapsed // 60, elapsed % 60
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now} | {mins}m {secs}s] {msg}", flush=True)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FILEPATH       = "data/t822bpd5q8g1deky.csv"
DELTA          = 0.03   # baseline constant half-spread
EWMA_LAM       = 0.94   # EWMA decay parameter
K0             = 0.01   # vol-adaptive intercept
K1             = 1.0    # vol-adaptive slope (spread = k0 + k1 * vol)
PHI_INVENTORY  = 0.001  # inventory-skew coefficient
ALPHA_AS       = 0.02   # adverse-selection cost coefficient (α·vol·mid per fill)
seed           = 42     # fixed seed → reproducible probabilistic fills

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(result: pd.DataFrame) -> dict:
    """Compute basic backtest performance metrics."""

    pnl = result["portfolio_value"].diff().dropna()

    total_pnl       = result["portfolio_value"].iloc[-1] - result["portfolio_value"].iloc[0]
    mean_pnl        = pnl.mean()
    std_pnl         = pnl.std()
    sharpe          = mean_pnl / std_pnl if std_pnl != 0 else 0.0
    inventory_var   = result["inventory"].var()
    max_inventory   = result["inventory"].abs().max()
    n_trades        = result["trade_count"].iloc[-1] if "trade_count" in result.columns else int((result["inventory"].diff().abs() > 0).sum())

    return {
        "total_pnl":          round(total_pnl, 4),
        "mean_pnl_per_bar":   round(mean_pnl, 6),
        "std_pnl_per_bar":    round(std_pnl, 6),
        "sharpe_ratio":       round(sharpe, 4),
        "inventory_variance": round(inventory_var, 4),
        "max_abs_inventory":  int(max_inventory),
        "n_trades":           int(n_trades),
    }


def print_comparison(strategies: dict[str, dict]):
    """Print side-by-side metric comparison for any number of strategies."""
    keys  = list(next(iter(strategies.values())).keys())
    names = list(strategies.keys())
    col_w = 18

    header = f"{'METRIC':<25}" + "".join(f"{n:>{col_w}}" for n in names)
    print("\n" + "=" * (25 + col_w * len(names)))
    print(header)
    print("=" * (25 + col_w * len(names)))
    for k in keys:
        row = f"  {k:<23}"
        base_val = list(strategies.values())[0][k]
        for i, (name, metrics) in enumerate(strategies.items()):
            v = metrics[k]
            row += f"{str(v):>{col_w}}"
            if i > 0 and isinstance(base_val, float) and base_val != 0:
                diff_pct = (v - base_val) / abs(base_val) * 100
                row += f" ({'+' if diff_pct >= 0 else ''}{diff_pct:.1f}%)"
        print(row)
    print("=" * (25 + col_w * len(names)))


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    start_heartbeat()

    try:
        # ── Step 1: Load and clean tick data ──────
        log("STEP 1 — Loading data...")
        df_tick = load_data(FILEPATH)
        log(f"✅ Loaded {len(df_tick):,} tick rows | {df_tick.index.min()} → {df_tick.index.max()}")

        # ── Step 2: Resample to 1-minute bars ─────
        log("STEP 2 — Resampling to 1-min realized vol bars...")
        df_1min = compute_realized_vol(df_tick, minute=1)
        log(f"✅ {len(df_1min):,} 1-min bars")
        print(df_1min.head())

        # ── Step 3: Realized variance from tick returns ──
        log("STEP 3 — Computing realized variance...")
        rvar = realized_variance_from_returns(df_tick["log_return"], interval="1min")
        log(f"✅ RVar bars: {len(rvar):,} | Mean RVar: {rvar.mean():.8f} | Mean RVol: {np.sqrt(rvar.mean()):.6f}")

        # ── Step 4: EWMA variance forecast ────────
        log(f"STEP 4 — Computing EWMA forecast (λ={EWMA_LAM})...")
        ewma_var = ewma_variance_forecast(rvar, lam=EWMA_LAM)
        ewma_vol = np.sqrt(ewma_var)
        log(f"✅ EWMA vol — mean: {ewma_vol.mean():.6f}, max: {ewma_vol.max():.6f}")

        # Align EWMA vol to 1-min bar index (used by all strategies)
        ewma_vol_aligned = ewma_vol.reindex(df_1min.index, method="ffill").fillna(ewma_vol.mean())

        # ── Step 5a: Baseline — constant spread ──────────────────
        log(f"STEP 5a — Baseline: constant δ={DELTA}, α_as={ALPHA_AS}...")
        mm = MarketMakerEngine(random_state=seed)
        result_baseline = mm.run(
            df_1min, delta=DELTA,
            volatility_series=ewma_vol_aligned, alpha_as=ALPHA_AS,
        )
        log(f"✅ Baseline complete: {len(result_baseline):,} bars, {result_baseline['trade_count'].iloc[-1]} trades")

        # ── Step 5b: Vol-adaptive spread ─────────────────────────
        log(f"STEP 5b — Vol-adaptive: δ = {K0} + {K1} * EWMA_vol, α_as={ALPHA_AS}...")
        delta_series = compute_spread(ewma_vol_aligned, k0=K0, k1=K1, min_spread=0.005)
        log(f"   δ stats — mean: {delta_series.mean():.4f}, min: {delta_series.min():.4f}, max: {delta_series.max():.4f}")
        mm = MarketMakerEngine(random_state=seed)
        result_voladapt = mm.run(
            df_1min, delta=delta_series,
            volatility_series=ewma_vol_aligned, alpha_as=ALPHA_AS,
        )
        log(f"✅ Vol-adaptive complete: {len(result_voladapt):,} bars, {result_voladapt['trade_count'].iloc[-1]} trades")

        # ── Step 5c: Vol-adaptive + Inventory skew ────────────────
        log(f"STEP 5c — Vol+Inv: δ = vol-adaptive, φ={PHI_INVENTORY}, α_as={ALPHA_AS}...")
        mm = MarketMakerEngine(random_state=seed)
        result_volinv = mm.run(
            df_1min, delta=delta_series, phi=PHI_INVENTORY,
            volatility_series=ewma_vol_aligned, alpha_as=ALPHA_AS,
        )
        log(f"✅ Vol+Inv complete: {len(result_volinv):,} bars, {result_volinv['trade_count'].iloc[-1]} trades")

        # ── Step 6: Metrics ───────────────────────
        log("STEP 6 — Computing performance metrics...")
        metrics_baseline  = compute_metrics(result_baseline)
        metrics_voladapt  = compute_metrics(result_voladapt)
        metrics_volinv    = compute_metrics(result_volinv)

        print_comparison({
            f"Baseline (δ={DELTA})":      metrics_baseline,
            "Vol-adaptive":               metrics_voladapt,
            f"Vol+Inv (φ={PHI_INVENTORY})": metrics_volinv,
        })

        log("✅ Pipeline complete")

    finally:
        stop_heartbeat()

    return result_baseline, result_voladapt, result_volinv, metrics_baseline, metrics_voladapt, metrics_volinv


if __name__ == "__main__":
    main()
