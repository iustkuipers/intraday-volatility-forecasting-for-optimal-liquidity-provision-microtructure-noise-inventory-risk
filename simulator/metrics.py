"""
simulator/metrics.py

Computes performance statistics from a simulator results DataFrame.

No plots, no file I/O — pure number computation.

Usage
-----
from simulator.metrics import Metrics

metrics = Metrics.compute(results_df)
print(metrics)

Returns
-------
dict with keys:
    total_pnl           final portfolio value (unrealized + realized)
    realized_pnl        profit locked in by closed trades
    sharpe              mean(Δportfolio) / std(Δportfolio)
    max_drawdown        largest peak-to-trough decline in portfolio value
    inventory_std       standard deviation of inventory over the run
    max_inventory       peak absolute inventory (inventory risk)
    fill_rate           fraction of events where a fill occurred
    n_fills             total number of fill events
    spread_capture      realized_pnl / n_round_trips  (avg profit per round trip)
    n_round_trips       number of completed round trips (inventory returns to 0)
"""

import math
import numpy as np
import pandas as pd


class Metrics:

    @staticmethod
    def compute(results: pd.DataFrame) -> dict:
        """
        Compute all performance metrics from a simulator results DataFrame.

        Parameters
        ----------
        results : pd.DataFrame
            Output of Simulator.run().  Must contain at minimum:
            portfolio_value, realized_pnl, inventory, fill_count.

        Returns
        -------
        dict
        """
        pv   = results["portfolio_value"]
        inv  = results["inventory"]
        rpnl = results["realized_pnl"]

        # ── PnL ──────────────────────────────────────────────────────────────
        total_pnl    = float(pv.iloc[-1])
        realized_pnl = float(rpnl.iloc[-1])

        # ── Sharpe ───────────────────────────────────────────────────────────
        # Event-level Sharpe: mean / std of portfolio value changes.
        # No annualisation here — events are irregular in time.
        returns = pv.diff().dropna()
        ret_std = float(returns.std())
        sharpe  = float(returns.mean() / ret_std) if ret_std > 0 else float("nan")

        # ── Max drawdown ──────────────────────────────────────────────────────
        running_max  = pv.cummax()
        drawdown     = pv - running_max
        max_drawdown = float(drawdown.min())

        # ── Inventory risk ────────────────────────────────────────────────────
        inventory_std = float(inv.std())
        max_inventory = int(inv.abs().max())

        # ── Fill statistics ───────────────────────────────────────────────────
        n_fills   = int(results["fill_count"].sum())
        fill_rate = n_fills / len(results) if len(results) > 0 else 0.0

        # ── Spread capture (realized PnL per round trip) ──────────────────────
        # A round trip is defined as inventory crossing back through zero.
        # Count sign changes in inventory (including flat → nonzero → flat).
        inv_vals     = inv.values
        sign_changes = int(np.sum(
            (np.sign(inv_vals[1:]) != np.sign(inv_vals[:-1])) &
            (np.sign(inv_vals[:-1]) != 0)
        ))
        # Each sign change represents completing one side of a round trip.
        # Two sign changes = one full round trip (flat→long→flat).
        n_round_trips   = max(sign_changes // 2, 1)
        spread_capture  = realized_pnl / n_round_trips

        return {
            "total_pnl":       round(total_pnl,    4),
            "realized_pnl":    round(realized_pnl, 4),
            "sharpe":          round(sharpe,        6) if not math.isnan(sharpe) else None,
            "max_drawdown":    round(max_drawdown,  4),
            "inventory_std":   round(inventory_std, 4),
            "max_inventory":   max_inventory,
            "fill_rate":       round(fill_rate,     6),
            "n_fills":         n_fills,
            "spread_capture":  round(spread_capture, 6),
            "n_round_trips":   n_round_trips,
        }
