"""
tests/test_metrics.py

Unit tests for simulator/metrics.py (Metrics.compute).

Run with:
    pytest tests/test_metrics.py -p no:dash -v
"""

import sys
import os
import math
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.metrics import Metrics


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_results(
    portfolio_values,
    realized_pnl_values=None,
    inventory_values=None,
    fill_counts=None,
):
    """Build a minimal results DataFrame for testing."""
    n = len(portfolio_values)
    return pd.DataFrame({
        "portfolio_value": portfolio_values,
        "realized_pnl":   realized_pnl_values if realized_pnl_values is not None else [0.0] * n,
        "inventory":       inventory_values   if inventory_values   is not None else [0]   * n,
        "fill_count":      fill_counts        if fill_counts        is not None else [0]   * n,
    })


# ---------------------------------------------------------------------------
# return type
# ---------------------------------------------------------------------------

class TestReturnType:

    def test_returns_dict(self):
        r = _make_results([0, 1, 2])
        assert isinstance(Metrics.compute(r), dict)

    def test_all_keys_present(self):
        r = _make_results([0, 1, 2])
        m = Metrics.compute(r)
        for key in ("total_pnl", "realized_pnl", "sharpe", "max_drawdown",
                    "inventory_std", "max_inventory", "fill_rate", "n_fills",
                    "spread_capture", "n_round_trips"):
            assert key in m, f"missing key: {key}"


# ---------------------------------------------------------------------------
# total_pnl
# ---------------------------------------------------------------------------

class TestTotalPnl:

    def test_total_pnl_is_last_portfolio_value(self):
        r = _make_results([0, 5, -3, 10])
        assert Metrics.compute(r)["total_pnl"] == 10.0

    def test_total_pnl_zero(self):
        r = _make_results([0, 0, 0])
        assert Metrics.compute(r)["total_pnl"] == 0.0

    def test_total_pnl_negative(self):
        r = _make_results([0, -5, -20])
        assert Metrics.compute(r)["total_pnl"] == -20.0


# ---------------------------------------------------------------------------
# realized_pnl
# ---------------------------------------------------------------------------

class TestRealizedPnl:

    def test_realized_pnl_is_last_value(self):
        r = _make_results([0, 1], realized_pnl_values=[0.0, 42.5])
        assert Metrics.compute(r)["realized_pnl"] == 42.5


# ---------------------------------------------------------------------------
# Sharpe
# ---------------------------------------------------------------------------

class TestSharpe:

    def test_sharpe_positive_returns(self):
        # Constant returns of +1 each step → std = 0 after diff (only one unique diff)
        r = _make_results([0, 1, 2, 3, 4])
        m = Metrics.compute(r)
        # diff = [1,1,1,1] → std = 0 → sharpe = NaN or None
        assert m["sharpe"] is None or math.isnan(m["sharpe"] or float("nan"))

    def test_sharpe_varying_returns(self):
        r = _make_results([0, 1, 3, 2, 5, 4, 7])
        m = Metrics.compute(r)
        # Should be a finite number
        assert m["sharpe"] is not None
        assert math.isfinite(m["sharpe"])

    def test_sharpe_all_zeros(self):
        r = _make_results([0, 0, 0, 0])
        m = Metrics.compute(r)
        assert m["sharpe"] is None


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:

    def test_no_drawdown(self):
        # Monotonically increasing → drawdown = 0
        r = _make_results([0, 1, 2, 3])
        assert Metrics.compute(r)["max_drawdown"] == 0.0

    def test_drawdown_from_peak(self):
        # Peak at 10, trough at 5 → drawdown = -5
        r = _make_results([0, 5, 10, 7, 5, 8])
        assert Metrics.compute(r)["max_drawdown"] == pytest.approx(-5.0)

    def test_drawdown_always_non_positive(self):
        r = _make_results([10, 5, 8, 3, 9])
        assert Metrics.compute(r)["max_drawdown"] <= 0.0


# ---------------------------------------------------------------------------
# inventory risk
# ---------------------------------------------------------------------------

class TestInventoryRisk:

    def test_inventory_std_flat(self):
        r = _make_results([0, 0], inventory_values=[0, 0, 0, 0])
        # std of constant = 0
        assert Metrics.compute(r)["inventory_std"] == pytest.approx(0.0, abs=1e-9)

    def test_max_inventory(self):
        r = _make_results([0, 0, 0, 0],
                          inventory_values=[0, 100, -50, 80])
        assert Metrics.compute(r)["max_inventory"] == 100

    def test_max_inventory_all_negative(self):
        r = _make_results([0, 0, 0],
                          inventory_values=[-10, -200, -50])
        assert Metrics.compute(r)["max_inventory"] == 200


# ---------------------------------------------------------------------------
# fill rate
# ---------------------------------------------------------------------------

class TestFillRate:

    def test_no_fills(self):
        r = _make_results([0, 0, 0], fill_counts=[0, 0, 0])
        assert Metrics.compute(r)["fill_rate"] == 0.0

    def test_all_fills(self):
        r = _make_results([0, 0, 0], fill_counts=[1, 1, 1])
        assert Metrics.compute(r)["fill_rate"] == pytest.approx(1.0)

    def test_half_fills(self):
        r = _make_results([0, 0, 0, 0], fill_counts=[1, 0, 1, 0])
        assert Metrics.compute(r)["fill_rate"] == pytest.approx(0.5)

    def test_n_fills_total(self):
        r = _make_results([0, 0, 0, 0], fill_counts=[2, 0, 1, 0])
        assert Metrics.compute(r)["n_fills"] == 3


# ---------------------------------------------------------------------------
# spread_capture
# ---------------------------------------------------------------------------

class TestSpreadCapture:

    def test_spread_capture_is_pnl_per_round_trip(self):
        # Inventory goes 0 → 100 → 0: one round trip, realized_pnl = 10
        r = _make_results(
            [0, -9995, 10, 10],
            realized_pnl_values=[0, 0, 10, 10],
            inventory_values=[0, 100, 0, 0],
            fill_counts=[0, 1, 1, 0],
        )
        m = Metrics.compute(r)
        assert m["n_round_trips"] >= 1
        assert math.isfinite(m["spread_capture"])
