"""
tests/test_simulator.py

Integration tests for simulator/simulator.py.

Tests verify the full event-driven cycle:
  MarketState → Strategy → ExecutionEngine → FillModel → Accounting

A minimal fake event stream is constructed in-memory so no file I/O is
required.

Run with:
    pytest tests/test_simulator.py -p no:dash -v
"""

import sys
import os
import math
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.simulator import Simulator
from simulator.market_state import MarketState
from simulator.execution_engine import ExecutionEngine
from simulator.accounting import Accounting
from simulator.strategy import ConstantSpreadStrategy
from simulator.fill_model import DeterministicFillModel


# ---------------------------------------------------------------------------
# event stream factory
# ---------------------------------------------------------------------------

_QUOTE_DEFAULTS = dict(
    event_type="quote",
    trade_price=float("nan"),
    trade_size=float("nan"),
    trade_direction=float("nan"),
    signed_volume=float("nan"),
    spread=0.10,
    rel_spread=0.0001,
    depth=200.0,
    imbalance=0.0,
    realized_vol=0.001,
    trade_intensity=0.0,
    trade_vol_intensity=0.0,
    queue_fraction_bid=1.0,
    queue_fraction_ask=1.0,
    bid_size=100.0,
    ask_size=100.0,
    sym_root="SPY",
)

_TRADE_DEFAULTS = dict(
    event_type="trade",
    spread=0.10,
    rel_spread=0.0001,
    depth=200.0,
    imbalance=0.0,
    realized_vol=0.001,
    trade_intensity=5.0,
    trade_vol_intensity=500.0,
    queue_fraction_bid=1.0,
    queue_fraction_ask=1.0,
    bid_size=100.0,
    ask_size=100.0,
    sym_root="SPY",
    signed_volume=-100.0,
)


def _events(*rows) -> pd.DataFrame:
    """
    Build a minimal events DataFrame from a list of dicts.
    Each dict overrides defaults for its event_type.
    """
    records = []
    ts = pd.Timestamp("2024-01-02 09:30:00")
    for i, row in enumerate(rows):
        base = _QUOTE_DEFAULTS.copy() if row.get("event_type", "quote") == "quote" \
               else _TRADE_DEFAULTS.copy()
        base.update(row)
        base.setdefault("timestamp", ts + pd.Timedelta(seconds=i))
        records.append(base)
    return pd.DataFrame(records)


def _make_sim(**strategy_kwargs):
    """Return a fresh Simulator with ConstantSpreadStrategy defaults."""
    kw = dict(
        base_half_spread=0.05,
        order_size=100,
        inventory_skew=0.0,
        max_inventory=10_000,
    )
    kw.update(strategy_kwargs)
    return Simulator(
        strategy   = ConstantSpreadStrategy(**kw),
        fill_model = DeterministicFillModel(),
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:

    def test_run_returns_dataframe(self):
        sim = _make_sim()
        events = _events({"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0})
        result = sim.run(events)
        assert isinstance(result, pd.DataFrame)

    def test_result_row_count_equals_event_count(self):
        sim = _make_sim()
        events = _events(
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            {"event_type": "quote", "bid": 99.94, "ask": 100.06, "mid": 100.0},
        )
        result = sim.run(events)
        assert len(result) == 2

    def test_required_columns_present(self):
        sim = _make_sim()
        events = _events({"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0})
        result = sim.run(events)
        for col in ("timestamp", "inventory", "cash", "portfolio_value", "realized_pnl"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# No-fill scenario: only quote events
# ---------------------------------------------------------------------------

class TestNoFills:

    def _run(self):
        sim = _make_sim()
        events = _events(
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            {"event_type": "quote", "bid": 99.94, "ask": 100.06, "mid": 100.0},
            {"event_type": "quote", "bid": 99.93, "ask": 100.07, "mid": 100.0},
        )
        return sim.run(events)

    def test_inventory_stays_zero(self):
        result = self._run()
        assert (result["inventory"] == 0).all()

    def test_cash_stays_zero(self):
        result = self._run()
        assert (result["cash"] == 0.0).all()

    def test_no_fills_recorded(self):
        result = self._run()
        assert (result["fill_count"] == 0).all()

    def test_bid_quote_posted(self):
        result = self._run()
        assert result["bid_quote"].notna().all()

    def test_ask_quote_posted(self):
        result = self._run()
        assert result["ask_quote"].notna().all()


# ---------------------------------------------------------------------------
# Single fill: trade hits bid
# ---------------------------------------------------------------------------

class TestSingleBidFill:

    def _run(self):
        sim = _make_sim(base_half_spread=0.05, order_size=100)
        # Quote sets bid at mid-0.05 = 99.95, ask at mid+0.05 = 100.05
        # Trade at 99.95 (= bid), best_bid = 99.95 → fill
        events = _events(
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            {
                "event_type": "trade",
                "bid": 99.95, "ask": 100.05, "mid": 100.0,
                "trade_price": 99.95, "trade_size": 100,
                "trade_direction": -1,
            },
        )
        return sim.run(events)

    def test_inventory_increases_on_bid_fill(self):
        result = self._run()
        assert result["inventory"].iloc[-1] == 100

    def test_cash_decreases_on_bid_fill(self):
        result = self._run()
        assert result["cash"].iloc[-1] == pytest.approx(-9995.0)

    def test_fill_recorded(self):
        result = self._run()
        assert result["fill_count"].iloc[-1] == 1

    def test_fill_side_is_bid(self):
        result = self._run()
        assert result["fill_side"].iloc[-1] == "bid"

    def test_no_fill_on_quote_row(self):
        result = self._run()
        assert result["fill_count"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Single fill: trade hits ask
# ---------------------------------------------------------------------------

class TestSingleAskFill:

    def _run(self):
        sim = _make_sim(base_half_spread=0.05, order_size=100)
        events = _events(
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            {
                "event_type": "trade",
                "bid": 99.95, "ask": 100.05, "mid": 100.0,
                "trade_price": 100.05, "trade_size": 100,
                "trade_direction": 1,
            },
        )
        return sim.run(events)

    def test_inventory_decreases(self):
        result = self._run()
        assert result["inventory"].iloc[-1] == -100

    def test_cash_increases(self):
        result = self._run()
        assert result["cash"].iloc[-1] == pytest.approx(10005.0)

    def test_fill_side_is_ask(self):
        result = self._run()
        assert result["fill_side"].iloc[-1] == "ask"


# ---------------------------------------------------------------------------
# Round-trip: buy then sell → realized PnL
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def _run(self):
        sim = _make_sim(base_half_spread=0.05, order_size=100)
        events = _events(
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            # sell trade hits our bid → we buy 100 @ 99.95
            {
                "event_type": "trade",
                "bid": 99.95, "ask": 100.05, "mid": 100.0,
                "trade_price": 99.95, "trade_size": 100,
                "trade_direction": -1,
            },
            # quote update (mid unchanged)
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            # buy trade lifts our ask → we sell 100 @ 100.05
            {
                "event_type": "trade",
                "bid": 99.95, "ask": 100.05, "mid": 100.0,
                "trade_price": 100.05, "trade_size": 100,
                "trade_direction": 1,
            },
        )
        return sim.run(events)

    def test_inventory_flat_after_round_trip(self):
        result = self._run()
        assert result["inventory"].iloc[-1] == 0

    def test_cash_equals_spread(self):
        result = self._run()
        # cash = -9995 + 10005 = +10
        assert result["cash"].iloc[-1] == pytest.approx(10.0)

    def test_realized_pnl_equals_spread(self):
        result = self._run()
        # realized = (100.05 - 99.95) * 100 = 10
        assert result["realized_pnl"].iloc[-1] == pytest.approx(10.0)

    def test_portfolio_value_equals_cash_when_flat(self):
        result = self._run()
        last = result.iloc[-1]
        assert last["portfolio_value"] == pytest.approx(last["cash"])


# ---------------------------------------------------------------------------
# Inventory cap disables side
# ---------------------------------------------------------------------------

class TestInventoryCap:

    def test_bid_disabled_at_max_inventory(self):
        """At max_inventory, strategy disables bid → no bid fills possible."""
        sim = _make_sim(
            base_half_spread=0.05,
            order_size=100,
            inventory_skew=0.0,
            max_inventory=100,
        )
        events = _events(
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            # first trade: fills bid, inventory = 100 = max
            {
                "event_type": "trade",
                "bid": 99.95, "ask": 100.05, "mid": 100.0,
                "trade_price": 99.95, "trade_size": 100,
                "trade_direction": -1,
            },
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            # second sell trade: bid should now be disabled → no fill
            {
                "event_type": "trade",
                "bid": 99.95, "ask": 100.05, "mid": 100.0,
                "trade_price": 99.95, "trade_size": 100,
                "trade_direction": -1,
            },
        )
        result = sim.run(events)
        # inventory must not exceed max_inventory
        assert result["inventory"].iloc[-1] == 100

    def test_bid_quote_is_none_when_cap_reached(self):
        sim = _make_sim(
            base_half_spread=0.05,
            order_size=100,
            inventory_skew=0.0,
            max_inventory=100,
        )
        events = _events(
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            {
                "event_type": "trade",
                "bid": 99.95, "ask": 100.05, "mid": 100.0,
                "trade_price": 99.95, "trade_size": 100,
                "trade_direction": -1,
            },
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
        )
        result = sim.run(events)
        assert pd.isna(result["bid_quote"].iloc[-1])


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------

class TestMarkToMarket:

    def test_portfolio_value_after_buy_and_price_move(self):
        sim = _make_sim(base_half_spread=0.05, order_size=100, inventory_skew=0.0)
        events = _events(
            {"event_type": "quote", "bid": 99.95, "ask": 100.05, "mid": 100.0},
            {
                "event_type": "trade",
                "bid": 99.95, "ask": 100.05, "mid": 100.0,
                "trade_price": 99.95, "trade_size": 100,
                "trade_direction": -1,
            },
            # mid moves up to 101 — portfolio value should improve
            {"event_type": "quote", "bid": 100.95, "ask": 101.05, "mid": 101.0},
        )
        result = sim.run(events)
        # cash = -9995, inventory = 100, mid = 101 → portfolio = -9995 + 100*101 = 105
        assert result["portfolio_value"].iloc[-1] == pytest.approx(105.0)
