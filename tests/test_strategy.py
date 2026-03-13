"""
tests/test_strategy.py

Unit tests for ConstantSpreadStrategy and VolatilityAdaptiveStrategy.

Run with:
    pytest tests/test_strategy.py -p no:dash -v
"""

import sys
import os
import pytest
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.strategy import ConstantSpreadStrategy, VolatilityAdaptiveStrategy
from simulator.order import Quote


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_state(mid=100.0, realized_vol=0.001):
    """Minimal mock of MarketState with only the attributes strategies read."""
    State = namedtuple("State", ["mid", "realized_vol"])
    return State(mid=mid, realized_vol=realized_vol)


# ---------------------------------------------------------------------------
# ConstantSpreadStrategy
# ---------------------------------------------------------------------------

class TestConstantSpreadStrategy:
    # mid=100, vol=0.001
    # half_spread = spread_frac * mid = 0.0005 * 100 = 0.05
    # skew        = risk_aversion * vol * mid * inv = 0.1 * 0.001 * 100 * inv = 0.01 * inv
    # max_inv     = max_position_value / mid = 50_000 / 100 = 500

    def _default(self, **kw):
        defaults = dict(
            spread_frac=0.0005,          # 0.05 at mid=100
            order_size=100,
            risk_aversion=0.1,           # 0.01/share skew at mid=100, vol=0.001
            max_position_value=50_000,   # 500 shares at mid=100
        )
        defaults.update(kw)
        return ConstantSpreadStrategy(**defaults)

    # --- basic quote shape ------------------------------------------------

    def test_bid_below_ask(self):
        strat = self._default()
        q = strat.compute_quote(_make_state(), inventory=0)
        assert q.bid_price < q.ask_price

    def test_symmetric_around_mid_zero_inventory(self):
        # spread_frac=0.0005, mid=100 → half_spread=0.05
        strat = self._default(spread_frac=0.0005, risk_aversion=0.1)
        state = _make_state(mid=100.0, realized_vol=0.001)
        q = strat.compute_quote(state, inventory=0)
        assert abs(q.bid_price - (100.0 - 0.05)) < 1e-9
        assert abs(q.ask_price - (100.0 + 0.05)) < 1e-9

    def test_spread_equals_twice_half_spread(self):
        # spread_frac=0.0003, mid=100 → half_spread=0.03, spread=0.06
        strat = self._default(spread_frac=0.0003)
        q = strat.compute_quote(_make_state(), inventory=0)
        assert abs((q.ask_price - q.bid_price) - 2 * 0.03) < 1e-9

    def test_spread_scales_with_mid_price(self):
        # same spread_frac at different mid prices → spread proportional to mid
        strat = self._default(spread_frac=0.0005)
        q_low  = strat.compute_quote(_make_state(mid=100.0), inventory=0)
        q_high = strat.compute_quote(_make_state(mid=500.0), inventory=0)
        spread_low  = q_low.ask_price  - q_low.bid_price
        spread_high = q_high.ask_price - q_high.bid_price
        assert abs(spread_high / spread_low - 5.0) < 1e-6

    def test_quote_sizes_correct(self):
        strat = self._default(order_size=200)
        q = strat.compute_quote(_make_state(), inventory=0)
        assert q.bid_size == 200
        assert q.ask_size == 200

    def test_both_sides_active_at_zero_inventory(self):
        strat = self._default()
        q = strat.compute_quote(_make_state(), inventory=0)
        assert q.bid_active
        assert q.ask_active

    # --- inventory skew ---------------------------------------------------

    def test_positive_inventory_shifts_quotes_down(self):
        strat = self._default()
        state = _make_state(mid=100.0, realized_vol=0.001)
        q_flat = strat.compute_quote(state, inventory=0)
        q_long = strat.compute_quote(state, inventory=100)
        assert q_long.bid_price < q_flat.bid_price
        assert q_long.ask_price < q_flat.ask_price

    def test_negative_inventory_shifts_quotes_up(self):
        strat = self._default()
        state = _make_state(mid=100.0, realized_vol=0.001)
        q_flat  = strat.compute_quote(state, inventory=0)
        q_short = strat.compute_quote(state, inventory=-100)
        assert q_short.bid_price > q_flat.bid_price
        assert q_short.ask_price > q_flat.ask_price

    def test_skew_magnitude(self):
        # risk_aversion=0.1, vol=0.001, mid=100, inv=10
        # skew = 0.1 * 0.001 * 100 * 10 = 0.1
        # half_spread = 0.0005 * 100 = 0.05
        strat = self._default(spread_frac=0.0005, risk_aversion=0.1)
        state = _make_state(mid=100.0, realized_vol=0.001)
        q = strat.compute_quote(state, inventory=10)
        expected_reservation = 100.0 - 0.1
        assert abs(q.bid_price - (expected_reservation - 0.05)) < 1e-9
        assert abs(q.ask_price - (expected_reservation + 0.05)) < 1e-9

    def test_zero_risk_aversion_produces_symmetric_quotes(self):
        strat = self._default(spread_frac=0.0005, risk_aversion=0.0)
        state = _make_state(mid=50.0)
        q = strat.compute_quote(state, inventory=9999)
        assert abs(q.bid_price - (50.0 - 0.025)) < 1e-9
        assert abs(q.ask_price - (50.0 + 0.025)) < 1e-9

    def test_skew_scales_with_mid(self):
        # same risk_aversion, doubled mid → doubled skew in dollar terms
        strat = self._default(risk_aversion=0.1)
        s100 = _make_state(mid=100.0, realized_vol=0.001)
        s200 = _make_state(mid=200.0, realized_vol=0.001)
        q100 = strat.compute_quote(s100, inventory=10)
        q200 = strat.compute_quote(s200, inventory=10)
        skew100 = 100.0 - (q100.bid_price + q100.ask_price) / 2
        skew200 = 200.0 - (q200.bid_price + q200.ask_price) / 2
        assert abs(skew200 / skew100 - 2.0) < 1e-6

    # --- inventory hard cap (dollar-based) --------------------------------

    def test_max_position_disables_bid(self):
        # max_position_value=50_000, mid=100 → max_inv=500
        strat = self._default(max_position_value=50_000)
        q = strat.compute_quote(_make_state(mid=100.0), inventory=500)
        assert not q.bid_active
        assert q.ask_active

    def test_above_max_position_disables_bid(self):
        strat = self._default(max_position_value=50_000)
        q = strat.compute_quote(_make_state(mid=100.0), inventory=999)
        assert not q.bid_active

    def test_negative_max_position_disables_ask(self):
        strat = self._default(max_position_value=50_000)
        q = strat.compute_quote(_make_state(mid=100.0), inventory=-500)
        assert not q.ask_active
        assert q.bid_active

    def test_below_negative_max_position_disables_ask(self):
        strat = self._default(max_position_value=50_000)
        q = strat.compute_quote(_make_state(mid=100.0), inventory=-999)
        assert not q.ask_active

    def test_max_inv_scales_with_mid(self):
        # same max_position_value, higher mid → fewer shares allowed
        # max_position_value=50_000, mid=500 → max_inv=100
        strat = self._default(max_position_value=50_000)
        q_at_cap = strat.compute_quote(_make_state(mid=500.0), inventory=100)
        assert not q_at_cap.bid_active

    def test_just_below_max_leaves_both_sides_active(self):
        strat = self._default(max_position_value=50_000)
        q = strat.compute_quote(_make_state(mid=100.0), inventory=499)
        assert q.bid_active
        assert q.ask_active

    def test_returns_quote_instance(self):
        strat = self._default()
        q = strat.compute_quote(_make_state(), inventory=0)
        assert isinstance(q, Quote)


# ---------------------------------------------------------------------------
# VolatilityAdaptiveStrategy
# ---------------------------------------------------------------------------

class TestVolatilityAdaptiveStrategy:
    # mid=100, vol=0.001
    # base half_spread = spread_frac * mid = 0.0005 * 100 = 0.05
    # extra spread     = vol_spread_coef * vol * mid = 2.0 * 0.001 * 100 = 0.20

    def _default(self, **kw):
        defaults = dict(
            spread_frac=0.0005,
            order_size=100,
            risk_aversion=0.1,
            max_position_value=50_000,
            vol_spread_coef=2.0,
        )
        defaults.update(kw)
        return VolatilityAdaptiveStrategy(**defaults)

    def test_bid_below_ask(self):
        strat = self._default()
        q = strat.compute_quote(_make_state(realized_vol=0.002), inventory=0)
        assert q.bid_price < q.ask_price

    def test_higher_vol_widens_spread(self):
        strat = self._default(vol_spread_coef=2.0)
        q_low  = strat.compute_quote(_make_state(mid=100.0, realized_vol=0.001), inventory=0)
        q_high = strat.compute_quote(_make_state(mid=100.0, realized_vol=0.010), inventory=0)
        assert (q_high.ask_price - q_high.bid_price) > (q_low.ask_price - q_low.bid_price)

    def test_zero_vol_spread_coef_matches_constant_strategy(self):
        # vol_spread_coef=0 → identical to ConstantSpreadStrategy
        strat = self._default(spread_frac=0.0005, vol_spread_coef=0.0)
        state = _make_state(mid=100.0, realized_vol=0.001)
        q = strat.compute_quote(state, inventory=0)
        # half_spread = (0.0005 + 0) * 100 = 0.05, spread = 0.10
        assert abs((q.ask_price - q.bid_price) - 0.10) < 1e-9

    def test_none_vol_does_not_crash(self):
        strat = self._default()
        State = namedtuple("State", ["mid", "realized_vol"])
        state = State(mid=100.0, realized_vol=None)
        q = strat.compute_quote(state, inventory=0)
        assert q.bid_price < q.ask_price

    def test_vol_spread_magnitude(self):
        # spread_frac=0.0005, vol_spread_coef=5.0, vol=0.02, mid=100
        # half_spread = (0.0005 + 5.0*0.02) * 100 = (0.1005)*100 = 10.05
        strat = self._default(spread_frac=0.0005, vol_spread_coef=5.0)
        state = _make_state(mid=100.0, realized_vol=0.02)
        q = strat.compute_quote(state, inventory=0)
        expected_half = (0.0005 + 5.0 * 0.02) * 100.0
        assert abs((q.ask_price - q.bid_price) - 2 * expected_half) < 1e-9

    def test_spread_scales_with_mid(self):
        # same spread_frac at doubled mid → doubled spread in dollars
        strat = self._default(spread_frac=0.0005, vol_spread_coef=0.0)
        q100 = strat.compute_quote(_make_state(mid=100.0, realized_vol=0.001), inventory=0)
        q200 = strat.compute_quote(_make_state(mid=200.0, realized_vol=0.001), inventory=0)
        spread100 = q100.ask_price - q100.bid_price
        spread200 = q200.ask_price - q200.bid_price
        assert abs(spread200 / spread100 - 2.0) < 1e-6

    def test_inventory_cap_disables_bid(self):
        # max_position_value=30_000, mid=100 → max_inv=300
        strat = self._default(max_position_value=30_000)
        q = strat.compute_quote(_make_state(mid=100.0), inventory=300)
        assert not q.bid_active

    def test_inventory_cap_disables_ask(self):
        strat = self._default(max_position_value=30_000)
        q = strat.compute_quote(_make_state(mid=100.0), inventory=-300)
        assert not q.ask_active

    def test_returns_quote_instance(self):
        strat = self._default()
        q = strat.compute_quote(_make_state(), inventory=0)
        assert isinstance(q, Quote)
