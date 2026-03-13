"""
tests/test_order.py

Unit tests for simulator/order.py (Quote) and simulator/strategy_base.py
(BaseStrategy raises NotImplementedError).

Run with:
    pytest tests/test_order.py -p no:dash -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.order import Quote
from simulator.strategy_base import BaseStrategy


# ---------------------------------------------------------------------------
# Quote — construction
# ---------------------------------------------------------------------------

class TestQuoteConstruction:

    def test_explicit_prices_stored(self):
        q = Quote(bid_price=99.95, ask_price=100.05, bid_size=100, ask_size=100)
        assert q.bid_price == 99.95
        assert q.ask_price == 100.05

    def test_explicit_sizes_stored(self):
        q = Quote(bid_price=99.0, ask_price=101.0, bid_size=50, ask_size=200)
        assert q.bid_size == 50
        assert q.ask_size == 200

    def test_both_sides_active_by_default(self):
        q = Quote(bid_price=99.0, ask_price=101.0)
        assert q.bid_active is True
        assert q.ask_active is True

    def test_default_prices_are_none(self):
        q = Quote()
        assert q.bid_price is None
        assert q.ask_price is None

    def test_default_sizes_are_zero(self):
        q = Quote()
        assert q.bid_size == 0
        assert q.ask_size == 0

    def test_active_flags_can_be_overridden_at_construction(self):
        q = Quote(bid_price=99.0, ask_price=101.0, bid_active=False, ask_active=False)
        assert not q.bid_active
        assert not q.ask_active


# ---------------------------------------------------------------------------
# Quote — disable_bid / disable_ask
# ---------------------------------------------------------------------------

class TestQuoteDisable:

    def test_disable_bid_sets_flag(self):
        q = Quote(bid_price=99.0, ask_price=101.0)
        q.disable_bid()
        assert not q.bid_active

    def test_disable_ask_sets_flag(self):
        q = Quote(bid_price=99.0, ask_price=101.0)
        q.disable_ask()
        assert not q.ask_active

    def test_disable_bid_does_not_affect_ask(self):
        q = Quote(bid_price=99.0, ask_price=101.0)
        q.disable_bid()
        assert q.ask_active

    def test_disable_ask_does_not_affect_bid(self):
        q = Quote(bid_price=99.0, ask_price=101.0)
        q.disable_ask()
        assert q.bid_active

    def test_disable_both_sides(self):
        q = Quote(bid_price=99.0, ask_price=101.0)
        q.disable_bid()
        q.disable_ask()
        assert not q.bid_active
        assert not q.ask_active

    def test_disable_bid_is_idempotent(self):
        q = Quote(bid_price=99.0, ask_price=101.0)
        q.disable_bid()
        q.disable_bid()
        assert not q.bid_active


# ---------------------------------------------------------------------------
# Quote — is_valid
# ---------------------------------------------------------------------------

class TestQuoteIsValid:

    def test_valid_normal_quote(self):
        q = Quote(bid_price=99.95, ask_price=100.05)
        assert q.is_valid()

    def test_invalid_crossed_quote(self):
        q = Quote(bid_price=100.05, ask_price=99.95)
        assert not q.is_valid()

    def test_invalid_equal_prices(self):
        q = Quote(bid_price=100.0, ask_price=100.0)
        assert not q.is_valid()

    def test_invalid_none_bid(self):
        q = Quote(bid_price=None, ask_price=100.05)
        assert not q.is_valid()

    def test_invalid_none_ask(self):
        q = Quote(bid_price=99.95, ask_price=None)
        assert not q.is_valid()

    def test_invalid_both_none(self):
        q = Quote()
        assert not q.is_valid()

    def test_valid_is_independent_of_active_flags(self):
        # is_valid checks prices only, not whether sides are active
        q = Quote(bid_price=99.0, ask_price=101.0)
        q.disable_bid()
        q.disable_ask()
        assert q.is_valid()


# ---------------------------------------------------------------------------
# Quote — __repr__
# ---------------------------------------------------------------------------

class TestQuoteRepr:

    def test_repr_both_sides_active(self):
        q = Quote(bid_price=99.95, ask_price=100.05, bid_size=100, ask_size=100)
        r = repr(q)
        assert "99.9500" in r
        assert "100.0500" in r
        assert "OFF" not in r

    def test_repr_bid_disabled_shows_off(self):
        q = Quote(bid_price=99.95, ask_price=100.05, bid_size=100, ask_size=100)
        q.disable_bid()
        r = repr(q)
        assert "OFF" in r
        assert "100.0500" in r

    def test_repr_ask_disabled_shows_off(self):
        q = Quote(bid_price=99.95, ask_price=100.05, bid_size=100, ask_size=100)
        q.disable_ask()
        r = repr(q)
        assert "99.9500" in r
        assert "OFF" in r

    def test_repr_is_string(self):
        q = Quote(bid_price=99.0, ask_price=101.0)
        assert isinstance(repr(q), str)


# ---------------------------------------------------------------------------
# BaseStrategy — NotImplementedError contract
# ---------------------------------------------------------------------------

class TestBaseStrategy:

    def test_compute_quote_raises_not_implemented(self):
        strat = BaseStrategy()
        with pytest.raises(NotImplementedError):
            strat.compute_quote(market_state=None, inventory=0)
