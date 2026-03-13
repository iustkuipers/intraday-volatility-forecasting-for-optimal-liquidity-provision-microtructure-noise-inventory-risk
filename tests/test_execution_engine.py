"""
tests/test_execution_engine.py

Unit tests for simulator/execution_engine.py.

Run with:
    pytest tests/test_execution_engine.py -p no:dash -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.order import Quote
from simulator.execution_engine import ExecutionEngine, Order


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_quote(bid=99.95, ask=100.05, bid_size=100, ask_size=100):
    return Quote(bid_price=bid, ask_price=ask, bid_size=bid_size, ask_size=ask_size)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_no_bid_at_start(self):
        assert ExecutionEngine().bid_order is None

    def test_no_ask_at_start(self):
        assert ExecutionEngine().ask_order is None

    def test_has_bid_false_at_start(self):
        assert not ExecutionEngine().has_bid()

    def test_has_ask_false_at_start(self):
        assert not ExecutionEngine().has_ask()

    def test_get_bid_price_none_at_start(self):
        assert ExecutionEngine().get_bid_price() is None

    def test_get_ask_price_none_at_start(self):
        assert ExecutionEngine().get_ask_price() is None

    def test_get_bid_size_none_at_start(self):
        assert ExecutionEngine().get_bid_size() is None

    def test_get_ask_size_none_at_start(self):
        assert ExecutionEngine().get_ask_size() is None


# ---------------------------------------------------------------------------
# update_quote — normal two-sided quote
# ---------------------------------------------------------------------------

class TestUpdateQuote:

    def test_bid_price_stored(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote(bid=99.90, ask=100.10))
        assert engine.get_bid_price() == 99.90

    def test_ask_price_stored(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote(bid=99.90, ask=100.10))
        assert engine.get_ask_price() == 100.10

    def test_bid_size_stored(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote(bid_size=200))
        assert engine.get_bid_size() == 200

    def test_ask_size_stored(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote(ask_size=300))
        assert engine.get_ask_size() == 300

    def test_has_bid_after_update(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        assert engine.has_bid()

    def test_has_ask_after_update(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        assert engine.has_ask()

    def test_orders_are_named_tuples(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        assert isinstance(engine.bid_order, Order)
        assert isinstance(engine.ask_order, Order)


# ---------------------------------------------------------------------------
# update_quote — disabled sides
# ---------------------------------------------------------------------------

class TestDisabledSides:

    def test_disabled_bid_clears_bid_order(self):
        engine = ExecutionEngine()
        q = _make_quote()
        q.disable_bid()
        engine.update_quote(q)
        assert engine.bid_order is None

    def test_disabled_bid_keeps_ask_order(self):
        engine = ExecutionEngine()
        q = _make_quote()
        q.disable_bid()
        engine.update_quote(q)
        assert engine.has_ask()

    def test_disabled_ask_clears_ask_order(self):
        engine = ExecutionEngine()
        q = _make_quote()
        q.disable_ask()
        engine.update_quote(q)
        assert engine.ask_order is None

    def test_disabled_ask_keeps_bid_order(self):
        engine = ExecutionEngine()
        q = _make_quote()
        q.disable_ask()
        engine.update_quote(q)
        assert engine.has_bid()

    def test_both_disabled_clears_both(self):
        engine = ExecutionEngine()
        q = _make_quote()
        q.disable_bid()
        q.disable_ask()
        engine.update_quote(q)
        assert engine.bid_order is None
        assert engine.ask_order is None


# ---------------------------------------------------------------------------
# update_quote — invalid quotes
# ---------------------------------------------------------------------------

class TestInvalidQuote:

    def test_crossed_quote_clears_both_sides(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())          # set valid orders first
        engine.update_quote(_make_quote(bid=101.0, ask=99.0))   # crossed
        assert engine.bid_order is None
        assert engine.ask_order is None

    def test_equal_prices_clears_both_sides(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        engine.update_quote(_make_quote(bid=100.0, ask=100.0))
        assert engine.bid_order is None
        assert engine.ask_order is None

    def test_none_bid_price_clears_both(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        engine.update_quote(Quote(bid_price=None, ask_price=100.0))
        assert engine.bid_order is None
        assert engine.ask_order is None

    def test_none_ask_price_clears_both(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        engine.update_quote(Quote(bid_price=99.0, ask_price=None))
        assert engine.bid_order is None
        assert engine.ask_order is None


# ---------------------------------------------------------------------------
# update_quote — replacement (quote2 always wins)
# ---------------------------------------------------------------------------

class TestQuoteReplacement:

    def test_second_quote_replaces_bid_price(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote(bid=99.90))
        engine.update_quote(_make_quote(bid=99.80))
        assert engine.get_bid_price() == 99.80

    def test_second_quote_replaces_ask_price(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote(ask=100.10))
        engine.update_quote(_make_quote(ask=100.20))
        assert engine.get_ask_price() == 100.20

    def test_second_quote_replaces_bid_size(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote(bid_size=100))
        engine.update_quote(_make_quote(bid_size=50))
        assert engine.get_bid_size() == 50

    def test_disabling_side_after_active_quote_clears_order(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        assert engine.has_bid()
        q2 = _make_quote()
        q2.disable_bid()
        engine.update_quote(q2)
        assert not engine.has_bid()

    def test_quote_not_mutated_by_engine(self):
        """ExecutionEngine must not touch the Quote object."""
        engine = ExecutionEngine()
        q = _make_quote()
        original_bid = q.bid_price
        original_active = q.bid_active
        engine.update_quote(q)
        assert q.bid_price == original_bid
        assert q.bid_active == original_active


# ---------------------------------------------------------------------------
# cancel helpers
# ---------------------------------------------------------------------------

class TestCancelHelpers:

    def test_cancel_bid_clears_bid(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        engine.cancel_bid()
        assert engine.bid_order is None

    def test_cancel_bid_keeps_ask(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        engine.cancel_bid()
        assert engine.has_ask()

    def test_cancel_ask_clears_ask(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        engine.cancel_ask()
        assert engine.ask_order is None

    def test_cancel_ask_keeps_bid(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        engine.cancel_ask()
        assert engine.has_bid()

    def test_cancel_all(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote())
        engine.cancel_all()
        assert engine.bid_order is None
        assert engine.ask_order is None

    def test_cancel_all_on_empty_engine_does_not_crash(self):
        engine = ExecutionEngine()
        engine.cancel_all()   # should be a no-op
        assert not engine.has_bid()
        assert not engine.has_ask()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_with_orders(self):
        engine = ExecutionEngine()
        engine.update_quote(_make_quote(bid=99.95, ask=100.05, bid_size=100, ask_size=100))
        r = repr(engine)
        assert "99.9500" in r
        assert "100.0500" in r

    def test_repr_no_orders(self):
        r = repr(ExecutionEngine())
        assert "None" in r

    def test_repr_is_string(self):
        assert isinstance(repr(ExecutionEngine()), str)
