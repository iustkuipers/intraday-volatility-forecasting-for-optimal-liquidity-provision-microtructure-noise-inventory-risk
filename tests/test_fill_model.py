"""
tests/test_fill_model.py

Unit tests for simulator/fill_model.py (Fill, BaseFillModel,
DeterministicFillModel).

Run with:
    pytest tests/test_fill_model.py -p no:dash -v
"""

import sys
import os
import math
import pytest
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.fill_model import Fill, BaseFillModel, DeterministicFillModel
from simulator.execution_engine import ExecutionEngine
from simulator.order import Quote


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_engine(bid=99.90, ask=100.10, bid_size=100, ask_size=100):
    """Return an ExecutionEngine with both sides active."""
    engine = ExecutionEngine()
    engine.update_quote(Quote(
        bid_price=bid, ask_price=ask,
        bid_size=bid_size, ask_size=ask_size,
    ))
    return engine


def _make_state(
    event_type="trade",
    trade_price=99.90,
    trade_size=50,
    trade_direction=-1,
    best_bid=99.90,
    best_ask=100.10,
):
    """Build a minimal fake MarketState namedtuple."""
    State = namedtuple("State", [
        "event_type", "trade_price", "trade_size", "trade_direction",
        "bid", "ask",
    ])
    s = State(
        event_type=event_type,
        trade_price=trade_price,
        trade_size=trade_size,
        trade_direction=trade_direction,
        bid=best_bid,
        ask=best_ask,
    )
    # Attach the helpers MarketState provides
    s.__class__.is_trade  = lambda self: self.event_type == "trade"
    s.__class__.is_quote  = lambda self: self.event_type == "quote"
    s.__class__.best_bid  = lambda self: self.bid
    s.__class__.best_ask  = lambda self: self.ask
    return s


# ---------------------------------------------------------------------------
# Fill namedtuple
# ---------------------------------------------------------------------------

class TestFill:

    def test_fill_fields(self):
        f = Fill(side="bid", price=99.90, size=100)
        assert f.side  == "bid"
        assert f.price == 99.90
        assert f.size  == 100

    def test_fill_is_immutable(self):
        f = Fill(side="bid", price=99.90, size=100)
        with pytest.raises(AttributeError):
            f.side = "ask"


# ---------------------------------------------------------------------------
# BaseFillModel — interface contract
# ---------------------------------------------------------------------------

class TestBaseFillModel:

    def test_evaluate_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BaseFillModel().evaluate(engine=None, state=None)


# ---------------------------------------------------------------------------
# DeterministicFillModel — quote events produce no fills
# ---------------------------------------------------------------------------

class TestQuoteEventNoFill:

    def test_quote_event_returns_empty(self):
        fm = DeterministicFillModel()
        state = _make_state(event_type="quote")
        fills = fm.evaluate(_make_engine(), state)
        assert fills == []


# ---------------------------------------------------------------------------
# DeterministicFillModel — bid side fills
# ---------------------------------------------------------------------------

class TestBidFill:

    def _fm(self):
        return DeterministicFillModel()

    def test_sell_trade_at_bid_price_fills_bid(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, ask=100.10)
        # trade_price == bid_price, best_bid == bid_price → fill
        state = _make_state(trade_price=99.90, trade_size=50,
                            best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert len(fills) == 1
        assert fills[0].side == "bid"

    def test_sell_trade_below_bid_price_fills_bid(self):
        """Trade executes below our bid — we are filled at our bid price."""
        fm = self._fm()
        engine = _make_engine(bid=99.90)
        state = _make_state(trade_price=99.80, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert any(f.side == "bid" for f in fills)

    def test_fill_price_is_our_bid_not_trade_price(self):
        """We fill at our posted bid, not the trade price."""
        fm = self._fm()
        engine = _make_engine(bid=99.90, bid_size=100)
        state = _make_state(trade_price=99.80, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        bid_fill = next(f for f in fills if f.side == "bid")
        assert bid_fill.price == 99.90

    def test_fill_size_limited_by_order_size(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, bid_size=30)
        state = _make_state(trade_price=99.90, trade_size=200,
                            best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        bid_fill = next(f for f in fills if f.side == "bid")
        assert bid_fill.size == 30

    def test_fill_size_limited_by_trade_size(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, bid_size=200)
        state = _make_state(trade_price=99.90, trade_size=40,
                            best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        bid_fill = next(f for f in fills if f.side == "bid")
        assert bid_fill.size == 40

    def test_no_bid_fill_when_trade_above_bid(self):
        """Buy trade at 100.10 should NOT fill our bid at 99.90."""
        fm = self._fm()
        engine = _make_engine(bid=99.90, ask=100.10)
        state = _make_state(trade_price=100.10, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert not any(f.side == "bid" for f in fills)

    def test_no_bid_fill_when_our_bid_behind_best(self):
        """Our bid is below the best bid — we are not at top of book."""
        fm = self._fm()
        engine = _make_engine(bid=99.80, ask=100.20)
        # best_bid is 99.90 (someone else is better)
        state = _make_state(trade_price=99.90, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert not any(f.side == "bid" for f in fills)

    def test_no_bid_fill_when_no_bid_order(self):
        fm = self._fm()
        engine = ExecutionEngine()   # empty
        state = _make_state(trade_price=99.90, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert fills == []


# ---------------------------------------------------------------------------
# DeterministicFillModel — ask side fills
# ---------------------------------------------------------------------------

class TestAskFill:

    def _fm(self):
        return DeterministicFillModel()

    def test_buy_trade_at_ask_price_fills_ask(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, ask=100.10)
        state = _make_state(trade_price=100.10, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert any(f.side == "ask" for f in fills)

    def test_buy_trade_above_ask_price_fills_ask(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, ask=100.10)
        state = _make_state(trade_price=100.20, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert any(f.side == "ask" for f in fills)

    def test_fill_price_is_our_ask_not_trade_price(self):
        fm = self._fm()
        engine = _make_engine(ask=100.10, ask_size=100)
        state = _make_state(trade_price=100.20, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        ask_fill = next(f for f in fills if f.side == "ask")
        assert ask_fill.price == 100.10

    def test_fill_size_limited_by_order_size(self):
        fm = self._fm()
        engine = _make_engine(ask=100.10, ask_size=25)
        state = _make_state(trade_price=100.10, trade_size=500,
                            best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        ask_fill = next(f for f in fills if f.side == "ask")
        assert ask_fill.size == 25

    def test_fill_size_limited_by_trade_size(self):
        fm = self._fm()
        engine = _make_engine(ask=100.10, ask_size=500)
        state = _make_state(trade_price=100.10, trade_size=10,
                            best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        ask_fill = next(f for f in fills if f.side == "ask")
        assert ask_fill.size == 10

    def test_no_ask_fill_when_trade_below_ask(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, ask=100.10)
        state = _make_state(trade_price=99.90, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert not any(f.side == "ask" for f in fills)

    def test_no_ask_fill_when_our_ask_behind_best(self):
        """Our ask is above the best ask — not at top of book."""
        fm = self._fm()
        engine = _make_engine(bid=99.70, ask=100.20)
        state = _make_state(trade_price=100.10, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert not any(f.side == "ask" for f in fills)

    def test_no_ask_fill_when_no_ask_order(self):
        fm = self._fm()
        engine = ExecutionEngine()
        state = _make_state(trade_price=100.10, best_bid=99.90, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert fills == []


# ---------------------------------------------------------------------------
# DeterministicFillModel — both sides, edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def _fm(self):
        return DeterministicFillModel()

    def test_both_sides_can_fill_on_same_trade(self):
        """A wash trade crossing the spread fills both sides."""
        fm = self._fm()
        engine = _make_engine(bid=100.00, ask=100.00)
        # Deliberately create a crossed-quote engine state for this test
        # by directly setting orders (bypassing update_quote validity check)
        from simulator.execution_engine import Order
        engine.bid_order = Order(price=100.00, size=100)
        engine.ask_order = Order(price=100.00, size=100)
        state = _make_state(trade_price=100.00, best_bid=100.00, best_ask=100.00)
        fills = fm.evaluate(engine, state)
        sides = {f.side for f in fills}
        assert "bid" in sides
        assert "ask" in sides

    def test_returns_list(self):
        fm = self._fm()
        result = fm.evaluate(_make_engine(), _make_state(event_type="quote"))
        assert isinstance(result, list)

    def test_no_fill_when_trade_price_is_none(self):
        fm = self._fm()
        state = _make_state(trade_price=None)
        fills = fm.evaluate(_make_engine(), state)
        assert fills == []

    def test_no_fill_when_trade_size_is_none(self):
        fm = self._fm()
        state = _make_state(trade_size=None)
        fills = fm.evaluate(_make_engine(), state)
        assert fills == []

    def test_engine_not_mutated_after_evaluate(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, ask=100.10, bid_size=100, ask_size=100)
        state = _make_state(trade_price=99.90, best_bid=99.90, best_ask=100.10)
        fm.evaluate(engine, state)
        # Engine should still hold the same orders
        assert engine.get_bid_price() == 99.90
        assert engine.get_ask_price() == 100.10
        assert engine.get_bid_size()  == 100
        assert engine.get_ask_size()  == 100

    def test_no_fill_when_best_bid_is_none(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, ask=100.10)
        state = _make_state(trade_price=99.90, best_bid=None, best_ask=100.10)
        fills = fm.evaluate(engine, state)
        assert not any(f.side == "bid" for f in fills)

    def test_no_fill_when_best_ask_is_none(self):
        fm = self._fm()
        engine = _make_engine(bid=99.90, ask=100.10)
        state = _make_state(trade_price=100.10, best_bid=99.90, best_ask=None)
        fills = fm.evaluate(engine, state)
        assert not any(f.side == "ask" for f in fills)
