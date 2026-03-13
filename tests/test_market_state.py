"""
tests/test_market_state.py

Unit tests for simulator.market_state.MarketState.

Run with:
    pytest tests/test_market_state.py -v
"""

import math
from collections import namedtuple
import pytest
from simulator.market_state import MarketState


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_row(**kwargs):
    """Build a minimal fake itertuples() row with sensible defaults."""
    defaults = dict(
        timestamp="2024-01-02 09:30:01",
        event_type="quote",
        bid=472.10,
        ask=472.12,
        mid=472.11,
        spread=0.02,
        rel_spread=0.02 / 472.11,
        bid_size=5,
        ask_size=3,
        depth=8,
        imbalance=(5 - 3) / (5 + 3),
        trade_price=float("nan"),
        trade_size=float("nan"),
        trade_direction=0,
        signed_volume=0,
        realized_vol=0.000123,
        trade_intensity=4.0,
        trade_vol_intensity=800.0,
        queue_fraction_bid=100 / 5,
        queue_fraction_ask=100 / 3,
    )
    defaults.update(kwargs)
    Row = namedtuple("Row", defaults.keys())
    return Row(**defaults)


# ── initialisation ────────────────────────────────────────────────────────────

class TestInit:
    def test_all_slots_none_on_init(self):
        state = MarketState()
        for attr in MarketState.__slots__:
            assert getattr(state, attr) is None, f"{attr} should be None after init"

    def test_cannot_set_arbitrary_attribute(self):
        state = MarketState()
        with pytest.raises(AttributeError):
            state.nonexistent_field = 99


# ── update_from_event ─────────────────────────────────────────────────────────

class TestUpdateFromEvent:
    def test_quote_fields_loaded(self):
        state = MarketState()
        row = _make_row()
        state.update_from_event(row)

        assert state.bid == 472.10
        assert state.ask == 472.12
        assert state.mid == 472.11
        assert state.spread == pytest.approx(0.02)
        assert state.bid_size == 5
        assert state.ask_size == 3
        assert state.depth == 8

    def test_imbalance_value(self):
        state = MarketState()
        row = _make_row(bid_size=5, ask_size=3)
        state.update_from_event(row)
        assert state.imbalance == pytest.approx((5 - 3) / (5 + 3))

    def test_microstructure_signals_loaded(self):
        state = MarketState()
        row = _make_row(realized_vol=0.000555, trade_intensity=12.0, trade_vol_intensity=2400.0)
        state.update_from_event(row)
        assert state.realized_vol == pytest.approx(0.000555)
        assert state.trade_intensity == 12.0
        assert state.trade_vol_intensity == 2400.0

    def test_queue_fractions_loaded(self):
        state = MarketState()
        row = _make_row(bid_size=10, ask_size=4,
                        queue_fraction_bid=100/10, queue_fraction_ask=100/4)
        state.update_from_event(row)
        assert state.queue_fraction_bid == pytest.approx(100 / 10)
        assert state.queue_fraction_ask == pytest.approx(100 / 4)

    def test_update_overwrites_previous_state(self):
        state = MarketState()
        state.update_from_event(_make_row(mid=472.00))
        state.update_from_event(_make_row(mid=473.50))
        assert state.mid == 473.50

    def test_timestamp_set(self):
        state = MarketState()
        state.update_from_event(_make_row(timestamp="2024-01-02 09:31:00"))
        assert state.timestamp == "2024-01-02 09:31:00"


# ── event type helpers ────────────────────────────────────────────────────────

class TestEventTypeHelpers:
    def test_is_quote_true(self):
        state = MarketState()
        state.update_from_event(_make_row(event_type="quote"))
        assert state.is_quote() is True
        assert state.is_trade() is False

    def test_is_trade_true(self):
        state = MarketState()
        state.update_from_event(_make_row(
            event_type="trade",
            trade_price=472.15,
            trade_size=100,
            trade_direction=-1,
            signed_volume=-100,
        ))
        assert state.is_trade() is True
        assert state.is_quote() is False

    def test_uninitialised_state_is_neither(self):
        state = MarketState()
        assert state.is_trade() is False
        assert state.is_quote() is False


# ── convenience accessors ─────────────────────────────────────────────────────

class TestConvenienceAccessors:
    def setup_method(self):
        self.state = MarketState()
        self.state.update_from_event(_make_row())

    def test_best_bid(self):
        assert self.state.best_bid() == self.state.bid

    def test_best_ask(self):
        assert self.state.best_ask() == self.state.ask

    def test_mid_price(self):
        assert self.state.mid_price() == self.state.mid

    def test_current_spread(self):
        assert self.state.current_spread() == self.state.spread


# ── trade-specific state ──────────────────────────────────────────────────────

class TestTradeState:
    def test_trade_direction_buy(self):
        state = MarketState()
        state.update_from_event(_make_row(
            event_type="trade",
            trade_price=472.20,
            trade_size=200,
            trade_direction=1,
            signed_volume=200,
            mid=472.11,
        ))
        assert state.trade_direction == 1
        assert state.signed_volume == 200

    def test_trade_direction_sell(self):
        state = MarketState()
        state.update_from_event(_make_row(
            event_type="trade",
            trade_price=472.05,
            trade_size=100,
            trade_direction=-1,
            signed_volume=-100,
            mid=472.11,
        ))
        assert state.trade_direction == -1
        assert state.signed_volume == -100

    def test_quote_event_trade_fields_zero(self):
        state = MarketState()
        state.update_from_event(_make_row(event_type="quote"))
        assert state.trade_direction == 0
        assert state.signed_volume == 0
        assert math.isnan(state.trade_price)


# ── repr ──────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_uninitialised(self):
        state = MarketState()
        assert "uninitialised" in repr(state)

    def test_repr_after_update(self):
        state = MarketState()
        state.update_from_event(_make_row())
        r = repr(state)
        assert "MarketState(" in r
        assert "quote" in r
