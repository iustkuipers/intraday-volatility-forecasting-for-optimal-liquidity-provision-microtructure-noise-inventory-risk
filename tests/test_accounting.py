"""
tests/test_accounting.py

Unit tests for simulator/accounting.py.

Run with:
    pytest tests/test_accounting.py -p no:dash -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.accounting import Accounting
from simulator.fill_model import Fill


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def bid(price, size):
    return Fill(side="bid", price=price, size=size)

def ask(price, size):
    return Fill(side="ask", price=price, size=size)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:

    def test_inventory_zero(self):
        assert Accounting().inventory == 0

    def test_cash_zero(self):
        assert Accounting().cash == 0.0

    def test_realized_pnl_zero(self):
        assert Accounting().realized_pnl == 0.0

    def test_portfolio_value_zero(self):
        assert Accounting().portfolio_value == 0.0

    def test_avg_cost_zero(self):
        assert Accounting().avg_cost == 0.0


# ---------------------------------------------------------------------------
# Single bid fill (buy)
# ---------------------------------------------------------------------------

class TestBidFill:

    def test_inventory_increases(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        assert acct.inventory == 100

    def test_cash_decreases(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        assert acct.cash == -1000.0

    def test_avg_cost_set(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        assert abs(acct.avg_cost - 10.0) < 1e-9

    def test_realized_pnl_unchanged(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        assert acct.realized_pnl == 0.0


# ---------------------------------------------------------------------------
# Single ask fill (sell)
# ---------------------------------------------------------------------------

class TestAskFill:

    def test_inventory_decreases(self):
        acct = Accounting()
        acct.apply_fill(ask(11.0, 50))
        assert acct.inventory == -50

    def test_cash_increases(self):
        acct = Accounting()
        acct.apply_fill(ask(11.0, 50))
        assert acct.cash == 550.0

    def test_avg_cost_set_on_short(self):
        acct = Accounting()
        acct.apply_fill(ask(11.0, 50))
        assert abs(acct.avg_cost - 11.0) < 1e-9

    def test_realized_pnl_unchanged_on_open_short(self):
        acct = Accounting()
        acct.apply_fill(ask(11.0, 50))
        assert acct.realized_pnl == 0.0


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------

class TestMarkToMarket:

    def test_flat_position_portfolio_equals_cash(self):
        acct = Accounting()
        acct.mark_to_market(mid_price=100.0)
        assert acct.portfolio_value == 0.0

    def test_long_position_portfolio_value(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))   # cash = -1000, inv = 100
        acct.mark_to_market(mid_price=11.0)
        # portfolio = -1000 + 100*11 = 100
        assert abs(acct.portfolio_value - 100.0) < 1e-9

    def test_short_position_portfolio_value(self):
        acct = Accounting()
        acct.apply_fill(ask(11.0, 100))   # cash = +1100, inv = -100
        acct.mark_to_market(mid_price=10.0)
        # portfolio = 1100 + (-100)*10 = 100
        assert abs(acct.portfolio_value - 100.0) < 1e-9

    def test_portfolio_updates_on_second_call(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        acct.mark_to_market(mid_price=10.5)
        acct.mark_to_market(mid_price=11.0)
        assert abs(acct.portfolio_value - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# Round-trip realized PnL
# ---------------------------------------------------------------------------

class TestRealizedPnL:

    def test_buy_then_sell_same_size(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        acct.apply_fill(ask(11.0, 100))
        # realized = (11 - 10) * 100 = 100
        assert abs(acct.realized_pnl - 100.0) < 1e-9

    def test_buy_then_sell_same_size_inventory_flat(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        acct.apply_fill(ask(11.0, 100))
        assert acct.inventory == 0

    def test_buy_then_sell_same_size_cash(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        acct.apply_fill(ask(11.0, 100))
        # cash = -1000 + 1100 = +100
        assert abs(acct.cash - 100.0) < 1e-9

    def test_loss_on_adverse_round_trip(self):
        acct = Accounting()
        acct.apply_fill(bid(11.0, 100))
        acct.apply_fill(ask(10.0, 100))
        # realized = (10 - 11) * 100 = -100
        assert abs(acct.realized_pnl - (-100.0)) < 1e-9

    def test_partial_sell_realizes_partial_pnl(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        acct.apply_fill(ask(12.0, 50))
        # realized = (12 - 10) * 50 = 100
        assert abs(acct.realized_pnl - 100.0) < 1e-9
        assert acct.inventory == 50

    def test_multiple_buys_average_cost(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        acct.apply_fill(bid(12.0, 100))
        # avg_cost = (10*100 + 12*100) / 200 = 11.0
        assert abs(acct.avg_cost - 11.0) < 1e-9

    def test_multiple_buys_then_full_sell(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        acct.apply_fill(bid(12.0, 100))
        acct.apply_fill(ask(13.0, 200))
        # realized = (13 - 11) * 200 = 400
        assert abs(acct.realized_pnl - 400.0) < 1e-9
        assert acct.inventory == 0

    def test_short_then_cover_profit(self):
        acct = Accounting()
        acct.apply_fill(ask(11.0, 100))   # short 100 @ 11
        acct.apply_fill(bid(10.0, 100))   # cover  100 @ 10
        # realized = (11 - 10) * 100 = 100
        assert abs(acct.realized_pnl - 100.0) < 1e-9
        assert acct.inventory == 0


# ---------------------------------------------------------------------------
# apply_fills (batch)
# ---------------------------------------------------------------------------

class TestApplyFills:

    def test_apply_empty_list(self):
        acct = Accounting()
        acct.apply_fills([])
        assert acct.inventory == 0
        assert acct.cash == 0.0

    def test_apply_multiple_fills(self):
        acct = Accounting()
        acct.apply_fills([bid(10.0, 100), ask(11.0, 100)])
        assert acct.inventory == 0
        assert abs(acct.realized_pnl - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# Position crossing zero
# ---------------------------------------------------------------------------

class TestCrossingZero:

    def test_long_reverses_to_short(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        acct.apply_fill(ask(11.0, 150))   # sells 100 at profit, opens short 50
        assert acct.inventory == -50
        # realized pnl from closing the long: (11-10)*100 = 100
        assert abs(acct.realized_pnl - 100.0) < 1e-9

    def test_short_reverses_to_long(self):
        acct = Accounting()
        acct.apply_fill(ask(11.0, 100))
        acct.apply_fill(bid(10.0, 150))   # covers 100, opens long 50
        assert acct.inventory == 50
        # realized pnl from covering: (11-10)*100 = 100
        assert abs(acct.realized_pnl - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# unknown fill side
# ---------------------------------------------------------------------------

class TestUnknownSide:

    def test_bad_side_raises(self):
        acct = Accounting()
        with pytest.raises(ValueError):
            acct.apply_fill(Fill(side="unknown", price=10.0, size=100))


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_is_string(self):
        assert isinstance(repr(Accounting()), str)

    def test_repr_contains_inventory(self):
        acct = Accounting()
        acct.apply_fill(bid(10.0, 100))
        assert "inv=100" in repr(acct)
