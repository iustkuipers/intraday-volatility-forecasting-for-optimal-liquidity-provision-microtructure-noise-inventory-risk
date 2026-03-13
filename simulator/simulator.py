"""
simulator/simulator.py

Event-driven market-making simulator.

Wires all core components together and replays a dataset event by event,
producing a time-series DataFrame of the system's full state.

Architecture
------------
for every row in events:
    MarketState.update_from_event(row)
    quote  = strategy.compute_quote(state, accounting.inventory)
    engine.update_quote(quote)
    fills  = fill_model.evaluate(engine, state)
    accounting.apply_fills(fills)
    accounting.mark_to_market(state.mid)
    → append snapshot to results

What the Simulator does NOT do
-------------------------------
- It does not interpret event data     (MarketState does)
- It does not decide quote prices      (strategy does)
- It does not judge fill probability   (fill model does)
- It does not compute PnL math         (accounting does)

The simulator is a pure orchestrator.
"""

import math

import pandas as pd

from simulator.market_state import MarketState
from simulator.execution_engine import ExecutionEngine
from simulator.accounting import Accounting
from simulator.strategy_base import BaseStrategy
from simulator.fill_model import BaseFillModel


class Simulator:
    """
    Event-driven market-making simulator.

    Parameters
    ----------
    strategy   : BaseStrategy   — computes quotes given market state + inventory
    fill_model : BaseFillModel  — decides whether trades execute against orders
    state      : MarketState    — mutable market snapshot (updated each row)
    engine     : ExecutionEngine— tracks active resting orders
    accounting : Accounting     — tracks inventory, cash, PnL

    Usage
    -----
    sim = Simulator(
        strategy   = ConstantSpreadStrategy(...),
        fill_model = DeterministicFillModel(),
        state      = MarketState(),
        engine     = ExecutionEngine(),
        accounting = Accounting(),
    )
    results_df = sim.run(events_df)
    """

    def __init__(
        self,
        strategy:   BaseStrategy,
        fill_model: BaseFillModel,
        state:      MarketState      = None,
        engine:     ExecutionEngine  = None,
        accounting: Accounting       = None,
    ):
        self.strategy   = strategy
        self.fill_model = fill_model
        self.state      = state      or MarketState()
        self.engine     = engine     or ExecutionEngine()
        self.accounting = accounting or Accounting()

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Replay all events and return a snapshot DataFrame.

        Parameters
        ----------
        events : pd.DataFrame
            The full event stream (output of data_merged.py).
            Must have columns matching MarketState.update_from_event().

        Returns
        -------
        pd.DataFrame
            One row per event with columns:
            timestamp, event_type, mid, bid, ask,
            bid_quote, ask_quote,
            inventory, cash, portfolio_value, realized_pnl,
            fill_count, fill_side.
        """
        snapshots   = []
        append      = snapshots.append

        strategy    = self.strategy
        fill_model  = self.fill_model
        state       = self.state
        engine      = self.engine
        accounting  = self.accounting

        for row in events.itertuples(index=False):

            # 1. Update market snapshot
            state.update_from_event(row)

            # 2. Strategy computes desired quote
            quote = strategy.compute_quote(state, accounting.inventory)

            # 3. Execution engine stores / replaces resting orders
            engine.update_quote(quote)

            # 4. Fill model checks whether any order was hit
            fills = fill_model.evaluate(engine, state)

            # 5. Accounting processes fills
            accounting.apply_fills(fills)

            # 5b. Commit last_trade_price AFTER fills so that the quote
            #     computed in step 2 used the price from the previous trade,
            #     not the trade that is about to be evaluated for fills.
            if state.is_trade() and state.trade_price is not None:
                if not math.isnan(float(state.trade_price)):
                    state.last_trade_price = state.trade_price

            # 6. Mark portfolio to current mid
            if state.mid is not None:
                accounting.mark_to_market(state.mid)

            # 7. Record snapshot
            append(self._snapshot(state, engine, accounting, fills))

        return pd.DataFrame(snapshots)

    # ── snapshot ──────────────────────────────────────────────────────────────

    def _snapshot(self, state, engine, accounting, fills) -> dict:
        """
        Capture a flat dict of the full system state after one event.

        All fields are scalars or None so DataFrame construction is cheap.
        """
        fill_count = len(fills)
        # Summarise fills: 'bid', 'ask', 'both', or None
        if fill_count == 0:
            fill_side = None
        elif fill_count == 1:
            fill_side = fills[0].side
        else:
            fill_side = "both"

        return {
            # market
            "timestamp":       state.timestamp,
            "event_type":      state.event_type,
            "mid":             state.mid,
            "bid":             state.bid,
            "ask":             state.ask,
            "spread":          state.spread,
            "realized_vol":    state.realized_vol,
            # quotes posted
            "bid_quote":       engine.get_bid_price(),
            "ask_quote":       engine.get_ask_price(),
            # fills
            "fill_count":      fill_count,
            "fill_side":       fill_side,
            # accounting
            "inventory":       accounting.inventory,
            "cash":            accounting.cash,
            "portfolio_value": accounting.portfolio_value,
            "realized_pnl":    accounting.realized_pnl,
            "avg_cost":        accounting.avg_cost,
        }
