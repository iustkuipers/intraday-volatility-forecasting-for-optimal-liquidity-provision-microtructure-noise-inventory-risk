"""
simulator/fill_model.py

Determines whether observed market trades execute against the market maker's
resting orders, and returns fill events consumed by Accounting.

Architecture position
---------------------
ExecutionEngine (holds resting orders)
       ↓
FillModel.evaluate(engine, state)
       ↓
list[Fill]  →  Accounting

Design constraints
------------------
- Stateless: no persistent state between calls
- Pure evaluation: never mutates engine, state, or Quote objects
- Modular: swap models by passing a different instance to the Simulator

Fill object
-----------
An immutable record of a single execution.

    Fill(side="bid", price=472.08, size=100)

`side` is "bid" when the market sold to us (we got long),
       "ask" when the market bought from us (we got short).

Available models
----------------
BaseFillModel           — abstract interface (raises NotImplementedError)
DeterministicFillModel  — baseline: price-competitive order always fills
"""

from collections import namedtuple

Fill = namedtuple("Fill", ["side", "price", "size"])


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseFillModel:
    """
    Abstract plug-in interface for all fill models.

    Subclasses must override evaluate().
    """

    def evaluate(self, engine, state) -> list:
        """
        Evaluate whether any resting orders were executed.

        Parameters
        ----------
        engine : ExecutionEngine
            Current resting orders (read-only).
        state : MarketState
            Current market event (read-only).

        Returns
        -------
        list[Fill]
            Empty list if no fills occurred.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DeterministicFillModel
# ---------------------------------------------------------------------------

class DeterministicFillModel(BaseFillModel):
    """
    Baseline deterministic fill model.

    Assumptions
    -----------
    - Only trade events can generate fills.
    - The market maker is assumed to be at the top of the queue whenever
      their price is competitive with the best market price.
    - A sell trade (direction == -1 or trade_price <= best_bid) hits bids.
    - A buy  trade (direction == +1 or trade_price >= best_ask) lifts asks.
    - Fill size = min(order_size, trade_size).
    - Both sides can fill on the same trade (e.g. a wash crossing the spread).

    Fill conditions
    ---------------
    Bid filled when:
        bid_order exists
        AND trade_price <= bid_price   (market sells below or at our bid)
        AND bid_price >= best_bid      (our quote is at the top of book)

    Ask filled when:
        ask_order exists
        AND trade_price >= ask_price   (market buys above or at our ask)
        AND ask_price <= best_ask      (our quote is at the top of book)

    Why trade_price vs. bid/ask_price rather than direction alone
    --------------------------------------------------------------
    `trade_direction` is derived via the Lee-Ready rule and can be 0 (at-mid)
    or NaN.  Using actual price comparison is more robust and works even when
    direction is missing.
    """

    def evaluate(self, engine, state) -> list:
        """
        Return fills triggered by the current market event.

        Parameters
        ----------
        engine : ExecutionEngine
        state  : MarketState

        Returns
        -------
        list[Fill]
        """
        # Only trade events interact with resting orders.
        if not state.is_trade():
            return []

        trade_price = state.trade_price
        trade_size  = state.trade_size

        # Guard: if trade data is missing, skip.
        if trade_price is None or trade_size is None:
            return []

        fills = []

        # ── bid side ──────────────────────────────────────────────────────────
        # A market sell order hits our bid when the trade price is at or below
        # our bid price AND our bid is at least as good as the prevailing best.
        if engine.has_bid():
            bid_price = engine.get_bid_price()
            bid_size  = engine.get_bid_size()
            best_bid  = state.best_bid()

            if (
                best_bid is not None
                and bid_price >= best_bid          # competitive at top of book
                and trade_price <= bid_price       # trade executes at our price
            ):
                fill_size = min(bid_size, int(trade_size))
                fills.append(Fill(side="bid", price=bid_price, size=fill_size))

        # ── ask side ──────────────────────────────────────────────────────────
        # A market buy order lifts our ask when the trade price is at or above
        # our ask price AND our ask is at most the prevailing best.
        if engine.has_ask():
            ask_price = engine.get_ask_price()
            ask_size  = engine.get_ask_size()
            best_ask  = state.best_ask()

            if (
                best_ask is not None
                and ask_price <= best_ask          # competitive at top of book
                and trade_price >= ask_price       # trade executes at our price
            ):
                fill_size = min(ask_size, int(trade_size))
                fills.append(Fill(side="ask", price=ask_price, size=fill_size))

        return fills
