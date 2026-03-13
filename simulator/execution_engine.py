"""
simulator/execution_engine.py

Manages the lifecycle of the market maker's resting orders.

Responsibilities
----------------
1. Store the current active bid and ask order (price + size tuples).
2. Accept a new Quote from the strategy and replace the previous orders.
3. Enforce validity: inactive sides and invalid quotes clear the relevant order.
4. Expose read-only helpers so FillModel can inspect resting orders without
   reaching into Quote internals.

What this module does NOT do
-----------------------------
- It does not compute quotes             (strategy's job)
- It does not decide fill probabilities  (fill model's job)
- It does not update inventory or PnL    (accounting's job)
- It does not mutate Quote objects       (quotes are immutable instructions)

Order representation
--------------------
Each resting order is stored as a plain named tuple:

    Order(price=472.08, size=100)

Using a named tuple rather than the Quote directly:
  - keeps ExecutionEngine independent of Quote internals
  - makes it trivial to extend (e.g. add queue position)
  - is cheap to construct (no dict allocation with __slots__)
"""

from collections import namedtuple

Order = namedtuple("Order", ["price", "size"])


class ExecutionEngine:
    """
    Maintains the market maker's current resting orders.

    Usage
    -----
    engine = ExecutionEngine()

    # on every quote-event:
    engine.update_quote(quote)          # stores or clears bid/ask orders

    # on every trade-event (called by FillModel):
    if engine.has_bid():
        price = engine.get_bid_price()
        size  = engine.get_bid_size()
    """

    __slots__ = ("bid_order", "ask_order")

    def __init__(self):
        self.bid_order = None   # Order | None
        self.ask_order = None   # Order | None

    # ── order update ─────────────────────────────────────────────────────────

    def update_quote(self, quote) -> None:
        """
        Replace resting orders from a new Quote.

        A side is set to None (cancelled) when:
          - the Quote side is inactive (disable_bid / disable_ask was called)
          - the Quote price for that side is None
          - the overall quote is crossed (bid >= ask)

        Quote objects are never mutated here.

        Parameters
        ----------
        quote : Quote
            The desired two-sided quote produced by the strategy.
        """
        # Validate overall quote integrity first.
        # A crossed or price-less quote is treated as a full cancel.
        if not quote.is_valid():
            self.bid_order = None
            self.ask_order = None
            return

        # Bid side
        if quote.bid_active and quote.bid_price is not None:
            self.bid_order = Order(price=quote.bid_price, size=quote.bid_size)
        else:
            self.bid_order = None

        # Ask side
        if quote.ask_active and quote.ask_price is not None:
            self.ask_order = Order(price=quote.ask_price, size=quote.ask_size)
        else:
            self.ask_order = None

    # ── presence helpers ──────────────────────────────────────────────────────

    def has_bid(self) -> bool:
        """True when there is an active resting bid order."""
        return self.bid_order is not None

    def has_ask(self) -> bool:
        """True when there is an active resting ask order."""
        return self.ask_order is not None

    # ── price helpers ─────────────────────────────────────────────────────────

    def get_bid_price(self) -> float | None:
        """Return the resting bid price, or None if no bid is active."""
        return self.bid_order.price if self.bid_order is not None else None

    def get_ask_price(self) -> float | None:
        """Return the resting ask price, or None if no ask is active."""
        return self.ask_order.price if self.ask_order is not None else None

    # ── size helpers ──────────────────────────────────────────────────────────

    def get_bid_size(self) -> int | None:
        """Return the resting bid size, or None if no bid is active."""
        return self.bid_order.size if self.bid_order is not None else None

    def get_ask_size(self) -> int | None:
        """Return the resting ask size, or None if no ask is active."""
        return self.ask_order.size if self.ask_order is not None else None

    # ── cancel helpers ────────────────────────────────────────────────────────

    def cancel_bid(self) -> None:
        """Explicitly cancel the resting bid order."""
        self.bid_order = None

    def cancel_ask(self) -> None:
        """Explicitly cancel the resting ask order."""
        self.ask_order = None

    def cancel_all(self) -> None:
        """Cancel both sides (e.g. end-of-day risk management)."""
        self.bid_order = None
        self.ask_order = None

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        bid_str = f"{self.bid_order.price:.4f}×{self.bid_order.size}" if self.bid_order else "None"
        ask_str = f"{self.ask_order.price:.4f}×{self.ask_order.size}" if self.ask_order else "None"
        return f"ExecutionEngine(bid={bid_str}, ask={ask_str})"
