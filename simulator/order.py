"""
simulator/order.py

Defines the Quote object — the output of every strategy call.
The execution engine reads this; it never contains strategy logic.
"""


class Quote:
    """
    A two-sided market-making quote.

    Both sides can be independently disabled (e.g. when inventory limits
    are reached).  The execution engine treats an inactive side as cancelled.
    """

    __slots__ = (
        "bid_price", "ask_price",
        "bid_size", "ask_size",
        "bid_active", "ask_active",
    )

    def __init__(
        self,
        bid_price=None,
        ask_price=None,
        bid_size=0,
        ask_size=0,
        bid_active=True,
        ask_active=True,
    ):
        self.bid_price  = bid_price
        self.ask_price  = ask_price
        self.bid_size   = bid_size
        self.ask_size   = ask_size
        self.bid_active = bid_active
        self.ask_active = ask_active

    # ── helpers ───────────────────────────────────────────────────────────────

    def disable_bid(self) -> None:
        self.bid_active = False

    def disable_ask(self) -> None:
        self.ask_active = False

    def is_valid(self) -> bool:
        """True when both prices are set and bid < ask (no crossed quote)."""
        return (
            self.bid_price is not None
            and self.ask_price is not None
            and self.bid_price < self.ask_price
        )

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        bid_str = f"{self.bid_price:.4f}×{self.bid_size}" if self.bid_active else "OFF"
        ask_str = f"{self.ask_price:.4f}×{self.ask_size}" if self.ask_active else "OFF"
        return f"Quote(bid={bid_str}, ask={ask_str})"
