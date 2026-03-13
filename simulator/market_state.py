"""
simulator/market_state.py

Converts a single event-stream row into a clean, attribute-based market
snapshot.  Contains no strategy logic, no accounting, no history.
"""

import math


class MarketState:
    """Current observable market snapshot derived from one event row."""

    __slots__ = (
        # event metadata
        "timestamp", "event_type",
        # quote state
        "bid", "ask", "mid", "spread", "rel_spread",
        "bid_size", "ask_size", "depth", "imbalance",
        # trade state  (NaN / 0 on quote events)
        "trade_price", "trade_size", "trade_direction", "signed_volume",
        # microstructure signals
        "realized_vol",
        "trade_intensity", "trade_vol_intensity",
        "queue_fraction_bid", "queue_fraction_ask",
        # persistent: last confirmed trade price (survives quote-only events)
        "last_trade_price",
    )

    def __init__(self):
        for attr in self.__slots__:
            object.__setattr__(self, attr, None)

    # ── update ────────────────────────────────────────────────────────────────

    def update_from_event(self, row) -> None:
        """Load state from a single itertuples() row of the event stream."""
        self.timestamp   = row.timestamp
        self.event_type  = row.event_type

        self.bid         = row.bid
        self.ask         = row.ask
        self.mid         = row.mid
        self.spread      = row.spread
        self.rel_spread  = row.rel_spread

        self.bid_size    = row.bid_size
        self.ask_size    = row.ask_size
        self.depth       = row.depth
        self.imbalance   = row.imbalance

        self.trade_price     = row.trade_price
        self.trade_size      = row.trade_size
        self.trade_direction = row.trade_direction
        self.signed_volume   = row.signed_volume

        self.realized_vol        = row.realized_vol
        self.trade_intensity     = row.trade_intensity
        self.trade_vol_intensity = row.trade_vol_intensity
        self.queue_fraction_bid  = row.queue_fraction_bid
        self.queue_fraction_ask  = row.queue_fraction_ask
        # last_trade_price is NOT updated here — it is committed by the
        # simulator AFTER fill evaluation so that quotes computed this tick
        # are anchored to the previous fill price, not the current one.

    # ── event type helpers ────────────────────────────────────────────────────

    def is_trade(self) -> bool:
        return self.event_type == "trade"

    def is_quote(self) -> bool:
        return self.event_type == "quote"

    # ── convenience accessors (aliases keep strategy code readable) ───────────

    def best_bid(self) -> float:
        return self.bid

    def best_ask(self) -> float:
        return self.ask

    def mid_price(self) -> float:
        return self.mid

    def current_spread(self) -> float:
        return self.spread

    def ref_mid(self) -> float:
        """
        Outlier-guarded mid price for strategy quoting.

        Uses the NBBO quote mid under normal conditions — this is the primary
        reference for where fair value is.  When a tick's mid deviates more
        than 2% from the last confirmed trade price (stale exchange, quote
        stuffer, data error), it is clamped to that trade price instead.

        Falls back to quote mid before the first trade of the session.
        """
        if self.last_trade_price is None or self.mid is None:
            return self.mid
        if 0.98 <= (self.mid / self.last_trade_price) <= 1.02:
            return self.mid          # normal — quote mid is reliable
        return self.last_trade_price # bad tick — snap to last real transaction

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MarketState("
            f"ts={self.timestamp}, type={self.event_type}, "
            f"bid={self.bid}, ask={self.ask}, mid={self.mid}, "
            f"vol={self.realized_vol:.6f})"
            if self.realized_vol is not None
            else f"MarketState(uninitialised)"
        )
