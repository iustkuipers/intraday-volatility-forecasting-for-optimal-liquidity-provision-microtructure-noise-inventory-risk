"""
simulator/accounting.py

Financial ledger for the market-making simulator.

Converts fill events (produced by FillModel) into running inventory, cash,
and PnL positions.  Every state variable is directly readable; no copies
or snapshots are made internally.

Tracked quantities
------------------
inventory       : int   — net share position (+long / -short)
cash            : float — cumulative cash flow from all fills
realized_pnl    : float — profit locked in by round-trip trades
portfolio_value : float — mark-to-market: cash + inventory × mid
avg_cost        : float — average entry price of current open position

What this module does NOT do
-----------------------------
- It does not decide fill prices or sizes   (fill model's job)
- It does not interact with market data     (market state's job)
- It does not enforce inventory limits      (strategy + engine's job)

Realized PnL accounting
-----------------------
Realized PnL tracks profit from completed round-trips using a FIFO
average-cost approach:

    When buying (bid fill):
        avg_cost = (avg_cost * old_inventory + price * size)
                   / (old_inventory + size)       [if inventory was long]

    When selling (ask fill) from a long position:
        realized_pnl += (price - avg_cost) * min(size, inventory)

    When inventory crosses zero, the residual portion opens a new short.
"""

from simulator.fill_model import Fill


class Accounting:
    """
    Running ledger for inventory, cash, and PnL.

    Usage
    -----
    acct = Accounting()

    for fill in fills:
        acct.apply_fill(fill)

    acct.mark_to_market(state.mid)
    print(acct.portfolio_value)
    """

    __slots__ = (
        "inventory",
        "cash",
        "realized_pnl",
        "portfolio_value",
        "avg_cost",
    )

    def __init__(self):
        self.inventory       = 0
        self.cash            = 0.0
        self.realized_pnl    = 0.0
        self.portfolio_value = 0.0
        self.avg_cost        = 0.0

    # ── fill processing ───────────────────────────────────────────────────────

    def apply_fill(self, fill: Fill) -> None:
        """
        Update inventory and cash from a single fill.

        Parameters
        ----------
        fill : Fill
            A Fill(side, price, size) produced by FillModel.
        """
        if fill.side == "bid":
            self._apply_bid(fill.price, fill.size)
        elif fill.side == "ask":
            self._apply_ask(fill.price, fill.size)
        else:
            raise ValueError(f"Unknown fill side: {fill.side!r}")

    def apply_fills(self, fills: list) -> None:
        """Convenience: apply an entire list of fills in order."""
        for fill in fills:
            self.apply_fill(fill)

    # ── mark-to-market ────────────────────────────────────────────────────────

    def mark_to_market(self, mid_price: float) -> None:
        """
        Recompute portfolio_value using the current mid price.

        portfolio_value = cash + inventory * mid_price
        """
        self.portfolio_value = self.cash + self.inventory * mid_price

    # ── private helpers ───────────────────────────────────────────────────────

    def _apply_bid(self, price: float, size: int) -> None:
        """Process a bid fill: we bought `size` shares at `price`."""
        old_inventory = self.inventory

        if old_inventory >= 0:
            # Adding to a long position (or opening from flat).
            # Update average cost as a weighted average.
            total = old_inventory + size
            self.avg_cost = (self.avg_cost * old_inventory + price * size) / total
        else:
            # Covering a short position.
            covered = min(size, -old_inventory)   # shares of short closed
            self.realized_pnl += (self.avg_cost - price) * covered

            residual = size - covered
            if residual > 0:
                # Position has crossed from short to long.
                self.avg_cost = price
            elif old_inventory + size == 0:
                # Position is now exactly flat.
                self.avg_cost = 0.0

        self.inventory += size
        self.cash      -= price * size

    def _apply_ask(self, price: float, size: int) -> None:
        """Process an ask fill: we sold `size` shares at `price`."""
        old_inventory = self.inventory

        if old_inventory <= 0:
            # Adding to a short position (or opening from flat).
            total = (-old_inventory) + size
            self.avg_cost = (self.avg_cost * (-old_inventory) + price * size) / total
        else:
            # Reducing a long position.
            closed = min(size, old_inventory)
            self.realized_pnl += (price - self.avg_cost) * closed

            residual = size - closed
            if residual > 0:
                # Position has crossed from long to short.
                self.avg_cost = price
            elif old_inventory - size == 0:
                # Position is now exactly flat.
                self.avg_cost = 0.0

        self.inventory -= size
        self.cash      += price * size

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Accounting("
            f"inv={self.inventory}, "
            f"cash={self.cash:.2f}, "
            f"realized={self.realized_pnl:.2f}, "
            f"portfolio={self.portfolio_value:.2f})"
        )
