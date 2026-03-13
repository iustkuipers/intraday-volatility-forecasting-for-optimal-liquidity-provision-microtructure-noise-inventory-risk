"""
simulator/strategy_base.py

Defines the plug-in interface every strategy must implement.
The simulator only ever calls compute_quote() — strategies are fully
interchangeable without touching any other module.
"""

from .order import Quote


class BaseStrategy:
    """
    Abstract base for all market-making strategies.

    Subclasses must override compute_quote().
    """

    def compute_quote(self, market_state, inventory: int) -> Quote:
        """
        Compute the desired two-sided quote given current market state.

        Parameters
        ----------
        market_state : MarketState
            Current observable market snapshot (read-only).
        inventory : int
            Current net inventory (positive = long, negative = short).

        Returns
        -------
        Quote
            Desired bid/ask prices and sizes.  Sides may be disabled.
        """
        raise NotImplementedError
