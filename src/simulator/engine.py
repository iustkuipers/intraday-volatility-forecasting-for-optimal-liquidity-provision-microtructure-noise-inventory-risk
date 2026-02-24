import pandas as pd
import numpy as np

from src.simulator.fill_model import fill_probability
from src.simulator.adverse_selection import adverse_selection_penalty


class MarketMakerEngine:
    """
    General market making engine.

    Supports:
        - Constant spread (scalar delta)
        - Time-varying spread (delta_series)
        - Optional inventory skew (phi)
        - Probabilistic fills via fill_model.fill_probability
          (when volatility_series is supplied)

    Fill logic:
        Deterministic (default) : fill whenever next_mid crosses quote.
        Probabilistic           : crossing is a necessary but not sufficient
                                  condition; fill occurs with probability
                                  P = fill_probability(delta_t, vol_t).

    Tracks:
        inventory
        cash
        portfolio value
        trade count
    """

    def __init__(self, initial_cash: float = 0.0, random_state: int | None = None):
        self.initial_cash = initial_cash
        self.rng = np.random.default_rng(random_state)

    def run(
        self,
        df: pd.DataFrame,
        delta,
        phi: float = 0.0,
        volatility_series: pd.Series | None = None,
        alpha_as: float = 0.0,
    ) -> pd.DataFrame:
        """
        Run market making simulation.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'mid'.
        delta : float or pd.Series
            Spread parameter (half-spread).
        phi : float
            Inventory skew coefficient.
        volatility_series : pd.Series or None
            Per-bar volatility forecasts.  When supplied, fills are
            probabilistic: a mid-crossing is still required, but the
            fill is then accepted with probability
            fill_probability(delta_t, vol_t).
            When None, fills are deterministic (original behaviour).
        alpha_as : float
            Adverse-selection scaling coefficient (alpha in α·vol·mid).
            When > 0, volatility_series must be supplied.
            Each fill subtracts adverse_selection_penalty(mid, vol, alpha_as)
            from cash regardless of side.

        Returns
        -------
        pd.DataFrame
            Contains: mid, bid, ask, inventory, cash,
            portfolio_value, trade_count
        """

        if "mid" not in df.columns:
            raise ValueError("DataFrame must contain 'mid' column")

        data = df.copy()

        # Handle constant vs series delta
        if np.isscalar(delta):
            data["delta"] = delta
        else:
            if not isinstance(delta, pd.Series):
                raise ValueError("delta must be float or pd.Series")
            data["delta"] = delta.reindex(data.index)
            if data["delta"].isna().any():
                raise ValueError("delta series misaligned with data index")

        inventory = 0
        cash = self.initial_cash
        trade_count = 0

        inventories = []
        cash_series = []
        portfolio_values = []
        trades_series = []

        mids = data["mid"].values
        deltas = data["delta"].values

        # Align volatility array (None → deterministic mode)
        if volatility_series is not None:
            if not isinstance(volatility_series, pd.Series):
                raise ValueError("volatility_series must be a pd.Series")
            vols = volatility_series.reindex(data.index).values
            if np.isnan(vols).any():
                raise ValueError("volatility_series misaligned with data index")
        else:
            vols = None

        if alpha_as > 0.0 and vols is None:
            raise ValueError(
                "alpha_as > 0 requires volatility_series (needed to scale adverse selection)."
            )

        for t in range(len(data) - 1):

            # --- Base quotes ---
            base_bid = mids[t] - deltas[t]
            base_ask = mids[t] + deltas[t]

            # --- Inventory skew ---
            skew = phi * inventory
            bid = base_bid - skew
            ask = base_ask - skew

            next_mid = mids[t + 1]

            # --- Fill logic (probabilistic when vols supplied) ---
            if next_mid <= bid:
                p_fill = (fill_probability(deltas[t], vols[t])
                          if vols is not None else 1.0)
                if self.rng.random() < p_fill:
                    inventory += 1
                    cash -= bid
                    if alpha_as > 0.0:
                        cash -= adverse_selection_penalty(mids[t], vols[t], alpha_as)
                    trade_count += 1

            if next_mid >= ask:
                p_fill = (fill_probability(deltas[t], vols[t])
                          if vols is not None else 1.0)
                if self.rng.random() < p_fill:
                    inventory -= 1
                    cash += ask
                    if alpha_as > 0.0:
                        cash -= adverse_selection_penalty(mids[t], vols[t], alpha_as)
                    trade_count += 1

            inventories.append(inventory)
            cash_series.append(cash)
            portfolio_values.append(cash + inventory * mids[t])
            trades_series.append(trade_count)

        # Last step
        inventories.append(inventory)
        cash_series.append(cash)
        portfolio_values.append(cash + inventory * mids[-1])
        trades_series.append(trade_count)

        data["inventory"] = inventories
        data["cash"] = cash_series
        data["portfolio_value"] = portfolio_values
        data["trade_count"] = trades_series

        return data