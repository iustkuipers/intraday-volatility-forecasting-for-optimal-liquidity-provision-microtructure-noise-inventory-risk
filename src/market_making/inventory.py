import pandas as pd
import numpy as np


def apply_inventory_skew(
    quotes: pd.DataFrame,
    inventory: pd.Series,
    phi: float,
    *,
    enforce_no_cross: bool = True,
    min_spread: float = 0.0,
) -> pd.DataFrame:
    """
    Apply inventory-based quote shift:
        ask += phi * I_t
        bid += phi * I_t

    Parameters
    ----------
    quotes : pd.DataFrame
        Must contain columns: ['bid', 'ask'] (and optionally 'delta').
        Indexed by timestamp.
    inventory : pd.Series
        Inventory path I_t aligned with quotes index.
    phi : float
        Inventory skew coefficient.
    enforce_no_cross : bool
        If True, ensures ask >= bid + min_spread after skew.
    min_spread : float
        Minimum spread to enforce if enforce_no_cross is True.

    Returns
    -------
    pd.DataFrame
        New quotes with columns: ['bid', 'ask', ...] plus:
        - 'inv_skew' (phi * I_t)
        - 'bid_skewed', 'ask_skewed' (as bid/ask overwritten)
    """
    if "bid" not in quotes.columns or "ask" not in quotes.columns:
        raise ValueError("quotes must contain 'bid' and 'ask' columns")

    if not isinstance(inventory, pd.Series):
        raise ValueError("inventory must be a pd.Series")

    inv = inventory.reindex(quotes.index)
    if inv.isna().any():
        raise ValueError("inventory series misaligned with quotes index")

    out = quotes.copy()
    out["inv_skew"] = phi * inv.astype(float)

    out["bid"] = out["bid"] + out["inv_skew"]
    out["ask"] = out["ask"] + out["inv_skew"]

    if enforce_no_cross:
        # Ensure ask is at least bid + min_spread
        out["ask"] = np.maximum(out["ask"].values, (out["bid"] + min_spread).values)

    return out