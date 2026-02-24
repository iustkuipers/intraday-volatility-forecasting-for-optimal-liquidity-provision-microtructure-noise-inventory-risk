import numpy as np


def adverse_selection_penalty(mid: float, vol: float, alpha: float) -> float:
    """
    Expected adverse selection cost per fill.

    Parameters
    ----------
    mid : float
        Current mid price (dollars).
    vol : float
        Volatility proxy for the period (e.g., EWMA vol of returns).
        Interpreted as approximate return std per bar.
    alpha : float
        Scaling coefficient (dimensionless). Typical starting range:
        0.0 to 0.5.

    Returns
    -------
    float
        Dollar penalty per filled share.
    """
    if alpha <= 0.0:
        return 0.0
    if vol <= 0.0 or not np.isfinite(vol):
        return 0.0
    if mid <= 0.0 or not np.isfinite(mid):
        return 0.0

    return float(alpha * vol * mid)