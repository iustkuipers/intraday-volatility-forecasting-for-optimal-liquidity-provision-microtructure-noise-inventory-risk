import numpy as np
import pandas as pd


def ewma_variance_forecast(
    realized_var: pd.Series,
    lam: float = 0.94,
    initial_var: float | None = None,
) -> pd.Series:
    """
    EWMA variance forecast using realized variance as the innovation input:

        sigma_{t+1}^2 = lam * sigma_t^2 + (1 - lam) * RVAR_t

    Parameters
    ----------
    realized_var : pd.Series
        Realized variance series (RV_t in your notation, but variance form),
        indexed by timestamp.
    lam : float
        Decay parameter in (0,1). Higher means smoother/longer memory.
    initial_var : float | None
        Starting sigma_0^2. If None, uses first realized_var value.

    Returns
    -------
    pd.Series
        Variance forecasts aligned with realized_var index as sigma_hat2_t
        meaning: forecast *for next period* based on info up to t.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0,1)")

    rv = realized_var.astype(float).copy()
    if rv.isna().any():
        rv = rv.dropna()

    if rv.empty:
        return pd.Series(dtype=float, name="ewma_var")

    sigma2 = np.empty(len(rv), dtype=float)

    sigma2[0] = rv.iloc[0] if initial_var is None else float(initial_var)

    for i in range(1, len(rv)):
        sigma2[i] = lam * sigma2[i - 1] + (1.0 - lam) * rv.iloc[i - 1]

    out = pd.Series(sigma2, index=rv.index, name="ewma_var")
    return out


def ewma_volatility_forecast(
    realized_var: pd.Series,
    lam: float = 0.94,
    initial_var: float | None = None,
) -> pd.Series:
    """
    Convenience wrapper: returns sqrt(ewma_variance_forecast).
    """
    var_fc = ewma_variance_forecast(realized_var, lam=lam, initial_var=initial_var)
    return np.sqrt(var_fc).rename("ewma_vol")