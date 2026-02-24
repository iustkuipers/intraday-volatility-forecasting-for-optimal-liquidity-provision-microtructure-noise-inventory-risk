import numpy as np
import pandas as pd


def realized_variance_from_returns(
    log_returns: pd.Series,
    interval: str = "1min",
    min_obs: int = 1,
) -> pd.Series:
    """
    Compute realized variance per interval from tick log-returns:
        RVAR_t = sum_{i in interval} r_{t,i}^2

    Parameters
    ----------
    log_returns : pd.Series
        Tick-level log returns indexed by timestamp.
    interval : str
        Resampling frequency (e.g. "1min", "5min").
    min_obs : int
        Minimum number of observations required per bucket to keep it.

    Returns
    -------
    pd.Series
        Realized variance per interval, indexed by interval timestamps.
    """
    if not isinstance(log_returns.index, pd.DatetimeIndex):
        raise TypeError("log_returns must have a DatetimeIndex")

    r2 = (log_returns.astype(float) ** 2)
    counts = r2.resample(interval).count()
    rvar = r2.resample(interval).sum()

    # Drop buckets with too few observations
    rvar = rvar.where(counts >= min_obs)
    return rvar.dropna()


def realized_volatility_from_returns(
    log_returns: pd.Series,
    interval: str = "1min",
    min_obs: int = 1,
) -> pd.Series:
    """
    Compute realized volatility per interval:
        RV_t = sqrt(sum r_{t,i}^2)

    Returns volatility (not variance).
    """
    rvar = realized_variance_from_returns(log_returns, interval=interval, min_obs=min_obs)
    return np.sqrt(rvar).rename("realized_vol")


def rolling_realized_volatility(
    realized_var: pd.Series,
    window: str = "60min",
    min_periods: int = 1,
    annualize: bool = False,
    periods_per_year: float = 252 * 390,  # 252 trading days * 390 minutes
) -> pd.Series:
    """
    Rolling realized volatility from realized variance.

    If realized_var is per-minute realized variance, then:
        rolling_rvar_t = sum_{last window} RVAR
        rolling_rv_t   = sqrt(rolling_rvar_t)

    Parameters
    ----------
    realized_var : pd.Series
        Realized variance series (e.g. per minute), DatetimeIndex.
    window : str
        Time-based rolling window (e.g. "30min", "60min", "1D").
    min_periods : int
        Minimum number of points in the rolling window.
    annualize : bool
        Whether to annualize the resulting volatility.
    periods_per_year : float
        Scaling factor for annualization. If series is per-minute variance,
        default is 252*390.

    Returns
    -------
    pd.Series
        Rolling realized volatility.
    """
    if not isinstance(realized_var.index, pd.DatetimeIndex):
        raise TypeError("realized_var must have a DatetimeIndex")

    roll_rvar = realized_var.rolling(window=window, min_periods=min_periods).sum()
    roll_rv = np.sqrt(roll_rvar)

    if annualize:
        # If realized_var is variance per minute, then rolling variance is in "per minute" units.
        # Annualize volatility by sqrt(periods_per_year).
        roll_rv = roll_rv * np.sqrt(periods_per_year)

    return roll_rv.rename(f"rv_{window}")