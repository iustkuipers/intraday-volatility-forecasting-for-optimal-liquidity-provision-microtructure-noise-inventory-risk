import numpy as np
import pandas as pd

def compute_spread(sigma_hat: pd.Series, k0: float, k1: float, min_spread: float = 0.0) -> pd.Series:
    delta = k0 + k1 * sigma_hat
    delta = delta.clip(lower=min_spread)
    return delta

def make_quotes(mid: pd.Series, delta: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "bid": mid - delta,
        "ask": mid + delta,
        "delta": delta
    }, index=mid.index)