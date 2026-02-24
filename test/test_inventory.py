import pandas as pd
import numpy as np

from src.market_making.inventory import apply_inventory_skew


def test_inventory_skew_direction():
    idx = pd.date_range("2024-01-01 09:30:00", periods=3, freq="1min")
    quotes = pd.DataFrame({"bid": [99, 99, 99], "ask": [101, 101, 101]}, index=idx)
    inv = pd.Series([0, 1, -2], index=idx)

    out = apply_inventory_skew(quotes, inv, phi=0.5, enforce_no_cross=False)

    # inv_skew = [0, 0.5, -1.0]
    assert np.isclose(out.loc[idx[0], "bid"], 99.0)
    assert np.isclose(out.loc[idx[1], "bid"], 99.5)
    assert np.isclose(out.loc[idx[2], "bid"], 98.0)

    assert np.isclose(out.loc[idx[0], "ask"], 101.0)
    assert np.isclose(out.loc[idx[1], "ask"], 101.5)
    assert np.isclose(out.loc[idx[2], "ask"], 100.0)


def test_enforce_no_cross():
    idx = pd.date_range("2024-01-01 09:30:00", periods=1, freq="1min")
    quotes = pd.DataFrame({"bid": [100.0], "ask": [100.01]}, index=idx)
    inv = pd.Series([0], index=idx)

    # Force negative spread via min_spread enforcement check
    out = apply_inventory_skew(quotes, inv, phi=0.0, enforce_no_cross=True, min_spread=0.05)
    assert out["ask"].iloc[0] >= out["bid"].iloc[0] + 0.05


def test_inventory_index_alignment_required():
    idx_q = pd.date_range("2024-01-01 09:30:00", periods=2, freq="1min")
    idx_i = pd.date_range("2024-01-01 09:31:00", periods=2, freq="1min")

    quotes = pd.DataFrame({"bid": [99, 99], "ask": [101, 101]}, index=idx_q)
    inv = pd.Series([0, 1], index=idx_i)

    try:
        apply_inventory_skew(quotes, inv, phi=0.1)
        assert False, "Expected ValueError due to misalignment"
    except ValueError:
        pass