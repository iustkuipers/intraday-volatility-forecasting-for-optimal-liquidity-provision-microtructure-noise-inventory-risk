import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.market_making.quoting import compute_spread, make_quotes


IDX = pd.date_range("2024-01-03 09:30:00", periods=5, freq="1min")


def test_compute_spread_linear():
    """Spread = k0 + k1 * sigma."""
    sigma = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=IDX)
    spread = compute_spread(sigma, k0=0.01, k1=10.0)

    expected = 0.01 + 10.0 * sigma
    assert np.allclose(spread.values, expected.values), f"Expected {expected.values}, got {spread.values}"


def test_compute_spread_min_floor():
    """Spread should never go below min_spread."""
    sigma = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=IDX)
    spread = compute_spread(sigma, k0=0.0, k1=0.0, min_spread=0.05)

    assert (spread >= 0.05).all(), "Spread fell below min_spread floor"


def test_compute_spread_clipped_non_negative():
    """Spread should always be >= 0 even with tiny sigma."""
    sigma = pd.Series([0.0001] * 5, index=IDX)
    spread = compute_spread(sigma, k0=0.0, k1=1.0, min_spread=0.0)

    assert (spread >= 0.0).all(), "Spread is negative"


def test_make_quotes_columns():
    """make_quotes should return bid, ask, delta columns."""
    mid = pd.Series([100.0, 101.0, 99.5, 100.5, 102.0], index=IDX)
    delta = pd.Series([0.05] * 5, index=IDX)

    quotes = make_quotes(mid, delta)

    assert "bid" in quotes.columns
    assert "ask" in quotes.columns
    assert "delta" in quotes.columns


def test_make_quotes_bid_ask_symmetry():
    """bid = mid - delta, ask = mid + delta."""
    mid = pd.Series([100.0, 101.0, 99.5, 100.5, 102.0], index=IDX)
    delta = pd.Series([0.05, 0.10, 0.08, 0.06, 0.07], index=IDX)

    quotes = make_quotes(mid, delta)

    assert np.allclose(quotes["bid"].values, (mid - delta).values), "bid != mid - delta"
    assert np.allclose(quotes["ask"].values, (mid + delta).values), "ask != mid + delta"


def test_make_quotes_ask_always_above_bid():
    """ask must always be strictly above bid (no locked/crossed quotes)."""
    mid = pd.Series([100.0, 101.0, 99.5, 100.5, 102.0], index=IDX)
    delta = pd.Series([0.05] * 5, index=IDX)

    quotes = make_quotes(mid, delta)

    assert (quotes["ask"] > quotes["bid"]).all(), "Crossed or locked quotes detected"


def test_make_quotes_index_preserved():
    """Output index should match input index."""
    mid = pd.Series([100.0] * 5, index=IDX)
    delta = pd.Series([0.05] * 5, index=IDX)

    quotes = make_quotes(mid, delta)

    assert quotes.index.equals(IDX), "Index not preserved"


if __name__ == "__main__":
    tests = [
        test_compute_spread_linear,
        test_compute_spread_min_floor,
        test_compute_spread_clipped_non_negative,
        test_make_quotes_columns,
        test_make_quotes_bid_ask_symmetry,
        test_make_quotes_ask_always_above_bid,
        test_make_quotes_index_preserved,
    ]
    print("=" * 60)
    print("TESTING src/market_making/quoting.py")
    print("=" * 60)
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
