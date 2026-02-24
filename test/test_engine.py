import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.simulator.engine import MarketMakerEngine


IDX = pd.date_range("2024-01-01 09:30:00", periods=5, freq="1min")
MID = pd.Series([100.0, 101.0, 99.0, 100.0, 102.0], index=IDX)


def test_engine_runs_constant_delta():
    """Engine runs with scalar delta and returns expected columns."""
    df = pd.DataFrame({"mid": MID})
    mm = MarketMakerEngine()
    result = mm.run(df, delta=1.0)

    for col in ["inventory", "cash", "portfolio_value", "trade_count", "bid", "ask"]:
        assert col in result.columns, f"Missing column: {col}"


def test_engine_output_length():
    """Output length matches input length."""
    df = pd.DataFrame({"mid": MID})
    mm = MarketMakerEngine()
    result = mm.run(df, delta=0.5)

    assert len(result) == len(df), "Output length mismatch"


def test_engine_bid_ask_symmetry():
    """bid = mid - delta, ask = mid + delta."""
    df = pd.DataFrame({"mid": MID})
    delta = 0.5
    mm = MarketMakerEngine()
    result = mm.run(df, delta=delta)

    assert np.allclose(result["bid"].values, (MID - delta).values)
    assert np.allclose(result["ask"].values, (MID + delta).values)


def test_engine_series_delta():
    """Engine accepts a pd.Series as delta (time-varying spread)."""
    df = pd.DataFrame({"mid": MID})
    delta_series = pd.Series([0.05, 0.10, 0.08, 0.06, 0.07], index=IDX)
    mm = MarketMakerEngine()
    result = mm.run(df, delta=delta_series)

    assert "bid" in result.columns
    assert "ask" in result.columns
    assert np.allclose(result["bid"].values, (MID - delta_series).values)
    assert np.allclose(result["ask"].values, (MID + delta_series).values)


def test_engine_trade_count_non_negative():
    """trade_count should be monotonically non-decreasing and >= 0."""
    df = pd.DataFrame({"mid": MID})
    mm = MarketMakerEngine()
    result = mm.run(df, delta=0.5)

    assert (result["trade_count"] >= 0).all()
    assert result["trade_count"].is_monotonic_increasing, "trade_count should never decrease"


def test_engine_misaligned_delta_raises():
    """Misaligned delta Series should raise ValueError."""
    df = pd.DataFrame({"mid": MID})
    bad_delta = pd.Series([0.05, 0.10], index=pd.date_range("2025-01-01", periods=2, freq="1min"))
    mm = MarketMakerEngine()
    try:
        mm.run(df, delta=bad_delta)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_engine_no_mid_raises():
    """Missing 'mid' column should raise ValueError."""
    df = pd.DataFrame({"price": MID})
    mm = MarketMakerEngine()
    try:
        mm.run(df, delta=0.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_engine_runs_constant_delta,
        test_engine_output_length,
        test_engine_bid_ask_symmetry,
        test_engine_series_delta,
        test_engine_trade_count_non_negative,
        test_engine_misaligned_delta_raises,
        test_engine_no_mid_raises,
    ]
    print("=" * 60)
    print("TESTING src/simulator/engine.py")
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