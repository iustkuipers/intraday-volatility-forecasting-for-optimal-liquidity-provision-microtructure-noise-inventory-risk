import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.volatility.realized_vol import (
    realized_variance_from_returns,
    realized_volatility_from_returns,
    rolling_realized_volatility,
)


def test_realized_variance_simple():
    # 2 minutes, 3 ticks each minute
    idx = pd.to_datetime([
        "2024-01-03 09:30:00.000",
        "2024-01-03 09:30:10.000",
        "2024-01-03 09:30:20.000",
        "2024-01-03 09:31:00.000",
        "2024-01-03 09:31:10.000",
        "2024-01-03 09:31:20.000",
    ])
    r = pd.Series([0.01, -0.02, 0.01, 0.00, 0.03, -0.04], index=idx)

    rvar = realized_variance_from_returns(r, interval="1min")
    # Minute 09:30: sum squares = 0.0001 + 0.0004 + 0.0001 = 0.0006
    # Minute 09:31: sum squares = 0 + 0.0009 + 0.0016 = 0.0025
    assert np.isclose(rvar.iloc[0], 0.0006)
    assert np.isclose(rvar.iloc[1], 0.0025)


def test_realized_vol_is_sqrt_rvar():
    idx = pd.to_datetime([
        "2024-01-03 09:30:00.000",
        "2024-01-03 09:30:30.000",
    ])
    r = pd.Series([0.03, 0.04], index=idx)

    rvar = realized_variance_from_returns(r, interval="1min")
    rv = realized_volatility_from_returns(r, interval="1min")

    assert np.isclose(rv.iloc[0], np.sqrt(rvar.iloc[0]))


def test_rolling_realized_vol():
    # 3 minutes of realized variance
    idx = pd.to_datetime([
        "2024-01-03 09:30:00",
        "2024-01-03 09:31:00",
        "2024-01-03 09:32:00",
    ])
    rvar = pd.Series([0.0001, 0.0004, 0.0009], index=idx)

    # rolling 2 minutes: at 09:31 => 0.0001+0.0004 = 0.0005
    # at 09:32 => 0.0004+0.0009 = 0.0013
    roll = rolling_realized_volatility(rvar, window="2min", min_periods=2)

    assert np.isclose(roll.loc["2024-01-03 09:31:00"], np.sqrt(0.0005))
    assert np.isclose(roll.loc["2024-01-03 09:32:00"], np.sqrt(0.0013))


if __name__ == "__main__":
    tests = [
        test_realized_variance_simple,
        test_realized_vol_is_sqrt_rvar,
        test_rolling_realized_vol,
    ]
    print("=" * 60)
    print("TESTING src/volatility/realized_vol.py")
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