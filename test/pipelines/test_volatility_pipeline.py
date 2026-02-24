import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from src.volatility.realized_vol import (
    realized_variance_from_returns,
    rolling_realized_volatility,
)
from src.volatility.ewma import ewma_variance_forecast


def test_full_volatility_pipeline():

    # Create synthetic tick data (1-min ticks)
    idx = pd.date_range("2024-01-01 09:30:00", periods=10, freq="1min")
    returns = pd.Series(
        [0.01, -0.02, 0.015, 0.0, 0.005, -0.01, 0.02, -0.01, 0.0, 0.01],
        index=idx
    )

    # Step 1: Realized variance (1min buckets)
    rvar = realized_variance_from_returns(returns, interval="1min")

    # Step 2: Rolling 5-min RV
    rolling_rv = rolling_realized_volatility(rvar, window="5min")

    # Step 3: EWMA forecast
    ewma_var = ewma_variance_forecast(rvar, lam=0.9)

    assert len(rvar) > 0
    assert len(rolling_rv) > 0
    assert len(ewma_var) > 0

    # Ensure no NaNs in forecast
    assert not ewma_var.isna().any()


if __name__ == "__main__":
    tests = [
        test_full_volatility_pipeline,
    ]
    print("=" * 60)
    print("TESTING volatility pipeline")
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