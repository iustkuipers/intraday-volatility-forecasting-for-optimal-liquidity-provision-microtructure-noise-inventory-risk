import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.volatility.ewma import ewma_variance_forecast, ewma_volatility_forecast


def test_ewma_variance_forecast_alignment():
    idx = pd.to_datetime([
        "2024-01-03 09:30:00",
        "2024-01-03 09:31:00",
        "2024-01-03 09:32:00",
    ])
    rv = pd.Series([1.0, 4.0, 9.0], index=idx)  # realized variance

    lam = 0.5
    sigma2 = ewma_variance_forecast(rv, lam=lam, initial_var=1.0)

    # sigma2[0] = initial_var = 1
    # sigma2[1] = 0.5*1 + 0.5*rv[0]=0.5*1+0.5*1=1
    # sigma2[2] = 0.5*1 + 0.5*rv[1]=0.5*1+0.5*4=2.5
    assert np.isclose(sigma2.iloc[0], 1.0)
    assert np.isclose(sigma2.iloc[1], 1.0)
    assert np.isclose(sigma2.iloc[2], 2.5)


def test_ewma_vol_is_sqrt_var():
    idx = pd.to_datetime(["2024-01-03 09:30:00", "2024-01-03 09:31:00"])
    rv = pd.Series([0.0004, 0.0009], index=idx)

    var_fc = ewma_variance_forecast(rv, lam=0.9, initial_var=0.0004)
    vol_fc = ewma_volatility_forecast(rv, lam=0.9, initial_var=0.0004)

    assert np.isclose(vol_fc.iloc[0], np.sqrt(var_fc.iloc[0]))
    assert np.isclose(vol_fc.iloc[1], np.sqrt(var_fc.iloc[1]))


if __name__ == "__main__":
    tests = [
        test_ewma_variance_forecast_alignment,
        test_ewma_vol_is_sqrt_var,
    ]
    print("=" * 60)
    print("TESTING src/volatility/ewma.py")
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