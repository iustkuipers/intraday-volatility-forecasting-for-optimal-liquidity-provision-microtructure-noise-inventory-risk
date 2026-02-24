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


# ── tests ─────────────────────────────────────────────────────────────────────

def test_zero_alpha_returns_zero():
    """No adverse selection when alpha=0."""
    assert adverse_selection_penalty(mid=500.0, vol=0.01, alpha=0.0) == 0.0


def test_zero_vol_returns_zero():
    """No penalty when vol is zero."""
    assert adverse_selection_penalty(mid=500.0, vol=0.0, alpha=0.3) == 0.0


def test_negative_vol_returns_zero():
    """Non-positive vol → zero penalty."""
    assert adverse_selection_penalty(mid=500.0, vol=-0.01, alpha=0.3) == 0.0


def test_zero_mid_returns_zero():
    """Non-positive mid → zero penalty."""
    assert adverse_selection_penalty(mid=0.0, vol=0.01, alpha=0.3) == 0.0


def test_negative_alpha_returns_zero():
    """Negative alpha → zero penalty (guard clause)."""
    assert adverse_selection_penalty(mid=500.0, vol=0.01, alpha=-0.1) == 0.0


def test_penalty_scales_linearly_with_alpha():
    """Doubling alpha doubles the penalty."""
    p1 = adverse_selection_penalty(mid=100.0, vol=0.02, alpha=0.1)
    p2 = adverse_selection_penalty(mid=100.0, vol=0.02, alpha=0.2)
    assert abs(p2 - 2 * p1) < 1e-10


def test_penalty_scales_linearly_with_mid():
    """Doubling mid doubles the penalty."""
    p1 = adverse_selection_penalty(mid=100.0, vol=0.02, alpha=0.3)
    p2 = adverse_selection_penalty(mid=200.0, vol=0.02, alpha=0.3)
    assert abs(p2 - 2 * p1) < 1e-10


def test_penalty_scales_linearly_with_vol():
    """Doubling vol doubles the penalty."""
    p1 = adverse_selection_penalty(mid=100.0, vol=0.01, alpha=0.3)
    p2 = adverse_selection_penalty(mid=100.0, vol=0.02, alpha=0.3)
    assert abs(p2 - 2 * p1) < 1e-10


def test_penalty_is_positive_for_valid_inputs():
    """Penalty is strictly positive for all-positive parameters."""
    p = adverse_selection_penalty(mid=500.0, vol=0.015, alpha=0.25)
    assert p > 0.0


def test_penalty_exact_value():
    """Spot-check: alpha=0.5, vol=0.02, mid=100 → 0.5*0.02*100 = 1.0."""
    p = adverse_selection_penalty(mid=100.0, vol=0.02, alpha=0.5)
    assert abs(p - 1.0) < 1e-10


def test_inf_vol_returns_zero():
    """Non-finite vol → zero penalty."""
    assert adverse_selection_penalty(mid=500.0, vol=float("inf"), alpha=0.3) == 0.0


def test_nan_mid_returns_zero():
    """Non-finite mid → zero penalty."""
    assert adverse_selection_penalty(mid=float("nan"), vol=0.01, alpha=0.3) == 0.0