"""
simulator/strategy.py

Market-making strategies.  All strategies inherit BaseStrategy and return
a Quote on every compute_quote() call.

All parameters are DIMENSIONLESS — they scale automatically with the current
mid price and realized volatility, so the same γ and spread_frac work for a
$10 stock or a $500 stock without re-calibration.

Parameter guide
---------------
spread_frac        — half-spread as a fraction of mid
                     e.g. 0.0001 → 1 bp → $0.047 at SPY $470
                     computed: half_spread = spread_frac × mid

risk_aversion (γ)  — dimensionless inventory-skew coefficient
                     e.g. 0.1 → reservation shifts $0.047 per 100 shares at SPY
                     computed: skew = γ × σ × mid × inventory

max_position_value — hard cap expressed in dollars, not shares
                     e.g. 50_000 → max_inventory = 50_000 / mid ≈ 106 shares at SPY
                     computed: max_inv = max_position_value / mid

vol_spread_coef    — (VolatilityAdaptiveStrategy only)
                     extra dimensionless multiplier on σ added to spread_frac
                     e.g. 2.0 → at σ=0.001 adds 0.2 bp extra spread

Available strategies
--------------------
ConstantSpreadStrategy      — baseline: price-normalised spread + inventory skew
VolatilityAdaptiveStrategy  — spread widens further with realised volatility
"""

from .order import Quote
from .strategy_base import BaseStrategy
from collections import deque
import statistics

_VOL_FLOOR       = 1e-6   # prevents zero-vol edge cases
_VOL_CAP         = 0.005  # default ceiling: ~5× typical SPY intraday vol (0.001)
                          # prevents bad ticks inflating realized_vol 50× and blowing
                          # out skew / spread.  Overridable per strategy instance.
_VOL_PERSISTENCE = 3      # default median-filter window: a spike must persist for
                          # this many consecutive events before it affects quoting.
_SKEW_CAP        = 0.95   # skew cannot exceed 95% of half_spread.
                          # Guarantees bid_quote < mid < ask_quote at all times:
                          # reservation = mid − skew; with |skew| < half_spread,
                          # bid = reservation − half_spread < mid and
                          # ask = reservation + half_spread > mid always hold.


class ConstantSpreadStrategy(BaseStrategy):
    """
    Baseline market-making strategy with price-normalised parameters.

    Formulas
    --------
    half_spread  = spread_frac × mid
    skew         = risk_aversion × vol × mid × inventory
    reservation  = mid − skew
    bid          = reservation − half_spread
    ask          = reservation + half_spread
    max_inv      = max_position_value / mid          (in shares, recomputed each tick)

    Parameters
    ----------
    spread_frac : float
        Half-spread as a fraction of mid price.  1 bp = 0.0001.
    order_size : int
        Shares quoted on each side (fixed lot size).
    risk_aversion : float
        Dimensionless skew coefficient γ.  Controls how aggressively quotes
        shift with inventory.  Typical range: 0.01 – 1.0.
    max_position_value : float
        Hard dollar cap on gross position.  Converted to shares each tick
        via max_inv = max_position_value / mid.
    vol_cap : float
        Hard ceiling on realized_vol fed to skew/spread formulas.
        Prevents single bad ticks in the vol series from blowing out quotes.
        Default: 0.005 (~5× typical SPY intraday vol).
    vol_persistence : int
        Median-filter window over recent vol readings.  A spike must appear
        in the majority of the last `vol_persistence` events before it
        influences quoting.  With the default of 3, a single-event spike is
        outvoted by the 2 surrounding normal readings.  Two consecutive spikes
        push the median to the spike level — confirming it is real.
    """

    def __init__(
        self,
        spread_frac: float,
        order_size: int,
        risk_aversion: float,
        max_position_value: float,
        vol_cap: float = _VOL_CAP,
        vol_persistence: int = _VOL_PERSISTENCE,
    ):
        self.spread_frac        = spread_frac
        self.order_size         = order_size
        self.risk_aversion      = risk_aversion
        self.max_position_value = max_position_value
        self.vol_cap            = vol_cap
        self.vol_persistence    = vol_persistence
        self._vol_buf           = deque(maxlen=vol_persistence)

    def compute_quote(self, state, inventory: int) -> Quote:
        mid     = state.ref_mid()
        raw_vol = max(state.realized_vol if state.realized_vol is not None else _VOL_FLOOR, _VOL_FLOOR)
        raw_vol = min(raw_vol, self.vol_cap)          # 1. hard cap  — kills extreme outliers
        self._vol_buf.append(raw_vol)
        vol     = statistics.median(self._vol_buf)    # 2. median filter — confirms persistence

        half_spread = self.spread_frac * mid
        raw_skew    = self.risk_aversion * vol * mid * inventory
        # Cap skew so bid_quote < mid < ask_quote always holds.
        # Without this, large inventory pushes reservation above mid, making
        # bid_quote > mid — we would buy above fair value on every fill.
        cap         = half_spread * _SKEW_CAP
        skew        = max(-cap, min(cap, raw_skew))
        reservation = mid - skew

        max_inv = int(self.max_position_value / mid)

        quote = Quote(
            bid_price = reservation - half_spread,
            ask_price = reservation + half_spread,
            bid_size  = self.order_size,
            ask_size  = self.order_size,
        )

        if inventory >= max_inv:
            quote.disable_bid()
        if inventory <= -max_inv:
            quote.disable_ask()

        return quote


class VolatilityAdaptiveStrategy(BaseStrategy):
    """
    Volatility-aware strategy with price-normalised parameters.

    Identical to ConstantSpreadStrategy but the half-spread widens further
    when realised volatility is elevated:

    half_spread = (spread_frac + vol_spread_coef × vol) × mid

    A higher vol_spread_coef makes the strategy more defensive during
    volatile periods — compensating for increased adverse-selection risk.

    Parameters
    ----------
    spread_frac : float
        Minimum half-spread fraction (at zero volatility).
    order_size : int
        Shares quoted on each side.
    risk_aversion : float
        Dimensionless skew coefficient γ.
    max_position_value : float
        Hard dollar cap on gross position.
    vol_spread_coef : float
        Dimensionless multiplier: extra spread fraction added per unit of σ.
        e.g. 2.0 → at σ=0.001 adds 0.2 bp on top of spread_frac.
    vol_cap : float
        Hard ceiling on realized_vol.  Same purpose as in ConstantSpreadStrategy.
    vol_persistence : int
        Median-filter window.  Same purpose as in ConstantSpreadStrategy.
    """

    def __init__(
        self,
        spread_frac: float,
        order_size: int,
        risk_aversion: float,
        max_position_value: float,
        vol_spread_coef: float,
        vol_cap: float = _VOL_CAP,
        vol_persistence: int = _VOL_PERSISTENCE,
    ):
        self.spread_frac        = spread_frac
        self.order_size         = order_size
        self.risk_aversion      = risk_aversion
        self.max_position_value = max_position_value
        self.vol_spread_coef    = vol_spread_coef
        self.vol_cap            = vol_cap
        self.vol_persistence    = vol_persistence
        self._vol_buf           = deque(maxlen=vol_persistence)

    def compute_quote(self, state, inventory: int) -> Quote:
        mid     = state.ref_mid()
        raw_vol = max(state.realized_vol if state.realized_vol is not None else _VOL_FLOOR, _VOL_FLOOR)
        raw_vol = min(raw_vol, self.vol_cap)
        self._vol_buf.append(raw_vol)
        vol     = statistics.median(self._vol_buf)

        half_spread = (self.spread_frac + self.vol_spread_coef * vol) * mid
        raw_skew    = self.risk_aversion * vol * mid * inventory
        cap         = half_spread * _SKEW_CAP
        skew        = max(-cap, min(cap, raw_skew))
        reservation = mid - skew

        max_inv = int(self.max_position_value / mid)

        quote = Quote(
            bid_price = reservation - half_spread,
            ask_price = reservation + half_spread,
            bid_size  = self.order_size,
            ask_size  = self.order_size,
        )

        if inventory >= max_inv:
            quote.disable_bid()
        if inventory <= -max_inv:
            quote.disable_ask()

        return quote
