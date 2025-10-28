"""
Volatility Indicators - CPU/Numba implementations

This module contains Numba JIT-compiled implementations for volatility indicators.
"""

import numpy as np
from numba import jit


__all__ = [
    "_natr_numba",
    "_trange_numba",
]


@jit(nopython=True, cache=True)
def _natr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled NATR calculation (in-place)

    Normalized Average True Range - ATR expressed as percentage of close price
    NATR = (ATR / close) * 100

    This normalization allows comparison of volatility across different price levels.
    A stock at $10 and $100 can be compared using NATR, while ATR would not be comparable.

    Formula:
    1. Calculate ATR using Wilder's smoothing
    2. Divide ATR by closing price
    3. Multiply by 100 to express as percentage
    """
    n = len(high)

    # Fill lookback period with NaN
    for i in range(timeperiod):
        output[i] = np.nan

    # Calculate True Range for all bars
    tr = np.empty(n, dtype=np.float64)

    # First TR value (no previous close)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Need at least timeperiod+1 bars
    if n < timeperiod + 1:
        return

    # Calculate first ATR as simple average of TR[1] through TR[timeperiod]
    atr_sum = 0.0
    for i in range(1, timeperiod + 1):
        atr_sum += tr[i]

    atr = atr_sum / timeperiod

    # Calculate first NATR
    if close[timeperiod] != 0.0:
        output[timeperiod] = (atr / close[timeperiod]) * 100.0
    else:
        output[timeperiod] = 0.0

    # Apply Wilder's smoothing for subsequent values
    for i in range(timeperiod + 1, n):
        atr = atr - (atr / timeperiod) + (tr[i] / timeperiod)
        # Calculate NATR as percentage
        if close[i] != 0.0:
            output[i] = (atr / close[i]) * 100.0
        else:
            output[i] = 0.0


@jit(nopython=True, cache=True)
def _trange_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, output: np.ndarray) -> None:
    """Numba-compiled TRANGE calculation (in-place)"""
    n = len(high)

    # First bar: just high - low
    output[0] = high[0] - low[0]

    # Subsequent bars: max of three ranges
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        output[i] = max(hl, hc, lc)
