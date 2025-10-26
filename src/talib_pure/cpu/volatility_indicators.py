"""
Volatility Indicators - CPU/Numba implementations

This module contains Numba JIT-compiled implementations for volatility indicators.
"""

import numpy as np
from numba import jit


__all__ = [
    "_trange_numba",
]


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
