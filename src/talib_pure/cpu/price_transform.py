"""
Price Transform - CPU/Numba implementations

This module contains Numba JIT-compiled implementations for price transformation functions.
"""

import numpy as np
from numba import jit


__all__ = [
    "_medprice_numba",
    "_midpoint_numba",
    "_midprice_numba",
]


@jit(nopython=True, cache=True)
def _medprice_numba(high: np.ndarray, low: np.ndarray, output: np.ndarray) -> None:
    """
    Numba-compiled MEDPRICE calculation (in-place)

    Formula: MEDPRICE = (High + Low) / 2
    """
    n = len(high)
    for i in range(n):
        output[i] = (high[i] + low[i]) / 2.0


@jit(nopython=True, cache=True)
def _midpoint_numba(data: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MIDPOINT calculation (in-place)

    Formula: MIDPOINT = (MAX + MIN) / 2 over timeperiod
    """
    n = len(data)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate midpoint for each window
    for i in range(timeperiod - 1, n):
        max_val = data[i]
        min_val = data[i]
        for j in range(i - timeperiod + 1, i + 1):
            if data[j] > max_val:
                max_val = data[j]
            if data[j] < min_val:
                min_val = data[j]
        output[i] = (max_val + min_val) / 2.0


@jit(nopython=True, cache=True)
def _midprice_numba(high: np.ndarray, low: np.ndarray, timeperiod: int, output: np.ndarray) -> None:
    """
    Numba-compiled MIDPRICE calculation (in-place)

    Formula: MIDPRICE = (MAX(high, period) + MIN(low, period)) / 2
    """
    n = len(high)

    # Fill lookback period with NaN
    for i in range(timeperiod - 1):
        output[i] = np.nan

    # Calculate midprice for each window
    for i in range(timeperiod - 1, n):
        max_high = high[i]
        min_low = low[i]
        for j in range(i - timeperiod + 1, i + 1):
            if high[j] > max_high:
                max_high = high[j]
            if low[j] < min_low:
                min_low = low[j]
        output[i] = (max_high + min_low) / 2.0
